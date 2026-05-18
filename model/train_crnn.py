""" - 2gpus, blah blah one
Lipi Snap — CRNN+CTC Training Pipeline
Word-level Ranjana Script Recognition (2x GPU + 3x3 Arch + Exact Match Logging)
"""

import os
import sys
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# --- Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- Configuration ---
IMG_HEIGHT = 64
IMG_WIDTH  = 512   
MAX_LR     = 3e-3
EPOCHS     = 99
GRAD_CLIP  = 5.0   

# --- Device ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

USE_AMP = (DEVICE.type == 'cuda')

# --- Paths & Env Config ---
if os.path.exists('/kaggle/input'):
    ENV         = "kaggle"
    MODEL_DIR   = Path("/kaggle/working")
    NUM_WORKERS = 4
    BATCH_SIZE  = 512
    LABELS_CSV  = '/kaggle/input/datasets/anisharyal09/ranjana-ls/labels.csv'
    IMAGES_DIR  = '/kaggle/input/datasets/anisharyal09/ranjana-ls/images/images'

elif 'google.colab' in sys.modules or os.path.exists('/content'):
    ENV         = "colab"
    MODEL_DIR   = Path("/content/drive/MyDrive/Lipi-Snap/model")
    NUM_WORKERS = 2
    BATCH_SIZE  = 256
    LABELS_CSV  = Path("/content/labels.csv")
    IMAGES_DIR  = Path("/content/data/images")

else:
    ENV, NUM_WORKERS = "local", 2
    BATCH_SIZE   = 128
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    MODEL_DIR    = PROJECT_ROOT / "model"
    DATA_DIR     = PROJECT_ROOT / "data" / "synthetic_words"
    LABELS_CSV   = DATA_DIR / "labels.csv"
    IMAGES_DIR   = DATA_DIR / "images"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = MODEL_DIR / "best_crnn.pth"

# --- Vocabulary ---
VOCAB_STR = "ँंःअआइईउऊऋएऐऒओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहाािीुूृेैोौ्०१२३४५६७८९"
CHARS = sorted(list(set(VOCAB_STR)))

class LabelConverter:
    def __init__(self, chars):
        self.chars = ["<blank>"] + chars
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}

    def encode(self, text):
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

    def decode(self, indices):
        return "".join([self.idx_to_char[i] for i in indices if i != 0])

    def num_classes(self):
        return len(self.chars)

converter = LabelConverter(CHARS)

# --- Preprocessing ---
def process_image_crnn(image_path):
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None: raise ValueError("Invalid image")

        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if np.sum(binary == 255) > np.sum(binary == 0):
            binary = cv2.bitwise_not(binary)

        h, w = binary.shape
        new_w = min(int(IMG_HEIGHT * (w / h)), IMG_WIDTH)
        resized = cv2.resize(binary, (new_w, IMG_HEIGHT), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
        canvas[:, :new_w] = resized

        return Image.fromarray(canvas)
    except Exception:
        return Image.new('L', (IMG_WIDTH, IMG_HEIGHT), 0)

# --- Dataset ---
class RanjanaCRNNDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.img_dir   = Path(img_dir)
        self.transform = transform
        self.samples   = []

        if not Path(csv_path).exists():
            print(f"⚠️  labels.csv not found: {csv_path}")
            return

        with open(csv_path, 'r', encoding='utf-8') as f:
            for row in csv.reader(f):
                if len(row) < 2: continue
                fname, label = row[0].strip(), row[1].strip()
                if fname.lower() in ('filename', 'file', 'image'): continue
                if label: self.samples.append((fname, label))

        print(f"✅ Loaded {len(self.samples)} samples")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        image = process_image_crnn(self.img_dir / fname)
        if self.transform: image = self.transform(image)
        encoded = converter.encode(label)
        return image, torch.IntTensor(encoded)


def crnn_collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    targets = torch.cat([l.long() for l in labels], 0)
    return images, targets, target_lengths

# --- Model ---
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Optimized 3x3 Architecture
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )

        self.dense_pre_rnn = nn.Sequential(
            nn.Linear(128 * 16, 64), nn.ReLU(True), nn.Dropout(0.2)
        )

        self.rnn1 = nn.LSTM(64,  256, bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(512, 128, bidirectional=True, batch_first=True)
        self.rnn3 = nn.LSTM(256, 128, bidirectional=True, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(True), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()

        conv = conv.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
        feats  = self.dense_pre_rnn(conv)
        out, _ = self.rnn1(feats)
        out, _ = self.rnn2(out)
        out, _ = self.rnn3(out)
        
        # RETURN [B, T, C] so DataParallel gathers batches correctly on dim 0
        return self.classifier(out)

# --- Loss + Decode ---
def compute_ctc_loss(criterion, preds, targets, target_lengths):
    # preds is [B, T, C]. Permute to [T, B, C] for CTCLoss computation
    preds = preds.permute(1, 0, 2)
    log_probs = torch.nn.functional.log_softmax(preds, dim=2)
    input_lengths = torch.full((preds.size(1),), preds.size(0), dtype=torch.long)
    
    if DEVICE.type == 'mps':
        return criterion(
            log_probs.cpu(), targets.cpu(),
            input_lengths.cpu(), target_lengths.cpu()
        )
    return criterion(log_probs, targets, input_lengths, target_lengths)

def greedy_decode(preds):
    _, idxs = preds.max(2)
    idxs = idxs.cpu() 
    
    results = []
    for seq in idxs:
        decoded, last = [], -1
        for i in seq.tolist():
            if i != last and i != 0: 
                decoded.append(i)
            last = i
        results.append(decoded)
    return results

# --- Training Helpers ---
class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class AddGaussianNoise:
    def __init__(self, std=0.05):
        self.std = std
    def __call__(self, tensor):
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std, 0.0, 1.0)

# --- Training ---
def train():
    train_transform = transforms.Compose([
        transforms.RandomApply([transforms.RandomAffine(degrees=2, translate=(0.01, 0.02), scale=(0.95, 1.05), shear=2)], p=0.9),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.5),
        transforms.ToTensor(),
        transforms.RandomApply([AddGaussianNoise(std=0.05)], p=0.3),
        transforms.Normalize((0.5,), (0.5,))
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = RanjanaCRNNDataset(LABELS_CSV, IMAGES_DIR, transform=None)
    if len(dataset) == 0: return

    train_size = int(0.9 * len(dataset))
    val_size   = len(dataset) - train_size
    train_sub, val_sub = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED)
    )

    train_ds = TransformSubset(train_sub, train_transform)
    val_ds   = TransformSubset(val_sub,   val_transform)
    kwargs = {'collate_fn': crnn_collate_fn, 'num_workers': NUM_WORKERS, 'pin_memory': (DEVICE.type == 'cuda')}
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  **kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    # --- Multi-GPU Support ---
    model = CRNN(converter.num_classes())
    if torch.cuda.device_count() > 1:
        print(f"🔥 Found {torch.cuda.device_count()} GPUs! Using DataParallel.")
        model = nn.DataParallel(model)
    model = model.to(DEVICE)

    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, steps_per_epoch=len(train_loader), epochs=EPOCHS,
    )

    scaler = torch.amp.GradScaler('cuda') if USE_AMP else None

    # Resume from best checkpoint
    start_epoch, best_loss = 1, float('inf')
    if BEST_MODEL_PATH.exists():
        try:
            ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
            
            # Handle DataParallel loading
            state = ckpt['model_state_dict']
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(state)
            else:
                model.load_state_dict(state)
                
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                
            start_epoch = ckpt.get('epoch', 0) + 1
            best_loss   = ckpt.get('loss', float('inf'))
            print(f"🔄 Resumed from epoch {start_epoch - 1}")
        except Exception as e:
            print(f"⚠️  Could not resume checkpoint: {e}")

    print(f"🚀 Training on {DEVICE.type.upper()} | {len(train_ds)} train / {len(val_ds)} val")

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for imgs, targets, target_lengths in pbar:
            imgs    = imgs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if USE_AMP:
                with torch.amp.autocast('cuda'):
                    preds = model(imgs)
                    loss  = compute_ctc_loss(criterion, preds, targets, target_lengths)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                
                if scale_before <= scaler.get_scale():
                    scheduler.step()
            else:
                preds = model(imgs)
                loss  = compute_ctc_loss(criterion, preds, targets, target_lengths)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        # --- Validation (Exact Match) ---
        val_loss, acc = validate(model, val_loader, criterion)
        avg_train_loss = running_loss / len(train_loader)
        
        print(f"Epoch {epoch:>3} | Train Loss {avg_train_loss:.4f} | Val Loss {val_loss:.4f} | Acc {acc:.2f}%")

        if val_loss < best_loss:
            best_loss = val_loss
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({
                'epoch'                : epoch,
                'model_state_dict'     : raw_model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'scheduler_state_dict' : scheduler.state_dict(),
                'loss'                 : val_loss,
                'accuracy'             : acc,
                'chars'                : CHARS,
            }, BEST_MODEL_PATH)
            print(f"  💾 Saved best model (val_loss={val_loss:.4f})")


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0

    for i, (imgs, targets, target_lengths) in enumerate(loader):
        imgs    = imgs.to(DEVICE)
        targets = targets.to(DEVICE)

        preds    = model(imgs)
        loss     = compute_ctc_loss(criterion, preds, targets, target_lengths)
        loss_sum += loss.item()

        decoded = greedy_decode(preds)
        targets_cpu = targets.cpu()
        gt_seqs, offset = [], 0
        for l in target_lengths:
            gt_seqs.append(targets_cpu[offset : offset + l.item()].tolist())
            offset += l.item()

        for pred_seq, gt_seq in zip(decoded, gt_seqs):
            if pred_seq == gt_seq: correct += 1
            total += 1

        if i == 0:
            print("\n" + "="*45)
            print("🧐 EPOCH LEARNINGS & SAMPLES")
            print("="*45)
            num_samples = min(5, len(decoded))
            for s in range(num_samples):
                pred_text  = converter.decode(decoded[s])
                gt_text    = converter.decode(gt_seqs[s])
                match_icon = "✅" if pred_text == gt_text else "❌"
                print(f"  {match_icon} GT:   {gt_text}")
                print(f"     Pred: {pred_text}")
            print("="*45 + "\n")

    return loss_sum / len(loader), 100.0 * correct / total

if __name__ == "__main__":
    train()