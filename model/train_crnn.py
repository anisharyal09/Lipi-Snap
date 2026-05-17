"""
Lipi Snap — CRNN+CTC Training Pipeline
Word-level Ranjana Script Recognition
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

# Allow unsupported MPS ops to fall back to CPU instead of crashing
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- Configuration ---
IMG_HEIGHT = 64
IMG_WIDTH  = 512   # wider = better resolution for long words
MAX_LR     = 3e-3  # peak LR for OneCycleLR warm-up scheduler
EPOCHS     = 99
GRAD_CLIP  = 5.0   # gradient clipping threshold to prevent exploding gradients

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
    LABELS_CSV  = '/kaggle/input/datasets/anisharyal09/ranjana-ls/labels.csv'   # my_private_dataset uploaded in kaggle
    IMAGES_DIR  = '/kaggle/input/datasets/anisharyal09/ranjana-ls/images/images'   # my_private_dataset uploaded in kaggle

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
    """Handles encoding text → indices and decoding indices → text."""
    def __init__(self, chars):
        # Index 0 is always reserved for the CTC blank token
        self.chars = ["<blank>"] + chars
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}

    def encode(self, text):
        # Skip any character not in our vocabulary (e.g. rare symbols)
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

    def decode(self, indices):
        # Index 0 is blank — skip it during decode
        return "".join([self.idx_to_char[i] for i in indices if i != 0])

    def num_classes(self):
        return len(self.chars)  # includes the blank token

converter = LabelConverter(CHARS)

# --- Preprocessing ---
def process_image_crnn(image_path):
    """
    Preprocessing pipeline (must match generate_word_images.py):
      1. Grayscale → Otsu binarize
      2. Smart invert → white text on black background
      3. Height-lock resize (preserve aspect ratio)
      4. Right-pad to IMG_WIDTH with black pixels
    """
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None: raise ValueError("Invalid image")

        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Smart invert: dominant colour = background
        if np.sum(binary == 255) > np.sum(binary == 0):
            binary = cv2.bitwise_not(binary)

        h, w = binary.shape
        new_w = min(int(IMG_HEIGHT * (w / h)), IMG_WIDTH)
        resized = cv2.resize(binary, (new_w, IMG_HEIGHT), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
        canvas[:, :new_w] = resized

        return Image.fromarray(canvas)

    except Exception:
        # Return a blank black image so the batch doesn't crash
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
    """
    Custom collate for CTC — labels have variable lengths so we can't
    stack them into a rectangle. Instead:
      - Images  → stacked normally into [B, C, H, W]
      - Labels  → flattened into one 1D tensor
      - Lengths → per-sample label lengths, needed by CTCLoss
    """
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    targets = torch.cat([l.long() for l in labels], 0)
    return images, targets, target_lengths

# --- Model ---
class CRNN(nn.Module):
    """
    CRNN Architecture:
      CNN  → strips vertical spatial info, produces a width-sequence of features
      LSTM → reads that sequence left+right, understands temporal context
      Head → per-timestep class scores (raw logits, log_softmax applied in loss)
    """
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 2, padding='same'), nn.ReLU(True),
            nn.Conv2d(32, 32, 2, padding='same'), nn.ReLU(True),
            nn.Conv2d(32, 64, 2, padding='same'), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 2, padding='same'), nn.ReLU(True),
            nn.Conv2d(64, 128, 2, padding='same'), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )

        # Flatten 128-channel × 16-height feature columns into a 1D vector per timestep
        self.dense_pre_rnn = nn.Sequential(
            nn.Linear(128 * 16, 64), nn.ReLU(True), nn.Dropout(0.2)
        )

        # 3-layer Bidirectional LSTM: reads the sequence in both directions
        self.rnn1 = nn.LSTM(64,  256, bidirectional=True, batch_first=True)  # out: 512
        self.rnn2 = nn.LSTM(512, 128, bidirectional=True, batch_first=True)  # out: 256
        self.rnn3 = nn.LSTM(256, 128, bidirectional=True, batch_first=True)  # out: 256

        # Per-timestep classifier: maps each column to a class score
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(True), nn.Dropout(0.2),
            nn.Linear(128, num_classes)  # raw logits — log_softmax applied at loss time
        )

    def forward(self, x):
        # CNN: [B, 1, H, W] → [B, 128, 16, W//4]
        conv = self.cnn(x)
        b, c, h, w = conv.size()

        # Reshape to sequence: [B, W, C*H] — each column becomes one timestep
        conv = conv.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
        feats  = self.dense_pre_rnn(conv)
        out, _ = self.rnn1(feats)
        out, _ = self.rnn2(out)
        out, _ = self.rnn3(out)
        return self.classifier(out).permute(1, 0, 2).contiguous()

# --- Loss + Decode ---
def compute_ctc_loss(criterion, preds, targets, target_lengths):
    log_probs = torch.nn.functional.log_softmax(preds, dim=2)
    input_lengths = torch.full((preds.size(1),), preds.size(0), dtype=torch.long)
    if DEVICE.type == 'mps':
        # MPS fallback: CTC not yet natively supported on Apple Silicon GPU
        return criterion(
            log_probs.cpu(), targets.cpu(),
            input_lengths.cpu(), target_lengths.cpu()
        )
    return criterion(log_probs, targets, input_lengths, target_lengths)


def greedy_decode(preds):
    _, idxs = preds.max(2)
    idxs = idxs.transpose(1, 0).cpu()
    results = []
    for seq in idxs:
        decoded, last = [], -1
        for i in seq.tolist():
            if i != last and i != 0:  # skip blank (0) and repeated chars
                decoded.append(i)
            last = i
        results.append(decoded)
    return results

# --- Training Helpers ---
class TransformSubset(Dataset):
    """Wraps a Subset with a different transform (needed because random_split shares one dataset)."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self): return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# Pickle-friendly noise transform (lambdas crash with NUM_WORKERS > 0)
class AddGaussianNoise:
    def __init__(self, std=0.05):
        self.std = std
    def __call__(self, tensor):
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std, 0.0, 1.0)

# --- Training ---
def train():
    # Dynamic Augmentation (train only)
    train_transform = transforms.Compose([
        # 1. GEOMETRY: 90% chance to tilt/shift/shear, 10% stays flat
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=2, translate=(0.01, 0.02), scale=(0.95, 1.05), shear=2
            )
        ], p=0.9),

        # 2. CAMERA EFFECTS: 50% chance to blur
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
        ], p=0.5),

        # 3. CONVERT
        transforms.ToTensor(),

        # 4. NOISE: 30% chance to add gaussian noise
        transforms.RandomApply([
            AddGaussianNoise(std=0.05)
        ], p=0.3),

        # 5. NORMALIZE
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Val: clean, no augmentation
    val_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])

    # Load dataset WITHOUT transform (raw PIL images)
    dataset = RanjanaCRNNDataset(LABELS_CSV, IMAGES_DIR, transform=None)
    if len(dataset) == 0:
        print("❌ No samples found.")
        return

    # 90/10 train/val split
    train_size = int(0.9 * len(dataset))
    val_size   = len(dataset) - train_size
    train_sub, val_sub = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_ds = TransformSubset(train_sub, train_transform)
    val_ds   = TransformSubset(val_sub,   val_transform)
    kwargs = {'collate_fn': crnn_collate_fn, 'num_workers': NUM_WORKERS, 'pin_memory': (DEVICE.type == 'cuda')}
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  **kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    # --- Model ---
    model = CRNN(converter.num_classes()).to(DEVICE)

    # CTCLoss: blank=0 matches LabelConverter; zero_infinity prevents early NaN crashes
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    # AdamW: proper weight decay to fight overfitting
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # OneCycleLR: slow warm-up → peak → gentle cool-down (must step every batch)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = MAX_LR,
        steps_per_epoch = len(train_loader),
        epochs          = EPOCHS,
    )

    scaler = torch.amp.GradScaler('cuda') if USE_AMP else None

    # --- Robust Checkpoint Loader ---
    # Search multiple potential paths (local, Kaggle inputs, Colab inputs, etc.)
    start_epoch, best_loss = 1, float('inf')
    
    candidate_paths = [
        BEST_MODEL_PATH,
        MODEL_DIR / "best_crnn6.pth",
        Path("best_crnn.pth"),
        Path("best_crnn6.pth")
    ]
    
    # Add Kaggle/Colab input directories as fallback search locations
    try:
        csv_parent = Path(LABELS_CSV).parent
        candidate_paths.append(csv_parent / "best_crnn.pth")
        candidate_paths.append(csv_parent / "best_crnn6.pth")
        # Recursively search the input dataset parent for any checkpoint matching best_crnn*.pth
        if csv_parent.exists():
            for pth in csv_parent.rglob("best_crnn*.pth"):
                candidate_paths.append(pth)
    except Exception:
        pass

    # Filter out duplicates while preserving order
    seen_paths = set()
    unique_candidates = []
    for p in candidate_paths:
        try:
            p_abs = p.resolve()
            if p_abs not in seen_paths:
                seen_paths.add(p_abs)
                unique_candidates.append(p)
        except Exception:
            if p not in unique_candidates:
                unique_candidates.append(p)

    loaded = False
    print("🔎 Searching for checkpoints to resume...")
    for path in unique_candidates:
        if path.exists():
            print(f"  📂 Found checkpoint file: {path}")
            try:
                ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
                if 'model_state_dict' in ckpt:
                    model.load_state_dict(ckpt['model_state_dict'])
                    if 'optimizer_state_dict' in ckpt:
                        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    if 'scheduler_state_dict' in ckpt:
                        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                    start_epoch = ckpt.get('epoch', 0) + 1
                    best_loss   = ckpt.get('loss', float('inf'))
                    print(f"🔄 Successfully resumed training from: {path} (Epoch {start_epoch - 1}, Val Loss: {best_loss:.4f})")
                    loaded = True
                    break
                else:
                    print(f"  ⚠️  {path} does not contain 'model_state_dict', skipping.")
            except Exception as e:
                print(f"  ⚠️  Could not load checkpoint from {path}: {e}")

    if not loaded:
        print("ℹ️  No valid checkpoint found or loaded. Starting training from scratch.")

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
                
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                
                # Only step scheduler if the scaler didn't skip the optimizer step (NaN/Inf)
                if scale <= scaler.get_scale():
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

        # --- Validation ---
        val_loss, acc = validate(model, val_loader, criterion)
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch:>3} | Train {avg_train_loss:.4f} | Val {val_loss:.4f} | Acc {acc:.2f}%")

        # Save best checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch'                : epoch,
                'model_state_dict'     : model.state_dict(),
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

        # Greedy decode predictions and compare with ground truth
        decoded = greedy_decode(preds)
        targets_cpu = targets.cpu()
        gt_seqs, offset = [], 0
        for l in target_lengths:
            gt_seqs.append(targets_cpu[offset : offset + l.item()].tolist())
            offset += l.item()

        for pred_seq, gt_seq in zip(decoded, gt_seqs):
            if pred_seq == gt_seq:
                correct += 1
            total += 1

        # Print epoch learning overview (first batch only)
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
