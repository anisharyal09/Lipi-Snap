"""
Lipi Snap — Standalone Evaluation Script
"""

import os
import sys
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import jiwer

# --- Configuration ---
IMG_HEIGHT = 64
IMG_WIDTH  = 512
BATCH_SIZE = 128

# --- Device ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# --- Paths & Env Config ---
if os.path.exists('/kaggle/input'):
    ENV         = "kaggle"
    MODEL_DIR   = Path("/kaggle/working")
    LABELS_CSV  = '/kaggle/input/datasets/anisharyal09/ranjana-ls/labels.csv'
    IMAGES_DIR  = '/kaggle/input/datasets/anisharyal09/ranjana-ls/images/images'
elif 'google.colab' in sys.modules or os.path.exists('/content'):
    ENV         = "colab"
    MODEL_DIR   = Path("/content/drive/MyDrive/Lipi-Snap/model")
    LABELS_CSV  = Path("/content/labels.csv")
    IMAGES_DIR  = Path("/content/data/images")
else:
    ENV         = "local"
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    MODEL_DIR    = PROJECT_ROOT / "model"
    DATA_DIR     = PROJECT_ROOT / "data" / "test_synthetic_words"
    LABELS_CSV   = DATA_DIR / "labels.csv"
    IMAGES_DIR   = DATA_DIR / "images"

MODEL_PATH = MODEL_DIR / "best_crnn.pth"

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
class RanjanaTestDataset(Dataset):
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

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        image = process_image_crnn(self.img_dir / fname)
        if self.transform: image = self.transform(image)
        encoded = converter.encode(label)
        return image, torch.IntTensor(encoded), label

def test_collate_fn(batch):
    images, labels, raw_texts = zip(*batch)
    images = torch.stack(images, 0)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    targets = torch.cat([l.long() for l in labels], 0)
    return images, targets, target_lengths, raw_texts

# --- Model ---
class CRNN(nn.Module):
    def __init__(self, num_classes, h1=256, h2=128, h3=128, kernel_size=2):
        super().__init__()
        padding = 'same' if kernel_size == 2 else (kernel_size // 2)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size, padding=padding), nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size, padding=padding), nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size, padding=padding), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size, padding=padding), nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size, padding=padding), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )

        self.dense_pre_rnn = nn.Sequential(
            nn.Linear(128 * 16, 64), nn.ReLU(True), nn.Dropout(0.2)
        )

        self.rnn1 = nn.LSTM(64,     h1, bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(h1 * 2, h2, bidirectional=True, batch_first=True)
        self.rnn3 = nn.LSTM(h2 * 2, h3, bidirectional=True, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(h3 * 2, 128), nn.ReLU(True), nn.Dropout(0.2),
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
        return self.classifier(out).permute(1, 0, 2).contiguous()

# --- Decoding ---
def greedy_decode(preds):
    _, idxs = preds.max(2)
    idxs = idxs.transpose(1, 0).cpu() 
    
    results = []
    for seq in idxs:
        decoded, last = [], -1
        for i in seq.tolist():
            if i != last and i != 0: 
                decoded.append(i)
            last = i
        results.append(decoded)
    return results

# --- Evaluation Loop ---
@torch.no_grad()
def evaluate_model():
    # --- Robust Checkpoint Search ---
    checkpoint_to_load = None
    candidate_paths = [
        MODEL_PATH,
        Path("best_crnn.pth"),
        Path("model/best_crnn.pth"),
        Path("best_crnn6.pth")
    ]
    
    # Try recursively matching files in workspace as fallback
    try:
        csv_parent = Path(LABELS_CSV).parent
        candidate_paths.append(csv_parent / "best_crnn.pth")
        if csv_parent.exists():
            for pth in csv_parent.rglob("best_crnn*.pth"):
                candidate_paths.append(pth)
    except Exception:
        pass

    for path in candidate_paths:
        if path.exists():
            checkpoint_to_load = path
            break

    if not checkpoint_to_load:
        print(f"❌ Model checkpoint not found. Checked candidate paths: {candidate_paths}")
        return

    print(f"🔄 Loading Model from: {checkpoint_to_load} ...")
    ckpt = torch.load(checkpoint_to_load, map_location=DEVICE, weights_only=False)
    sd   = ckpt['model_state_dict']

    # Auto-detect architecture parameters
    h1 = sd['rnn1.weight_ih_l0'].shape[0] // 4
    h2 = sd['rnn2.weight_ih_l0'].shape[0] // 4
    h3 = sd['rnn3.weight_ih_l0'].shape[0] // 4
    num_classes = sd['classifier.3.weight'].shape[0]
    kernel_size = sd['cnn.0.weight'].shape[2]

    # Initialize vocabulary from checkpoint (preferred)
    chars = ckpt.get('chars', CHARS)
    local_converter = LabelConverter(chars)

    model = CRNN(num_classes, h1, h2, h3, kernel_size=kernel_size).to(DEVICE)
    model.load_state_dict(sd)
    model.eval()
    print("✅ Model loaded successfully.\n")

    val_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])

    print("📂 Loading Test Dataset...")
    dataset = RanjanaTestDataset(LABELS_CSV, IMAGES_DIR, transform=val_transform)
    if len(dataset) == 0: return
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=test_collate_fn, num_workers=2)

    all_predictions = []
    all_ground_truths = []
    incorrect_samples = []

    print("🚀 Running Inference...")
    for imgs, targets, target_lengths, raw_texts in loader:
        imgs = imgs.to(DEVICE)
        preds = model(imgs)
        decoded = greedy_decode(preds)

        for i, pred_seq in enumerate(decoded):
            pred_text = local_converter.decode(pred_seq)
            gt_text   = raw_texts[i]
            
            if len(gt_text.strip()) == 0: gt_text = "<empty>"
            if len(pred_text.strip()) == 0: pred_text = "<empty>"
            
            all_predictions.append(pred_text)
            all_ground_truths.append(gt_text)

            # Store some failures for review
            if pred_text != gt_text and len(incorrect_samples) < 10:
                incorrect_samples.append((gt_text, pred_text))

    print("\n" + "="*45)
    print("📊 FINAL EVALUATION METRICS")
    print("="*45)
    
    cer = jiwer.cer(all_ground_truths, all_predictions) * 100.0
    wer = jiwer.wer(all_ground_truths, all_predictions) * 100.0
    
    # Calculate exact match accuracy just for extra context
    exact_matches = sum(1 for gt, pred in zip(all_ground_truths, all_predictions) if gt == pred)
    acc = (exact_matches / len(all_predictions)) * 100.0

    print(f"Total Samples Tested : {len(all_predictions)}")
    print(f"Character Error Rate : {cer:.2f}%")
    print(f"Word Error Rate      : {wer:.2f}%")
    print(f"Exact Match Accuracy : {acc:.2f}%")
    
    if incorrect_samples:
        print("\n" + "="*45)
        print("🧐 SAMPLE ERRORS (Where the model failed)")
        print("="*45)
        for gt, pred in incorrect_samples:
            print(f"  ❌ GT:   {gt}")
            print(f"     Pred: {pred}")
        print("="*45 + "\n")

if __name__ == "__main__":
    evaluate_model()
