"""
Train on Dataset-B (RHCD-like) following the paper's settings:
- 64x64 grayscale
- Adam, lr≈7.75e-4
- batch=32
- moderate augmentation (can be increased)
- optional binarize+invert to approximate RHCD preproc
Saves: model/best_char_cnn_data_b.pth
"""

import os, random, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# -------------------------------
# CONFIG
# -------------------------------
TRAIN_DIR = "data/data_B_relabel/train"
VAL_DIR   = "data/data_B_relabel/val"

MODEL_SAVE_PATH = "model/best_char_cnn_data_b.pth"

IMG_SIZE = 64
BATCH_SIZE = 32
LEARNING_RATE = 7.75e-4
EPOCHS = 50
SEED = 42

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)

# -------------------------------
# Binarize + Invert
# -------------------------------
BINARIZE = True
BIN_THRESH = 0.5   

class BinarizeInvert(object):
    def __call__(self, tensor: torch.Tensor):
        t = tensor.clone()
        t = (t > BIN_THRESH).float()
        t = 1.0 - t
        return t

# -------------------------------
# Augmentation
# -------------------------------
train_transforms = [
    transforms.Grayscale(1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),                      
]
if BINARIZE:
    train_transforms.append(BinarizeInvert())   
train_transforms += [
    transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 0.6))], p=0.10),
    transforms.RandomAffine(
        degrees=12,
        translate=(0.05, 0.05),
        scale=(0.90, 1.10),
        shear=6
    ),
    transforms.Normalize([0.5],[0.5])          
]

val_transforms = [
    transforms.Grayscale(1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
]
if BINARIZE:
    val_transforms.append(BinarizeInvert())
val_transforms += [
    transforms.Normalize([0.5],[0.5])
]

train_transform = transforms.Compose(train_transforms)
val_transform   = transforms.Compose(val_transforms)

# -------------------------------
# DATA
# -------------------------------
def make_loaders(num_workers=2):
    if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(VAL_DIR):
        raise FileNotFoundError(
            f"Dataset B paths not found. Expecting:\n  {TRAIN_DIR}\n  {VAL_DIR}\n" \
            "Make sure you relabeled Dataset B into canonical class folders."
        )

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_transform)
    pin = (DEVICE.type == "cuda")
    
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=num_workers, pin_memory=pin)
    val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=num_workers, pin_memory=pin)
    
    return train_ds, val_ds, train_ld, val_ld

# -------------------------------
# MODEL (with Batch Normalization)
# -------------------------------
class CharCNN(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            nn.MaxPool2d(2, 2),  # 64->32
            
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(2, 2),  # 32->16
            
            nn.Conv2d(64, 128, 3, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            nn.MaxPool2d(2, 2),  # 16->8
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), 
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n)
        )
        
    def forward(self, x):
        x = self.net(x)
        return self.head(x)

def evaluate(model, loader, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return (loss_sum / total), (correct / total)

def main():
    print(f"Device: {DEVICE}")
    set_seed(SEED)
    train_ds, val_ds, train_ld, val_ld = make_loaders(num_workers=2)

    classes = train_ds.classes
    print(f"Classes ({len(classes)}): {classes[:10]}{' ...' if len(classes)>10 else ''}")

    model = CharCNN(len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best = 0.0
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    for ep in range(1, EPOCHS + 1):
        model.train()
        run_loss = 0.0
        for x, y in train_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            run_loss += loss.item()

        val_loss, val_acc = evaluate(model, val_ld, criterion)
        print(f"Epoch {ep:02d} | train_loss={run_loss/len(train_ld):.4f} | val_loss={val_loss:.4f} | val_acc={val_acc*100:.2f}%")
        scheduler.step(val_loss)

        if val_acc > best:
            best = val_acc
            torch.save({"state_dict": model.state_dict(), "classes": classes}, MODEL_SAVE_PATH)
            print(f"  -> saved best: {MODEL_SAVE_PATH} (val_acc={val_acc*100:.2f}%)")

    print(f"\nBest val_acc={best*100:.2f}% | saved at {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()