"""
Minimal PyTorch Character CNN Training for Dataset A (Ranjana Lipi).
Trains on data_A/train, validates on data_A/val.
Saves best checkpoint to model/best_char_cnn_data_a.pth.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -------------------------------
# CONFIG
# -------------------------------
TRAIN_DIR = "data/data_A/train"
VAL_DIR   = "data/data_A/val"
MODEL_SAVE_PATH = "model/best_char_cnn_data_a.pth"

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 10
IMG_SIZE = 64
SEED = 42

# Prefer CUDA, else Apple MPS, else CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)

class CharCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2),   # 64->32
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2),  # 32->16
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2), # 16->8
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8*8, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.net(x)
        return self.head(x)

def make_loaders(num_workers=2):
    # (Why) normalize keeps inputs centered; grayscale & resize standardize

    train_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 0.6))], p=0.10),
        transforms.RandomAffine(degrees=7, translate=(0.04, 0.04), shear=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    val_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_transform)

    # (Why) pin_memory helps only on CUDA; MPS/CPU don’t benefit
    pin = (DEVICE.type == "cuda")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=pin
    )
    return train_ds, val_ds, train_loader, val_loader

def evaluate(model, loader, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred==y).sum().item()
            total += y.numel()
    return (loss_sum/total), (correct/total)

def main():
    print(f"Using device: {DEVICE}")
    set_seed(SEED)

    train_ds, val_ds, train_loader, val_loader = make_loaders(num_workers=2)

    classes = train_ds.classes
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes[:8]}{' ...' if num_classes>8 else ''}")

    model = CharCNN(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    for epoch in range(1, EPOCHS+1):
        model.train()
        run_loss = 0.0
        for x,y in train_loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            run_loss += loss.item()

        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch:02d} | train_loss={run_loss/len(train_loader):.4f} "
              f"| val_loss={val_loss:.4f} | val_acc={val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"state_dict": model.state_dict(), "classes": classes}, MODEL_SAVE_PATH)
            print(f"  -> Saved best: {MODEL_SAVE_PATH} (val_acc={val_acc*100:.2f}%)")

    print(f"\nDone. Best val_acc={best_acc*100:.2f}% | saved at {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()