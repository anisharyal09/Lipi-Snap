"""
PyTorch Training Script for Ranjana Script OCR (CharCNN)

Optional script to continue training from a previously saved best model.
Specially used for fine-tuning, resuming an interrupted run, 
or extending training for more epochs when initial results are promising and 
further accuracy improvements are expected.
"""

import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# --- Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Hyperparameters ---
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
FINE_TUNE_LR = 0.00005  # Lower LR for resuming
TOTAL_EPOCHS = 36       # Target goal
IMAGE_SIZE = 64

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "merged_dataset", "train")
VAL_DIR = os.path.join(PROJECT_ROOT, "data", "merged_dataset", "val")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_char_cnn.pth")


# --- Model Architecture ---
class CharCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        # Block 1: 64x64 -> 32x32
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 5, padding=2), nn.ReLU(inplace=True),
            nn.BatchNorm2d(32), nn.MaxPool2d(2, 2), nn.Dropout2d(p=0.45),
        )
        # Block 2: 32x32 -> 16x16
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5, padding=2), nn.ReLU(inplace=True),
            nn.BatchNorm2d(64), nn.MaxPool2d(2, 2), nn.Dropout2d(p=0.40),
        )
        # Block 3: 16x16 -> 8x8
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(64), nn.MaxPool2d(2, 2), nn.Dropout2d(p=0.45),
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512), nn.ReLU(inplace=True), nn.BatchNorm1d(512), nn.Dropout(p=0.50),
            nn.Linear(512, 512), nn.ReLU(inplace=True), nn.BatchNorm1d(512), nn.Dropout(p=0.45),
            nn.Linear(512, 64), nn.ReLU(inplace=True), nn.BatchNorm1d(64), nn.Dropout(p=0.35),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


# --- Device Selection ---
def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --- Dataloaders ---
def build_dataloaders():
    train_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomAffine(degrees=20, scale=(0.8, 1.2), shear=0.51),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    val_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    train_ds = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
    val_ds = datasets.ImageFolder(root=VAL_DIR, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=2, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, train_ds.classes


# --- Loop Functions ---
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc=f"Training Epoch {epoch}"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, 100.0 * correct / total

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, 100.0 * correct / total


# --- Main Pipeline ---
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = select_device()
    print(f"✓ Training on: {device}")

    train_loader, val_loader, classes = build_dataloaders()
    num_classes = len(classes)
    model = CharCNN(num_classes=num_classes).to(device)

    # --- RESUME LOGIC ---
    start_epoch = 1
    best_val_acc = 0.0
    current_lr = LEARNING_RATE

    if os.path.exists(BEST_MODEL_PATH):
        print(f"Found checkpoint at {BEST_MODEL_PATH}. Loading weights...")
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        current_lr = FINE_TUNE_LR # Switch to fine-tuning rate
        print(f"Resuming from Epoch {start_epoch}. Previous Best Val Acc: {best_val_acc:.2f}%")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=current_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    print(f"Targeting {TOTAL_EPOCHS} total epochs with LR: {current_lr}")

    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        epoch_start = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_acc)
        elapsed = time.time() - epoch_start

        print(f"Epoch [{epoch:>3}/{TOTAL_EPOCHS}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:6.2f}% | "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:6.2f}% ({elapsed:.1f}s)", end="")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes,
                "epoch": epoch,
                "best_val_acc": best_val_acc,
            }, BEST_MODEL_PATH)
            print(" ★ saved")
        else:
            print()

    print(f"\nFinished! Best Validation Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()