"""
PyTorch Training Script for Ranjana Script OCR (CharCNN)

Implements the convolutional neural network architecture and training procedure 
for recognizing Ranjana (Nepal Lipi) script characters.
"""


# Imports

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

# Set random seeds for reproducible and deterministic results

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Training Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 18
IMAGE_SIZE = 64

# Directory Paths
# Subfolders represent class labels
# Resolve paths relative to the project root (one level above this script).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "merged_dataset", "train")
VAL_DIR = os.path.join(PROJECT_ROOT, "data", "merged_dataset", "val")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_char_cnn.pth")


# --- Model Architecture ---
class CharCNN(nn.Module):

    """
    Convolutional Neural Network for character-level classification of Ranjana script glyphs.

    Architecture summary:
    - Three convolutional blocks, each containing two Conv2d layers, BatchNorm, MaxPool, and Dropout2d.
    - A fully-connected classifier head with three hidden layers, followed by ReLU, BatchNorm, and Dropout.

    Args:
        num_classes (int): Number of distinct character classes to classify.
    """
    def __init__(self, num_classes: int) -> None:
        super().__init__()

        # Block 1
        # Spatial dims: 64x64 -> 32x32
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=3, padding=1),       # 64×64 → 64×64
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=5, padding=2),       # 64×64 → 64×64
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),     # 64×64 → 32×32
            nn.Dropout2d(p=0.45),
        )

        # Block 2
        # Spatial dims: 32x32 -> 16x16
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1),       # 32×32 → 32×32
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=5, padding=2),       # 32×32 → 32×32
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),     # 32×32 → 16×16
            nn.Dropout2d(p=0.40),
        )

        # Block 3
        # Spatial dims: 16x16 -> 8x8
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=5, padding=2),       # 16×16 → 16×16
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64,
                      kernel_size=3, padding=1),       # 16×16 → 16×16
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),     # 16×16 → 8×8
            nn.Dropout2d(p=0.45),
        )

        # Classifier Head
        # Input dimension: 64 channels * 8 * 8 = 4096 (output of Block 3)
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(in_features=64 * 8 * 8, out_features=512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.50),

            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.45),

            nn.Linear(in_features=512, out_features=64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.35),

            nn.Linear(in_features=64, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the three conv blocks and classifier head."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


# --- DEvice Selection ---

def select_device() -> torch.device:
    """
    Auto-detect the best available hardware accelerator.
    Prioritizes CUDA, then Apple MPS, and falls back to CPU.

    Returns:
        torch.device: The selected device object.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("⚠ No GPU detected — falling back to CPU")
    return device


# --- Data Loading and Augmentation ---

def build_dataloaders():
    """
    Create training and validation DataLoaders using ImageFolder.

    Training transforms include:
      - Grayscale conversion (ensures single-channel tensor)
      - Resize to 64x64
      - RandomAffine augmentation (rotation, scale, shear) for better generalization
      - Normalization to scale pixel values from [0, 1] to [-1, 1]

    Note: Images are expected to be externally preprocessed (e.g. Otsu thresholding and inversion).

    Returns:
        tuple: (train_loader, val_loader, class_names)
    """
    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomAffine(
            degrees=20,          # random rotation in [−20°, +20°]
            scale=(0.8, 1.2),    # random zoom between 80% and 120%
            shear=0.51,          # slight shear distortion
        ),
        transforms.ToTensor(),              # [0, 255] → [0.0, 1.0]
        transforms.Normalize([0.5], [0.5]), # [0.0, 1.0] → [−1.0, 1.0]
    ])

    # Validation transforms without data augmentation
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # Initialize Datasets
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR,
                                         transform=train_transform)
    val_dataset = datasets.ImageFolder(root=VAL_DIR,
                                       transform=val_transform)

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,           # shuffle every epoch for stochastic training
        num_workers=2,          # parallel data loading
        pin_memory=True,        # speeds up host → device transfers on CUDA
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,          # no need to shuffle for evaluation
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset.classes


# --- Training and Validation Loops ---

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """
    Train the model for one full epoch.

    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.train()  # enable dropout and batch normalization layers
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"Training Epoch {epoch}")):
        # Move data to the target device (GPU / MPS / CPU)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()   # Clear previous gradients
        loss.backward()         # Compute new gradients
        optimizer.step()        # Update weights

        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()  # disable gradient computation for efficiency
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate the model on the validation set.

    Returns:
        tuple: (val_loss, val_accuracy)
    """
    model.eval()  # disable dropout and batch normalization layers
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    val_loss = running_loss / total
    val_acc = 100.0 * correct / total
    return val_loss, val_acc


# --- Main Training Pipeline ---

def main() -> None:
    """Orchestrate the full training pipeline."""

    print("=" * 70)
    print("  CharCNN — Ranjana Script Character Recognition")
    print("  Architecture & hyperparameters: Jen Bati & Pankaj Raj Dawadi")
    print("=" * 70)

    # 1. Setup Device
    device = select_device()

    # 2. Setup Data Loaders
    print(f"\nLoading data …")
    print(f"  Train : {TRAIN_DIR}")
    print(f"  Val   : {VAL_DIR}")
    train_loader, val_loader, classes = build_dataloaders()
    num_classes = len(classes)
    print(f"  Classes detected : {num_classes}")
    print(f"  Training samples : {len(train_loader.dataset)}")
    print(f"  Validation samples : {len(val_loader.dataset)}")

    # 3. Initialize Model
    model = CharCNN(num_classes=num_classes).to(device)

    # Print a quick parameter count for sanity-checking
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters     : {total_params:,}")
    print(f"  Trainable parameters : {trainable_params:,}")

    # 4. Define Loss, Optimizer, and Learning Rate Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Scheduler: Reduces learning rate if validation accuracy plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    # 5. Prepare Output Directory
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 6. Execute Training Loop
    best_val_acc = 0.0  # track the best validation accuracy for checkpointing

    print(f"\nStarting training for {NUM_EPOCHS} epochs …")
    print(f"  Batch size     : {BATCH_SIZE}")
    print(f"  Learning rate  : {LEARNING_RATE}")
    print(f"  Optimiser      : Adam")
    print(f"  Loss function  : CrossEntropyLoss")
    print("-" * 70)

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        # Training phase
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validation phase
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update learning rate scheduler
        scheduler.step(val_acc)

        elapsed = time.time() - epoch_start

        # Logging
        print(
            f"Epoch [{epoch:>3}/{NUM_EPOCHS}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:6.2f}%  │  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:6.2f}%  "
            f"({elapsed:.1f}s)",
            end="",
        )

        # Save model checkpoint if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": classes,            # preserve class names & order
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                },
                BEST_MODEL_PATH,
            )
            print(f"  ★ saved (best)")
        else:
            print()

    # 7. Training Summary
    print("-" * 70)
    print(f"Training complete.")
    print(f"  Best validation accuracy : {best_val_acc:.2f}%")
    print(f"  Best model saved to      : {BEST_MODEL_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
