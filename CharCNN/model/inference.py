# inference.py — Single-image inference for the Ranjana Script CharCNN model.
# Usage: python model/inference.py <path_to_image>

import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# --- Model Architecture ---
class CharCNN(nn.Module):
    # Convolutional Neural Network for Ranjana script character classification.

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        # Block 1 — 64×64 → 32×32
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 5, padding=2), nn.ReLU(inplace=True),
            nn.BatchNorm2d(32), nn.MaxPool2d(2, 2), nn.Dropout2d(p=0.45),
        )

        # Block 2 — 32×32 → 16×16
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5, padding=2), nn.ReLU(inplace=True),
            nn.BatchNorm2d(64), nn.MaxPool2d(2, 2), nn.Dropout2d(p=0.40),
        )

        # Block 3 — 16×16 → 8×8
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(64), nn.MaxPool2d(2, 2), nn.Dropout2d(p=0.45),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512), nn.ReLU(inplace=True),
            nn.BatchNorm1d(512), nn.Dropout(p=0.50),
            nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.BatchNorm1d(512), nn.Dropout(p=0.45),
            nn.Linear(512, 64), nn.ReLU(inplace=True),
            nn.BatchNorm1d(64), nn.Dropout(p=0.35),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


# --- Directory and File Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "model", "best_char_cnn.pth")


# --- Device Selection ---
def select_device() -> torch.device:
    # Pick the best available accelerator: CUDA → MPS → CPU.
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --- Image Preprocessing (OpenCV) ---
def clean_image(image_path: str) -> Image.Image:
    # Read image, grayscale, otsu binarize, invert, and resize to 64x64.
    # Step 1 — Read
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Step 2 — Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3 — Otsu binarisation
    # cv2.threshold returns (threshold_value, thresholded_image).
    # THRESH_BINARY + THRESH_OTSU lets OpenCV pick the threshold automatically.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 4 — Invert so that strokes are white (255) on a black (0) background
    inverted = cv2.bitwise_not(binary)

    # Step 5 — Resize to the model's expected input dimensions
    resized = cv2.resize(inverted, (64, 64), interpolation=cv2.INTER_AREA)

    # Convert the NumPy array to a PIL Image (mode "L" = 8-bit grayscale)
    pil_img = Image.fromarray(resized, mode="L")
    return pil_img


# --- PyTorch Transforms ---
# (Must match validation transforms from train.py)
inference_transform = transforms.Compose([
    transforms.ToTensor(),              # [0, 255] → [0.0, 1.0]
    transforms.Normalize([0.5], [0.5]), # [0.0, 1.0] → [−1.0, 1.0]
])


# --- Model Loading ---
def load_model(device: torch.device) -> tuple[CharCNN, list[str]]:
    # Load the trained CharCNN from the checkpoint file.
    if not os.path.isfile(CHECKPOINT_PATH):
        print(f"✗ Checkpoint not found: {CHECKPOINT_PATH}")
        print("  Run train.py first to train and save a model.")
        sys.exit(1)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    classes = checkpoint["classes"]
    num_classes = len(classes)

    model = CharCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()  # disable dropout / use running batch-norm stats

    return model, classes


# --- Prediction ---
def predict(
    image_path: str,
    model: CharCNN,
    classes: list[str],
    device: torch.device,
) -> tuple[str, float]:
    # Run inference on a single image.
    # Preprocess with OpenCV (binarise + invert + resize)
    pil_img = clean_image(image_path)

    # Apply PyTorch transforms and add a batch dimension: (1, 1, 64, 64)
    tensor = inference_transform(pil_img).unsqueeze(0).to(device)

    # Forward pass — no gradients needed during inference
    with torch.no_grad():
        logits = model(tensor)                      # (1, num_classes)
        probabilities = torch.softmax(logits, dim=1)  # convert to [0, 1]

    # Extract the top prediction
    confidence, predicted_idx = torch.max(probabilities, dim=1)
    predicted_class = classes[predicted_idx.item()]
    confidence_pct = confidence.item() * 100.0

    return predicted_class, confidence_pct


# --- Main Entry Point ---
def main() -> None:
    # 1. Argument Check
    if len(sys.argv) < 2:
        print("Usage: python model/inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.isfile(image_path):
        print(f"✗ File not found: {image_path}")
        sys.exit(1)

    # 2. Setup Device and Model
    device = select_device()
    model, classes = load_model(device)

    # 3. Perform Inference
    predicted_class, confidence = predict(image_path, model, classes, device)

    # 4. Display Output
    print(f"Predicted : {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")


if __name__ == "__main__":
    main()
