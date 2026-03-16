"""
Minimal inference script for trained Character CNN (Dataset A).
Usage:
  python model/inference_A.py path/to/image.png
"""

import sys
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

MODEL_PATH = "model/best_char_cnn_data_a.pth"
IMG_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CharCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8*8, 256), nn.ReLU(),
            nn.Linear(256, 1)  # placeholder; we will reset based on checkpoint
        )
    def forward(self, x):
        x = self.net(x)
        return self.head(x)

transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5]),
])

def predict(image_path: str):
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    classes = ckpt["classes"]
    num_classes = len(classes)

    # rebuild head with correct num_classes
    model = CharCNN(num_classes)
    # replace final layer to correct output size
    model.head[-1] = nn.Linear(256, num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(DEVICE)

    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)
    return classes[idx.item()], float(conf.item()*100)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python model/inference_A.py <image_path>")
        sys.exit(1)
    label, conf = predict(sys.argv[1])
    print(f"Predicted: {label}  |  Confidence: {conf:.2f}%")