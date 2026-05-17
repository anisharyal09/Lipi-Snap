"""
Lipi Snap — CRNN+CTC Inference
Word-level Ranjana Script Recognition
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import transforms

# --- Configuration ---
IMG_HEIGHT = 64
IMG_WIDTH  = 512

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
BEST_MODEL_PATH = PROJECT_ROOT / "model" / "best_crnn.pth"

# --- Vocabulary (fallback — checkpoint's own chars are preferred) ---
VOCAB_STR = "ँंःअआइईउऊऋएऐऒओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहाािीुूृेैोौ्०१२३४५६७८९"
CHARS = sorted(list(set(VOCAB_STR)))

class LabelConverter:
    def __init__(self, chars):
        self.chars = ["<blank>"] + chars
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}

    def decode(self, indices):
        return "".join([self.idx_to_char[i] for i in indices if i != 0])

    def num_classes(self):
        return len(self.chars)

# --- Preprocessing ---
def preprocess_image(image_path):
    """
    Preprocessing pipeline (must match training):
      1. Grayscale → Otsu binarize
      2. Smart invert → white text on black background
      3. Height-lock resize (preserve aspect ratio)
      4. Right-pad to IMG_WIDTH with black pixels
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None: raise ValueError(f"Cannot read: {image_path}")

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

# --- Load Model ---
def load_model(model_path=None):
    """
    Load the CRNN model from a checkpoint.
    Architecture and vocabulary are inferred automatically from the checkpoint,
    so any model version can be loaded without code changes.
    """
    path = Path(model_path) if model_path else BEST_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"No model found at {path}")

    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    sd   = ckpt['model_state_dict']

    # Auto-detect architecture from the saved weights
    h1 = sd['rnn1.weight_ih_l0'].shape[0] // 4
    h2 = sd['rnn2.weight_ih_l0'].shape[0] // 4
    h3 = sd['rnn3.weight_ih_l0'].shape[0] // 4
    num_classes = sd['classifier.3.weight'].shape[0]
    kernel_size = sd['cnn.0.weight'].shape[2]  # Auto-detect convolution kernel size

    # Use vocabulary stored in checkpoint (preferred) or fall back to default
    chars = ckpt.get('chars', CHARS)
    converter = LabelConverter(chars)

    # Rebuild the model with the correct sizes
    model = CRNN(num_classes, h1, h2, h3, kernel_size=kernel_size).to(DEVICE)
    model.load_state_dict(sd)
    model.eval()
    return model, converter


@torch.no_grad()
def predict(image_path, model, converter):
    """Run inference on a single image and return the predicted word."""
    pil_img = preprocess_image(image_path)
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])
    img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    preds = model(img_tensor)
    _, preds_idx = preds.max(2)
    preds_idx = preds_idx.squeeze(1).tolist()

    # Greedy CTC decode: collapse repeated chars, remove blanks
    decoded, last = [], -1
    for idx in preds_idx:
        if idx != last and idx != 0: decoded.append(idx)
        last = idx

    return converter.decode(decoded)


if __name__ == "__main__":
    import sys
    try:
        model, converter = load_model()
        print(f"✅ Model loaded on {DEVICE.type.upper()}")
        if len(sys.argv) > 1:
            result = predict(sys.argv[1], model, converter)
            print(f"🔍 Prediction: {result}")
    except Exception as e:
        print(f"❌ {e}")
