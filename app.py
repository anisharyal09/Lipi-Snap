import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
from torchvision import transforms

# --- Configuration and Constants ---
CONFIG = {
    "MODEL_PATH": "model/best_char_cnn.pth",
    "MAPPING_PATH": "mapping/ranjana_to_devanagari.json",
    "IMG_SIZE": 64,
}

@st.cache_resource
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

# --- Model Architecture ---
class CharCNN(nn.Module):
    # Convolutional Neural Network matching the trained architecture.
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 5, padding=2), nn.ReLU(inplace=True),
            nn.BatchNorm2d(32), nn.MaxPool2d(2, 2), nn.Dropout2d(p=0.45),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5, padding=2), nn.ReLU(inplace=True),
            nn.BatchNorm2d(64), nn.MaxPool2d(2, 2), nn.Dropout2d(p=0.40),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(64), nn.MaxPool2d(2, 2), nn.Dropout2d(p=0.45),
        )
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

# --- Utilities and Preprocessing ---
@st.cache_data
def load_mapping(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def preprocess_image(uploaded_file, apply_opencv=True, already_dark_bg=False) -> Image.Image:
    # Simulates training conditions: Grayscale -> Otsu Threshold -> Bitwise Invert -> 64x64 Resize.
    # Read raw bytes into numpy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    if apply_opencv:
        # Decode image using OpenCV
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # 1. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Otsu's Binarisation
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 3. Invert (strokes become white, background becomes black)
        if already_dark_bg:
            inverted = binary
        else:
            inverted = cv2.bitwise_not(binary)
        
        # 4. Resize to 64x64
        resized = cv2.resize(inverted, (CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]), interpolation=cv2.INTER_AREA)
        
        return Image.fromarray(resized, mode="L")
    else:
        # Fallback: Just load via PIL, resize and grayscale
        img = Image.open(uploaded_file).convert("L")
        return img.resize((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]))

# PyTorch normalisation (matches validation transforms)
inference_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# --- Core Logic: Model Loading and Prediction ---
@st.cache_resource
def load_trained_model():
    path = CONFIG["MODEL_PATH"]
    
    if not os.path.exists(path):
        return None, None

    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    classes = checkpoint['classes']
    
    model = CharCNN(len(classes)).to(DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, classes

def predict(pil_img: Image.Image, model: nn.Module, classes: list):
    # Prepare image for the model
    img_tensor = inference_transform(pil_img).unsqueeze(0).to(DEVICE)
    
    # Run forward pass and get prediction confidence
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        conf, idx = torch.max(probs, 1)
    
    return classes[idx.item()], conf.item() * 100

# --- Streamlit Interface ---
def main():
    st.set_page_config(page_title="Lipi Snap - Ranjana OCR", page_icon="📝", layout="centered")
    
    # Custom CSS for styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #ff4b4b;
            text-align: center;
            font-weight: 800;
            margin-bottom: 0px;
        }
        .sub-header {
            text-align: center;
            font-size: 1.2rem;
            color: #aaaaaa;
            margin-bottom: 30px;
        }
        .metric-card {
            background-color: #262730;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            border: 1px solid #3d3f4b;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="main-header">📝 Lipi Snap</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ranjana Script Character Recognition</p>', unsafe_allow_html=True)
    
    # --- Sidebar ---
    st.sidebar.header("⚙️ Model Status")
    
    model, classes = load_trained_model()
    mapping = load_mapping(CONFIG["MAPPING_PATH"])

    if not model:
        st.sidebar.error(f"⚠️ Model checkpoint not found at `{CONFIG['MODEL_PATH']}`.")
        st.sidebar.info("Please run `train.py` first until it saves a new best accuracy.")
        st.warning("Model weights are missing. Cannot perform inference.")
        return
    else:
        st.sidebar.success(f"✅ Model Loaded Successfully")
        st.sidebar.metric("Total Classes", len(classes))
        st.sidebar.info(f"Running on: {str(DEVICE).upper()}")

    st.sidebar.markdown("---")
    st.sidebar.header("🧪 Preprocessing Options")
    apply_opencv = st.sidebar.checkbox(
        "Apply OpenCV Preprocessing", 
        value=True,
        help="Applies Otsu's thresholding and binarisation to match training conditions."
    )
    already_dark_bg = st.sidebar.checkbox(
        "Image already has Dark Background", 
        value=False,
        help="Check this if uploaded image already has white text on a black background to prevent improper inversion."
    )

    # --- Main UI ---
    st.markdown("### Upload Character Image")
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    if uploaded_file:
        # Preprocess using OpenCV (like in inference.py)
        processed_pil_img = preprocess_image(uploaded_file, apply_opencv=apply_opencv, already_dark_bg=already_dark_bg)
        
        st.markdown("### Vision Pipeline")
        col1, col2 = st.columns(2)
        with col1:
            # We need to rewind the file pointer to read it again for raw display
            uploaded_file.seek(0)
            raw_img = Image.open(uploaded_file).convert('RGB')
            st.image(raw_img, caption="1. Raw Uploaded Image", use_container_width=True)
        
        with col2:
            st.image(processed_pil_img, caption="2. Processed Input (64x64, Binarised, Inverted)", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Prediction
        if st.button("🔮 Recognize Character", use_container_width=True, type="primary"):
            with st.spinner("Running CharCNN forward pass..."):
                label, confidence = predict(processed_pil_img, model, classes)
                devanagari = mapping.get(label, "No mapping found")

                st.markdown("---")
                st.markdown("### Results")
                
                # Display metrics in custom cards
                m1, m2, m3 = st.columns(3)
                
                with m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p style="color: #aaaaaa; margin: 0;">Predicted Class</p>
                        <h2 style="margin: 0; color: white;">{label}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with m2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p style="color: #aaaaaa; margin: 0;">Confidence</p>
                        <h2 style="margin: 0; color: #4bffa5;">{confidence:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with m3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p style="color: #aaaaaa; margin: 0;">Devanagari</p>
                        <h2 style="margin: 0; color: white;">{devanagari}</h2>
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()