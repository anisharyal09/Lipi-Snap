import os
import json
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
from torchvision import transforms

# ============================================================================
# 🎯 CONFIGURATION & CONSTANTS
# ============================================================================
CONFIG = {
    "MODEL_A": {"path": "model/best_char_cnn_data_a.pth", "label": "Dataset A"},
    "MODEL_B": {"path": "model/best_char_cnn_data_b.pth", "label": "Dataset B"},
    "MAPPING_PATH": "mapping/ranjana_to_devanagari.json",
    "IMG_SIZE": 64,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# ============================================================================
# 🧠 MODEL ARCHITECTURES
# ============================================================================
class CharCNN_A(nn.Module):
    """Original architecture without Batch Normalization."""
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.head(self.net(x))

class CharCNN_B(nn.Module):
    """Improved architecture with Batch Normalization and Dropout."""
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.head(self.net(x))

# ============================================================================
# 🛠️ UTILS & PREPROCESSING
# ============================================================================
class BinarizeInvert:
    """Custom transform to binarize and invert images for better OCR contrast."""
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, tensor: torch.Tensor):
        return 1.0 - (tensor > self.threshold).float()

@st.cache_data
def load_mapping(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_transform(apply_binarize: bool):
    t_list = [
        transforms.Grayscale(1),
        transforms.Resize((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"])),
        transforms.ToTensor(),
    ]
    if apply_binarize:
        t_list.append(BinarizeInvert())
    t_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    return transforms.Compose(t_list)

# ============================================================================
# 🚀 CORE LOGIC: MODEL LOADING & PREDICTION
# ============================================================================
@st.cache_resource
def load_trained_model(model_choice: str):
    config_key = "MODEL_A" if "Original" in model_choice else "MODEL_B"
    path = CONFIG[config_key]["path"]
    ModelClass = CharCNN_A if config_key == "MODEL_A" else CharCNN_B

    if not os.path.exists(path):
        return None, None

    checkpoint = torch.load(path, map_location=CONFIG["DEVICE"])
    classes = checkpoint['classes']
    model = ModelClass(len(classes)).to(CONFIG["DEVICE"])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, classes

def predict(image: Image, model: nn.Module, classes: list, use_binarize: bool):
    transform = get_transform(use_binarize)
    img_tensor = transform(image).unsqueeze(0).to(CONFIG["DEVICE"])
    
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        conf, idx = torch.max(probs, 1)
    
    return classes[idx.item()], conf.item() * 100

# ============================================================================
# 🖥️ STREAMLIT INTERFACE
# ============================================================================
def main():
    st.set_page_config(page_title="Lipi Snap", page_icon="🌟")
    st.title("🌟 Lipi Snap")
    st.markdown("Digitizing ancient scripts with modern Neural Networks.")
    
    # --- Sidebar ---
    st.sidebar.header("🛠️ Settings")
    choice = st.sidebar.radio("Model Version", [CONFIG["MODEL_A"]["label"], CONFIG["MODEL_B"]["label"]])
    
    do_binarize = st.sidebar.checkbox(
        "Binarize & Invert", 
        value=("Improved" in choice),
        help="Recommended for Model B to match training data style."
    )

    # Load resources
    mapping = load_mapping(CONFIG["MAPPING_PATH"])
    model, classes = load_trained_model(choice)

    if not model:
        st.sidebar.error(f"⚠️ weights not found at the specified path.")
        return

    st.sidebar.success(f"✅ Model Loaded")

    # --- Main UI ---
    uploaded_file = st.file_uploader("Upload a Ranjana character snippet", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original Image", use_container_width=True)
        
        with col2:
            # Preview what the model actually processes
            preview_t = get_transform(do_binarize)
            # Re-normalize for display: (x * 0.5) + 0.5
            viewable_tensor = (preview_t(img) * 0.5) + 0.5
            st.image(transforms.ToPILImage()(viewable_tensor), caption="Processed Input", use_container_width=True)

        if st.button("Recognize Character", use_container_width=True, type="primary"):
            with st.spinner("Analyzing strokes..."):
                label, confidence = predict(img, model, classes, do_binarize)
                devanagari = mapping.get(label, "N/A")

                # Result Display
                st.markdown("---")
                res_col1, res_col2 = st.columns(2)
                res_col1.metric("Predicted Label", label)
                res_col2.metric("Confidence", f"{confidence:.1f}%")
                
                st.info(f"**Devanagari Mapping:** {devanagari}")

if __name__ == "__main__":
    main()