import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import base64
from PIL import Image
from torchvision import transforms

# --- Configuration ---
CONFIG = {
    "MODEL_PATH": "model/best_char_cnn.pth",
    "MAPPING_DEV": "mapping/ranjana_to_devanagari.json",
    "MAPPING_NEWA": "mapping/ranjana_unicode.json",
    "IMG_SIZE": 64,
    "FONTS": {
        "Normal": "font/NithyaRanjanaDU-Regular.otf",
        "Stylish": "font/ranjana_NLG_v3_fixed.ttf"
    }
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

# --- Utilities ---
@st.cache_data
def load_mapping(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def get_base64_font(font_path: str):
    """Reads a local font file and returns the Base64 string for CSS injection."""
    if not os.path.exists(font_path):
        return ""
    with open(font_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def preprocess_image(uploaded_file, apply_opencv=True, already_dark_bg=False) -> Image.Image:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    if apply_opencv:
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if already_dark_bg:
            inverted = binary
        else:
            inverted = cv2.bitwise_not(binary)
        resized = cv2.resize(inverted, (CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]), interpolation=cv2.INTER_AREA)
        return Image.fromarray(resized, mode="L")
    else:
        img = Image.open(uploaded_file).convert("L")
        return img.resize((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]))

inference_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

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
    img_tensor = inference_transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        conf, idx = torch.max(probs, 1)
    return classes[idx.item()], conf.item() * 100

def get_top_predictions(pil_img: Image.Image, model: nn.Module, classes: list, top_k=5):
    img_tensor = inference_transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        top_probs, top_idxs = torch.topk(probs, top_k, dim=1)
    results = []
    for i in range(top_k):
        results.append((classes[top_idxs[0][i].item()], top_probs[0][i].item() * 100))
    return results


# --- Static CSS ---
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    .stApp { font-family: 'Inter', sans-serif; }
    #MainMenu, header, footer { visibility: hidden; }
    
    /* Title */
    .app-title {
        text-align: center;
        padding: 2.5rem 0 0.5rem 0;
    }
    .app-title h1 {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #a78bfa 0%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.15;
    }
    .app-title p {
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 4px;
    }
    
    /* Divider */
    .sep {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
        margin: 20px 0;
    }
    
    /* Section label */
    .sec-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #4b5563;
        margin-bottom: 12px;
    }
    
    /* Image frame */
    .img-frame {
        background: rgba(0,0,0,0.25);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 10px;
        text-align: center;
    }
    .img-frame p {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #4b5563;
        margin: 8px 0 0 0;
    }
    
    /* Result card */
    .r-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 24px 16px;
        text-align: center;
    }
    .r-card .lbl {
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #4b5563;
        margin-bottom: 6px;
    }
    .r-card .val {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.2;
    }
    .r-card .val.purple { color: #a78bfa; }
    .r-card .val.green { color: #34d399; }
    .r-card .val.pink { color: #f472b6; }
    
    /* Newa char: .copy-only = invisible Newa Unicode (for copy-paste), .visual-only = rendered via DynamicRanjana font */
    .newa-char-container {
        position: relative;
        display: inline-flex;
        justify-content: center;
        align-items: center;
    }
    .copy-only {
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        color: transparent;
        z-index: 10;
        overflow: hidden;
        white-space: nowrap;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .copy-only::selection {
        background: rgba(139, 92, 246, 0.4);
        color: transparent;
    }
    .visual-only {
        pointer-events: none;
        user-select: none;
        -webkit-user-select: none;
    }
    .newa-char {
        font-size: 2.4rem;
        color: #fbbf24;
        line-height: 1.3;
    }
    
    /* Confidence bar */
    .conf-bg {
        width: 100%;
        height: 6px;
        background: rgba(255,255,255,0.05);
        border-radius: 100px;
        margin-top: 10px;
        overflow: hidden;
    }
    .conf-fill {
        height: 100%;
        border-radius: 100px;
    }
    .conf-fill.hi { background: linear-gradient(90deg, #34d399, #6ee7b7); }
    .conf-fill.mid { background: linear-gradient(90deg, #fbbf24, #fcd34d); }
    .conf-fill.lo { background: linear-gradient(90deg, #f87171, #fca5a5); }
    
    /* Prediction rows */
    .pred-row {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 12px;
        border-radius: 8px;
        margin-bottom: 4px;
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.04);
    }
    .pred-row .pred-newa-container {
        font-size: 1.2rem;
        color: #fbbf24;
        width: 28px;
        text-align: center;
        display: inline-block;
        position: relative;
        flex-shrink: 0;
    }
    .pred-row .pred-name {
        font-weight: 600;
        color: #d1d5db;
        font-size: 0.85rem;
        min-width: 50px;
    }
    .pred-row .pred-dev {
        color: #9ca3af;
        font-size: 0.85rem;
        min-width: 20px;
    }
    .pred-row .pred-bar-bg {
        flex-grow: 1;
        height: 3px;
        background: rgba(255,255,255,0.04);
        border-radius: 100px;
        overflow: hidden;
    }
    .pred-row .pred-bar-fill {
        height: 100%;
        border-radius: 100px;
        background: linear-gradient(90deg, #8b5cf6, #a78bfa);
    }
    .pred-row .pred-pct {
        font-weight: 700;
        font-size: 0.8rem;
        color: #6b7280;
        min-width: 52px;
        text-align: right;
    }
    
    /* Upload area */
    .stFileUploader > div > div {
        border: 2px dashed rgba(139,92,246,0.25) !important;
        border-radius: 14px !important;
        background: rgba(139,92,246,0.03) !important;
    }
    .stFileUploader > div > div:hover {
        border-color: rgba(139,92,246,0.45) !important;
        background: rgba(139,92,246,0.06) !important;
    }
    
    /* Button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #8b5cf6, #7c3aed) !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        box-shadow: 0 4px 14px rgba(139,92,246,0.25) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
        box-shadow: 0 6px 24px rgba(139,92,246,0.35) !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(17,17,27,0.95);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
</style>
"""

# --- App ---
def main():
    st.set_page_config(page_title="Lipi Snap", page_icon="📸", layout="centered")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### Typography")
        selected_font_name = st.radio(
            "Display Font", 
            options=list(CONFIG["FONTS"].keys()),
            help="Choose which font to render the Ranjana text with."
        )
        
        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
        st.markdown("### Preprocessing")
        apply_opencv = st.toggle("OpenCV Pipeline", value=True,
            help="Otsu threshold + binarise to match training conditions.")
        already_dark_bg = st.toggle("Dark Background", value=False,
            help="Enable if image already has white strokes on black.")
        
        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
        
        model, classes = load_trained_model()
        st.markdown("### Info")
        st.markdown(f"""
            <div style="font-size: 0.82rem; color: #6b7280; line-height: 1.7;">
                <strong style="color: #9ca3af;">Classes:</strong> {len(classes) if classes else '—'}<br>
                <strong style="color: #9ca3af;">Device:</strong> {str(DEVICE).upper()}<br>
                <strong style="color: #9ca3af;">Model Input:</strong> {CONFIG['IMG_SIZE']}×{CONFIG['IMG_SIZE']}
            </div>
        """, unsafe_allow_html=True)

    # --- Dynamic Font Injection ---
    font_path = CONFIG["FONTS"][selected_font_name]
    b64_font = get_base64_font(font_path)
    
    if b64_font:
        font_format = "truetype" if font_path.endswith(".ttf") else "opentype"
        mime_type = "font/ttf" if font_path.endswith(".ttf") else "font/opentype"
        st.markdown(f"""
        <style>
            @font-face {{
                font-family: 'DynamicRanjana';
                src: url('data:{mime_type};base64,{b64_font}') format('{font_format}');
            }}
            .visual-only {{
                font-family: 'DynamicRanjana', sans-serif !important;
            }}
        </style>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.error(f"⚠️ Font file not found at {font_path}")


    # --- Main UI ---
    st.markdown("""
        <div class="app-title">
            <h1>Lipi Snap</h1>
            <p>Ranjana Script Character Recognition</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
    
    dev_mapping = load_mapping(CONFIG["MAPPING_DEV"])
    newa_mapping = load_mapping(CONFIG["MAPPING_NEWA"])
    
    if not model:
        st.warning("Model weights not found. Run training first.")
        return
    
    # Upload
    st.markdown('<p class="sec-label">Upload</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    if uploaded_file:
        processed = preprocess_image(uploaded_file, apply_opencv=apply_opencv, already_dark_bg=already_dark_bg)
        
        st.markdown('<p class="sec-label" style="margin-top:20px;">Pipeline</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            uploaded_file.seek(0)
            raw = Image.open(uploaded_file).convert('RGB')
            st.markdown('<div class="img-frame">', unsafe_allow_html=True)
            st.image(raw, width="stretch")
            st.markdown('<p>Uploaded</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="img-frame">', unsafe_allow_html=True)
            st.image(processed, width="stretch")
            st.markdown('<p>Processed (64×64 Grayscale)</p></div>', unsafe_allow_html=True)
        
        st.markdown("")
        
        if st.button("Recognize", type="primary"):
            with st.spinner(""):
                label, confidence = predict(processed, model, classes)
                devanagari = dev_mapping.get(label, "—")
                newa_char = newa_mapping.get(label, "—")
                top_preds = get_top_predictions(processed, model, classes, top_k=5)
            
            st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
            st.markdown('<p class="sec-label">Result</p>', unsafe_allow_html=True)
            
            bar_cls = "hi" if confidence >= 90 else ("mid" if confidence >= 60 else "lo")
            
            r1, r2, r3, r4 = st.columns(4, gap="small")
            
            with r1:
                st.markdown(f"""
                    <div class="r-card">
                        <p class="lbl">Class</p>
                        <p class="val purple">{label}</p>
                    </div>
                """, unsafe_allow_html=True)
            with r2:
                st.markdown(f"""
                    <div class="r-card">
                        <p class="lbl">Ranjana</p>
                        <div class="val newa-char newa-char-container">
                            <span class="copy-only">{newa_char}</span>
                            <span class="visual-only">{devanagari}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            with r3:
                st.markdown(f"""
                    <div class="r-card">
                        <p class="lbl">Devanagari</p>
                        <p class="val pink">{devanagari}</p>
                    </div>
                """, unsafe_allow_html=True)
            with r4:
                st.markdown(f"""
                    <div class="r-card">
                        <p class="lbl">Confidence</p>
                        <p class="val green">{confidence:.1f}%</p>
                        <div class="conf-bg">
                            <div class="conf-fill {bar_cls}" style="width:{confidence}%;"></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Top predictions
            st.markdown('<p class="sec-label" style="margin-top:24px;">Top Predictions</p>', unsafe_allow_html=True)
            
            max_conf = top_preds[0][1] if top_preds else 100
            for p_label, p_conf in top_preds:
                p_dev = dev_mapping.get(p_label, "—")
                p_newa = newa_mapping.get(p_label, "—")
                bar_w = (p_conf / max_conf) * 100 if max_conf > 0 else 0
                st.markdown(f"""
                    <div class="pred-row">
                        <span class="pred-newa-container">
                            <span class="copy-only">{p_newa}</span>
                            <span class="visual-only">{p_dev}</span>
                        </span>
                        <span class="pred-name">{p_label}</span>
                        <span class="pred-dev">{p_dev}</span>
                        <div class="pred-bar-bg">
                            <div class="pred-bar-fill" style="width:{bar_w}%;"></div>
                        </div>
                        <span class="pred-pct">{p_conf:.2f}%</span>
                    </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()