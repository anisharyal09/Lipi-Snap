"""
Lipi Snap — CRNN + CTC Word Recognition UI
============================================
Streamlit app for recognizing full Ranjana script words.
Uses the CRNN model trained with CTC loss.
"""

import os
import cv2
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import base64
from PIL import Image
from torchvision import transforms

# --- Configuration ---
CONFIG = {
    "MODEL_PATH": "model/best_crnn.pth",
    "MAPPING_DEV": "mapping/ranjana_to_devanagari.json",
    "MAPPING_NEWA": "mapping/ranjana_unicode.json",
    "IMG_HEIGHT": 64,
    "IMG_WIDTH": 512,
    "FONTS": {
        "Normal": "font/NithyaRanjanaDU-Regular.otf",
        "Stylish": "font/ranjana_NLG_v3_fixed.ttf"
    }
}

# --- Vocabulary ---
VOCAB_STR = "ँंःअआइईउऊऋएऐऒओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहाािीुूृेैोौ्०१२३४५६७८९"
CHARS = sorted(list(set(VOCAB_STR)))

# --- Device ---
@st.cache_resource
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

# --- Label Converter ---
class LabelConverter:
    def __init__(self, chars):
        self.chars = ["<blank>"] + chars
        self.char_to_idx = {char: i for i, char in enumerate(self.chars)}
        self.idx_to_char = {i: char for i, char in enumerate(self.chars)}

    def decode(self, indices):
        return "".join([self.idx_to_char[idx] for idx in indices if idx != 0])

    def num_classes(self):
        return len(self.chars)

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

# --- Preprocessing ---
def preprocess_image(source, apply_opencv=True, already_dark_bg=False):
    if isinstance(source, str):
        img = cv2.imread(source, cv2.IMREAD_COLOR)
    else:
        file_bytes = np.asarray(bytearray(source.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
    if img is None: return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if apply_opencv:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if not already_dark_bg and np.sum(binary == 255) > np.sum(binary == 0):
            binary = cv2.bitwise_not(binary)
        inverted = binary
    else:
        inverted = gray

    h, w = inverted.shape
    target_h, target_w = CONFIG["IMG_HEIGHT"], CONFIG["IMG_WIDTH"]
    new_w = min(int(target_h * (w / h)), target_w)
    resized = cv2.resize(inverted, (new_w, target_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    canvas[:, :new_w] = resized
    return Image.fromarray(canvas, mode="L"), Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

inference_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# --- Model Loading ---
@st.cache_resource
def load_trained_model():
    path = CONFIG["MODEL_PATH"]
    if not os.path.exists(path):
        return None, None
    
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    sd   = ckpt['model_state_dict']

    h1 = sd['rnn1.weight_ih_l0'].shape[0] // 4
    h2 = sd['rnn2.weight_ih_l0'].shape[0] // 4
    h3 = sd['rnn3.weight_ih_l0'].shape[0] // 4
    num_classes = sd['classifier.3.weight'].shape[0]
    kernel_size = sd['cnn.0.weight'].shape[2]  

    chars = ckpt.get('chars', CHARS)
    converter = LabelConverter(chars)

    model = CRNN(num_classes, h1, h2, h3, kernel_size=kernel_size).to(DEVICE)
    model.load_state_dict(sd)
    model.eval()
    return model, converter

# --- Inference ---
@torch.no_grad()
def predict_word(pil_img, model, converter):
    img_tensor = inference_transform(pil_img).unsqueeze(0).to(DEVICE)
    preds = model(img_tensor)

    probs = torch.softmax(preds, dim=2)
    max_probs, preds_idx = probs.max(2)
    preds_idx = preds_idx.squeeze(1).cpu().tolist()
    max_probs = max_probs.squeeze(1).cpu().tolist()

    decoded_indices = []
    decoded_confidences = []
    raw_path_chars = []
    
    last_idx = -1
    for i, idx in enumerate(preds_idx):
        char = converter.idx_to_char[idx]
        is_valid_char = (idx != 0 and idx != last_idx)
        raw_path_chars.append({"char": char, "valid": is_valid_char})
        
        if is_valid_char:
            decoded_indices.append(idx)
            decoded_confidences.append(max_probs[i])
        last_idx = idx

    text = converter.decode(decoded_indices)
    avg_confidence = sum(decoded_confidences) / len(decoded_confidences) * 100 if decoded_confidences else 0.0

    return text, avg_confidence, list(zip(text, [c * 100 for c in decoded_confidences])), raw_path_chars

# --- Utilities ---
@st.cache_data
def get_base64_font(font_path):
    if not os.path.exists(font_path):
        return ""
    with open(font_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# --- CSS (Minimal) ---
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp { font-family: 'Inter', sans-serif; }
    #MainMenu, footer { visibility: hidden; }

    /* Title */
    .app-title { text-align: center; padding: 1.5rem 0 0.5rem 0; width: 100%; }
    .app-title h1 { font-size: 2.2rem; font-weight: 700; color: #f3f4f6; margin: 0 auto; letter-spacing: -0.5px; }
    .app-title .subtitle { color: #9ca3af; font-size: 0.95rem; margin: 4px auto 0 auto; }

    /* Divider */
    .sep { height: 1px; background: rgba(255,255,255,0.08); margin: 24px 0; }
    
    /* Section label */
    .sec-label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: #9ca3af; margin-bottom: 12px; }

    /* Image frame */
    .img-frame { background: rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 10px; text-align: center; }
    .img-frame p { font-size: 0.7rem; font-weight: 500; color: #6b7280; margin: 8px 0 0 0; }

    /* Match Random Button Height to Uploader */
    [data-testid="stButton"] button[kind="secondary"] { height: 68px !important; }

    /* Word display (Neotic Minimal) */
    .word-display {
        background: linear-gradient(145deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 30px 20px;
        text-align: center;
        margin: 16px 0;
        box-shadow: 0 8px 32px 0 rgba(14, 165, 233, 0.15);
        backdrop-filter: blur(4px);
    }
    .word-display .word-text { font-size: 2.2rem; font-weight: 600; color: #f3f4f6; margin: 0; }
    .word-display .word-ranjana { font-size: 2.4rem; color: #fbbf24; margin: 8px 0 0 0; }

    /* Result cards */
    .r-card {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .r-card .lbl { font-size: 0.65rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: #9ca3af; margin-bottom: 6px; }
    .r-card .val { font-size: 1.4rem; font-weight: 600; margin: 0; color: #f3f4f6; }

    /* CTC Path Visualizer */
    .ctc-path-container { display: flex; overflow-x: auto; gap: 6px; padding: 12px; background: rgba(0,0,0,0.2); border-radius: 8px; scrollbar-width: none; }
    .ctc-path-container::-webkit-scrollbar { display: none; }
    .ctc-chip { padding: 4px 10px; border-radius: 6px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); font-size: 0.85rem; color: #6b7280; white-space: nowrap; flex-shrink: 0; }
    .ctc-chip.valid { background: rgba(52,211,153,0.1); border-color: rgba(52,211,153,0.3); color: #34d399; font-weight: 600; }
    .ctc-chip.blank { opacity: 0.4; font-size: 0.75rem; }

    /* Newa font rendering */
    .newa-char-container { position: relative; display: inline-flex; justify-content: center; align-items: center; }
    .visual-only { pointer-events: none; user-select: none; -webkit-user-select: none; }

    /* Character breakdown table */
    .char-row { display: flex; align-items: center; gap: 12px; padding: 8px 12px; border-radius: 8px; background: rgba(255,255,255,0.01); border: 1px solid rgba(255,255,255,0.03); margin-bottom: 4px; }
    .char-row .char-symbol { font-size: 1.1rem; font-weight: 600; color: #d1d5db; width: 30px; text-align: center; flex-shrink: 0; }
    .char-row .char-ranjana { font-size: 1.2rem; color: #fbbf24; width: 30px; text-align: center; flex-shrink: 0; }
    .char-row .char-bar-bg { flex-grow: 1; height: 3px; background: rgba(255,255,255,0.05); border-radius: 10px; overflow: hidden; }
    .char-row .char-bar-fill { height: 100%; border-radius: 10px; background: #60a5fa; }
    .char-row .char-pct { font-weight: 600; font-size: 0.75rem; color: #9ca3af; min-width: 45px; text-align: right; }

    /* Buttons */
    .stButton > button { border-radius: 8px !important; font-weight: 500 !important; padding: 6px 14px !important; }
    .stButton > button[kind="primary"] { background: #4f46e5 !important; border: none !important; color: white !important; }
    .stButton > button[kind="primary"]:hover { background: #4338ca !important; }
    .stButton > button[kind="secondary"] { background: transparent !important; border: 1px solid rgba(255,255,255,0.2) !important; color: #d1d5db !important; }
    .stButton > button[kind="secondary"]:hover { border-color: rgba(255,255,255,0.4) !important; color: white !important; }
    
    /* Expanders styling */
    .streamlit-expanderHeader { font-weight: 500 !important; color: #9ca3af !important; font-size: 0.85rem !important; }
</style>
"""

# --- App ---
def main():
    st.set_page_config(page_title="Lipi Snap", page_icon="📜", layout="centered")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### Typography")
        font_options = list(CONFIG["FONTS"].keys())
        default_idx = font_options.index("Normal") if "Normal" in font_options else 0
        selected_font_name = st.radio("Display Font", options=font_options, index=default_idx)
        
        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
        st.markdown("### Preprocessing")
        apply_opencv = st.checkbox("OpenCV Pipeline", value=True)
        already_dark_bg = st.checkbox("Dark Background Input", value=False)

        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

        st.markdown("### Model Info")
        model, converter = load_trained_model()
        if model and converter:
            total_params = sum(p.numel() for p in model.parameters())
            st.markdown(f"""
                <div style="font-size: 0.8rem; color: #9ca3af; line-height: 1.8;">
                    <strong>Architecture:</strong> CRNN + CTC<br>
                    <strong>Vocab Size:</strong> {converter.num_classes()}<br>
                    <strong>Parameters:</strong> {total_params:,}<br>
                    <strong>Device:</strong> {str(DEVICE).upper()}<br>
                    <strong>Input:</strong> {CONFIG['IMG_HEIGHT']}×{CONFIG['IMG_WIDTH']}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Model not loaded.")

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

    # --- Main UI ---
    st.markdown("""
        <div class="app-title">
            <h1>Lipi Snap</h1>
            <p class="subtitle">Ranjana Script Word Recognition</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

    if not model:
        st.warning("⚠️ Model weights not found at `model/best_crnn.pth`.")
        return

    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    with col2:
        use_random = st.button("Test Random Image", use_container_width=True, type="secondary")
        
    if 'img_source' not in st.session_state:
        st.session_state.img_source = None
        
    if uploaded_file:
        st.session_state.img_source = uploaded_file
    elif use_random:
        sample_images = glob.glob("data/test_synthetic_words/images/*.png")
        if sample_images:
            st.session_state.img_source = random.choice(sample_images)

    if st.session_state.img_source is not None:
        processed_img, raw_img = preprocess_image(st.session_state.img_source, apply_opencv=apply_opencv, already_dark_bg=already_dark_bg)

        with st.expander("View Vision Pipeline"):
            c1, c2 = st.columns(2, gap="small")
            with c1:
                st.markdown('<div class="img-frame">', unsafe_allow_html=True)
                st.image(raw_img, width="stretch")
                st.markdown('<p>Raw Input</p></div>', unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="img-frame">', unsafe_allow_html=True)
                st.image(processed_img, width="stretch")
                st.markdown(f'<p>Model Input ({CONFIG["IMG_HEIGHT"]}×{CONFIG["IMG_WIDTH"]})</p></div>', unsafe_allow_html=True)

        if st.button("Run OCR", type="primary", use_container_width=True):
            with st.spinner("Decoding..."):
                text, confidence, char_confs, raw_path_chars = predict_word(processed_img, model, converter)

            st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

            # --- Word Display ---
            st.markdown(f"""
                <div class="word-display">
                    <p class="word-text">{text if text else '—'}</p>
                    <div class="word-ranjana newa-char-container">
                        <span class="visual-only">{text if text else '—'}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # --- Stats Cards ---
            r1, r2, r3 = st.columns(3, gap="small")
            with r1:
                st.markdown(f"""<div class="r-card"><p class="lbl">Characters</p><p class="val">{len(text) if text else 0}</p></div>""", unsafe_allow_html=True)
            with r2:
                st.markdown(f"""<div class="r-card"><p class="lbl">Confidence</p><p class="val">{confidence:.1f}%</p></div>""", unsafe_allow_html=True)
            with r3:
                st.markdown(f"""<div class="r-card"><p class="lbl">Devanagari</p><p class="val">{text if text else '—'}</p></div>""", unsafe_allow_html=True)

            # --- Advanced Details Expander ---
            if raw_path_chars or char_confs:
                with st.expander("Detailed Analysis"):
                    if raw_path_chars:
                        st.markdown('<p class="sec-label" style="margin-top:8px;">CTC Decoding Path</p>', unsafe_allow_html=True)
                        chips_html = ""
                        for item in raw_path_chars:
                            if item["char"] == "<blank>":
                                chips_html += '<div class="ctc-chip blank">_</div>'
                            elif item["valid"]:
                                chips_html += f'<div class="ctc-chip valid">{item["char"]}</div>'
                            else:
                                chips_html += f'<div class="ctc-chip">{item["char"]}</div>'
                        
                        st.markdown(f"""<div class="ctc-path-container">{chips_html}</div>""", unsafe_allow_html=True)

                    if char_confs:
                        st.markdown('<p class="sec-label" style="margin-top:24px;">Confidence per Character</p>', unsafe_allow_html=True)
                        max_conf = max(c for _, c in char_confs) if char_confs else 100
                        for char, conf in char_confs:
                            bar_w = (conf / max_conf) * 100 if max_conf > 0 else 0
                            st.markdown(f"""
                                <div class="char-row">
                                    <span class="char-symbol">{char}</span>
                                    <span class="char-ranjana newa-char-container"><span class="visual-only">{char}</span></span>
                                    <div class="char-bar-bg"><div class="char-bar-fill" style="width:{bar_w}%;"></div></div>
                                    <span class="char-pct">{conf:.1f}%</span>
                                </div>
                            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
