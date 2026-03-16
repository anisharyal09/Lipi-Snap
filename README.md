# 🌟 Lipi Snap 
*As a part of Python project - SIC, IoE, Pulchowk Campus*

## Phase 1
**Ranjana Script Recognition** - Streamlit app + PyTorch CNNs for ancient Ranjana characters → Devanagari mapping.

(*Not yet hosted*)

## ✨ Features
- Inference with 2 models: (Dataset A) vs (Dataset B)
- Image upload + preprocessing (binarize/invert option)
- Devanagari character mapping
- Training scripts included

## 🚀 Quick Start

### 1. Setup Virtual Environment
```bash
python -m venv lipi-snap-venv
source lipi-snap-venv/bin/activate  # macOS/Linux
# lipi-snap-venv\Scripts\activate  # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```
Open http://localhost:8501

## 📁 Project Structure
```
Lipi-Snap/
├── app.py                 # Streamlit UI + inference
├── model/                 # PyTorch CNNs
│   ├── train_A.py        # Train Dataset A model
│   └── train_B.py        # Train Dataset B model (recommended)
├── tools/                 # Data processing
│   ├── data_split_A.py
│   └── relabel_dataset_B.py
├── mapping/               # JSON mappings
│   ├── ranjana_to_devanagari.json
│   └── relabel_dataset_B.json
├── data/                  # Datasets here after prep
├── requirements.txt
└── DATASET.md            # Dataset download/processing
```

## 🎯 Usage

### Inference
Upload Ranjana character image in the app. Toggle binarize for Model B.

### Training
See trained models in `model/best_char_cnn_data_a.pth` or `_b.pth`.
 *"Note: Models are generated after running the training scripts and are excluded"*

**Dataset A:**
```bash
python model/train_A.py
```

**Dataset B (Recommended):**
```bash
python model/train_B.py
```

### Data Preparation
**Refer to [DATASET.md](DATASET.md)** for Kaggle download (kagglehub), splitting & relabeling.

## 🔗 Mappings
- Ranjana labels → Devanagari: `mapping/ranjana_to_devanagari.json`
- Dataset B indices → labels: `mapping/relabel_dataset_B.json`