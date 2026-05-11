# 🌟 Lipi Snap 
*As a part of Python project - SIC at IOE, Pulchowk Campus*

## 📜 Overview
**Ranjana Script Recognition** — A deep learning project implementing a Convolutional Neural Network (CharCNN) to classify and recognize ancient Ranjana (Nepal Lipi) characters and map them to their corresponding Devanagari labels - utilizing PyTorch, OpenCV for preprocessing, and Streamlit for a beautiful real-time inference  UI.

## ✨ Features
- **Research-Backed Architecture** as per the Bati & Dawadi paper.
- **Robust Preprocessing**: Real-time OpenCV pipeline (Grayscale → Otsu's Binarisation → Inversion → 64x64 Resize) to handle raw images perfectly.
- **Devanagari Transliteration**: Automatically maps the recognized Ranjana character to its Devanagari equivalent.

## 🚀 Quick Start

### 1. Setup Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Training
```bash
python model/train.py
``` 
*Automatically saves the best performing model to model/best_char_cnn.pth.*

### Run the App
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501)

## 📁 Project Structure
```
Lipi-Snap/
├── app.py                 # Streamlit UI & Visual Inference
├── model/                 # Neural Network Code
│   ├── train.py           # CharCNN Training Script
│   ├── extend_training.py # Optional script to continue training from a previously saved best model.
│   ├── inference.py       # CLI Single-Image Inference
│   ├── evaluate.py        # Test the model
│   ├── audit.py           # Audit the model
│   └── best_char_cnn.pth  # Saved Model Weights (Generated)
├── mapping/               # JSON Dictionary Mappings
├── data/                  # Merged Datasets
├── generated_outputs/     # Generated Outputs (Reports, Predictions, etc.)
├── tools/                 # Data Processing Tools
├── requirements.txt       # Project Dependencies
├── .gitignore
├── DATASET.md             # Dataset documentation
└── README.md              # Overall documentation

```

## 🎯 Usage

### Training
To train the model from scratch on the merged dataset:
```bash
python model/train.py
```
*Note: The script will automatically save the best performing model to `model/best_char_cnn.pth`.*

### Web Interface
Run the Streamlit app to test the model interactively. You can upload any Ranjana character image, and the app will handle the binarisation and inversion automatically before feeding it to the model.
```bash
streamlit run app.py
```

### CLI Inference
You can also run inference directly from the terminal on a single image:
```bash
python model/inference.py path/to/image.png
```

## 📊 Dataset
Refer to [DATASET.md](DATASET.md) for details on how the datasets were sourced, merged, and structured.