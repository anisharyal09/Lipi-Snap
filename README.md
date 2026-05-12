# 🌟 Lipi Snap - Ranjana Script Character Recognition Model
*A Capstone Python Project for the Samsung Innovation Campus (SIC) at IOE, Pulchowk Campus*

## 📜 Overview
**Ranjana Script Character Recognition Model** — A deep learning project implementing a Convolutional Neural Network (CharCNN) to classify and recognize ancient Ranjana (Nepal Lipi) characters and map them to their corresponding Devanagari labels - utilizing PyTorch, OpenCV for preprocessing, and Streamlit for a beautiful real-time inference UI.


## ✨ Features
- **Character-level recognition** — The model is trained to recognize individual Ranjana characters (handwritten or typed), not whole words or sentences.
- **PyTorch-based CharCNN** — A custom implementation of the Bati & Dawadi architecture, optimized for high-precision recognition of 62 Ranjana classes.
- **Streamlit Web UI** — Dark-themed interface with real-time image upload, preprocessing preview, and top-5 predictions with confidence bars.
- **Robust Preprocessing** — OpenCV pipeline (Otsu → Binarise → Invert → 64×64) with dark-background toggle.
- **Dual Script Output** — Displays both predicted Ranjana (Newa) Transcription and Devanagari Transliteration side-by-side.
- **Dynamic Font & Newa Unicode (U+11400) Support** — Switch between Normal and Stylish Ranjana display fonts (dynamically injected via base64) while maintaining copy-pasteable Ranjana Unicode under the hood via CSS overlay idea.


## 📈 Performance
- **Test Accuracy**: **99.70%** (34,617 / 34,720 correct on the test set)
- **Validation Accuracy**: **98.68%** (achieved at Epoch 61)

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
├── font/                  # Ranjana Display Fonts (.otf/.ttf)
├── mapping/               # JSON Dictionary Mappings
├── data/                  # Merged Datasets
├── generated_outputs/     # Generated Outputs (Reports, Predictions, etc.)
├── tools/                 # Data Processing & Font Patching Scripts
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
To run inference on a single image, execute:
```bash
python model/inference.py path/to/image.png
```

## 📊 Dataset
Refer to [DATASET.md](DATASET.md) for details on how the datasets were sourced, merged, and structured.

## Acknowledgments
- **Ranjana Fonts** — NithyaRanjanaDU and Ranjana NLG fonts sourced via [Callijatra](https://www.facebook.com/callijatra/posts/883335911018659/).
- **Bati & Dawadi's CNN Architecture** — [Ranjana Script Handwritten Character Recognition using CNN](https://www.researchgate.net/publication/374169328_Ranjana_Script_Handwritten_Character_Recognition_using_CNN)