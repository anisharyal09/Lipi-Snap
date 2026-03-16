# 📜 Ranjana Lipi Dataset Management

This project utilizes two primary datasets for script recognition. Throughout the codebase and documentation, the terms **Dataset A/B** and **data_A/B** are used interchangeably to refer to the respective sources.

---

## 📁 Dataset A: Ranjana Lipi
* **Source:** [Kaggle: Ranjana Lipi](https://www.kaggle.com/datasets/shyboy123maharjan/ranjana-lipi)
* **Processing:** * Split into an **80/20** ratio for training and validation.
    * **Execution:** `python tools/data_split_A.py`
    * **Output:** `data/data_A/train` and `data/data_A/val`

## 📁 Dataset B: Ranjana Script 64 (Primary)
* **Source:** [Kaggle: Ranjana Script 64](https://www.kaggle.com/datasets/a1a3bb5d8fc063cbed7fba4d5662df60b63c501be5766d30ef1dc7a441ef8fdb)
* **Processing:**
    * Relabels numeric folder names into their corresponding **Devanagari classes**.
    * **Execution:** `python tools/relabel_dataset_B.py` 
    * **Mapping Logic:** `mapping/relabel_dataset_B.json`
    * **Output:** `data/data_B_relabel/{train,val,test}`

---

## ⚡ VS Code Quick Start

### 1. Environment Setup
Ensure `kagglehub` is installed in your active Python environment. You can initialize the data download directly from the VS Code terminal:
```bash
python tools/download_kaggle_dataset.py