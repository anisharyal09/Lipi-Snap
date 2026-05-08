# 📜 Ranjana Lipi Dataset Management

This project utilizes a merged dataset comprised of two primary Kaggle sources. Dataset A is processed (binarised using Otsu's thresholding, inverted, and resized to 64x64 grayscale images) specifically to match Dataset B's exact format. For dataset A, labeling & splitting is done manually *(refer to additional notes)*, for dataset B, relabeling is done through script `relabel_dataset_B.py` as they are provided with numeric class labels. Then the processed datasets are merged together using `merge_datasets.py`.


## The Merged Dataset
Instead of training separate models on segmented datasets, we combined multiple sources into a single, unified dataset located at:
* `data/merged_dataset/train`
* `data/merged_dataset/val`

The merged dataset contains over 100,000 training images spread across 68 distinct character classes. 

### Preprocessing Standards
All images in the final dataset adhere to the following preprocessing format, which is required by the `CharCNN` architecture:
1. **Grayscale**: Single channel (`L` mode).
2. **Binarised**: Processed using Otsu's automatic thresholding.
3. **Inverted**: The strokes of the character are **white** (pixel value 255), and the background is **black** (pixel value 0).
4. **Resized**: Exactly 64x64 pixels.

*Note: When using `app.py` or `inference.py`, this exact preprocessing pipeline is automatically applied to raw user uploads using OpenCV before the image is fed into the Neural Network.*

---

## Source Datasets

### 1. Dataset A: Ranjana Lipi
* **Source:** [Kaggle: Ranjana Lipi](https://www.kaggle.com/datasets/shyboy123maharjan/ranjana-lipi)
*

### 2. Dataset B: Ranjana Script 64
* **Source:** [Kaggle: Ranjana Script 64](https://www.kaggle.com/datasets/a1a3bb5d8fc063cbed7fba4d5662df60b63c501be5766d30ef1dc7a441ef8fdb)
* **Mapping:** The original dataset used numeric folder names. These were programmatically mapped to their corresponding Devanagari/Ranjana class labels using `mapping/relabel_dataset_B.json`.

---

## Data Loading in PyTorch
The `model/train.py` script automatically loads the merged dataset using PyTorch's `ImageFolder`. 

Because the images are already binarised and inverted on disk, the training script only applies standard Deep Learning regularizations:
* Conversion to single channel (`transforms.Grayscale(1)`)
* Normalization to `[-1, 1]` (`transforms.Normalize([0.5], [0.5])`)
* Data Augmentation: Random rotations (±20°), scaling (0.8-1.2), and shearing to prevent overfitting (`transforms.RandomAffine`).


### Additional Note:

1. For the dataset A['shyboy123maharjan/ranjana-lipi'] — Cross-check ta & tta, da & dda, na & nga, tha & ttha, etc. are correctly labelled - ensuring labels follow the standard Devanagari mapping (see mapping/ranjana_to_devanagari.json)

  - Also rename 'lu' with 'li', 'luu' with 'lli', 'na' by 'nga' and 'nnna' by 'na', 'sa' by 'ssa' and 'saa' by 'sa', 'rii' by 'rri', etc. by tallying with standard devanagari & ranjana mapping in dataset A before splitting & merging!

2. Make sure to check and verify the datasets are completely cleaned and processed before training.

    *(Comment: Hard lesson for us as we didn't do it first and it frustrated us a lot, leading to wasted time - specially for datasets A, be careful with labeling.)*
