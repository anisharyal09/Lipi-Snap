"""
data_split_A.py — Dataset Splitter

Splits Dataset A (ranjana-lipi) into train and validation folders for easier handling and model training.
"""
import os, shutil, random

random.seed(42)

SRC = "data/ranjana-lipi/Dataset"   # Dataset A path
DST = "data/data_A"          # Destination base directory
TRAIN_RATIO = 0.8

def is_img(x):
    x = x.lower()
    return x.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))

for cls in os.listdir(SRC):
    src_cls = os.path.join(SRC, cls)
    if not os.path.isdir(src_cls): 
        continue

    images = [f for f in os.listdir(src_cls) if is_img(f)]
    random.shuffle(images)

    n_train = int(len(images) * TRAIN_RATIO)
    train_imgs = images[:n_train]
    val_imgs   = images[n_train:]

    for split, items in [("train", train_imgs), ("val", val_imgs)]:
        dst_cls = os.path.join(DST, split, cls)
        os.makedirs(dst_cls, exist_ok=True)
        for img in items:
            shutil.copy(os.path.join(src_cls, img), os.path.join(dst_cls, img))

print("DONE → Dataset A is now split into data_A/train and data_A/val.")