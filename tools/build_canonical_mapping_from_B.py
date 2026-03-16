import os, json

DATASET_A_DIR = "data/data_B_relabel/train"

classes = sorted([
    d for d in os.listdir(DATASET_A_DIR)
    if os.path.isdir(os.path.join(DATASET_A_DIR, d))
])

os.makedirs("mapping", exist_ok=True)

with open("mapping/class_to_idx.json", "w", encoding="utf-8") as f:
    json.dump({c:i for i,c in enumerate(classes)}, f, ensure_ascii=False, indent=2)
with open("mapping/idx_to_class.json", "w", encoding="utf-8") as f:
    json.dump({i:c for i,c in enumerate(classes)}, f, ensure_ascii=False, indent=2)

print("Canonical classes:", classes)
print("Saved mapping/class_to_idx.json & mapping/idx_to_class.json")