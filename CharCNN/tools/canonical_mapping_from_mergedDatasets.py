"""
Generates JSON mappings of merged dataset classes(deva) to indices (numeric)
and vice versa (class -> idx and idx -> class).
"""

import json
import os

# load all classes from merged_dataset in sorted order
train_path = "data/merged_dataset/train"
classes = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])

print(f"Total classes: {len(classes)}")
print(f"Classes: {classes}")

# Create mapping
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
idx_to_class = {str(idx): cls for cls, idx in class_to_idx.items()}

# Define mapping directory
mapping_dir = "mapping"

# Save mappings to JSON
class_to_idx_path = os.path.join(mapping_dir, "class_to_idx.json")
idx_to_class_path = os.path.join(mapping_dir, "idx_to_class.json")

with open(class_to_idx_path, 'w') as f:
    json.dump(class_to_idx, f, indent=2)
print(f"\nUpdated: {class_to_idx_path}")

with open(idx_to_class_path, 'w') as f:
    json.dump(idx_to_class, f, indent=2)
print(f"Updated: {idx_to_class_path}")

print("\nclass_to_idx:")
print(json.dumps(class_to_idx, indent=2))

print("\nidx_to_class:")
print(json.dumps(idx_to_class, indent=2))