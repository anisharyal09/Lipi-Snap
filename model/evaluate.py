"""
evaluate.py — PyTorch Evaluation Script

Loads a trained model, performs inference on the test dataset, and reports accuracy.
Outputs a CSV of wrong predictions.
"""
import os
import json
import argparse
import csv
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


class CharCNN(nn.Module):
    # Convolutional Neural Network for Ranjana script character classification.
    def __init__(self, num_classes=68):
        super(CharCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


def load_mapping(mapping_path):
    # Load a class mapping from a JSON file.
    if not mapping_path or not os.path.exists(mapping_path):
        return None
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    # Normalize mapping to dict of str(index) -> class_name
    # Accept formats: {"0": "a"} or {"a": 0} or list
    if isinstance(mapping, list):
        return {str(i): v for i, v in enumerate(mapping)}

    # If mapping keys look like ints -> values are class names
    keys_are_ints = all(k.isdigit() for k in map(str, mapping.keys()))
    if keys_are_ints:
        return {str(int(k)): v for k, v in mapping.items()}

    # Otherwise assume class->idx, invert to idx->class
    inv = {str(v): k for k, v in mapping.items()}
    return inv


def evaluate(model, loader, device, mapping, dataset):
    # Evaluate the model over the given DataLoader.
    model.eval()
    correct = 0
    total = 0
    wrong_samples = []
    sample_paths = [path for path, _ in dataset.samples]

    # If mapping is provided: mapping maps model_idx (str) -> class_name
    use_name_compare = mapping is not None

    # Prepare dataset idx->class name
    dataset_idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    seen = 0

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1).cpu()
            preds = outputs.argmax(1).cpu()
            targets_cpu = targets.cpu()

            # Compare predictions with true labels
            for i, (p, t) in enumerate(zip(preds.tolist(), targets_cpu.tolist())):
                pred_idx = int(p)
                true_idx = int(t)
                
                # Resolve class names if mapping is provided
                pred_name = mapping.get(str(pred_idx), str(pred_idx)) if use_name_compare else str(pred_idx)
                true_name = dataset_idx_to_class.get(true_idx, str(true_idx))

                # Check accuracy based on names or indices
                if use_name_compare:
                    is_correct = str(pred_name) == str(true_name)
                else:
                    is_correct = pred_idx == true_idx

                if is_correct:
                    correct += 1
                else:
                    # Record details of wrong predictions for analysis
                    sample_idx = seen + i
                    image_path = sample_paths[sample_idx] if sample_idx < len(sample_paths) else ''
                    confidence = float(probs[i, pred_idx].item())
                    wrong_samples.append({
                        'image_path': image_path,
                        'true_label': true_name,
                        'pred_label': str(pred_name),
                        'true_idx': true_idx,
                        'pred_idx': pred_idx,
                        'confidence': confidence,
                    })

                total += 1

            seen += len(targets_cpu)

    return correct, total, wrong_samples


def save_wrong_samples(wrong_samples, output_path):
    # Save the list of wrong predictions to a CSV file.
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['image_path', 'true_label', 'pred_label', 'true_idx', 'pred_idx', 'confidence']
        )
        writer.writeheader()
        writer.writerows(wrong_samples)


def infer_num_classes_from_state_dict(state_dict):
    # Inspect the model state dict to infer the number of output classes.
    # Match the final classifier layer in this architecture.
    key = 'classifier.13.weight'
    if key in state_dict and hasattr(state_dict[key], 'shape'):
        return int(state_dict[key].shape[0])
    return None


def main():
    # Main evaluation pipeline
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='model/best_char_cnn.pth')
    parser.add_argument('--mapping', default='mapping/idx_to_class.json')
    parser.add_argument('--data', default='data/merged_dataset/test')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default=None)
    parser.add_argument('--num-classes', type=int, default=None)
    parser.add_argument('--wrong-out', default='model/wrong_predictions.csv')
    parser.add_argument('--show-wrong-limit', type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    if not os.path.exists(args.data):
        print(f"ERROR: data folder not found: {args.data}")
        return

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.ImageFolder(args.data, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load the saved model weights
    try:
        ckpt = torch.load(args.model, map_location=device)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            state = ckpt

        ckpt_classes = None
        if isinstance(ckpt, dict) and isinstance(ckpt.get('classes'), list):
            ckpt_classes = ckpt['classes']

        # handle possible 'module.' prefixes
        new_state = {}
        for k, v in state.items():
            nk = k.replace('module.', '')
            new_state[nk] = v

        inferred_num_classes = infer_num_classes_from_state_dict(new_state)
        num_classes = args.num_classes or inferred_num_classes or (len(ckpt_classes) if ckpt_classes else 68)
        model = CharCNN(num_classes=num_classes).to(device)
        model.load_state_dict(new_state)
        print(f"Loaded model: {args.model}")
        print(f"Model output classes: {num_classes}")

        if ckpt_classes:
            print(f"Checkpoint classes found: {len(ckpt_classes)}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Prefer checkpoint classes to ensure order matches training exactly
    if 'ckpt_classes' in locals() and ckpt_classes:
        mapping = {str(i): cls_name for i, cls_name in enumerate(ckpt_classes)}
        print("Using class mapping from checkpoint metadata.")
    else:
        mapping = load_mapping(args.mapping)
        if mapping is None:
            print(f"No mapping loaded from {args.mapping}. Comparing indices directly.")
        else:
            print(f"Loaded mapping from {args.mapping} (using name comparison).")
            if inferred_num_classes is not None and len(mapping) != inferred_num_classes:
                print(
                    f"WARNING: mapping has {len(mapping)} classes but model outputs {inferred_num_classes}. "
                    "This can cause false 'wrong' reports."
                )

    correct, total, wrong_samples = evaluate(model, loader, device, mapping, dataset)

    if total > 0:
        acc = (correct / total) * 100
        print(f"\nRESULT: {correct}/{total} correct — Accuracy: {acc:.2f}%")
        print(f"Wrong predictions: {len(wrong_samples)}")

        if wrong_samples:
            save_wrong_samples(wrong_samples, args.wrong_out)
            print(f"Saved wrong predictions to: {args.wrong_out}")
            limit = min(args.show_wrong_limit, len(wrong_samples))
            print(f"\nFirst {limit} wrong predictions:")
            for i in range(limit):
                row = wrong_samples[i]
                print(
                    f"{i + 1:>3}. {row['image_path']} | "
                    f"true={row['true_label']} ({row['true_idx']}) | "
                    f"pred={row['pred_label']} ({row['pred_idx']}) | "
                    f"conf={row['confidence']:.4f}"
                )
    else:
        print("No images evaluated.")


if __name__ == '__main__':
    main()
