"""
Relabel Dataset-B numeric class folders to canonical label folders.

Reads mapping from: mapping/relabel_dataset_B.json  (index -> label)
Source:             data/data_B/{train,val,test}/<numeric>/
Destination:        data/data_B_relabel/{train,val,test}/<label>/
"""

import os
import json
import shutil
import argparse
from pathlib import Path

# --- Configuration ---
MAPPING_PATH = "mapping/relabel_dataset_B.json"
DATA_B_ROOT  = "data/data_B"          # Source folder
TARGET_B_ROOT = "data/data_B_relabel" # Destination folder
SPLITS = ["train", "val", "test"]

# Supported image extensions
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def load_mapping(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    # Ensure keys are strings for dictionary lookups
    return {str(k): v for k, v in mapping.items()}

def relabel_split(src_split_dir: Path, dst_split_dir: Path, mapping: dict, mode: str, dry_run: bool) -> None:
    # Iterates over numeric folders in src_split_dir and moves/copies content to dst_split_dir.
    if not src_split_dir.is_dir():
        print(f"[WARN] Source split missing: {src_split_dir}")
        return

    # Find numeric folders (e.g., '0', '1', '10') in the SOURCE
    numeric_folders = [d for d in src_split_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    
    if not numeric_folders:
        print(f"[INFO] No numeric folders found in {src_split_dir}")
        return

    print(f"\n--- Processing Split: {src_split_dir.name} ---")
    
    for num_dir in sorted(numeric_folders, key=lambda p: int(p.name)):
        num = num_dir.name
        if num not in mapping:
            print(f"[SKIP] {num} -> No mapping found in {MAPPING_PATH}")
            continue

        label = mapping[num]
        final_dst_dir = dst_split_dir / label

        if dry_run:
            count = sum(1 for p in num_dir.iterdir() if p.is_file() and is_image(p))
            print(f"[DRY] {num_dir.relative_to(DATA_B_ROOT)} -> {label} ({count} files)")
            continue

        # Ensure the target label directory exists
        final_dst_dir.mkdir(parents=True, exist_ok=True)

        moved_count = 0
        for p in num_dir.iterdir():
            if p.is_file() and is_image(p):
                target = final_dst_dir / p.name
                
                # Resolve naming conflicts by appending an incrementing suffix
                if target.exists():
                    stem, suffix = p.stem, p.suffix
                    i = 1
                    while (final_dst_dir / f"{stem}_{i}{suffix}").exists():
                        i += 1
                    target = final_dst_dir / f"{stem}_{i}{suffix}"

                if mode == "move":
                    shutil.move(str(p), str(target))
                else:
                    shutil.copy2(str(p), str(target))
                moved_count += 1

        print(f"[OK]  {num} -> {label} ({moved_count} files, mode={mode})")

        # Remove the source folder if it is now empty
        if mode == "move" and not any(num_dir.iterdir()):
            try:
                num_dir.rmdir()
            except Exception as e:
                print(f"[WARN] Could not remove empty folder {num_dir}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["copy", "move"], default="copy",
                        help="copy (keep source) or move (delete source) files")
    parser.add_argument("--dry-run", action="store_true",
                        help="preview actions without modifying files")
    args = parser.parse_args()

    # 1. Check Mapping
    if not os.path.isfile(MAPPING_PATH):
        print(f"[ERR] Mapping file not found: {MAPPING_PATH}")
        return
    mapping = load_mapping(MAPPING_PATH)

    # 2. Check Source Root
    src_root = Path(DATA_B_ROOT)
    if not src_root.exists():
        print(f"[ERR] Source directory '{DATA_B_ROOT}' does not exist.")
        return

    # 3. Process Splits
    for split in SPLITS:
        src_split_path = src_root / split
        dst_split_path = Path(TARGET_B_ROOT) / split
        
        relabel_split(src_split_path, dst_split_path, mapping, args.mode, args.dry_run)

    print("\nProcessing complete.")
    if args.dry_run:
        print("NOTE: This was a dry run. No files were actually moved/copied.")

if __name__ == "__main__":
    main()