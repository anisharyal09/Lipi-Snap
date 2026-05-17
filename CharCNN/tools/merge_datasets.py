# Merge Dataset A (standardized) and Dataset B (relabelled) into one dataset.

import os
import shutil

def merge_datasets(source_dir, target_base_dir, prefix):
    """
    Copy images from a source directory to a target directory, maintaining
    the internal directory structure and prefixing filenames to prevent collisions.
    
    Args:
        source_dir (str): Path to the source dataset directory.
        target_base_dir (str): Path to the merged target directory.
        prefix (str): String prefix added to every copied file's name.
    """
    if not os.path.exists(source_dir):
        print(f"Warning: Cannot find {source_dir}. Skipping...")
        return

    print(f"⚙️ Merging {source_dir} into {target_base_dir} with prefix '{prefix}'...")
    
    count = 0
    # Walk through every subfolder in the source directory
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Figure out the relative path (e.g., 'train/ka')
                relative_path = os.path.relpath(root, source_dir)
                
                # Create corresponding output directory
                target_dir = os.path.join(target_base_dir, relative_path)
                os.makedirs(target_dir, exist_ok=True)
                
                src_file = os.path.join(root, filename)
                
                # Add the prefix to prevent overwriting files with the same name!
                new_filename = f"{prefix}_{filename}"
                dest_file = os.path.join(target_dir, new_filename)
                
                # Copy to merged directory
                shutil.copy2(src_file, dest_file)
                count += 1
                
    print(f"✅ Successfully copied {count} files from {source_dir}.")

# paths definition
dataset_b_dir = "data/data_B_relabel"   # labelled dataset B 
dataset_a_dir = "data/data_A_standardized" 
merged_output_dir = "data/merged_dataset"

# Run the merge
print("Starting the grand merge...\n")
merge_datasets(dataset_b_dir, merged_output_dir, "B")
merge_datasets(dataset_a_dir, merged_output_dir, "A")

print("\n🎉 Merge complete! Your combined dataset is ready in 'data/merged_dataset'.")