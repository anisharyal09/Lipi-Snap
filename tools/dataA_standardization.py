"""
dataA_standardization.py — Image Standardization Script
"""

import cv2
import os

# Base directories
input_base_dir = "data/data_A"
output_base_dir = "data/data_A_standardized"

print("Starting deep scan for images...")

count = 0
# Recursively process all subdirectories
for root, dirs, files in os.walk(input_base_dir):
    for filename in files:
        # Only process image files
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(root, filename)
            
            # Read the image
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Determine relative path to maintain directory structure
            relative_path = os.path.relpath(root, input_base_dir)
            
            # Create corresponding output directory
            output_folder = os.path.join(output_base_dir, relative_path)
            os.makedirs(output_folder, exist_ok=True)
            
            # Apply preprocessing pipeline: grayscale -> binarize -> invert -> resize
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            inverted = cv2.bitwise_not(binary)
            resized = cv2.resize(inverted, (64, 64), interpolation=cv2.INTER_AREA)
            
            # Save processed image
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, resized)
            count += 1

print(f"\n✅ SUCCESS: Processed {count} images and preserved all alphabet subfolders!")