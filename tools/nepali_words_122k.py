

from datasets import load_dataset

print("Downloading the 122k Brihat Sabdakosh dataset...")

ds = load_dataset("Titung/nepali-brihat-sabdakosh")

# Extract just the 'word' column
words = ds['train']['word']

# Save to your OCR project directory
output_file = "nepali_words.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for word in words:
        # Strip any accidental whitespace just to be safe
        clean_word = word.strip()
        if clean_word:
            f.write(f"{clean_word}\n")

print(f"✅ Successfully saved {len(words)} words to {output_file}!")