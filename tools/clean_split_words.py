"""
Lipi Snap — Word Splitter & Cleaner Tool (Order Preserving)

This script reads a text file and:
1. Splits any lines containing multiple words (separated by spaces).
2. Filters out ANY word that contains characters not in our VOCAB_STR.
3. PRESERVES the original top-to-bottom order of the words.txt file (for no good reason yooo).
"""

import sys
from pathlib import Path

# The exact same vocabulary from your model
VOCAB_STR = "ँंःअआइईउऊऋएऐऒओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहाािीुूृेैोौ्०१२३४५६७८९"
VALID_CHARS = set(VOCAB_STR)

def split_and_clean_words():
    input_file  = Path("data/new_test_words.txt")
    output_file = Path("data/new_test_words_clean.txt")

    if not input_file.exists():
        print(f"❌ Error: Input file '{input_file}' not found.")
        return

    print(f"📖 Reading from: {input_file}")
    
    cleaned_words = []
    seen = set()  # To keep track of duplicates while preserving order
    total_words_found = 0
    invalid_words_dropped = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Split by whitespace (handles multiple words in one line)
            words = line.strip().split()
            for w in words:
                clean_word = w.strip()
                if not clean_word:
                    continue
                
                total_words_found += 1
                
                # Check if every character is in the VOCAB_STR
                if all(char in VALID_CHARS for char in clean_word):
                    if clean_word not in seen:
                        cleaned_words.append(clean_word)
                        seen.add(clean_word)
                else:
                    invalid_words_dropped += 1

    print(f"⚙️  Found {total_words_found} total words.")
    print(f"🗑️  Dropped {invalid_words_dropped} words (invalid symbols).")
    print(f"✨ Kept {len(cleaned_words)} unique words (Order Preserved).")

    print(f"💾 Saving to: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for w in cleaned_words:
            f.write(f"{w}\n")

    print("✅ Done! Order preserved.")

if __name__ == "__main__":
    split_and_clean_words()
