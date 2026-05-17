# for generating more words dataset for testing on new datasets 
# (to determine - test accuracy)

import urllib.request
import re
import ssl
import sys

VOCAB_STR = "ँंःअआइईउऊऋएऐऒओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहाािीुूृेैोौ्०१२३४५६७८९"
valid_chars = set(VOCAB_STR)

# Load existing words
existing_file = 'data/nepali_words_clean.txt'
try:
    with open(existing_file, 'r', encoding='utf-8') as f:
        existing = set([line.strip() for line in f if line.strip()])
except FileNotFoundError:
    print(f"Could not find {existing_file}")
    sys.exit(1)

print(f"Existing words in dataset: {len(existing)}")

urls = [
    "https://ne.wikipedia.org/wiki/%E0%A4%AE%E0%A5%81%E0%A4%96%E0%A5%8D%E0%A4%AF_%E0%A4%AA%E0%A5%83%E0%A4%B7%E0%A5%8D%E0%A4%A0", # Main page
    "https://ne.wikipedia.org/wiki/%E0%A4%A8%E0%A5%87%E0%A4%AA%E0%A4%BE%E0%A4%B2", # Nepal
    "https://ne.wikipedia.org/wiki/%E0%A4%95%E0%A4%BE%E0%A4%A0%E0%A4%AE%E0%A4%BE%E0%A4%A1%E0%A5%8C%E0%A4%82", # Kathmandu
    "https://ne.wikipedia.org/wiki/%E0%A4%B8%E0%A4%97%E0%A4%B0%E0%A4%AE%E0%A4%BE%E0%A4%A5%E0%A4%BE", # Mount Everest
    "https://ne.wikipedia.org/wiki/%E0%A4%97%E0%A5%8C%E0%A4%A4%E0%A4%AE_%E0%A4%AC%E0%A5%81%E0%A4%A6%E0%A5%8D%E0%A4%A7", # Gautam Buddha
]

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

new_words = set()
for url in urls:
    print(f"Fetching from {url}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        html = urllib.request.urlopen(req, context=ctx).read().decode('utf-8')
        words = re.findall(r'[\u0900-\u097F]+', html)
        for w in words:
            # Must be purely valid characters and > 1 character long
            if len(w) > 1 and all(c in valid_chars for c in w):
                if w not in existing:
                    new_words.add(w)
    except Exception as e:
        print(f"Error fetching {url}: {e}")

new_words_list = sorted(list(new_words))

out_file = 'data/new_test_words.txt'
with open(out_file, 'w', encoding='utf-8') as f:
    for w in new_words_list:
        f.write(w + '\n')

print(f"\nFound {len(new_words_list)} NEW valid words!")
print(f"Saved to: {out_file}")
