"""Fix ranjana NLG_v3.ttf for web embedding (removes corrupt kern table, adds dev2 script)."""
import copy
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables.otTables import ScriptRecord

SRC = "font/ranjana NLG_v3.ttf"
DST = "font/ranjana_NLG_v3_fixed.ttf"

font = TTFont(SRC)

# Remove corrupt/empty tables that cause OTS parsing errors in browsers
for tag in ('kern', 'PCLT'):
    if hasattr(font, 'reader') and font.reader and tag in font.reader:
        del font.reader.tables[tag]

# Force-load all remaining tables for clean serialization
for tag in list(font.keys()):
    try: _ = font[tag]
    except: pass

# Add dev2 script tag (required by modern browsers for Devanagari shaping)
for table_tag in ("GSUB", "GPOS"):
    if table_tag not in font:
        continue
    records = font[table_tag].table.ScriptList.ScriptRecord
    has_deva = next((r for r in records if r.ScriptTag == "deva"), None)
    has_dev2 = any(r.ScriptTag == "dev2" for r in records)
    if has_deva and not has_dev2:
        dev2 = ScriptRecord()
        dev2.ScriptTag = "dev2"
        dev2.Script = copy.deepcopy(has_deva.Script)
        records.append(dev2)

font.save(DST)
print(f"Saved {DST}")
