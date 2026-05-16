import json
import sys
sys.path.append('.')
from preprocessing.cleaner import preprocess

# Load Nole's data
with open('data/raw/tenders.json', 'r', encoding='utf-8') as f:
    tenders = json.load(f)

# Test on first 3 tenders
for tender in tenders[:3]:
    print("Original:", tender['all_data'][:80])
    print("Cleaned: ", preprocess(tender))
    print("---")