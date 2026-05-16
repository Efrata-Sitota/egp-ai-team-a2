# EGP AI Data Intelligence Tool
import re
import spacy

nlp = spacy.load("en_core_web_sm")

def clean_text(text: str) -> str:
    if not text:
        return ""
    # Lowercase
    text = text.lower()
    # Remove Lot: 1 tags and extra whitespace
    text = re.sub(r'lot:\s*\d+', '', text)
    # Keep Amharic unicode
    text = re.sub(r'[^\w\s\u1200-\u137F]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lemmatize(text: str) -> str:
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and len(token.text) > 2
    ]
    return " ".join(tokens)

def preprocess(tender: dict) -> str:
    # Handle Nole's format — all data in one field
    raw = tender.get("all_data", "")
    cleaned = clean_text(raw)
    return lemmatize(cleaned)