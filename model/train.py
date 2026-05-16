import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.cleaner import preprocess

KEYWORDS = {
    "CAT-01": ["computer", "server", "network", "software", "it", "cyber", "laptop", "printer"],
    "CAT-02": ["arduino", "sensor", "motor", "relay", "robot", "electronic", "circuit"],
    "CAT-03": ["building", "construction", "road", "bridge", "civil", "concrete"],
    "CAT-04": ["hospital", "medical", "laboratory", "diagnostic", "clinic", "health"],
    "CAT-05": ["furniture", "printer", "stationery", "office", "chair", "desk"],
    "CAT-06": ["vehicle", "truck", "bus", "fuel", "transport", "car", "ambulance"],
    "CAT-07": ["training", "consulting", "audit", "management", "service", "consultancy"],
}

def auto_label(text: str) -> str:
    text_lower = text.lower()
    scores = {}
    for cat, keywords in KEYWORDS.items():
        scores[cat] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "CAT-08"

def train_model(data_path: str):
    df = pd.read_csv(data_path)
    X = df["text"].tolist()
    y = df["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, C=1.0))
    ])

    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, "model/classifier_v1.joblib")
    print("✅ Model saved!")
    return pipeline, X_test, y_test

if __name__ == "__main__":
    train_model("data/labeled/tenders_labeled.csv")