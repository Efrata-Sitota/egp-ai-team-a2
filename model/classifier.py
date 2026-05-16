import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.cleaner import preprocess

CATEGORY_NAMES = {
    "CAT-01": "ICT & Software",
    "CAT-02": "Engineering & Electronics",
    "CAT-03": "Construction & Civil",
    "CAT-04": "Medical Equipment",
    "CAT-05": "Office & Stationery",
    "CAT-06": "Vehicles & Transport",
    "CAT-07": "Services & Consultancy",
    "CAT-08": "Other / Unclassified",
}

def classify_tender(tender: dict) -> dict:
    model_path = "model/classifier_v1.joblib"

    if not os.path.exists(model_path):
        return {"error": "Model not trained yet. Run train.py first."}

    pipeline = joblib.load(model_path)
    text = preprocess(tender)
    category = pipeline.predict([text])[0]
    probabilities = pipeline.predict_proba([text])[0]
    confidence = max(probabilities)

    if confidence < 0.70:
        category = "CAT-08"

    return {
        "category_id": category,
        "category_name": CATEGORY_NAMES[category],
        "confidence": round(confidence * 100, 1)
    }