from fastapi import APIRouter
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.classifier import classify_tender

router = APIRouter()

@router.post("/classify")
def classify(tender: dict):
    result = classify_tender(tender)
    return result