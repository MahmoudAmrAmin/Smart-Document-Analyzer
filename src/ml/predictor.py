import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, ROOT_DIR)

from src.ml.classifier import load_model
from src.preprocessing.text_cleaner import clean_text

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


def predict_document_type(text: str) -> dict:
    """
    Predict document type from text.

    Returns:
        {
            "label":      "contract" | "invoice" | "research",
            "confidence": float,
            "all_scores": {"contract": float, ...}
        }
    """
    model   = _get_model()
    cleaned = clean_text(text)
    snippet = " ".join(cleaned.split()[:1000])

    proba      = model.predict_proba([snippet])[0]
    classes    = model.classes_
    label      = classes[proba.argmax()]
    confidence = float(proba.max())
    all_scores = {c: round(float(p), 4)
                  for c, p in zip(classes, proba)}

    return {
        "label":      label,
        "confidence": round(confidence, 4),
        "all_scores": all_scores
    }


def predict_from_pdf(pdf_path: str) -> dict:
    """Predict directly from a PDF file path."""
    from src.ingestion.pdf_loader import extract_text_from_pdf
    text = extract_text_from_pdf(pdf_path)
    return predict_document_type(text)