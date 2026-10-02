"""Load the trained model and classify text.

CLI:  python -m src.predict "your text here"
"""
import sys
from functools import lru_cache

import joblib

from src.config import MODEL_PATH


@lru_cache(maxsize=1)
def load_model():
    """Load the saved model once."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Train it first with: python -m src.train")
    return joblib.load(MODEL_PATH)


def predict_text(text: str) -> dict:
    """Return label, confidence and probabilities for one text."""
    from src.preprocess import clean_text

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter some text to classify.")
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Text has no usable words after cleaning.")
    model = load_model()
    probs = model.predict_proba([cleaned])[0]
    classes = list(model.classes_)
    best = int(probs.argmax())
    return {
        "label": classes[best],
        "confidence": round(float(probs[best]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python -m src.predict "your text here"')
        sys.exit(1)
    try:
        print(predict_text(" ".join(sys.argv[1:])))
    except (ValueError, FileNotFoundError) as err:
        print(f"Error: {err}")
        sys.exit(1)
