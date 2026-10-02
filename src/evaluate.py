"""Evaluation helpers: metrics, confusion matrix, error listing."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)

from src.config import LABELS, REPORTS_DIR


def compute_metrics(y_true, y_pred) -> dict:
    """Accuracy, macro/weighted F1 and per-class precision/recall/F1."""
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "per_class": classification_report(
            y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
        ),
    }


def save_confusion_matrix(y_true, y_pred, name="confusion_matrix.png"):
    """Save a readable confusion matrix image to reports/."""
    REPORTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=LABELS, cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title("Confusion Matrix (test set)")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / name, dpi=150)
    plt.close(fig)


def list_errors(model, texts, y_true):
    """Return misclassified examples as (text, true, predicted, confidence)."""
    preds = model.predict(texts)
    probs = model.predict_proba(texts).max(axis=1)
    return [
        (t, true, p, round(float(c), 3))
        for t, true, p, c in zip(texts, y_true, preds, probs)
        if true != p
    ]
