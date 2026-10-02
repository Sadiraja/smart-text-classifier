"""Text cleaning and dataset loading."""
import re

import pandas as pd

from src.config import LABELS

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
# keep letters, digits, spaces and "?" (questions are a strong Inquiry signal)
PUNCT_RE = re.compile(r"[^\w\s?]")
SPACE_RE = re.compile(r"\s+")


def clean_text(text) -> str:
    """Lowercase, remove URLs/emails/punctuation (keep '?'), collapse spaces.

    Stopwords are NOT removed: in short texts words like "not", "how" or "why"
    carry the meaning ("not working" vs "working").
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = PUNCT_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def load_dataset(path) -> pd.DataFrame:
    """Load the CSV, clean text, drop empty rows/duplicates, validate labels."""
    df = pd.read_csv(path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Dataset must have 'text' and 'label' columns.")
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"] != ""].drop_duplicates(subset="text").reset_index(drop=True)
    bad = set(df["label"]) - set(LABELS)
    if bad:
        raise ValueError(f"Unknown labels in dataset: {bad}")
    return df
