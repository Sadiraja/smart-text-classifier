"""Build a dataset from real public data.

Run once (needs internet):  python -m scripts.build_dataset

Sources:
- Complaint / Inquiry : Banking77 (PolyAI, CC BY 4.0), real customer-service queries,
                        read from the authors' GitHub CSV files. The 77 intents are mapped
                        to Complaint or Inquiry by keyword (see COMPLAINT_KEYWORDS).
                        REVIEW THE PRINTED MAPPING.
- Feedback            : Amazon Polarity (Apache-2.0), positive short reviews (Hugging Face).
- Other               : UCI SMS Spam Collection, ordinary personal texts and spam
                        (read from a public GitHub mirror of the UCI file).
"""
import pandas as pd

from src.config import DATA_PATH, RANDOM_SEED

N_PER_CLASS = 500
MIN_WORDS, MAX_WORDS = 4, 30

BANKING_BASE = ("https://raw.githubusercontent.com/PolyAI-LDN/"
                "task-specific-datasets/master/banking_data/")
SMS_URL = ("https://raw.githubusercontent.com/justmarkham/"
           "pycon-2016-tutorial/master/data/sms.tsv")
AMAZON_TEST_PARQUET = ("https://huggingface.co/datasets/fancyzhx/amazon_polarity/"
                       "resolve/refs%2Fconvert%2Fparquet/amazon_polarity/test/0000.parquet")

# An intent whose name contains any of these describes a PROBLEM -> Complaint.
COMPLAINT_KEYWORDS = [
    "not_working", "declined", "failed", "not_received", "not_recognised",
    "charged_twice", "extra_charge", "wrong", "not_showing", "pending",
    "reverted", "swallowed", "compromised", "lost_or_stolen", "blocked",
    "not_updated", "fee_charged", "withdrawal_charge", "not_allowed",
    "cannot", "unable",
]


def intent_to_label(intent_name: str) -> str:
    name = intent_name.lower()
    return "Complaint" if any(k in name for k in COMPLAINT_KEYWORDS) else "Inquiry"


def n_words(text: str) -> int:
    return len(text.split())


def load_banking() -> pd.DataFrame:
    df = pd.concat([pd.read_csv(BANKING_BASE + "train.csv"),
                    pd.read_csv(BANKING_BASE + "test.csv")])
    df["label"] = df["category"].map(intent_to_label)

    print("\nIntent -> label mapping (edit COMPLAINT_KEYWORDS if something looks wrong):")
    for label in ("Complaint", "Inquiry"):
        intents = sorted(i for i in df["category"].unique() if intent_to_label(i) == label)
        print(f"\n{label} ({len(intents)}): {', '.join(intents)}")

    parts = [df[df["label"] == lab].sample(N_PER_CLASS, random_state=RANDOM_SEED)
             for lab in ("Complaint", "Inquiry")]
    out = pd.concat(parts)[["text", "label"]]
    out["source"] = "banking77"
    return out


def load_feedback() -> pd.DataFrame:
    """Positive short Amazon reviews. Tries the datasets library, then a parquet file."""
    try:
        from datasets import load_dataset
        stream = load_dataset("fancyzhx/amazon_polarity", split="train", streaming=True)
        stream = stream.shuffle(seed=RANDOM_SEED, buffer_size=20000)
        rows = (r for r in stream)
    except Exception as err:
        print(f"datasets library failed ({err}); reading parquet file directly...")
        rows = (r for r in pd.read_parquet(AMAZON_TEST_PARQUET)
                .sample(frac=1, random_state=RANDOM_SEED).to_dict("records"))

    texts = []
    for row in rows:
        text = row["content"].strip()
        if row["label"] == 1 and MIN_WORDS <= n_words(text) <= MAX_WORDS:
            texts.append(text)
        if len(texts) >= N_PER_CLASS:
            break
    return pd.DataFrame({"text": texts, "label": "Feedback", "source": "amazon_polarity"})


def load_other() -> pd.DataFrame:
    df = pd.read_csv(SMS_URL, sep="\t", header=None, names=["kind", "text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].map(n_words) >= 2]
    half = N_PER_CLASS // 2
    parts = [df[df["kind"] == k].sample(half, random_state=RANDOM_SEED)
             for k in ("ham", "spam")]
    out = pd.concat(parts)[["text"]]
    out["label"] = "Other"
    out["source"] = "sms_spam"
    return out


def main():
    df = pd.concat([load_banking(), load_feedback(), load_other()])
    df = df.drop_duplicates(subset="text").sample(frac=1, random_state=RANDOM_SEED)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"\nSaved {len(df)} rows to {DATA_PATH}")
    print(df.groupby(["label", "source"]).size())


if __name__ == "__main__":
    main()