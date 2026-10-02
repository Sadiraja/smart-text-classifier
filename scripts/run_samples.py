"""Run demo predictions. Run: python -m scripts.run_samples"""
import pandas as pd

from src.config import REPORTS_DIR
from src.predict import predict_text

# (text, expected label) - none of these are in the training data
SAMPLES = [
    ("The charger I received stopped working after a week.", "Complaint"),
    ("My refund has not arrived and support keeps ignoring me.", "Complaint"),
    ("Can you tell me when the sale ends?", "Inquiry"),
    ("how do i update my phone number", "Inquiry"),
    ("Is cash on delivery available in Rawalpindi?", "Inquiry"),
    ("Really enjoyed the quality, thank you!", "Feedback"),
    ("Please add a bigger search button on mobile.", "Feedback"),
    ("The team was helpful, though the wait was a little long.", "Feedback"),
    ("Hi there", "Other"),
    ("Win a free vacation, click now!!!", "Other"),
    ("See you at the meeting tomorrow.", "Other"),
    ("Great, now my order is late again. Thanks a lot.", "Complaint"),  # sarcasm
    ("Yes", "Other"),  # very short
    ("Why is the app so good and yet so slow?", "Feedback"),  # tricky
]


def main():
    rows = []
    for text, expected in SAMPLES:
        r = predict_text(text)
        rows.append({"text": text, "predicted_label": r["label"],
                     "confidence": r["confidence"], "expected_label": expected})
    df = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    df.to_csv(REPORTS_DIR / "sample_predictions.csv", index=False)
    print(df.to_string(index=False))
    matches = (df["predicted_label"] == df["expected_label"]).sum()
    print(f"\nMatched expected label: {matches}/{len(df)}")


if __name__ == "__main__":
    main()
