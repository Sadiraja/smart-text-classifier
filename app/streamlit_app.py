"""Streamlit interface. Run: streamlit run app/streamlit_app.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src.config import LOW_CONFIDENCE, REPORTS_DIR
from src.predict import load_model, predict_text

st.set_page_config(page_title="Smart Text Classifier", page_icon="💬")

EXAMPLES = [
    "My parcel arrived broken and I want a refund.",
    "Do you deliver to Islamabad on weekends?",
    "Love the new design, maybe add dark mode.",
    "Good morning everyone",
]


@st.cache_resource
def get_model():
    return load_model()


st.title("💬 Smart Text Classifier")
st.write("Enter a short message and get its category: "
         "**Complaint, Inquiry, Feedback or Other**.")

with st.sidebar:
    st.header("About")
    st.markdown(
        "- **Complaint**: dissatisfaction or a problem\n"
        "- **Inquiry**: asks for information or help\n"
        "- **Feedback**: opinion or suggestion\n"
        "- **Other**: greetings, spam, off-topic")
    metrics_file = REPORTS_DIR / "metrics.json"
    if metrics_file.exists():
        data = json.loads(metrics_file.read_text())
        best = data["models"][data["best_model"]]
        st.header("Model performance")
        st.write(f"Model: `{data['best_model']}`")
        st.metric("Accuracy", best["accuracy"])
        st.metric("Macro F1", best["macro_f1"])

try:
    get_model()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

st.write("Try an example:")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex[:22] + "...", help=ex):
        st.session_state["text"] = ex

text = st.text_area("Your text", key="text", height=120)

if st.button("Classify", type="primary"):
    try:
        result = predict_text(text)
        st.success(f"Predicted category: **{result['label']}** "
                   f"({result['confidence']:.0%} confidence)")
        if result["confidence"] < LOW_CONFIDENCE:
            st.warning("Low confidence: the model is unsure about this text.")
        probs = pd.Series(result["probabilities"], name="probability")
        st.bar_chart(probs)
    except ValueError as err:
        st.warning(str(err))
