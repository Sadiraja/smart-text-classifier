import pytest

from src.config import LABELS, MODEL_PATH
from src.predict import predict_text


@pytest.fixture(scope="module", autouse=True)
def ensure_model():
    if not MODEL_PATH.exists():
        from src.train import main
        main()


def test_returns_valid_label():
    assert predict_text("Where is my order?")["label"] in LABELS


def test_probabilities_sum_to_one():
    probs = predict_text("I love this app")["probabilities"]
    assert abs(sum(probs.values()) - 1) < 0.01


def test_empty_input_raises():
    with pytest.raises(ValueError):
        predict_text("   ")
