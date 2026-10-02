"""Central settings: paths, labels and the random seed."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "dataset.csv"
EXTERNAL_TEST_PATH = BASE_DIR / "data" / "external_test.csv"
MODEL_PATH = BASE_DIR / "models" / "classifier.joblib"
REPORTS_DIR = BASE_DIR / "reports"

LABELS = ["Complaint", "Inquiry", "Feedback", "Other"]
RANDOM_SEED = 42
TEST_SIZE = 0.2
LOW_CONFIDENCE = 0.5
