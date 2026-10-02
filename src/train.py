"""Train two models, compare them, save the best one.

Run with:  python -m src.train
"""
import json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.config import (DATA_PATH, EXTERNAL_TEST_PATH, MODEL_PATH, RANDOM_SEED,
                        REPORTS_DIR, TEST_SIZE)
from src.evaluate import compute_metrics, list_errors, save_confusion_matrix
from src.preprocess import load_dataset


def make_vectorizer():
    # token_pattern keeps "?" as its own token; bigrams capture phrases like "not working"
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                           token_pattern=r"(?u)\b\w+\b|\?")


def build_models() -> dict:
    return {
        "logistic_regression": Pipeline([
            ("tfidf", make_vectorizer()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=RANDOM_SEED)),
        ]),
        "naive_bayes": Pipeline([
            ("tfidf", make_vectorizer()),
            ("clf", MultinomialNB()),
        ]),
    }


def main():
    df = load_dataset(DATA_PATH)
    print(f"Loaded {len(df)} examples:\n{df['label'].value_counts()}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=TEST_SIZE,
        stratify=df["label"], random_state=RANDOM_SEED)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    results, fitted = {}, {}
    for name, model in build_models().items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
        model.fit(X_train, y_train)
        metrics = compute_metrics(y_test, model.predict(X_test))
        metrics["cv_macro_f1_mean"] = round(float(scores.mean()), 4)
        metrics["cv_macro_f1_std"] = round(float(scores.std()), 4)
        results[name], fitted[name] = metrics, model
        print(f"{name}: accuracy={metrics['accuracy']} macro_f1={metrics['macro_f1']} "
              f"CV={metrics['cv_macro_f1_mean']} +/- {metrics['cv_macro_f1_std']}")

    best = max(results, key=lambda n: results[n]["macro_f1"])
    print(f"\nBest model: {best}")
    best_model = fitted[best]

    MODEL_PATH.parent.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    save_confusion_matrix(y_test, best_model.predict(X_test))

    output = {"best_model": best, "train_size": len(X_train),
              "test_size": len(X_test), "models": results}

    # Optional harder test: hand-written examples from a different source.
    if EXTERNAL_TEST_PATH.exists():
        ext = load_dataset(EXTERNAL_TEST_PATH)
        ext_metrics = compute_metrics(ext["label"], best_model.predict(ext["text"]))
        output["external_test"] = {"size": len(ext), **ext_metrics}
        save_confusion_matrix(ext["label"], best_model.predict(ext["text"]),
                              name="confusion_matrix_external.png")
        print(f"\nExternal hand-written test ({len(ext)} examples): "
              f"accuracy={ext_metrics['accuracy']} macro_f1={ext_metrics['macro_f1']}")
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(output, indent=2))

    errors = list_errors(best_model, list(X_test), list(y_test))
    print(f"\nMisclassified test examples ({len(errors)}):")
    for text, true, pred, conf in errors:
        print(f"  [{true} -> {pred}, conf={conf}] {text}")


if __name__ == "__main__":
    main()
