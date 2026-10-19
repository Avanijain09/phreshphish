from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data/processed/combined_training_enhanced_features.parquet"
MODEL_FILE = ROOT / "artifacts/url_token_model.joblib"

NUMERIC_FEATURES = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "dot_count",
    "hyphen_count",
    "digit_count",
    "subdomain_count",
    "uses_https",
    "has_ip_address",
    "suspicious_term_count",
    "registered_domain_length",
    "subdomain_length",
    "is_hosting_platform",
]


def main():
    df = pd.read_parquet(DATA_FILE)

    # The target column is intentionally excluded.
    X = df[["url"] + NUMERIC_FEATURES].copy()
    X["url"] = X["url"].fillna("").astype(str)
    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].fillna(0)

    y = df["label"].map({"benign": 0, "phish": 1})
    if y.isna().any():
        raise ValueError("Unexpected values in label column.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # Word and character patterns are learned from training URLs only.
    preprocess = ColumnTransformer(
        transformers=[
            (
                "url_words",
                TfidfVectorizer(
                    analyzer="word",
                    token_pattern=r"(?u)[A-Za-z0-9]+",
                    ngram_range=(1, 2),
                    max_features=20000,
                    sublinear_tf=True,
                    dtype="float32",
                ),
                "url",
            ),
            (
                "url_chars",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    max_features=30000,
                    min_df=2,
                    sublinear_tf=True,
                    dtype="float32",
                ),
                "url",
            ),
            ("numeric", MaxAbsScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    model = Pipeline([
        ("features", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    print("Training URL token model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n=== URL TOKEN MODEL ===")
    print(f"Rows: {len(df)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Evaluation rows: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, target_names=["benign", "phish"],
        zero_division=0,
    ))

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "numeric_features": NUMERIC_FEATURES,
            "text_column": "url",
        },
        MODEL_FILE,
    )
    print(f"\nModel saved to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
