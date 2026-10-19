from pathlib import Path

import joblib
import pandas as pd
import tldextract

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data/processed/combined_training_enhanced_features.parquet"
MODEL_FILE = ROOT / "artifacts/url_token_model_domain_grouped.joblib"

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
    df = pd.read_parquet(DATA_FILE).reset_index(drop=True)

    # Registered domains are computed offline; no websites are contacted.
    extractor = tldextract.TLDExtract(suffix_list_urls=())

    def get_domain(url):
        parts = extractor(str(url))
        if parts.domain and parts.suffix:
            return f"{parts.domain}.{parts.suffix}".lower()
        return parts.domain.lower()

    df["registered_domain"] = df["url"].fillna("").map(get_domain)
    df = df[df["registered_domain"] != ""].copy()
    df = df.reset_index(drop=True)

    y = df["label"].map({"benign": 0, "phish": 1})
    if y.isna().any():
        raise ValueError("Unexpected values in label column.")

    X = df[["url"] + NUMERIC_FEATURES].copy()
    X["url"] = X["url"].fillna("").astype(str)
    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].fillna(0)

    groups = df["registered_domain"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    train_domains = set(groups.iloc[train_idx])
    test_domains = set(groups.iloc[test_idx])
    overlap = train_domains & test_domains

    print("=== DOMAIN SPLIT ===")
    print("Total rows:", len(df))
    print("Training rows:", len(train_idx))
    print("Test rows:", len(test_idx))
    print("Training domains:", len(train_domains))
    print("Test domains:", len(test_domains))
    print("Domain overlap:", len(overlap))
    print("\nTraining labels:")
    print(df.iloc[train_idx]["label"].value_counts())
    print("\nTest labels:")
    print(df.iloc[test_idx]["label"].value_counts())

    if overlap:
        raise ValueError("Domain overlap detected!")
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("Train or test split contains only one class.")

    # Fit all learned text preprocessing only on training URLs.
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

    print("\nTraining a NEW model on domain-grouped training data...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n=== DOMAIN-GROUPED MODEL RESULTS ===")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["benign", "phish"],
        zero_division=0,
    ))

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "numeric_features": NUMERIC_FEATURES,
            "text_column": "url",
            "split_strategy": "GroupShuffleSplit by registered domain",
            "random_state": 42,
        },
        MODEL_FILE,
    )
    print("\nSaved model to:", MODEL_FILE)


if __name__ == "__main__":
    main()
