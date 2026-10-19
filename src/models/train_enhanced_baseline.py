from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data/processed/combined_training_enhanced_features.parquet"
MODEL_FILE = ROOT / "artifacts/random_forest_enhanced.joblib"

FEATURES = [
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

    X = df[FEATURES].fillna(0)
    y = df["label"].map({"benign": 0, "phish": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("=== ENHANCED RANDOM FOREST ===")
    print(f"Rows: {len(df)}")
    print(f"Features: {len(FEATURES)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    print("\nFeature importances:")
    importances = pd.Series(
        model.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)
    print(importances.to_string())

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": FEATURES},
        MODEL_FILE,
    )
    print(f"\nModel saved to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
