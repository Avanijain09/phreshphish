
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "combined_training_features.parquet"
)
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

FEATURE_COLUMNS = [
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
]


def main():
    df = pd.read_parquet(DATA_FILE)

    df = df.dropna(subset=["label", "url"]).copy()
    df = df[df["label"].isin(["benign", "phish"])].copy()

    X = df[FEATURE_COLUMNS]
    y = df["label"].map({"benign": 0, "phish": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        print(f"\n{'=' * 60}")
        print(name)
        print("=" * 60)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        print("\nClassification report:")
        print(classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["benign", "phish"],
            zero_division=0,
        ))

        print("Confusion matrix (rows=actual, columns=predicted):")
        print(confusion_matrix(y_test, predictions, labels=[0, 1]))

        print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")

        filename = name.lower().replace(" ", "_") + "_13k.joblib"
        model_path = ARTIFACT_DIR / filename
        joblib.dump(
            {
                "model": model,
                "features": FEATURE_COLUMNS,
                "label_mapping": {"benign": 0, "phish": 1},
            },
            model_path,
        )
        print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
