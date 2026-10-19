
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = (
    ROOT / "data" / "processed"
    / "combined_training_features.parquet"
)

MODEL_FILE = ROOT / "artifacts" / "random_forest_13k.joblib"
OUTPUT_DIR = ROOT / "artifacts"

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
    df = df.dropna(subset=["label", "url"])
    df = df[df["label"].isin(["benign", "phish"])].copy()

    X = df[FEATURE_COLUMNS]
    y = df["label"].map({"benign": 0, "phish": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    saved = joblib.load(MODEL_FILE)
    model = saved["model"]

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLUMNS,
    ).sort_values(ascending=False)

    print("=== RANDOM FOREST FEATURE IMPORTANCE ===")
    print(importance.round(4).to_string())

    predictions = model.predict(X_test)
    errors = df.loc[X_test.index].copy()
    errors["actual"] = errors["label"]
    errors["predicted"] = pd.Series(predictions, index=X_test.index).map(
        {0: "benign", 1: "phish"}
    )

    missed = errors[
        (errors["actual"] == "phish") &
        (errors["predicted"] == "benign")
    ]

    false_alarms = errors[
        (errors["actual"] == "benign") &
        (errors["predicted"] == "phish")
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    missed[["url", "actual", "predicted"]].to_csv(
        OUTPUT_DIR / "missed_phishing_13k.csv", index=False
    )
    false_alarms[["url", "actual", "predicted"]].to_csv(
        OUTPUT_DIR / "false_positives_13k.csv", index=False
    )

    print(f"\nMissed phishing URLs: {len(missed)}")
    print(f"False positives: {len(false_alarms)}")
    print("\nError examples (URLs truncated):")
    print(
        missed[["url", "actual", "predicted"]]
        .assign(url=lambda d: d["url"].str.slice(0, 120))
        .head(10)
        .to_string(index=False)
    )

    print("\nSaved error-analysis CSV files in artifacts/.")


if __name__ == "__main__":
    main()
