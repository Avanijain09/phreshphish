from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from src.data.extract_url_features import extract_url_features
from src.data.extract_enhanced_url_features import extract_enhanced_features

ROOT = Path(__file__).resolve().parents[2]
MODEL_FILE = ROOT / "artifacts" / "url_token_model_domain_grouped.joblib"
TEST_FILE = ROOT / "data" / "processed" / "official_test_5k.parquet"
OUTPUT_FILE = ROOT / "artifacts" / "official_test_5k_predictions.csv"


def main():
    print("Loading domain-grouped model...", flush=True)
    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]
    numeric_features = bundle["numeric_features"]
    text_column = bundle["text_column"]

    print("Loading saved test sample...", flush=True)
    df = pd.read_parquet(TEST_FILE)
    df = df.drop_duplicates(subset=["sha256"]).copy()
    df = df[
        df["label"].isin(["benign", "phish"]) & df["url"].notna()
    ].copy()

    feature_rows = []
    for url in df["url"]:
        features = {
            **extract_url_features(url),
            **extract_enhanced_features(url),
        }
        feature_rows.append(features)

    numeric_df = pd.DataFrame(feature_rows, index=df.index)

    X = numeric_df[numeric_features].copy()
    X[text_column] = df[text_column].astype(str)

    y_true = df["label"].map({"benign": 0, "phish": 1})
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    print(f"\nRecords evaluated: {len(df)}")
    print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["benign", "phish"],
        zero_division=0,
    ))
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    if y_true.nunique() == 2:
        print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.4f}")

    df["predicted_label"] = pd.Series(
        y_pred, index=df.index
    ).map({0: "benign", 1: "phish"})
    df["phish_probability"] = y_prob

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nPredictions saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()