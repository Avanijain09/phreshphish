from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data/processed/combined_training_enhanced_features.parquet"
MODEL_FILE = ROOT / "artifacts/url_token_model.joblib"

df = pd.read_parquet(DATA_FILE)
bundle = joblib.load(MODEL_FILE)

features = bundle["numeric_features"]
X = df[["url"] + features].copy()
X["url"] = X["url"].fillna("").astype(str)
X[features] = X[features].fillna(0)

y = df["label"].map({"benign": 0, "phish": 1})

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

predictions = bundle["model"].predict(X_test)

errors = pd.DataFrame({
    "url": X_test["url"].values,
    "actual": y_test.values,
    "predicted": predictions,
})

missed = errors[(errors.actual == 1) & (errors.predicted == 0)]
false_positives = errors[(errors.actual == 0) & (errors.predicted == 1)]

missed.to_csv(ROOT / "artifacts/missed_phishing_tokens.csv", index=False)
false_positives.to_csv(ROOT / "artifacts/false_positives_tokens.csv", index=False)

print("Missed phishing:", len(missed))
print("False positives:", len(false_positives))

print("\n=== FIRST 15 MISSED PHISHING ===")
print(missed["url"].head(15).to_string(index=False))

print("\n=== FIRST 15 FALSE POSITIVES ===")
print(false_positives["url"].head(15).to_string(index=False))
