from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data/processed/combined_training_enhanced_features.parquet"
MODEL_FILE = ROOT / "artifacts/random_forest_enhanced.joblib"

MISSED_FILE = ROOT / "artifacts/missed_phishing_enhanced.csv"
FP_FILE = ROOT / "artifacts/false_positives_enhanced.csv"

bundle = joblib.load(MODEL_FILE)
model = bundle["model"]
features = bundle["features"]

df = pd.read_parquet(DATA_FILE)
X = df[features].fillna(0)
y = df["label"].map({"benign": 0, "phish": 1})

if y.isna().any():
    raise ValueError("Unexpected labels found in the label column.")

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

predictions = model.predict(X_test)

errors = pd.DataFrame({
    "url": df.loc[X_test.index, "url"].values,
    "actual": y_test.values,
    "predicted": predictions,
})

missed = errors[(errors["actual"] == 1) & (errors["predicted"] == 0)]
false_positives = errors[(errors["actual"] == 0) & (errors["predicted"] == 1)]

MISSED_FILE.parent.mkdir(parents=True, exist_ok=True)
missed.to_csv(MISSED_FILE, index=False)
false_positives.to_csv(FP_FILE, index=False)

print("=== ENHANCED ERROR ANALYSIS ===")
print(f"Missed phishing: {len(missed)}")
print(f"False positives: {len(false_positives)}")
print(f"Saved missed phishing: {MISSED_FILE}")
print(f"Saved false positives: {FP_FILE}")

print("\n=== FIRST 15 MISSED PHISHING URLS ===")
print(missed["url"].head(15).to_string(index=False))

print("\n=== FIRST 15 FALSE POSITIVE URLS ===")
print(false_positives["url"].head(15).to_string(index=False))
