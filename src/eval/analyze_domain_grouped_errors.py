from pathlib import Path
import joblib
import pandas as pd
import tldextract
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data/processed/combined_training_enhanced_features.parquet"
MODEL_FILE = ROOT / "artifacts/url_token_model_domain_grouped.joblib"

df = pd.read_parquet(DATA_FILE).reset_index(drop=True)
extractor = tldextract.TLDExtract(suffix_list_urls=())

def get_domain(url):
    parts = extractor(str(url))
    if parts.domain and parts.suffix:
        return f"{parts.domain}.{parts.suffix}".lower()
    return parts.domain.lower()

df["registered_domain"] = df["url"].fillna("").map(get_domain)
df = df[df["registered_domain"] != ""].copy().reset_index(drop=True)

y = df["label"].map({"benign": 0, "phish": 1})
splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(
    splitter.split(df, y, groups=df["registered_domain"])
)

test = df.iloc[test_idx].copy()
bundle = joblib.load(MODEL_FILE)
features = bundle["numeric_features"]

X_test = test[["url"] + features].copy()
X_test["url"] = X_test["url"].fillna("").astype(str)
X_test[features] = X_test[features].fillna(0)

model = bundle["model"]
pred = model.predict(X_test)

errors = pd.DataFrame({
    "url": test["url"].values,
    "actual": y.iloc[test_idx].values,
    "predicted": pred,
})

missed = errors[(errors.actual == 1) & (errors.predicted == 0)]
false_positives = errors[(errors.actual == 0) & (errors.predicted == 1)]

missed.to_csv(ROOT / "artifacts/missed_phishing_domain_grouped.csv", index=False)
false_positives.to_csv(ROOT / "artifacts/false_positives_domain_grouped.csv", index=False)

print("Missed phishing:", len(missed))
print("\n=== FIRST 20 MISSED PHISHING ===")
print(missed["url"].head(20).to_string(index=False))
print("\nFalse positives:", len(false_positives))
print("\n=== FIRST 20 FALSE POSITIVES ===")
print(false_positives["url"].head(20).to_string(index=False))
