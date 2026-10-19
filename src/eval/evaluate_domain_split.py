from pathlib import Path
import joblib
import pandas as pd
import tldextract
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data/processed/combined_training_enhanced_features.parquet"
MODEL_FILE = ROOT / "artifacts/url_token_model.joblib"

df = pd.read_parquet(DATA_FILE).reset_index(drop=True)
bundle = joblib.load(MODEL_FILE)

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
groups = df["registered_domain"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42,
)
train_idx, test_idx = next(splitter.split(df, y, groups))

train = df.iloc[train_idx]
test = df.iloc[test_idx]

train_domains = set(train["registered_domain"])
test_domains = set(test["registered_domain"])

print("Train rows:", len(train))
print("Test rows:", len(test))
print("Train domains:", len(train_domains))
print("Test domains:", len(test_domains))
print("Domain overlap:", len(train_domains & test_domains))
print("\nTest label distribution:")
print(test["label"].value_counts())

# Evaluate only if both classes are present.
if y.iloc[test_idx].nunique() < 2:
    raise ValueError("Test split contains only one class; ROC-AUC is undefined.")

features = bundle["numeric_features"]
X_test = test[["url"] + features].copy()
X_test["url"] = X_test["url"].fillna("").astype(str)
X_test[features] = X_test[features].fillna(0)

model = bundle["model"]
pred = model.predict(X_test)
scores = model.predict_proba(X_test)[:, 1]

print("\n=== DOMAIN-GROUPED EVALUATION ===")
print("Accuracy:", round(accuracy_score(y.iloc[test_idx], pred), 4))
print("ROC-AUC:", round(roc_auc_score(y.iloc[test_idx], scores), 4))
print("Confusion matrix:")
print(confusion_matrix(y.iloc[test_idx], pred))
print("\nClassification report:")
print(classification_report(
    y.iloc[test_idx],
    pred,
    target_names=["benign", "phish"],
    zero_division=0,
))
