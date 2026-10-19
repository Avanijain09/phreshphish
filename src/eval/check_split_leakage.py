from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data/processed/combined_training_enhanced_features.parquet"

df = pd.read_parquet(DATA_FILE)

# Never use target as a feature: it contains phishing-category information.
y = df["label"].map({"benign": 0, "phish": 1})

train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

train = df.loc[train_idx]
test = df.loc[test_idx]

print("=== DATASET CHECK ===")
print("Total rows:", len(df))
print("Duplicate SHA256 values:", df["sha256"].duplicated().sum())
print("Duplicate URLs:", df["url"].duplicated().sum())
print("Missing labels:", df["label"].isna().sum())

print("\n=== SPLIT CHECK ===")
print("Train rows:", len(train))
print("Test rows:", len(test))
print("Train/test SHA256 overlap:",
      len(set(train["sha256"]) & set(test["sha256"])))
print("Train/test exact URL overlap:",
      len(set(train["url"]) & set(test["url"])))

print("\n=== LABEL DISTRIBUTION ===")
print("Train:")
print(train["label"].value_counts())
print("\nTest:")
print(test["label"].value_counts())
