from pathlib import Path
import pandas as pd
import tldextract
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data/processed/combined_training_enhanced_features.parquet"

df = pd.read_parquet(DATA_FILE)
y = df["label"].map({"benign": 0, "phish": 1})

train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

train = df.loc[train_idx].copy()
test = df.loc[test_idx].copy()

extractor = tldextract.TLDExtract(suffix_list_urls=())

def get_domain(url):
    parts = extractor(url)
    if parts.domain and parts.suffix:
        return f"{parts.domain}.{parts.suffix}".lower()
    return parts.domain.lower()

for split in (train, test):
    split["registered_domain"] = split["url"].fillna("").map(get_domain)

train_domains = set(train["registered_domain"]) - {""}
test_domains = set(test["registered_domain"]) - {""}
overlap = train_domains & test_domains

print("Unique train domains:", len(train_domains))
print("Unique test domains:", len(test_domains))
print("Domains appearing in both splits:", len(overlap))
print("Test rows whose domain appears in training:",
      test["registered_domain"].isin(train_domains).sum(),
      "out of", len(test))

print("\nExample overlapping domains:")
for domain in sorted(overlap)[:20]:
    print(domain)
