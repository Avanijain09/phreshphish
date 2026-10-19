
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "dev_features.parquet"

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

    print("=== FEATURE VALIDATION ===")
    print(f"Dataset shape: {df.shape}")

    print("\nLabel distribution:")
    print(df["label"].value_counts().to_string())

    print("\nMissing values in features:")
    print(df[FEATURE_COLUMNS].isna().sum().to_string())

    print("\nFeature summary:")
    print(df[FEATURE_COLUMNS].describe().round(2).to_string())

    print("\nInvalid label values:")
    print(sorted(set(df["label"].dropna()) - {"benign", "phish"}))

    print("\nDuplicate SHA256 hashes:", df["sha256"].duplicated().sum())

    print("\nFeature validation complete.")


if __name__ == "__main__":
    main()
