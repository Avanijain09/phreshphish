
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "dev_sample.parquet"


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Sample not found: {DATA_FILE}")

    df = pd.read_parquet(DATA_FILE)

    print("=== DATASET VALIDATION ===")
    print(f"Shape: {df.shape}")

    print("\nLabel distribution:")
    print(df["label"].value_counts(dropna=False).to_string())

    print("\nMissing values:")
    print(df.isna().sum().to_string())

    print("\nDuplicate SHA256 hashes:", df["sha256"].duplicated().sum())

    empty_text = df["page_text"].fillna("").str.strip().eq("")
    print("Empty page-text records:", int(empty_text.sum()))

    print("\nSample records (text limited to 100 characters):")
    preview = df[["label", "target", "page_text"]].head(3).copy()
    preview["page_text"] = preview["page_text"].fillna("").str.slice(0, 100)
    print(preview.to_string(index=False))

    print("\nEmpty-text record labels:")
    print(df.loc[empty_text, "label"].value_counts().to_string())


    print("\nEmpty-text record examples (first 10):")

    columns = ["sha256", "url", "label", "target", "lang"]
    examples = df.loc[empty_text, columns].head(10)

    for _, row in examples.iterrows():
        print({
            "sha256": row["sha256"],
            "url": str(row["url"])[:180],
            "label": row["label"],
            "target": row["target"],
            "lang": row["lang"],
        })



if __name__ == "__main__":
    main()
