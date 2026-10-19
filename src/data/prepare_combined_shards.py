from pathlib import Path

import pandas as pd

from src.data.extract_url_features import extract_url_features


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "processed" / "combined_training_shards.parquet"

OUTPUT_FILE = ROOT / "data" / "processed" / "combined_training_features.parquet"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE)

    print("Input shape:", df.shape)

    features = pd.DataFrame(
        df["url"].apply(extract_url_features).tolist(),
        index=df.index,
    )

    # Preserve labels and metadata, but exclude raw HTML.
    result = pd.concat(
        [df.drop(columns=["html"], errors="ignore"), features],
        axis=1,
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_FILE, index=False)

    print("\nFeature processing complete!")
    print("Output shape:", result.shape)
    print("\nLabel distribution:")
    print(result["label"].value_counts(dropna=False))
    print("\nHTML column present:", "html" in result.columns)
    print("Duplicate SHA256 hashes:", result["sha256"].duplicated().sum())
    print("Missing labels:", result["label"].isna().sum())
    print("URL feature columns:", list(features.columns))
    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
