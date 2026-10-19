
from pathlib import Path

import pandas as pd

from src.data.extract_url_features import extract_url_features


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "raw" / "huggingface"
    / "data" / "train-000.parquet"
)

OUTPUT_FILE = (
    ROOT / "data" / "processed"
    / "train_shard_000_features.parquet"
)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE)

    features = pd.DataFrame(
        df["url"].apply(extract_url_features).tolist(),
        index=df.index,
    )

    result = pd.concat(
        [df.drop(columns=["html"]), features],
        axis=1,
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_FILE, index=False)

    print("Shard processing complete!")
    print("Shape:", result.shape)
    print("\nLabel distribution:")
    print(result["label"].value_counts())
    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
