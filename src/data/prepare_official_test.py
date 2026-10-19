from pathlib import Path
import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/processed/official_test_5k.parquet"
TARGET_ROWS = 5000

print("Loading official test split in streaming mode...")

dataset = load_dataset(
    "phreshphish/phreshphish",
    split="test",
    streaming=True,
)

rows = []
for row in dataset.shuffle(seed=42, buffer_size=10_000).take(TARGET_ROWS):
    rows.append({
        "sha256": row.get("sha256"),
        "url": row.get("url"),
        "label": row.get("label"),
        "target": row.get("target"),
        "date": str(row.get("date")) if row.get("date") is not None else None,
        "lang": row.get("lang"),
        "lang_score": row.get("lang_score"),
    })
    if len(rows) % 500 == 0:
        print(f"Collected {len(rows)} records")

df = pd.DataFrame(rows)

if len(df) != TARGET_ROWS:
    raise RuntimeError(f"Expected 5000 rows, got {len(df)}")

if df["url"].isna().any() or df["label"].isna().any():
    raise ValueError("Missing URL or label")

if df["sha256"].duplicated().any():
    raise ValueError("Duplicate hashes found")

df.to_parquet(OUTPUT, index=False)

print(f"\nSaved: {OUTPUT}")
print("Total records:", len(df))
print("\nLabel distribution:")
print(df["label"].value_counts(dropna=False))
