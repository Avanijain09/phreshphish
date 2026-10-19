
from itertools import islice
from pathlib import Path
import re

import pandas as pd
from bs4 import BeautifulSoup
from datasets import load_dataset


DATASET_NAME = "phreshphish/phreshphish"
SAMPLE_SIZE = 2_000
MAX_TEXT_CHARS = 8_000

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "dev_sample.parquet"


def extract_page_text(html, max_chars=MAX_TEXT_CHARS):
    """Extract text without rendering or executing webpage HTML."""
    if not isinstance(html, str) or not html:
        return ""

    soup = BeautifulSoup(html, "lxml")

    for element in soup(["script", "style", "noscript", "svg"]):
        element.decompose()

    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)

    return text[:max_chars]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Opening training dataset stream...", flush=True)

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    records = []

    print(f"Reading at most {SAMPLE_SIZE} records...", flush=True)

    for index, row in enumerate(islice(dataset, SAMPLE_SIZE), start=1):
        records.append({
            "sha256": row.get("sha256"),
            "url": row.get("url"),
            "label": row.get("label"),
            "target": row.get("target"),
            "date": row.get("date"),
            "lang": row.get("lang"),
            "lang_score": row.get("lang_score"),
            "page_text": extract_page_text(row.get("html")),
        })

        if index % 100 == 0:
            print(f"Processed {index} records", flush=True)

    if not records:
        raise RuntimeError("No records were received from the dataset.")

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["sha256"], keep="first")
    df = df[df["label"].isin(["benign", "phish"])].reset_index(drop=True)

    df.to_parquet(OUTPUT_FILE, index=False)

    print("\nSample preparation complete!", flush=True)
    print(f"Records saved: {len(df)}", flush=True)
    print("\nLabel distribution:")
    print(df["label"].value_counts().to_string())
    print(f"\nSaved to: {OUTPUT_FILE}", flush=True)
    print(
        f"File size: {OUTPUT_FILE.stat().st_size / (1024 * 1024):.2f} MB",
        flush=True,
    )


if __name__ == "__main__":
    main()
