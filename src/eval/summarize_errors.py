
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MISSED_FILE = ROOT / "artifacts" / "missed_phishing.csv"
FP_FILE = ROOT / "artifacts" / "false_positives.csv"


def summarize_file(path, title):
    df = pd.read_csv(path)

    print(f"\n=== {title} ===")
    print(f"Total records: {len(df)}")

    if df.empty:
        print("No examples to analyse.")
        return

    # Summarize URL structure without visiting the URLs.
    urls = df["url"].fillna("").astype(str)

    print("\nURL length summary:")
    print(urls.str.len().describe().round(2).to_string())

    print("\nCommon hostname endings:")
    hostnames = (
        urls.str.replace(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", regex=True)
        .str.split("/", n=1)
        .str[0]
        .str.split("@")
        .str[-1]
        .str.split(":")
        .str[0]
    )

    endings = hostnames.str.extract(r"(\.[a-zA-Z]{2,})$", expand=False)
    print(endings.value_counts().head(10).to_string())

    print("\nLongest URL examples (truncated):")
    for url in urls.loc[urls.str.len().sort_values(ascending=False).index].head(5):
        print(url[:150])


def main():
    summarize_file(MISSED_FILE, "MISSED PHISHING")
    summarize_file(FP_FILE, "FALSE POSITIVES")


if __name__ == "__main__":
    main()
