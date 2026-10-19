
from collections import Counter

from datasets import load_dataset


DATASET_NAME = "phreshphish/phreshphish"
SAMPLE_SIZE = 500


def main():
    print("Opening official test split stream...", flush=True)

    dataset = load_dataset(
        DATASET_NAME,
        split="test",
        streaming=True,
    )

    labels = Counter()
    targets = Counter()
    records_seen = 0

    for row in dataset:
        labels[str(row.get("label"))] += 1

        target = row.get("target")
        if target:
            targets[str(target)] += 1

        records_seen += 1

        if records_seen >= SAMPLE_SIZE:
            break

    print(f"\nRecords inspected: {records_seen}")
    print("Label counts:", dict(labels))
    print("Top 10 targets:", targets.most_common(10))


if __name__ == "__main__":
    main()
