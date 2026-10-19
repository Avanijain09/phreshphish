
from collections import Counter

from datasets import load_dataset


def inspect_training_data(limit: int = 500) -> None:
    dataset = load_dataset(
        "phreshphish/phreshphish",
        split="train",
        streaming=True,
    )

    label_counts = Counter()
    brand_counts = Counter()

    for index, row in enumerate(dataset):
        label_counts[str(row.get("label"))] += 1

        brand = row.get("target")
        if brand:
            brand_counts[str(brand)] += 1

        if index + 1 >= limit:
            break

    print(f"Records inspected: {sum(label_counts.values())}")
    print("Label counts:", dict(label_counts))
    print("Top 10 brands:", brand_counts.most_common(10))


if __name__ == "__main__":
    inspect_training_data()
