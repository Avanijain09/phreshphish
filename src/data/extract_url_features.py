
from pathlib import Path
from urllib.parse import urlparse
import ipaddress

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "dev_sample.parquet"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "dev_features.parquet"

SUSPICIOUS_TERMS = (
    "login",
    "verify",
    "update",
    "secure",
    "account",
    "password",
    "signin",
    "confirm",
    "wallet",
)


def extract_url_features(url):
    """Extract simple structural features from a URL without visiting it."""
    url = str(url).strip()
    candidate = url if "://" in url else "//" + url
    parsed = urlparse(candidate)

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    url_lower = url.lower()

    try:
        ipaddress.ip_address(hostname)
        has_ip_address = 1
    except ValueError:
        has_ip_address = 0

    return {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "digit_count": sum(char.isdigit() for char in url),
        "subdomain_count": max(0, len(hostname.split(".")) - 2),
        "uses_https": int(parsed.scheme.lower() == "https"),
        "has_ip_address": has_ip_address,
        "suspicious_term_count": sum(
            term in url_lower for term in SUSPICIOUS_TERMS
        ),
    }


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE)

    features = pd.DataFrame(
        df["url"].apply(extract_url_features).tolist(),
        index=df.index,
    )

    result = pd.concat(
        [df.drop(columns=["page_text"]), features],
        axis=1,
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_FILE, index=False)

    print("URL feature extraction complete!")
    print(f"Records: {len(result)}")
    print(f"Features created: {len(features.columns)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("\nFeature preview:")
    print(result[["label", "url", "url_length",
                  "uses_https", "has_ip_address",
                  "suspicious_term_count"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
