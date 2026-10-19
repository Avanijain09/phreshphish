from pathlib import Path
from urllib.parse import urlparse
import ipaddress

import pandas as pd
import tldextract


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "combined_training_features.parquet"
)
OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "combined_training_enhanced_features.parquet"
)

# No network requests for public suffix data.
EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())

HOSTING_PLATFORMS = {
    "github.io",
    "pages.dev",
    "vercel.app",
    "glitch.me",
}

def extract_enhanced_features(url):
    url = str(url).strip()
    candidate = url if "://" in url else "//" + url
    parsed = urlparse(candidate)

    hostname = (parsed.hostname or "").lower()
    parts = EXTRACTOR(hostname)

    registered_domain = ".".join(
        part for part in (parts.domain, parts.suffix) if part
    )
    subdomain = parts.subdomain or ""

    try:
        ipaddress.ip_address(hostname)
        has_ip_address = 1
    except ValueError:
        has_ip_address = 0

    is_hosting_platform = int(
        any(
            hostname == platform or hostname.endswith("." + platform)
            for platform in HOSTING_PLATFORMS
        )
    )

    return {
        "registered_domain_length": len(registered_domain),
        "subdomain_length": len(subdomain),
        "is_hosting_platform": is_hosting_platform,
    }


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE)

    enhanced = pd.DataFrame(
        df["url"].apply(extract_enhanced_features).tolist(),
        index=df.index,
    )

    result = pd.concat([df, enhanced], axis=1)
    result.to_parquet(OUTPUT_FILE, index=False)

    print("Enhanced feature extraction complete!")
    print(f"Rows: {len(result)}")
    print(f"New features: {list(enhanced.columns)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("\nHosting platform counts:")
    print(result["is_hosting_platform"].value_counts())
    print("\nPreview:")
    print(
        result[
            ["url", "registered_domain_length",
             "subdomain_length", "is_hosting_platform"]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
