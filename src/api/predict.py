from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse
import ipaddress

import joblib
import pandas as pd
import tldextract

from src.schemas import PhishingAnalysis


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts" / "url_token_model_domain_grouped.joblib"

SUSPICIOUS_TERMS = (
    "login", "verify", "update", "secure", "account",
    "password", "signin", "confirm", "wallet",
)
HOSTING_PLATFORMS = {
    "github.io", "pages.dev", "vercel.app", "glitch.me",
}
DOMAIN_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())

# Heuristic decision thresholds; not calibrated probabilities.
PHISHING_THRESHOLD = 0.75
BENIGN_THRESHOLD = 0.25


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"]


def extract_features(url: str) -> dict:
    url = url.strip()
    candidate = url if "://" in url else "//" + url
    parsed = urlparse(candidate)
    hostname = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""

    parts = DOMAIN_EXTRACTOR(hostname)
    registered_domain = ".".join(
        part for part in (parts.domain, parts.suffix) if part
    )
    subdomain = parts.subdomain or ""

    try:
        ipaddress.ip_address(hostname)
        has_ip_address = 1
    except ValueError:
        has_ip_address = 0

    is_hosting_platform = int(any(
        hostname == platform or hostname.endswith("." + platform)
        for platform in HOSTING_PLATFORMS
    ))

    return {
        "url": url,
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "digit_count": sum(c.isdigit() for c in url),
        "subdomain_count": max(0, len(hostname.split(".")) - 2),
        "uses_https": int(parsed.scheme.lower() == "https"),
        "has_ip_address": has_ip_address,
        "suspicious_term_count": sum(
            term in url.lower() for term in SUSPICIOUS_TERMS
        ),
        "registered_domain_length": len(registered_domain),
        "subdomain_length": len(subdomain),
        "is_hosting_platform": is_hosting_platform,
    }


def predict_url(url: str) -> PhishingAnalysis:
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Please provide a non-empty URL.")

    url = url.strip()
    candidate = url if "://" in url else "//" + url
    parsed = urlparse(candidate)

    if not parsed.hostname or any(ch.isspace() for ch in url):
        raise ValueError("Invalid URL: provide a valid hostname without spaces.")

    features = extract_features(url)
    row = pd.DataFrame([features])
    model = load_model()
    probability = float(model.predict_proba(row)[0][1])

    if probability >= PHISHING_THRESHOLD:
        verdict = "phishing"
    elif probability <= BENIGN_THRESHOLD:
        verdict = "benign"
    else:
        verdict = "uncertain"

    evidence = []
    if features["has_ip_address"]:
        evidence.append("Hostname is an IP address.")
    if features["suspicious_term_count"]:
        evidence.append(
            f"URL contains {features['suspicious_term_count']} "
            "account/security-related keyword matches."
        )
    if features["is_hosting_platform"]:
        evidence.append("Hostname matches a configured hosting-platform suffix.")
    if features["hyphen_count"] >= 3:
        evidence.append(
            f"URL contains {features['hyphen_count']} hyphens."
        )
    if features["subdomain_count"] >= 3:
        evidence.append(
            f"Hostname contains {features['subdomain_count']} subdomain levels."
        )
    if not evidence:
        evidence.append(
            "No configured URL heuristic triggered; verdict is based on "
            "the trained URL classifier."
        )

    return PhishingAnalysis(
        verdict=verdict,
        risk_score=round(probability, 4),
        confidence=round(max(probability, 1 - probability), 4),
        evidence=evidence,
        explanation=(
            f"The trained URL classifier assigned a phishing score of "
            f"{probability:.1%}. Verdict thresholds are heuristic and the "
            "score is not calibrated confidence. This analyzes URL structure "
            "only; it does not establish whether the live page is safe."
        ),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PhreshPhish URL detector")
    parser.add_argument("url", help="URL to analyze without visiting it")
    args = parser.parse_args()

    print(predict_url(args.url).model_dump_json(indent=2))
