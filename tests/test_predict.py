import pytest

from src.api.predict import predict_url
from src.schemas import PhishingAnalysis


def test_valid_url_returns_validated_schema():
    result = predict_url("https://example.com")

    assert isinstance(result, PhishingAnalysis)
    assert result.verdict in {"phishing", "benign", "uncertain"}
    assert 0.0 <= result.risk_score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.evidence, list)
    assert result.explanation


def test_suspicious_url_returns_validated_schema():
    result = predict_url(
        "http://secure-login-verify-example.test/account"
    )

    assert isinstance(result, PhishingAnalysis)
    assert 0.0 <= result.risk_score <= 1.0
    assert result.evidence


@pytest.mark.parametrize("url", ["", "   ", "/account", "https:///missing-host"])
def test_invalid_url_is_rejected(url):
    with pytest.raises(ValueError):
        predict_url(url)


def test_feature_extraction_does_not_visit_website():
    from src.api.predict import extract_features

    features = extract_features("https://example.com/login")

    assert features["url"] == "https://example.com/login"
    assert features["uses_https"] == 1
    assert features["suspicious_term_count"] >= 1
