from typing import Literal

from pydantic import BaseModel, Field


class PhishingAnalysis(BaseModel):
    verdict: Literal["phishing", "benign", "uncertain"]
    risk_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    explanation: str
    model_version: str = "phreshphish-v1"
