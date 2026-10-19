
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api.predict import predict_url
from src.schemas import PhishingAnalysis

app = FastAPI(
    title="PhreshPhish API",
    description="Research prototype for URL-based phishing detection.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    url: str = Field(
        min_length=1,
        description="URL to analyze. The API does not visit the website.",
    )


@app.get("/")
def home():
    return {
        "message": "Welcome to PhreshPhish API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PhishingAnalysis)
def predict(request: PredictionRequest):
    try:
        return predict_url(request.url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
