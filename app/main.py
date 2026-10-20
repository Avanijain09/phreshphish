
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api.predict import load_model, predict_url
from src.schemas import PhishingAnalysis
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(
    title="PhreshPhish API",
    description="Research prototype for URL-based phishing detection.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
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
    try:
        load_model()
        return {
            "status": "ok",
            "model_status": "loaded",
        }
    except Exception:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "error",
                "model_status": "unavailable",
            },
        )


@app.post("/predict", response_model=PhishingAnalysis)
def predict(request: PredictionRequest):
    try:
        return predict_url(request.url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
