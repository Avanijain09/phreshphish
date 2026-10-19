# PhreshPhish — Small-Model Phishing URL Detector

PhreshPhish is a research prototype that analyzes URL structure and text patterns to estimate phishing risk. It exposes predictions through a REST API built with FastAPI.

> **Research use only:** This model is not a browser security product and must not be used as the sole basis for deciding whether a website is safe. Scores are not calibrated confidence estimates.

## Features

- URL-based phishing risk classification.
- Numeric URL feature extraction and word/character TF-IDF features.
- Domain-grouped model evaluation to reduce domain overlap between training and evaluation data.
- JSON prediction responses with a verdict, risk score, evidence, and explanation.
- FastAPI endpoints and interactive API documentation.
- Automated tests for prediction responses, invalid URLs, and URL feature extraction.

## Architecture

1. **Dataset:** PhreshPhish dataset from Hugging Face.
2. **Preprocessing:** Prepare URL records and labels for model training and evaluation.
3. **Feature engineering:** Extract URL characteristics and text representations.
4. **Model:** Random Forest classifier with numeric and TF-IDF features.
5. **Evaluation:** Evaluate on a domain-grouped split and a sampled official-test-shard set.
6. **API:** FastAPI exposes the trained model through `/predict`.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Health check |
| POST | `/predict` | Predict phishing risk for a URL |
| GET | `/docs` | Interactive API documentation |

### Example request

```json
{
  "url": "http://secure-login-verify-example.test/account"
}
```

### Example response

```json
{
  "verdict": "phishing",
  "risk_score": 0.94,
  "confidence": 0.94,
  "evidence": [
    "URL contains 4 account/security-related keyword matches.",
    "URL contains 3 hyphens."
  ],
  "explanation": "The classifier assigned a high phishing score. The score is not calibrated confidence.",
  "model_version": "phreshphish-v1"
}
```

The response above is an illustrative synthetic demo, not a guarantee of detection accuracy. Actual evidence and scores depend on the input and saved model.

## Local Setup

Use Python 3.11 and a virtual environment.

```bash
git clone <your-repository-url>
cd phreshphish

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt
```

Ensure the trained model artifact exists at:

```text
artifacts/url_token_model_domain_grouped.joblib
```

## Run the API

From the project root:

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to test the API interactively.

## Run Tests

Run the focused prediction tests:

```bash
PYTHONPATH=. pytest -v tests/test_predict.py
```

## Evaluation Results

The current recorded evaluation results are:

| Evaluation | Accuracy | ROC-AUC |
|---|---:|---:|
| Domain-grouped internal evaluation | 0.9302 | 0.9802 |
| Sampled official-test-shard set (5,000 records) | 0.8940 | 0.9647 |

The 5,000-record evaluation set was sampled from the first two downloaded official test shards; it should not be described as a random sample of the entire official test split. The evaluation has already been inspected, so it should not be used repeatedly for model selection.

### Sampled official-test-set classification results

- Benign: precision 0.88, recall 0.94, F1-score 0.91.
- Phishing: precision 0.92, recall 0.84, F1-score 0.88.

These metrics are dataset-dependent and do not guarantee performance on newly observed URLs or unseen phishing campaigns.

## Limitations and Responsible Use

- The detector analyzes URL structure and text; it does not inspect the live webpage.
- A phishing verdict is a risk signal, not proof that a URL is malicious.
- A benign verdict does not guarantee a website is safe.
- Scores and confidence fields are not calibrated probabilities of real-world harm.
- Dataset sampling, class distribution, domain overlap, and distribution shift can affect performance.
- The current API is a local research prototype, not a production-hardened security service.
- The project does not currently claim a connected LLM, QLoRA fine-tuning, or live webpage analysis.

## Dataset Attribution

The project uses the PhreshPhish dataset published by `phreshphish` on Hugging Face. The dataset is identified as CC BY 4.0 and for research use. Review the dataset card and license terms before redistributing dataset contents or deploying a model trained on the data.

## Project Status

The URL classification model, evaluation scripts, prediction wrapper, FastAPI endpoints, and focused prediction tests are implemented. Production deployment, calibration, broader independent validation, and any LLM fine-tuning remain separate work.
