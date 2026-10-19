# PhreshPhish — Project Plan

## 1. Project Objective
Build a machine-learning-based phishing URL detection system that classifies URLs as phishing or benign using URL-derived features.

## 2. Current Implementation
- Prepared and validated URL datasets.
- Implemented URL feature extraction and data preparation scripts.
- Trained a URL-token classification model using a domain-grouped split.
- Evaluated model performance on internal and sampled official test data.
- Built a FastAPI application for URL predictions.
- Added prediction schemas and automated API tests.
- Documented setup and usage in the README.

## 3. Technology Stack
- Python 3.11
- Pandas and scikit-learn
- Joblib
- FastAPI and Pydantic
- Pytest
- Git

## 4. Evaluation Results
- Domain-grouped internal test accuracy: 93.02%
- Domain-grouped internal test ROC-AUC: 0.9802
- Sampled official-test-shard accuracy: 89.40%
- Sampled official-test-shard ROC-AUC: 0.9647

The official-test-shard evaluation used 5,000 sampled records from the first two official test shards. These results should not be interpreted as production performance.

## 5. Testing
The targeted prediction test suite passes all 7 tests.

## 6. Current Limitations
- Predictions are based on URL information, not live webpage content.
- Model scores are not calibrated probabilities.
- The system has not been established as production-ready.
- LLM integration and fine-tuning are not implemented.

## 7. Future Improvements
- Add more robust input validation and error handling.
- Expand evaluation across broader, representative datasets.
- Add monitoring and structured prediction logging.
- Consider containerization and deployment.
- Explore LLM-based analysis as a separate, evaluated extension.
