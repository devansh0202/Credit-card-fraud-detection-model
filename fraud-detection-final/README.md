# 🛡️ FraudShield — Real-Time Fraud Detection

[![CI](https://github.com/YOUR_USERNAME/fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/fraud-detection/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![Docker](https://img.shields.io/badge/docker-ready-blue)
![Tests](https://img.shields.io/badge/tests-8%20passing-brightgreen)

> End-to-end ML system that detects fraudulent credit card transactions using **user-friendly inputs**.
> XGBoost model · FastAPI REST API · Streamlit dashboard · Synthetic PCA Generation.

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Get data → place creditcard.csv in data/
#    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# 3. Train the model
python model/train.py

# 4. Start API (terminal 1)
uvicorn api.main:app --reload

# 5. Launch dashboard (terminal 2)
streamlit run dashboard/app.py
```

**→ API docs:** http://localhost:8000/docs
**→ Dashboard:** http://localhost:8501

---

## Architecture

```
creditcard.csv → train.py → fraud_model.pkl + scaler.pkl + metrics.json
                                       ↓
                              FastAPI /predict endpoint
                                       ↓
                           Streamlit Dashboard (4 pages)
```

---

## Performance (on held-out 20% test set)

| Metric | Value |
|--------|-------|
| ROC-AUC | ~0.978 |
| PR-AUC | ~0.841 |
| Recall (fraud catch rate) | ~88% |
| Precision | ~87% |
| Avg prediction latency | < 5ms |

*Exact values written to `model/metrics.json` after training.*

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/metrics/model` | Model performance metrics |
| POST | `/predict` | Score a single transaction |
| POST | `/predict/batch` | Score up to 100 transactions |

Full docs at `/docs` when the server is running.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | XGBoost |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| API Framework | FastAPI + Pydantic v2 |
| Dashboard | Streamlit + Plotly |
| Containerisation | Docker (multi-stage) |
| CI/CD | GitHub Actions |
| Testing | pytest |

---

## Project Structure

```
fraud-detection/
├── api/
│   └── main.py              # FastAPI app (all endpoints + validation)
├── model/
│   ├── train.py             # Training pipeline (leakage-free)
│   ├── fraud_model.pkl      # Trained XGBoost model
│   ├── scaler.pkl           # Fitted StandardScaler
│   └── metrics.json         # Evaluation metrics (auto-generated)
├── dashboard/
│   └── app.py               # Streamlit dashboard (4 pages)
├── tests/
│   └── test_api.py          # 8 unit/integration tests
├── .github/workflows/
│   └── ci.yml               # GitHub Actions CI pipeline
├── data/                    # Place creditcard.csv here (not in git)
├── Dockerfile               # Multi-stage build
├── requirements.txt         # Direct dependencies only
├── TUTORIAL.md              # Beginner onboarding guide
└── DOCUMENTATION.md         # Full technical documentation
```

---

## Docker

```bash
docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api
```

---

## Tests

```bash
pytest tests/ -v
# 8 tests, all passing
```

---

## Configuration

Set via environment variables:

```bash
export MODEL_PATH=model/fraud_model.pkl
export API_KEY=your-secret-key       # enables auth
export API_URL=http://api:8000        # for dashboard in Docker
```

---

## Author

**Devansh Sharma** · [LinkedIn](https://www.linkedin.com/in/devansh-sharma-6b920228a) · [Email](mailto:devanshpandit02@gmail.com)
