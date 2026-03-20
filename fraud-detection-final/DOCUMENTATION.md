# FraudShield — Project Documentation

## Table of Contents
1. [System Architecture](#1-system-architecture)
2. [Complete Workflow](#2-complete-workflow)
3. [Bug Fixes Applied](#3-bug-fixes-applied)
4. [Architecture Improvements](#4-architecture-improvements)
5. [ML/Data Science Improvements](#5-mldata-science-improvements)
6. [UI/UX Improvements](#6-uiux-improvements)
7. [How the Machine Learning Works](#7-how-the-machine-learning-works)
8. [API Reference](#8-api-reference)
9. [Configuration Reference](#9-configuration-reference)
10. [Test Suite](#10-test-suite)

---

## 1. System Architecture

```
creditcard.csv (Kaggle)
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  model/train.py                                     │
│  1. Split data (80/20) — no leakage                 │
│  2. StandardScaler fit on train only                │
│  3. SMOTE oversampling                              │
│  4. XGBoost training                               │
│  5. Save fraud_model.pkl + scaler.pkl + metrics.json│
└──────────────────────────┬──────────────────────────┘
                           │ joblib.load()
                           ▼
┌─────────────────────────────────────────────────────┐
│  api/main.py (FastAPI + Uvicorn)                    │
│  GET  /health          — liveness probe             │
│  GET  /metrics/model   — performance metrics        │
│  POST /predict         — single transaction         │
│  POST /predict/batch   — up to 100 transactions     │
└──────────────────────────┬──────────────────────────┘
                           │ HTTP POST /predict
                           ▼
┌─────────────────────────────────────────────────────┐
│  dashboard/app.py (Streamlit)                       │
│  Page 1: Score Transaction — interactive scoring    │
│  Page 2: Batch Analysis — CSV upload                │
│  Page 3: Model Performance — live metrics           │
│  Page 4: Tutorial — onboarding guide                │
└─────────────────────────────────────────────────────┘
```

---

## 2. Complete Workflow

### Training workflow
```
Load CSV → Train/Test Split → Fit Scaler (train only) → SMOTE →
Train XGBoost → Evaluate on test set → Save artifacts
```

### Prediction workflow
```
User input (friendly) → Backend mapping → Synthetic PCA generation (V1-V28) → 
Scale (StandardScaler) → XGBoost.predict_proba() → Risk level assignment → Return JSON
```

### CI/CD workflow
```
Git push → GitHub Actions →
  [test job]   pip install → pytest (8 tests) →
  [docker job] docker build → smoke test /health
```

---

## 3. Bug Fixes Applied

### Bug 1: Missing `/health` endpoint
**File:** `api/main.py`
**Impact:** 2 tests failed + CI smoke test broke
**Fix:** Added `GET /health` route returning `{"status": "healthy", "model_loaded": true}`

### Bug 2: Missing `recommendation` and `latency_ms` response fields
**File:** `api/main.py`
**Impact:** `test_predict_returns_expected_keys` failed
**Fix:** Added both fields to `Prediction` Pydantic model and populated them in `_run_inference()`

### Bug 3: Missing `/predict/batch` endpoint
**File:** `api/main.py`
**Impact:** `test_batch_predict` and `test_batch_limit` both returned 404
**Fix:** Implemented `POST /predict/batch` accepting `List[Transaction]`, max 100 items

### Bug 4: `Amount` field had no validation
**File:** `api/main.py`
**Impact:** `test_invalid_amount_rejected` expected 422 but got 200 (negative amounts passed through)
**Fix:** Changed `Amount: float` to `Amount: float = Field(gt=0)`

### Bug 5: Test helper `make_transaction()` missing `Time` field
**File:** `tests/test_api.py`
**Impact:** Every single test that used the helper returned 422 before reaching any logic
**Fix:** Added `tx["Time"] = 0.0` to the helper function

### Bug 6: Model filename mismatch between `train.py` and `main.py`
**Files:** `model/train.py`, `api/main.py`
**Impact:** API could never find the model file — constant 503 errors at runtime
**Fix:** Standardized to `fraud_model.pkl` in both files

### Bug 7: Startup silently swallowed model load failures
**File:** `api/main.py`
**Impact:** App started "successfully" even when model was missing, then returned 503 on every request
**Fix:** Replaced `except Exception → model = None` pattern with `raise RuntimeError` in lifespan

### Bug 8: CI smoke test pointed at non-existent `/health` endpoint
**File:** `.github/workflows/ci.yml`
**Impact:** Docker build job permanently broken
**Fix:** Resolved automatically by Bug 1 fix above

### Bug 9: Random generation not updating UI
**File:** `dashboard/app.py`
**Impact:** "Generate Legit/Fraud" buttons failed to update PCA numeric inputs
**Fix:** Refactored dashboard to use user-friendly inputs and updated `st.session_state` correctly.

---

## 4. Architecture Improvements

### Config via environment variables
**Before:** `API_URL = "http://127.0.0.1:8000"` hardcoded in dashboard; paths hardcoded in API
**After:** All config from `os.getenv()` with sensible defaults. Set `API_URL`, `MODEL_PATH`, `SCALER_PATH`, `API_KEY`, `ALLOWED_ORIGINS` in environment.

### Structured logging
**Before:** `print()` statements scattered throughout
**After:** Python `logging` module with timestamped, leveled, named logger output in `api/main.py`

### CORS middleware
**Before:** No CORS — any cross-origin request blocked by browser
**After:** `CORSMiddleware` added. Configure `ALLOWED_ORIGINS` env var for production.

### Optional API key authentication
**Before:** No authentication — any caller could hit `/predict` unlimited times
**After:** Set `API_KEY` environment variable to enable. Requests must include `X-API-Key` header.

### Multi-stage Dockerfile
**Before:** Single-stage, ran as root, no HEALTHCHECK, ~1.2GB image
**After:** Builder stage + slim runtime, `appuser` non-root, `HEALTHCHECK` instruction, ~400MB image

### Requirements cleanup
**Before:** Full `pip freeze` output with 60+ packages including `reportlab`, `GitPython`, `annotated-doc` (unused)
**After:** 14 direct dependencies only, with version pins appropriate for production

### Dynamic model metrics
**Before:** ROC-AUC, Recall etc. hardcoded as string literals in dashboard and README — would diverge from actual model
**After:** `train.py` saves `metrics.json`. Dashboard reads it via `GET /metrics/model` API endpoint.

### Batch endpoint
**Before:** Only single-transaction scoring
**After:** `POST /predict/batch` processes up to 100 transactions in one HTTP call

### Duplicate directory removed
**Before:** Both `model/` and `models/` directories existed with identical files
**After:** Single canonical `model/` directory

---

## 5. ML/Data Science Improvements

### Critical: Data leakage fix
**Before:**
```python
# WRONG — scaler sees test data statistics
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)           # fit on full dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, ...)
```
**After:**
```python
# CORRECT — scaler only sees training data
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, ...)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)   # fit on train only
X_test_scaled  = scaler.transform(X_test_raw)        # transform only
```
**Why it matters:** Fitting the scaler on the full dataset leaks test-set statistics (mean, std) into the model's training environment. This makes test metrics optimistically biased — the model indirectly "knows" about the test set.

### Full evaluation suite
**Before:** Only PR-AUC printed; `roc_auc_score` was imported but never called
**After:** Prints ROC-AUC, PR-AUC, Precision, Recall, F1, confusion matrix (TN/FP/FN/TP), false alarm rate, fraud catch rate

### Saved metrics artifact
`train.py` now writes `model/metrics.json` so metrics are version-controlled alongside the model and accessible to the dashboard at runtime.

### Improved XGBoost hyperparameters
Added `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5` for better regularization. Increased `n_estimators=200` with lower `learning_rate=0.05` for a better bias-variance tradeoff.

---

## 6. UI/UX Improvements

### Design system
- Dark theme (`#0d0f1a` background) with indigo (`#6366f1`) primary color
- DM Sans + DM Mono typography (professional, readable)
- Card-based layout with subtle borders and depth
- Consistent color coding: green=safe, amber=medium, red=fraud

### New pages
- **Batch Analysis page** — upload a CSV, score up to 100 transactions, download results
- **Tutorial page** — complete onboarding with 4 tabs covering system overview, scoring, batch usage, and output interpretation

### Improved results display
- Risk card shows probability in large monospace font with color-matching background
- Plotly gauge chart replaces the basic one with better color zones
- Feature magnitude bar chart shows which V features drove the transaction
- Latency badge shows API response time inline

### API status indicator
Sidebar shows a live green/red dot indicating whether the FastAPI server is reachable. Users know immediately if something is wrong before trying to score.

### Better error handling
All API calls wrapped with connection error detection and user-friendly error messages including the exact command needed to fix the issue.

---

## 7. How the Machine Learning Works

### Why XGBoost?
XGBoost (Extreme Gradient Boosting) is an ensemble of decision trees trained sequentially, where each new tree corrects the errors of the previous ones. It's the industry standard for tabular fraud detection because it:
- Handles the mix of PCA features and raw Amount/Time natively
- Is robust to outliers (fraud often has extreme feature values)
- Trains fast and predicts in microseconds
- Provides calibrated probability outputs

### Why SMOTE?
The dataset has 284,315 legitimate transactions and only 492 fraudulent ones (0.17%). A naive model would achieve 99.83% accuracy by predicting "legit" for everything. SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic fraud examples by interpolating between real fraud transactions, giving the model a balanced training set to learn from.

### Why StandardScaler?
V1–V28 are already PCA-transformed (zero-mean). However, `Amount` ranges from $0 to $25,000 and `Time` ranges from 0 to 172,800. Without scaling, XGBoost's split decisions would be dominated by these large-magnitude features. StandardScaler normalizes all features to zero-mean and unit variance.

### The feature vector
The XGBoost model expects 30 technical features. If the user does not provide V1–V28 (PCA components), the API automatically generates them:
- **Amount**: Transaction dollar value (Required)
- **Time**: Seconds elapsed since dataset start (Required)
- **User-Friendly Context**: Merchant Category, Location, Mode of entry, and Device context are used to synthesize a plausible V1–V28 profile.

---

## 8. API Reference

### GET /health
Returns the API health status.
```json
{"status": "healthy", "model_loaded": true}
```

### GET /metrics/model
Returns training metrics from `model/metrics.json`.
```json
{
  "roc_auc": 0.9780,
  "pr_auc": 0.8410,
  "precision": 0.8700,
  "recall": 0.8800,
  "f1_score": 0.8750,
  "false_alarm_rate": 0.03,
  "fraud_catch_rate": 88.0,
  "confusion_matrix": {"tn": 56848, "fp": 18, "fn": 12, "tp": 88}
}
```

### POST /predict
Score a single transaction.
**Request body (User-Friendly):**
```json
{
  "Amount": 149.62,
  "Time": 0.0,
  "merchant_category": "Electronics",
  "transaction_location": "Home",
  "transaction_type": "In-person (Chip)",
  "is_new_device": false
}
```
**Request body (Technical - optional):**
```json
{
  "V1": -1.35, "V2": -0.07, ..., "V28": 0.02,
  "Amount": 149.62,
  "Time": 0.0
}
```
**Response:**
```json
{
  "is_fraud": false,
  "fraud_probability": 0.0312,
  "risk_level": "LOW",
  "recommendation": "✅ Transaction looks legitimate. Safe to approve.",
  "latency_ms": 3.4
}
```

### POST /predict/batch
Score 1–100 transactions.
**Request body:** Array of Transaction objects (same schema as /predict).
**Response:** Array of Prediction objects.
**Error:** HTTP 400 if array is empty or > 100 items.

---

## 9. Configuration Reference

All configuration is via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `model/fraud_model.pkl` | Path to trained model file |
| `SCALER_PATH` | `model/scaler.pkl` | Path to fitted scaler file |
| `METRICS_PATH` | `model/metrics.json` | Path to metrics JSON file |
| `API_KEY` | `""` (disabled) | Set to enable X-API-Key auth |
| `ALLOWED_ORIGINS` | `"*"` | CORS allowed origins (comma-separated) |
| `API_URL` | `http://127.0.0.1:8000` | Dashboard: FastAPI server URL |

---

## 10. Test Suite

8 tests in `tests/test_api.py`, all passing:

| Test | What it verifies |
|------|-----------------|
| `test_health_check` | `/health` returns 200 with `status` field |
| `test_predict_returns_expected_keys` | All 5 response fields present |
| `test_fraud_probability_in_range` | Probability is between 0 and 1 |
| `test_risk_level_is_valid` | Risk level is LOW, MEDIUM, or HIGH |
| `test_invalid_amount_rejected` | Amount ≤ 0 returns 422 |
| `test_missing_field_rejected` | Missing V1 returns 422 |
| `test_batch_predict` | Batch of 5 returns 5 results |
| `test_batch_limit` | Batch of 101 returns 400 |

Run with: `pytest tests/ -v`
