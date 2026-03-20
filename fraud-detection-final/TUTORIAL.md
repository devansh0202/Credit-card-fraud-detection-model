# 🎓 FraudShield — Beginner Tutorial

Welcome! This guide will walk you through the entire system from zero to a running fraud detector in under 10 minutes.

---

## What You're Building

A real-time fraud detection system that:
1. Trains an AI model on real credit card transaction data
2. Serves that model as a REST API
3. Provides a web dashboard to score transactions

---

## Step 1 — Install Dependencies

```bash
git clone https://github.com/devansh0202/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt
```

**What this installs:**
- `fastapi` + `uvicorn` — the web API server
- `streamlit` + `plotly` — the visual dashboard
- `xgboost` + `scikit-learn` — machine learning libraries
- `pytest` — for running tests

---

## Step 2 — Get the Training Data

1. Go to [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv` (144 MB)
3. Create a `data/` folder in the project root
4. Move `creditcard.csv` into `data/creditcard.csv`

The dataset has **284,807 transactions** with only **492 fraudulent** ones (0.17%). This extreme imbalance is what makes fraud detection challenging.

---

## Step 3 — Train the Model

```bash
python model/train.py
```

**What happens:**
1. Loads the CSV and splits it 80% train / 20% test
2. Fits a StandardScaler **only on training data** (important: prevents data leakage)
3. Uses SMOTE to generate synthetic fraud examples so the model sees balanced classes
4. Trains an XGBoost classifier with tuned hyperparameters
5. Evaluates on the held-out test set and prints metrics
6. Saves three files: `model/fraud_model.pkl`, `model/scaler.pkl`, `model/metrics.json`

**Expected output:**
```
📂  Loading dataset…
     Rows: 284,807 | Fraud rate: 0.173%
✂️   Splitting train/test (80/20, stratified)…
⚖️   Balancing classes with SMOTE…
🚀  Training XGBoost classifier…
  ROC-AUC   : 0.9780
  PR-AUC    : 0.8400
  Recall    : 0.8800
✅  Saved: model/fraud_model.pkl
```

**Key metrics explained:**
| Metric | What it means | Target |
|--------|--------------|--------|
| ROC-AUC | Overall ranking ability (1.0 = perfect) | > 0.95 |
| PR-AUC | Precision-recall tradeoff on imbalanced data | > 0.80 |
| Recall | % of frauds actually caught | > 0.85 |
| Precision | % of fraud alerts that were real | > 0.80 |

---

## Step 4 — Start the API Server

Open a terminal and run:

```bash
uvicorn api.main:app --reload
```

You should see:
```
INFO:     Loading model artifacts…
INFO:     Model and scaler loaded successfully.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**Test it's working:**
```bash
curl http://127.0.0.1:8000/health
# → {"status": "healthy", "model_loaded": true}
```

**Explore the API docs:**
Open your browser to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the full interactive documentation.

---

## Step 5 — Launch the Dashboard

Open a **second terminal** and run:

```bash
streamlit run dashboard/app.py
```

The dashboard opens automatically at [http://localhost:8501](http://localhost:8501).

---

## Step 6 — Score Your First Transaction

1. Click **"🔍 Score Transaction"** in the sidebar
2. Click **"🚨 Generate Fraud"** to load fraud-like values
3. Click **"🔮 Run Analysis"**
4. You should see a HIGH risk result with a red gauge

Then:
1. Click **"🎲 Generate Legit"** to load normal values
2. Click **"🔮 Run Analysis"** again
3. You should see a LOW risk result with a green gauge

---

## Understanding the Features

Since technical PCA features (**V1–V28**) are not user-friendly, FraudShield uses a hybrid approach:
1. **User-Friendly Inputs**: You select the Merchant Category, Location, and Transaction Type.
2. **Synthetic PCA Generation**: The backend converts these choices into a mathematical profile (V1-V28) that the AI understands.

**Key concepts:**
- **V14, V12, V17** are the most predictive internal features.
- **International (High-risk)** locations or **New Devices** combined with **Luxury Goods** will trigger these internal features to look "fraudulent".
- **Amount** still plays a direct role in the final score.

---

## Run the Tests

```bash
pytest tests/ -v
```

All 8 tests should pass:
```
✓ test_health_check
✓ test_predict_returns_expected_keys
✓ test_fraud_probability_in_range
✓ test_risk_level_is_valid
✓ test_invalid_amount_rejected
✓ test_missing_field_rejected
✓ test_batch_predict
✓ test_batch_limit
```

---

## Docker (Optional)

If you have Docker installed:

```bash
# Build
docker build -t fraud-detection-api .

# Run (put your trained model files in the model/ folder first)
docker run -p 8000:8000 fraud-detection-api

# Test
curl http://localhost:8000/health
```

---

## Common Issues

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: creditcard.csv` | Place the file in `data/creditcard.csv` |
| `Model not loaded` in API | Run `python model/train.py` first |
| Dashboard says "API Offline" | Start the API: `uvicorn api.main:app --reload` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Tests fail | Make sure you haven't changed the API schema |

---

## What Happens When You Score a Transaction

```
User selects Merchant, Location, Type + Amount
         ↓
Dashboard sends details to FastAPI
         ↓
API Synthesizes V1–V28 PCA features if missing
         ↓
StandardScaler normalizes everything
         ↓
XGBoost predicts fraud_probability (0.0–1.0)
         ↓
Risk level assigned: < 0.2 = LOW, < 0.6 = MEDIUM, else HIGH
         ↓
Dashboard renders gauge chart + risk card
```

---

*Built with XGBoost · FastAPI · Streamlit · scikit-learn*
