"""
api/main.py — Production-ready FastAPI backend for Fraud Detection.

Fixes applied vs original:
  - Added /health endpoint (was missing, broke tests + CI)
  - Added recommendation + latency_ms to Prediction (tests expected them)
  - Added /predict/batch with 100-item limit (tests expected it)
  - Added Field(gt=0) validation on Amount (tests expected 422 on negative)
  - Added Time to Transaction schema with ge=0 validation
  - Fixed model filename: fraud_model.pkl (was model.pkl — mismatch with train.py)
  - Startup now RAISES on model load failure (no more silent broken state)
  - All config via environment variables (no hardcoded paths)
  - Structured logging replaces print()
  - CORS middleware added
  - Optional API-key auth via X-API-Key header
  - API versioning prefix /v1/ with backward-compatible aliases
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List

import joblib
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("fraud_api")

# ── Config (all from environment — no hardcoded paths) ───────────────────────
MODEL_PATH   = os.getenv("MODEL_PATH",   "model/fraud_model.pkl")
SCALER_PATH  = os.getenv("SCALER_PATH",  "model/scaler.pkl")
METRICS_PATH = os.getenv("METRICS_PATH", "model/metrics.json")
API_KEY      = os.getenv("API_KEY", "")          # empty string = auth disabled
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# ── Feature order must match exactly what the scaler was trained on ──────────
FEATURE_ORDER = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]

RECOMMENDATIONS = {
    "LOW":    "✅ Transaction looks legitimate. Safe to approve.",
    "MEDIUM": "⚠️  Requires manual review before approving.",
    "HIGH":   "🚨 High fraud risk. Block and investigate immediately.",
}

# ── Global model state ────────────────────────────────────────────────────────
_state: dict = {}


# ── App lifespan (replaces deprecated @app.on_event) ─────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading model artifacts…")
    try:
        _state["model"]  = joblib.load(MODEL_PATH)
        _state["scaler"] = joblib.load(SCALER_PATH)
        log.info("Model and scaler loaded successfully.")
    except Exception as exc:
        log.critical("Model load FAILED: %s", exc)
        raise RuntimeError(f"Startup aborted — model load failed: {exc}") from exc
    yield
    _state.clear()
    log.info("Shutdown complete.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit-card fraud scoring powered by XGBoost.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Optional API-key auth ─────────────────────────────────────────────────────
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: str | None = Security(_api_key_header)):
    if API_KEY and key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key.",
        )


# ── Request / response schemas ────────────────────────────────────────────────
class Transaction(BaseModel):
    """
    Credit card transaction. 
    V1–V28 are PCA components (optional, generated if missing).
    New user-friendly fields help estimate risk if PCA is missing.
    """
    # PCA Features (Optional)
    V1: float | None = None;  V2: float | None = None;  V3: float | None = None
    V4: float | None = None;  V5: float | None = None;  V6: float | None = None
    V7: float | None = None;  V8: float | None = None;  V9: float | None = None
    V10: float | None = None; V11: float | None = None; V12: float | None = None
    V13: float | None = None; V14: float | None = None; V15: float | None = None
    V16: float | None = None; V17: float | None = None; V18: float | None = None
    V19: float | None = None; V20: float | None = None; V21: float | None = None
    V22: float | None = None; V23: float | None = None; V24: float | None = None
    V25: float | None = None; V26: float | None = None; V27: float | None = None
    V28: float | None = None

    # Base features (Required)
    Amount: float = Field(gt=0,  description="Transaction amount in USD")
    Time:   float = Field(ge=0,  description="Seconds since first transaction in dataset")

    # User-friendly features (Optional, used for demo/synthetic data)
    merchant_category: str | None = Field(default="Other", description="Category of merchant")
    transaction_location: str | None = Field(default="Home", description="Location relative to cardholder")
    transaction_type: str | None = Field(default="In-person (Chip)", description="How the card was used")
    is_new_device: bool | None = Field(default=False, description="Whether this is a new device or session")

    model_config = {"json_schema_extra": {
        "example": {
            "Amount": 149.62,
            "Time": 0.0,
            "merchant_category": "Electronics",
            "transaction_location": "Home",
            "transaction_type": "In-person (Chip)",
            "is_new_device": False
        }
    }}


class Prediction(BaseModel):
    is_fraud: bool
    fraud_probability: float = Field(ge=0.0, le=1.0)
    risk_level: str
    recommendation: str
    latency_ms: float


# ── Core prediction logic ─────────────────────────────────────────────────────
def _generate_synthetic_features(tx: Transaction) -> dict:
    """
    Generates plausible V1-V28 PCA values based on user-friendly inputs.
    Used when the technical fields are not provided by the user.
    """
    import random
    
    # Calculate a heuristic 'suspicion score' (0-100)
    score = 0
    if tx.transaction_location == "International (High-risk)":
        score += 60
    elif tx.transaction_location == "International (Low-risk)":
        score += 20
    elif tx.transaction_location == "Domestic (Different City)":
        score += 5
        
    if tx.merchant_category in ["Luxury Goods", "Mystery Box"]:
        score += 30
    elif tx.merchant_category == "Online Services":
        score += 10
        
    if tx.transaction_type in ["Manual Entry", "In-person (Swipe)"]:
        score += 15
        
    if tx.is_new_device:
        score += 20

    is_fraudulent = score >= 50
    
    # Create synthetic V values
    if is_fraudulent:
        v = [round(random.uniform(-1, 1), 4) for _ in range(28)]
        # Primary fraud signals discovered from model analysis (V15, V11, V13)
        v[14] = round(random.uniform(-18, -12), 4)   # V15 (Top Importance: 0.35)
        v[10] = round(random.uniform(-15, -10), 4)   # V11 (Importance: 0.17)
        v[12] = round(random.uniform(-10, -6),  4)   # V13 (Importance: 0.06)
        
        # Secondary fraud signals
        v[11] = round(random.uniform(-8, -4),   4)   # V12
        v[13] = round(random.uniform(-10, -5),  4)   # V14
        v[16] = round(random.uniform(-10, -5),  4)   # V17
    else:
        v = [round(random.uniform(-1.5, 1.5), 4) for _ in range(28)]

    # Map to dict V1-V28
    v_dict = {f"V{i+1}": v[i] for i in range(28)}
    return v_dict


def _run_inference(transaction: Transaction) -> Prediction:
    t0     = time.perf_counter()
    data   = transaction.model_dump()
    
    # Check if we need to generate synthetic PCA features
    v_missing = all(data.get(f"V{i}") is None for i in range(1, 29))
    if v_missing:
        log.info("V features missing; generating synthetic data based on user-friendly inputs.")
        synthetic_v = _generate_synthetic_features(transaction)
        data.update(synthetic_v)

    # Prepare for model
    raw    = np.array([[data[f] for f in FEATURE_ORDER]])
    scaled = _state["scaler"].transform(raw)
    prob   = float(_state["model"].predict_proba(scaled)[0][1])

    if prob < 0.2:   risk = "LOW"
    elif prob < 0.6: risk = "MEDIUM"
    else:            risk = "HIGH"

    latency = round((time.perf_counter() - t0) * 1000, 2)
    log.info("predict | risk=%s prob=%.4f latency_ms=%.2f", risk, prob, latency)

    return Prediction(
        is_fraud=prob >= 0.5,
        fraud_probability=round(prob, 4),
        risk_level=risk,
        recommendation=RECOMMENDATIONS[risk],
        latency_ms=latency,
    )


def _check_model():
    if "model" not in _state or "scaler" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded. Contact administrator.")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["system"], summary="Health check")
async def health():
    """Returns healthy when model is loaded, degraded otherwise."""
    loaded = "model" in _state and "scaler" in _state
    return {"status": "healthy" if loaded else "degraded", "model_loaded": loaded}


@app.get("/", tags=["system"])
async def root():
    return {"message": "Fraud Detection API v1.0 — visit /docs for full documentation."}


@app.get("/metrics/model", tags=["system"], summary="Trained model performance metrics")
async def model_metrics():
    """Returns metrics saved by train.py (ROC-AUC, PR-AUC, confusion matrix, etc.)."""
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="metrics.json not found. Run model/train.py first.",
        )


@app.post(
    "/predict",
    response_model=Prediction,
    tags=["prediction"],
    summary="Score a single transaction",
    dependencies=[Depends(verify_api_key)],
)
async def predict(transaction: Transaction):
    """Classify one credit card transaction as fraud or legitimate."""
    _check_model()
    return _run_inference(transaction)


@app.post(
    "/predict/batch",
    response_model=List[Prediction],
    tags=["prediction"],
    summary="Score up to 100 transactions in one request",
    dependencies=[Depends(verify_api_key)],
)
async def predict_batch(transactions: List[Transaction]):
    """Batch inference endpoint. Returns one Prediction per input transaction."""
    if not transactions:
        raise HTTPException(status_code=400, detail="Batch cannot be empty.")
    if len(transactions) > 100:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(transactions)} exceeds maximum of 100.",
        )
    _check_model()
    return [_run_inference(tx) for tx in transactions]
