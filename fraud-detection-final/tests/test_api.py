"""
tests/test_api.py — Unit + integration tests for the Fraud Detection API.

Run:  pytest tests/ -v

All 8 tests now pass. Fixes applied vs original:
  - make_transaction() now includes Time=0.0 (was missing, caused 422 on all tests)
  - test_health_check hits /health which now exists in main.py
  - test_predict_returns_expected_keys includes recommendation + latency_ms
  - test_invalid_amount_rejected: Amount now has Field(gt=0) validation
  - test_batch_predict + test_batch_limit: /predict/batch endpoint now implemented
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api.main import app

client = TestClient(app)


# ── Helpers ───────────────────────────────────────────────────────────────────
def make_transaction(**overrides) -> dict:
    """Return a valid transaction with default user-friendly fields."""
    tx = {
        "Amount": 100.0,
        "Time": 0.0,
        "merchant_category": "Other",
        "transaction_location": "Home",
        "transaction_type": "In-person (Chip)",
        "is_new_device": False
    }
    tx.update(overrides)
    return tx


@pytest.fixture(scope="module")
def client():
    """Fixture to provide a TestClient with lifespan events triggered."""
    with TestClient(app) as c:
        yield c


# ── Tests ─────────────────────────────────────────────────────────────────────
def test_health_check(client):
    """/health must return 200 with a status field."""
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert data["status"] in {"healthy", "degraded"}


def test_predict_returns_expected_keys(client):
    """Prediction response must include all five required fields."""
    r = client.post("/predict", json=make_transaction())
    assert r.status_code == 200
    data = r.json()
    for key in ["is_fraud", "fraud_probability", "risk_level", "recommendation", "latency_ms"]:
        assert key in data, f"Missing key: {key}"


def test_fraud_probability_in_range(client):
    """Fraud probability must be a float between 0.0 and 1.0 inclusive."""
    r = client.post("/predict", json=make_transaction())
    assert r.status_code == 200
    prob = r.json()["fraud_probability"]
    assert 0.0 <= prob <= 1.0


def test_risk_level_is_valid(client):
    """Risk level must be one of LOW, MEDIUM, or HIGH."""
    r = client.post("/predict", json=make_transaction())
    assert r.status_code == 200
    assert r.json()["risk_level"] in {"LOW", "MEDIUM", "HIGH"}


def test_invalid_amount_rejected(client):
    """Amount <= 0 must return HTTP 422 (Pydantic validation error)."""
    r = client.post("/predict", json=make_transaction(Amount=-10.0))
    assert r.status_code == 422


def test_missing_field_rejected(client):
    """Omitting a required field (Amount) must return HTTP 422."""
    tx = make_transaction()
    del tx["Amount"]
    r = client.post("/predict", json=tx)
    assert r.status_code == 422


def test_batch_predict(client):
    """Batch endpoint must return exactly one result per input transaction."""
    txs = [make_transaction(Amount=float(i * 10 + 1)) for i in range(5)]
    r = client.post("/predict/batch", json=txs)
    assert r.status_code == 200
    results = r.json()
    assert len(results) == 5
    for item in results:
        assert "fraud_probability" in item
        assert "risk_level" in item


def test_batch_limit(client):
    """Batch endpoint must reject more than 100 transactions with HTTP 400."""
    txs = [make_transaction() for _ in range(101)]
    r = client.post("/predict/batch", json=txs)
    assert r.status_code == 400
    assert "100" in r.json()["detail"]
