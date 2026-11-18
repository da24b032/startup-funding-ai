"""FastAPI inference server tests using DummyModel (no trained model needed)."""

from fastapi.testclient import TestClient
from src.serving.app import app

client = TestClient(app)

VALID_CUSTOMER = {
    "tenure_months": 24,
    "monthly_charges": 55.0,
    "total_charges": 1320.0,
    "num_services": 4,
    "contract_type": 1,
    "paperless_billing": 1,
    "tech_support": 1,
    "online_security": 0,
    "payment_method": 2,
    "avg_monthly_gb": 15.0,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_prediction():
    r = client.post("/predict", json=VALID_CUSTOMER)
    assert r.status_code == 200
    data = r.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert data["churn_prediction"] in (0, 1)
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_predict_batch():
    r = client.post("/predict/batch", json={"customers": [VALID_CUSTOMER, VALID_CUSTOMER]})
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 2
    assert "churn_rate" in data


def test_predict_invalid_input():
    r = client.post("/predict", json={"tenure_months": -1, "monthly_charges": 50.0})
    assert r.status_code == 422


def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"churn_predictions_total" in r.content


def test_drift_endpoint():
    r = client.get("/drift")
    assert r.status_code == 200
    assert "buffer_size" in r.json()
