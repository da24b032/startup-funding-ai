"""API tests: contract, validation errors, monitoring endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.serving.api import app


@pytest.fixture()
def client(require_model):
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_predict_returns_valid_probabilities(client, profile):
    resp = client.post("/predict", json=profile)
    assert resp.status_code == 200
    body = resp.json()
    assert body["outcome"] in {"Acquired", "Closed"}
    assert abs(body["acquired_probability"] + body["closed_probability"] - 1) < 1e-3
    assert "x-request-id" in resp.headers


def test_explain_returns_factors(client, profile):
    body = client.post("/predict/explain", json=profile).json()
    assert body["explanation_method"].startswith("occlusion")
    assert all(f["direction"] in {"raises", "lowers"} for f in body["top_factors"])


def test_batch_summary_counts_add_up(client, profile):
    resp = client.post("/predict/batch", json={"records": [profile, profile, profile]})
    body = resp.json()
    assert body["total"] == 3
    assert body["predicted_acquired"] + body["predicted_closed"] == 3


@pytest.mark.parametrize(
    "change",
    [
        {"funding_rounds": -1},
        {"has_VC": 2},
        {"state_code": "California"},
        {"age_last_funding_year": 0.1},  # earlier than first funding
        {"milestones": 0},  # milestone ages present but milestones == 0
        {"unknown_field": 1},
    ],
)
def test_invalid_input_returns_422_with_request_id(client, profile, change):
    resp = client.post("/predict", json={**profile, **change})
    assert resp.status_code == 422
    assert resp.json()["error"] == "Invalid input" and resp.json()["request_id"]


def test_missing_field_returns_422(client, profile):
    profile.pop("relationships")
    assert client.post("/predict", json=profile).status_code == 422


def test_oversized_batch_rejected(client, profile):
    resp = client.post("/predict/batch", json={"records": [profile] * 501})
    assert resp.status_code == 422


def test_schema_describes_all_18_fields(client):
    fields = client.get("/schema").json()["fields"]
    assert len(fields) == 18
    assert {"state_code", "has_VC", "funding_total_usd"} <= {f["name"] for f in fields}


def test_model_info_carries_traceability(client):
    info = client.get("/model/info").json()
    assert info["mlflow_run_id"] and "git_commit" in info and info["train_data_sha256"]


def test_drift_endpoint_and_reset(client, profile):
    assert client.get("/drift").status_code == 200
    assert client.post("/drift/reset").json() == {"status": "reset"}


def test_prometheus_metrics_exposed(client, profile):
    client.post("/predict", json=profile)
    text = client.get("/metrics").text
    for name in ("startup_predictions_total", "startup_http_requests_total", "startup_model_info"):
        assert name in text


def test_pipeline_summary_shape(client):
    body = client.get("/pipeline/summary").json()
    assert {"data_report", "model_comparison", "dvc_stages", "run_counts"} <= set(body)
    assert {s["name"] for s in body["dvc_stages"]} >= {"prepare", "train"}
