"""
FastAPI inference server with:
  - POST /predict        — single customer churn prediction
  - POST /predict/batch  — batch predictions
  - GET  /health         — model status + metadata
  - GET  /metrics        — Prometheus metrics endpoint
  - GET  /drift          — current drift check results
"""

from __future__ import annotations

import logging
import time

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from src.config import settings
from src.data.validate import BatchRequest, CustomerFeatures
from src.drift.detector import DriftDetector
from src.features.engineering import prepare_features
from src.monitoring.metrics import (
    BATCH_SIZE,
    record_prediction,
    register_model,
    update_churn_rate,
)
from src.serving.model_loader import get_model_meta, get_model_version, load_model
from src.serving.schemas import BatchPredictionResponse, PredictionResponse

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Churn Prediction — MLOps Template",
    description="Production-grade ML inference server with Prometheus monitoring and drift detection.",
    version="0.1.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_drift_detector = DriftDetector(threshold=settings.drift_threshold)


@app.on_event("startup")
async def startup() -> None:
    model = load_model()
    meta = get_model_meta()
    register_model(
        model_version=get_model_version(),
        model_type=meta.get("model_type", "unknown"),
    )
    logger.info("Inference server ready | model_version=%s", get_model_version())


@app.get("/health")
async def health() -> dict:
    meta = get_model_meta()
    return {
        "status": "ok",
        "model_version": get_model_version(),
        "model_type": meta.get("model_type", "unknown"),
        "roc_auc": meta.get("roc_auc"),
        "drift_buffer_size": len(_drift_detector),
    }


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus scrape endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerFeatures) -> PredictionResponse:
    """Predict churn probability for a single customer."""
    model = load_model()
    model_version = get_model_version()

    row = pd.DataFrame([customer.model_dump()])
    X, _ = prepare_features(row)

    t0 = time.perf_counter()
    try:
        prob = float(model.predict_proba(X)[0, 1])
        pred = int(prob >= 0.5)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    latency = time.perf_counter() - t0

    record_prediction(predicted_class=pred, model_version=model_version, latency_seconds=latency)
    _drift_detector.add(customer.model_dump())

    return PredictionResponse(
        customer_features=customer.model_dump(),
        churn_probability=round(prob, 4),
        churn_prediction=pred,
        model_version=model_version,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchRequest) -> BatchPredictionResponse:
    """Predict churn for a batch of customers (up to 1,000)."""
    model = load_model()
    model_version = get_model_version()

    BATCH_SIZE.observe(len(request.customers))
    rows = pd.DataFrame([c.model_dump() for c in request.customers])
    X, _ = prepare_features(rows)

    t0 = time.perf_counter()
    probs = model.predict_proba(X)[:, 1]
    latency = time.perf_counter() - t0

    predictions = []
    churn_count = 0
    for i, (customer, prob) in enumerate(zip(request.customers, probs)):
        pred = int(float(prob) >= 0.5)
        churn_count += pred
        record_prediction(pred, model_version, latency / len(request.customers))
        _drift_detector.add(customer.model_dump())
        predictions.append(
            PredictionResponse(
                customer_features=customer.model_dump(),
                churn_probability=round(float(prob), 4),
                churn_prediction=pred,
                model_version=model_version,
            )
        )

    churn_rate = churn_count / len(predictions)
    update_churn_rate(churn_rate)

    return BatchPredictionResponse(
        predictions=predictions,
        model_version=model_version,
        total=len(predictions),
        predicted_churn_count=churn_count,
        churn_rate=round(churn_rate, 4),
    )


@app.get("/drift")
async def drift_check() -> dict:
    """Run a drift check against the baseline and return per-feature results."""
    results = _drift_detector.check()
    drifted = [f for f, v in results.items() if v.get("drifted")]
    return {
        "buffer_size": len(_drift_detector),
        "drifted_features": drifted,
        "details": results,
    }
