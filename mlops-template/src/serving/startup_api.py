"""Startup acquisition prediction API."""

import os
import pickle
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from starlette.responses import Response

MODEL_PATH = Path(os.getenv(
    "STARTUP_MODEL_PATH",
    "../Success-Prediction-System/best_classical_model.pkl"
)).resolve()

DATA_PATH = Path(os.getenv(
    "STARTUP_TRAIN_DATA",
    "../Success-Prediction-System/X_train.csv"
)).resolve()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = list(model.feature_names_in_)

with open(DATA_PATH, "r") as f:
    baseline = pd.read_csv(f)

PREDICTIONS = Counter(
    "startup_predictions_total",
    "Total startup acquisition predictions",
    ["predicted_class"]
)

LATENCY = Histogram(
    "startup_prediction_latency_seconds",
    "Startup prediction latency"
)

ACQUISITION_RATE = Gauge(
    "startup_acquisition_prediction_rate",
    "Fraction of predictions classified as acquired"
)

app = FastAPI(
    title="Startup Acquisition Prediction API",
    description="Random Forest startup acquisition prediction with monitoring.",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ..., description="Exactly 41 model features, using their original names."
    )


class BatchRequest(BaseModel):
    records: list[PredictionRequest] = Field(..., min_length=1, max_length=500)


def prepare_input(features):
    missing = set(FEATURES) - set(features)
    extra = set(features) - set(FEATURES)

    if missing or extra:
        raise ValueError({
            "missing_features": sorted(missing),
            "extra_features": sorted(extra)
        })

    return pd.DataFrame(
        [[float(features[name]) for name in FEATURES]],
        columns=FEATURES
    )


def predict_one(features):
    X = prepare_input(features)

    start = time.perf_counter()
    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]
    latency = time.perf_counter() - start

    probability = float(probabilities[list(model.classes_).index(1)])

    PREDICTIONS.labels(predicted_class=str(prediction)).inc()
    LATENCY.observe(latency)

    return {
        "prediction": prediction,
        "outcome": "Acquired" if prediction == 1 else "Closed",
        "acquired_probability": round(probability, 4),
        "model": "RandomForestClassifier",
        "latency_seconds": round(latency, 6)
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "RandomForestClassifier",
        "features": len(FEATURES),
        "model_path": str(MODEL_PATH)
    }


@app.get("/features")
def features():
    return {"count": len(FEATURES), "feature_names": FEATURES}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        result = predict_one(request.features)
        ACQUISITION_RATE.set(result["prediction"])
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    try:
        results = [predict_one(record.features) for record in request.records]
        rate = sum(r["prediction"] for r in results) / len(results)
        ACQUISITION_RATE.set(rate)

        return {
            "total": len(results),
            "predicted_acquired": sum(r["prediction"] for r in results),
            "predicted_closed": len(results) - sum(r["prediction"] for r in results),
            "acquired_prediction_rate": round(rate, 4),
            "predictions": results
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4"
    )
