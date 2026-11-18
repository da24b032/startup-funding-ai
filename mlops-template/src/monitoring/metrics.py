"""
Prometheus metrics for the inference server.
Exposes a /metrics endpoint that Prometheus scrapes every 15 seconds.
Import and call these functions from the FastAPI app.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Summary

# Prediction counters
PREDICTIONS_TOTAL = Counter(
    "churn_predictions_total",
    "Total number of churn predictions served",
    ["model_version", "predicted_class"],
)

# Latency histogram (buckets in seconds)
PREDICTION_LATENCY = Histogram(
    "churn_prediction_latency_seconds",
    "Inference latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# Churn rate gauge — rolling average over last N predictions
CHURN_RATE_GAUGE = Gauge(
    "churn_rate_rolling",
    "Rolling churn rate (predicted positives / total)",
)

# Batch size histogram
BATCH_SIZE = Histogram(
    "prediction_batch_size",
    "Number of customers per batch prediction request",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
)

# Model info
MODEL_INFO = Gauge(
    "model_info",
    "Model metadata",
    ["model_version", "model_type"],
)

# Feature drift score per feature
FEATURE_DRIFT_SCORE = Gauge(
    "feature_drift_score",
    "KS-test drift score for each input feature",
    ["feature_name"],
)


def record_prediction(predicted_class: int, model_version: str, latency_seconds: float) -> None:
    """Record a single prediction event."""
    PREDICTIONS_TOTAL.labels(
        model_version=model_version,
        predicted_class=str(predicted_class),
    ).inc()
    PREDICTION_LATENCY.observe(latency_seconds)


def update_churn_rate(churn_rate: float) -> None:
    CHURN_RATE_GAUGE.set(churn_rate)


def register_model(model_version: str, model_type: str) -> None:
    MODEL_INFO.labels(model_version=model_version, model_type=model_type).set(1)
