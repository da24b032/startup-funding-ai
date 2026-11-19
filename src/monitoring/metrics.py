"""Prometheus instrumentation for the inference service.

What is monitored (all scraped from ``GET /metrics`` every 15 s):

Service health
    startup_http_requests_total{method,endpoint,status}
    startup_http_request_duration_seconds{endpoint}
    startup_errors_total{endpoint,error_type}

Model behaviour
    startup_predictions_total{predicted_class}
    startup_prediction_latency_seconds            (per-row inference latency)
    startup_acquired_probability                  (distribution of scores)
    startup_prediction_batch_size
    startup_acquired_rate_window                  (share of recent predictions = Acquired)

Model quality (static, from the training run that produced the served model)
    startup_model_info{model_name,run_id,git_commit}
    startup_model_test_metric{metric}

Data drift
    startup_feature_drift_statistic{feature}
    startup_feature_drift_detected{feature}
    startup_drifted_features
    startup_drift_window_size

Data pipeline
    startup_pipeline_throughput_rows_per_second
    startup_pipeline_last_run_timestamp{stage,status}
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter(
    "startup_http_requests_total", "HTTP requests", ["method", "endpoint", "status"]
)
HTTP_LATENCY = Histogram(
    "startup_http_request_duration_seconds",
    "HTTP request duration",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5],
)
ERRORS = Counter("startup_errors_total", "Errors by endpoint and type", ["endpoint", "error_type"])

PREDICTIONS = Counter(
    "startup_predictions_total", "Predictions served", ["predicted_class"]
)
PREDICTION_LATENCY = Histogram(
    "startup_prediction_latency_seconds",
    "Per-row inference latency",
    buckets=[0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)
ACQUIRED_PROBABILITY = Histogram(
    "startup_acquired_probability",
    "Distribution of predicted acquisition probability",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
BATCH_SIZE = Histogram(
    "startup_prediction_batch_size",
    "Records per batch request",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500],
)
ACQUIRED_RATE = Gauge(
    "startup_acquired_rate_window", "Share of recent predictions classified Acquired"
)

MODEL_INFO = Gauge(
    "startup_model_info", "Served model metadata", ["model_name", "run_id", "git_commit"]
)
MODEL_TEST_METRIC = Gauge(
    "startup_model_test_metric", "Held-out test metric of the served model", ["metric"]
)

DRIFT_STATISTIC = Gauge(
    "startup_feature_drift_statistic", "Drift statistic per feature", ["feature"]
)
DRIFT_DETECTED = Gauge(
    "startup_feature_drift_detected", "1 if drift detected for feature", ["feature"]
)
DRIFTED_FEATURES = Gauge("startup_drifted_features", "Number of drifted features")
DRIFT_WINDOW = Gauge("startup_drift_window_size", "Predictions currently in the drift window")

PIPELINE_THROUGHPUT = Gauge(
    "startup_pipeline_throughput_rows_per_second", "Data pipeline throughput"
)
PIPELINE_LAST_RUN = Gauge(
    "startup_pipeline_last_run_timestamp",
    "Unix time of the latest pipeline run",
    ["stage", "status"],
)


def record_prediction(label: int, probability: float, latency_seconds: float) -> None:
    PREDICTIONS.labels(predicted_class="acquired" if label == 1 else "closed").inc()
    PREDICTION_LATENCY.observe(latency_seconds)
    ACQUIRED_PROBABILITY.observe(probability)


def set_model_info(info: dict[str, Any]) -> None:
    MODEL_INFO.labels(
        model_name=str(info.get("model_name")),
        run_id=str(info.get("mlflow_run_id"))[:12],
        git_commit=str(info.get("git_commit"))[:10],
    ).set(1)
    for metric, value in (info.get("test_metrics") or {}).items():
        MODEL_TEST_METRIC.labels(metric=metric).set(value)


def update_drift(result: dict[str, Any]) -> None:
    DRIFT_WINDOW.set(result["window_size"])
    if not result["ready"]:
        return
    for feature, values in result["features"].items():
        DRIFT_STATISTIC.labels(feature=feature).set(values["statistic"])
        DRIFT_DETECTED.labels(feature=feature).set(1 if values["drifted"] else 0)
    DRIFTED_FEATURES.set(len(result["drifted_features"]))


def update_pipeline(runs: list[dict[str, Any]]) -> None:
    """Expose the latest pipeline runs (newest first) as gauges."""
    seen: set[tuple[str, str]] = set()
    for run in runs:
        key = (run.get("stage", "?"), run.get("status", "?"))
        if key in seen:
            continue
        seen.add(key)
        try:
            ts = datetime.fromisoformat(run["timestamp"]).timestamp()
        except (KeyError, ValueError):
            continue
        PIPELINE_LAST_RUN.labels(stage=key[0], status=key[1]).set(ts)
        if key == ("prepare", "success") and "throughput_rows_per_second" in run:
            PIPELINE_THROUGHPUT.set(run["throughput_rows_per_second"])
