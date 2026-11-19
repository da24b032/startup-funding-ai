"""REST API for startup outcome prediction.

This is the ONLY interface the web frontend uses. The frontend and this
backend are independent programs connected solely by HTTP (the frontend reads
the backend URL from configuration), so either can be replaced or scaled
without touching the other.

Endpoints
---------
GET  /health              liveness + model availability
GET  /model/info          which model is served, how it was trained, test metrics
GET  /schema              input field definitions (labels, ranges, dropdown options)
POST /predict             one startup -> prediction
POST /predict/explain     one startup -> prediction + top factors
POST /predict/batch       1..500 startups -> predictions + summary
GET  /drift               current data-drift assessment
POST /drift/reset         clear the drift window
GET  /pipeline/summary    data report, model comparison, DVC stages, latest runs
GET  /pipeline/runs       pipeline run history (successes and failures)
GET  /metrics             Prometheus metrics

Run:  uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.config import PROJECT_ROOT, load_params, resolve, setup_logging
from src.data.schema import VALUE_RANGES
from src.monitoring import metrics
from src.monitoring.drift import DriftMonitor
from src.serving.field_meta import BINARY_FIELDS, FIELD_META
from src.serving.predictor import (
    InvalidInputError,
    ModelNotAvailableError,
    Predictor,
)
from src.serving.schemas import (
    BatchRequest,
    BatchResponse,
    ExplainedPrediction,
    PredictionResponse,
    StartupProfile,
)
from src.tracking.run_log import read_runs

logger = setup_logging("api")
START_TIME = time.time()


# --------------------------------------------------------------------- state
class AppState:
    """Objects shared by all requests (created once at startup)."""

    predictor: Predictor | None = None
    drift: DriftMonitor | None = None
    load_error: str | None = None
    recent_labels: deque[int] = deque(maxlen=200)
    predictions_since_drift_check: int = 0
    lock = threading.Lock()


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once. A missing model must not crash the process:
    /health reports 'degraded' and prediction endpoints return 503."""
    monitoring_cfg = load_params()["monitoring"]
    try:
        state.predictor = Predictor.from_settings()
        state.drift = DriftMonitor(
            state.predictor.baseline,
            p_value_threshold=monitoring_cfg["drift_p_value_threshold"],
            min_samples=monitoring_cfg["drift_min_samples"],
            window_size=monitoring_cfg["drift_window_size"],
            tvd_threshold=monitoring_cfg["categorical_tvd_threshold"],
        )
        metrics.set_model_info({**state.predictor.metadata, "mlflow_run_id": state.predictor.metadata.get("mlflow_model_run_id")})
        metrics.update_pipeline(read_runs(50))
    except ModelNotAvailableError as exc:
        state.load_error = str(exc)
        logger.error("Model could not be loaded: %s", exc)
    yield


app = FastAPI(
    title="Startup Outcome Prediction API",
    description="Predicts whether a startup will be acquired or closed, "
    "with monitoring and pipeline-status endpoints.",
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------- middleware
@app.middleware("http")
async def observe_requests(request: Request, call_next):
    """Request id, structured access log and HTTP metrics for every call."""
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    request.state.request_id = request_id
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        route = request.scope.get("route")
        endpoint = getattr(route, "path", "unmatched")
        duration = time.perf_counter() - start
        metrics.HTTP_REQUESTS.labels(request.method, endpoint, str(status)).inc()
        metrics.HTTP_LATENCY.labels(endpoint).observe(duration)
        logger.info(
            "request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
            request_id,
            request.method,
            request.url.path,
            status,
            duration * 1000,
        )


# ------------------------------------------------------------ error handling
def _error(request: Request, status_code: int, error: str, detail: Any = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": error,
            "detail": detail,
            "request_id": getattr(request.state, "request_id", None),
        },
    )


def _endpoint(request: Request) -> str:
    return getattr(request.scope.get("route"), "path", request.url.path)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    metrics.ERRORS.labels(_endpoint(request), "validation").inc()
    problems = [
        {"field": ".".join(str(p) for p in err["loc"] if p != "body"), "problem": err["msg"]}
        for err in exc.errors()
    ]
    return _error(request, 422, "Invalid input", problems)


@app.exception_handler(InvalidInputError)
async def invalid_input_handler(request: Request, exc: InvalidInputError):
    metrics.ERRORS.labels(_endpoint(request), "invalid_input").inc()
    return _error(request, 422, "Invalid input", str(exc))


@app.exception_handler(ModelNotAvailableError)
async def model_unavailable_handler(request: Request, exc: ModelNotAvailableError):
    metrics.ERRORS.labels(_endpoint(request), "model_unavailable").inc()
    return _error(request, 503, "Model unavailable", str(exc))


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    metrics.ERRORS.labels(_endpoint(request), "unhandled").inc()
    logger.exception("Unhandled error | request_id=%s", getattr(request.state, "request_id", None))
    return _error(request, 500, "Internal server error", "See server logs with the request id.")


def _require_predictor() -> Predictor:
    if state.predictor is None:
        raise ModelNotAvailableError(state.load_error or "Model not loaded")
    return state.predictor


# ---------------------------------------------------------------- monitoring
def _after_predictions(records: list[dict], results: list[dict]) -> None:
    """Update metrics, rolling rate and drift window after serving predictions."""
    check_every = load_params()["monitoring"]["check_every_n_predictions"]
    for record, result in zip(records, results, strict=False):
        metrics.record_prediction(
            result["prediction"], result["acquired_probability"], result["latency_ms"] / 1000
        )
        if state.drift is not None:
            state.drift.add(record)
    with state.lock:
        state.recent_labels.extend(r["prediction"] for r in results)
        metrics.ACQUIRED_RATE.set(sum(state.recent_labels) / len(state.recent_labels))
        state.predictions_since_drift_check += len(results)
        due = state.predictions_since_drift_check >= check_every
        if due:
            state.predictions_since_drift_check = 0
    if due and state.drift is not None:
        metrics.update_drift(state.drift.check())


# ----------------------------------------------------------------- endpoints
@app.get("/health", tags=["service"])
def health():
    """Liveness probe. ``degraded`` means the process is up but has no model."""
    ok = state.predictor is not None
    body = {
        "status": "ok" if ok else "degraded",
        "model_loaded": ok,
        "uptime_seconds": round(time.time() - START_TIME, 1),
    }
    if not ok:
        body["detail"] = state.load_error
    return JSONResponse(status_code=200 if ok else 503, content=body)


@app.get("/model/info", tags=["service"])
def model_info():
    return _require_predictor().info()


@app.get("/schema", tags=["service"])
def input_schema():
    """Field definitions so a client can build its form without hard-coding them."""
    predictor = _require_predictor()
    categories = predictor.categories
    fields = []
    for name, meta in FIELD_META.items():
        spec: dict[str, Any] = {"name": name, **meta}
        if name in BINARY_FIELDS:
            spec.update(type="flag", options=[0, 1])
        elif name in categories:
            spec.update(type="category", options=categories[name])
        else:
            low, high = VALUE_RANGES[name]
            is_int = name in {"relationships", "milestones", "funding_rounds"}
            spec.update(type="integer" if is_int else "number", min=low, max=high)
        spec["required"] = name not in {"age_first_milestone_year", "age_last_milestone_year"}
        fields.append(spec)
    return {"fields": fields}


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(profile: StartupProfile):
    predictor = _require_predictor()
    record = profile.model_dump()
    results = predictor.predict([record])
    _after_predictions([record], results)
    return results[0]


@app.post("/predict/explain", response_model=ExplainedPrediction, tags=["prediction"])
def predict_explain(profile: StartupProfile):
    predictor = _require_predictor()
    record = profile.model_dump()
    result = predictor.predict([record])[0]
    _after_predictions([record], [result])
    explanation = predictor.explain(record)
    return {
        **result,
        "top_factors": explanation["top_factors"],
        "explanation_method": explanation["method"],
    }


@app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
def predict_batch(request: BatchRequest):
    predictor = _require_predictor()
    max_size = load_params()["serving"]["max_batch_size"]
    if len(request.records) > max_size:
        raise InvalidInputError(f"Batch too large (max {max_size}).")
    records = [r.model_dump() for r in request.records]
    results = predictor.predict(records)
    metrics.BATCH_SIZE.observe(len(records))
    _after_predictions(records, results)
    acquired = sum(r["prediction"] for r in results)
    return {
        "total": len(results),
        "predicted_acquired": acquired,
        "predicted_closed": len(results) - acquired,
        "acquired_rate": round(acquired / len(results), 4),
        "predictions": results,
    }


@app.get("/drift", tags=["monitoring"])
def drift():
    _require_predictor()
    result = state.drift.check()
    metrics.update_drift(result)
    return result


@app.post("/drift/reset", tags=["monitoring"])
def drift_reset():
    _require_predictor()
    state.drift.reset()
    return {"status": "reset"}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@app.get("/pipeline/summary", tags=["pipeline"])
def pipeline_summary():
    """Everything the pipeline console needs in one call."""
    data_dir = resolve(load_params()["data"]["processed_dir"], "PROCESSED_DIR")
    reports_dir = resolve("reports", "REPORTS_DIR")

    comparison_path = reports_dir / "model_comparison.csv"
    comparison = (
        pd.read_csv(comparison_path).to_dict("records") if comparison_path.exists() else []
    )

    stages = []
    dvc_file = PROJECT_ROOT / "dvc.yaml"
    if dvc_file.exists():
        for name, spec in (yaml.safe_load(dvc_file.read_text()) or {}).get("stages", {}).items():
            outs = [o if isinstance(o, str) else next(iter(o)) for o in spec.get("outs", [])]
            deps = [d if isinstance(d, str) else next(iter(d)) for d in spec.get("deps", [])]
            stages.append({"name": name, "cmd": spec.get("cmd"), "deps": deps, "outs": outs})

    runs = read_runs(50)
    metrics.update_pipeline(runs)
    return {
        "data_report": _read_json(data_dir / "data_report.json"),
        "training_metrics": _read_json(reports_dir / "metrics.json"),
        "model_comparison": comparison,
        "dvc_stages": stages,
        "feature_importance": _read_json(reports_dir / "feature_importance.json"),
        "recent_runs": runs[:10],
        "run_counts": {
            "success": sum(1 for r in runs if r.get("status") == "success"),
            "failed": sum(1 for r in runs if r.get("status") == "failed"),
        },
    }


@app.get("/pipeline/runs", tags=["pipeline"])
def pipeline_runs(limit: int = Query(50, ge=1, le=500)):
    return {"runs": read_runs(limit)}


@app.get("/metrics", tags=["monitoring"])
def prometheus_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
