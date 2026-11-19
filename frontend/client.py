"""HTTP client for the prediction backend.

The frontend talks to the backend ONLY through this module, over REST. The
backend address comes from the ``BACKEND_URL`` environment variable, so the UI
has no import-time or code-level dependency on the model code. Every call
returns plain Python data or raises :class:`BackendError` with a message that
is safe to show to a non-technical user.
"""

from __future__ import annotations

import os
from typing import Any

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
TIMEOUT_SECONDS = float(os.getenv("BACKEND_TIMEOUT", "30"))


class BackendError(RuntimeError):
    """The backend could not be reached or rejected the request."""


def _request(method: str, path: str, **kwargs: Any) -> Any:
    url = f"{BACKEND_URL}{path}"
    try:
        response = requests.request(method, url, timeout=TIMEOUT_SECONDS, **kwargs)
    except requests.ConnectionError as exc:
        raise BackendError(
            "The prediction service is not reachable right now. "
            "Please check that it is running and try again."
        ) from exc
    except requests.Timeout as exc:
        raise BackendError("The prediction service took too long to answer.") from exc

    if response.status_code == 422:
        detail = response.json().get("detail")
        raise BackendError(_format_validation(detail))
    if response.status_code == 503:
        raise BackendError("The model is not loaded yet. Train the model first, then retry.")
    if not response.ok:
        raise BackendError(f"Unexpected error from the service (HTTP {response.status_code}).")
    return response.json()


def _format_validation(detail: Any) -> str:
    """Turn API validation details into one readable sentence."""
    if isinstance(detail, list):
        parts = [f"{d.get('field') or 'input'}: {d.get('problem')}" for d in detail]
        return "Please fix the following: " + "; ".join(parts)
    return f"Please check your input: {detail}"


def health() -> dict[str, Any]:
    return _request("GET", "/health")


def get_schema() -> list[dict[str, Any]]:
    return _request("GET", "/schema")["fields"]


def model_info() -> dict[str, Any]:
    return _request("GET", "/model/info")


def predict(profile: dict[str, Any], explain: bool = True) -> dict[str, Any]:
    return _request("POST", "/predict/explain" if explain else "/predict", json=profile)


def predict_batch(records: list[dict[str, Any]]) -> dict[str, Any]:
    return _request("POST", "/predict/batch", json={"records": records})


def drift() -> dict[str, Any]:
    return _request("GET", "/drift")


def reset_drift() -> dict[str, Any]:
    return _request("POST", "/drift/reset")


def pipeline_summary() -> dict[str, Any]:
    return _request("GET", "/pipeline/summary")


def pipeline_runs(limit: int = 100) -> list[dict[str, Any]]:
    return _request("GET", "/pipeline/runs", params={"limit": limit})["runs"]
