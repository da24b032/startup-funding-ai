"""Prediction service (no web-framework dependency).

``Predictor`` owns the saved model pipeline and turns raw startup profiles into
predictions. The FastAPI layer in ``api.py`` is a thin wrapper around it, which
keeps all behaviour unit-testable without starting a server.

Explanations use *occlusion*: each input in turn is replaced by a typical
training value and the change in predicted probability is measured. It is a
simple, model-agnostic way to show "what pushed this prediction up or down"
without extra dependencies. It describes the model, it does not prove cause.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import load_params, resolve
from src.data.schema import (
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    OPTIONAL_COLUMNS,
    RAW_FEATURE_COLUMNS,
)
from src.serving.field_meta import FIELD_META

logger = logging.getLogger("predictor")

OUTCOME_LABELS = {1: "Acquired", 0: "Closed"}


def _to_python(value: Any) -> Any:
    """Convert numpy/pandas scalars (and NaN) to plain JSON-friendly values."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return value.item() if hasattr(value, "item") else value


class ModelNotAvailableError(RuntimeError):
    """The model artifacts are missing or unreadable."""


class InvalidInputError(ValueError):
    """The input record is missing fields or has unusable values."""


class Predictor:
    def __init__(
        self,
        model_path: Path,
        metadata_path: Path,
        baseline_path: Path,
        threshold: float = 0.5,
    ):
        try:
            self.model = joblib.load(model_path)
            self.metadata: dict[str, Any] = json.loads(Path(metadata_path).read_text())
            self.baseline: dict[str, Any] = json.loads(Path(baseline_path).read_text())
        except FileNotFoundError as exc:
            raise ModelNotAvailableError(
                f"Model artifact not found: {exc.filename}. Run the training pipeline first "
                f"(`dvc repro` or `python -m src.training.train`)."
            ) from exc
        except Exception as exc:  # corrupt pickle / json
            raise ModelNotAvailableError(f"Could not load model artifacts: {exc}") from exc

        self.threshold = float(threshold)
        self._typical = self._typical_values()
        self._lock = threading.Lock()
        logger.info(
            "Predictor ready | model=%s | run=%s | threshold=%.2f",
            self.metadata.get("model_name"),
            self.metadata.get("mlflow_model_run_id"),
            self.threshold,
        )

    # --------------------------------------------------------- construction
    @classmethod
    def from_settings(cls) -> Predictor:
        cfg = load_params()["serving"]
        return cls(
            model_path=resolve(cfg["model_path"], "MODEL_PATH"),
            metadata_path=resolve(cfg["metadata_path"], "MODEL_METADATA_PATH"),
            baseline_path=resolve(cfg["baseline_path"], "DRIFT_BASELINE_PATH"),
            threshold=cfg["positive_threshold"],
        )

    def _typical_values(self) -> dict[str, Any]:
        """A 'typical' startup used as the reference for explanations."""
        typical: dict[str, Any] = {}
        for col in NUMERIC_COLUMNS:
            typical[col] = float(np.median(self.baseline["numeric"][col]["sample"]))
        for col in BINARY_COLUMNS:
            typical[col] = 1 if self.baseline["binary"][col] >= 0.5 else 0
        for col in CATEGORICAL_COLUMNS:
            freq = self.baseline["categorical"][col]
            typical[col] = max(freq, key=freq.get)
        return typical

    # ------------------------------------------------------------- helpers
    @property
    def categories(self) -> dict[str, list[str]]:
        """Category values seen in training, most common first."""
        return {
            col: sorted(freq, key=freq.get, reverse=True)
            for col, freq in self.baseline["categorical"].items()
        }

    def _to_frame(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        rows = []
        for i, record in enumerate(records):
            missing = [
                c
                for c in RAW_FEATURE_COLUMNS
                if c not in record and c not in OPTIONAL_COLUMNS
            ]
            if missing:
                raise InvalidInputError(f"Record {i}: missing fields {missing}")
            row = {c: record.get(c) for c in RAW_FEATURE_COLUMNS}
            row["state_code"] = str(row["state_code"]).upper()
            rows.append(row)
        frame = pd.DataFrame(rows, columns=RAW_FEATURE_COLUMNS)
        for col in NUMERIC_COLUMNS + BINARY_COLUMNS:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        required_numeric = [c for c in NUMERIC_COLUMNS + BINARY_COLUMNS if c not in OPTIONAL_COLUMNS]
        bad = frame[required_numeric].isna().any(axis=1)
        if bad.any():
            raise InvalidInputError(
                f"Record(s) {frame.index[bad].tolist()} contain non-numeric or empty required values"
            )
        return frame

    # ---------------------------------------------------------- prediction
    def predict(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Predict for one or more raw startup profiles."""
        frame = self._to_frame(records)
        start = time.perf_counter()
        try:
            prob_acquired = self.model.predict_proba(frame)[:, 1]
        except Exception as exc:  # model/pipeline failure
            logger.exception("Model inference failed")
            raise ModelNotAvailableError(f"Model inference failed: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000 / max(len(frame), 1)

        results = []
        for p in prob_acquired:
            label = int(p >= self.threshold)
            results.append(
                {
                    "prediction": label,
                    "outcome": OUTCOME_LABELS[label],
                    "acquired_probability": round(float(p), 4),
                    "closed_probability": round(float(1 - p), 4),
                    "threshold": self.threshold,
                    "model_name": self.metadata.get("model_name", "unknown"),
                    "model_run_id": self.metadata.get("mlflow_model_run_id", "unknown"),
                    "latency_ms": round(latency_ms, 3),
                }
            )
        return results

    # --------------------------------------------------------- explanation
    def explain(self, record: dict[str, Any], top_k: int = 6) -> dict[str, Any]:
        """Which inputs pushed this prediction up or down vs. a typical startup."""
        frame = self._to_frame([record])
        variants = [frame.iloc[0].to_dict()]
        for col in RAW_FEATURE_COLUMNS:
            variant = frame.iloc[0].to_dict()
            variant[col] = self._typical[col]
            variants.append(variant)
        batch = pd.DataFrame(variants, columns=RAW_FEATURE_COLUMNS)
        probs = self.model.predict_proba(batch)[:, 1]
        actual, replaced = float(probs[0]), probs[1:]

        impacts = []
        for col, p_replaced in zip(RAW_FEATURE_COLUMNS, replaced, strict=False):
            delta = actual - float(p_replaced)
            impacts.append(
                {
                    "field": col,
                    "label": FIELD_META[col]["label"],
                    "value": _to_python(frame.iloc[0][col]),
                    "impact": round(delta, 4),
                    "direction": "raises" if delta > 0 else "lowers" if delta < 0 else "neutral",
                }
            )
        impacts.sort(key=lambda d: abs(d["impact"]), reverse=True)
        return {
            "acquired_probability": round(actual, 4),
            "method": "occlusion vs typical training startup",
            "top_factors": [i for i in impacts[:top_k] if i["direction"] != "neutral"],
        }

    # ------------------------------------------------------------ metadata
    def info(self) -> dict[str, Any]:
        return {
            "model_name": self.metadata.get("model_name"),
            "trained_at": self.metadata.get("trained_at"),
            "mlflow_run_id": self.metadata.get("mlflow_model_run_id"),
            "git_commit": self.metadata.get("git_commit"),
            "train_data_sha256": self.metadata.get("train_data_sha256"),
            "tracking_backend": self.metadata.get("tracking_backend"),
            "n_train": self.metadata.get("n_train"),
            "n_test": self.metadata.get("n_test"),
            "test_metrics": self.metadata.get("test_metrics"),
            "cv_roc_auc_mean": self.metadata.get("cv_roc_auc_mean"),
            "threshold": self.threshold,
        }
