"""
Model loader — loads the trained sklearn Pipeline from disk.
Falls back to a dummy model if no trained model exists (for CI/tests).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)

_model = None
_model_meta: dict = {}


class DummyModel:
    """Placeholder that returns random predictions — used when no trained model exists."""

    def predict(self, X) -> np.ndarray:
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X) -> np.ndarray:
        return np.column_stack([np.ones(len(X)) * 0.8, np.ones(len(X)) * 0.2])


def load_model():
    """Load the trained model from disk; return a DummyModel if not found."""
    global _model, _model_meta

    if _model is not None:
        return _model

    model_path = Path(settings.model_dir) / f"{settings.model_name}.joblib"
    meta_path = Path(settings.model_dir) / f"{settings.model_name}_meta.json"

    if model_path.exists():
        _model = joblib.load(model_path)
        if meta_path.exists():
            _model_meta = json.loads(meta_path.read_text())
        logger.info("Model loaded from %s | AUC=%.4f", model_path, _model_meta.get("roc_auc", 0))
    else:
        logger.warning("No trained model at %s — using DummyModel.", model_path)
        _model = DummyModel()
        _model_meta = {"model_type": "dummy", "mlflow_run_id": "N/A"}

    return _model


def get_model_meta() -> dict:
    load_model()
    return _model_meta


def get_model_version() -> str:
    return _model_meta.get("mlflow_run_id", "unknown")[:8]
