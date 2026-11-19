"""Training-artifact checks (run after the pipeline has produced a model)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.config import resolve
from src.data.schema import RAW_FEATURE_COLUMNS


def test_model_metadata_is_traceable(require_model):
    meta = json.loads(resolve("models/model_metadata.json", "MODEL_METADATA_PATH").read_text())
    for key in ("mlflow_model_run_id", "git_commit", "train_data_sha256", "test_metrics"):
        assert meta.get(key), f"metadata missing {key}"


def test_selected_model_beats_chance(require_model):
    metrics = json.loads((resolve("reports", "REPORTS_DIR") / "metrics.json").read_text())
    assert metrics["test_roc_auc"] > 0.7


def test_saved_pipeline_accepts_raw_rows(require_model):
    import joblib

    model = joblib.load(resolve("models/model.joblib", "MODEL_PATH"))
    test_csv = resolve("data/processed/test.csv", "PROCESSED_DIR")
    if not test_csv.exists():
        pytest.skip("processed test set missing")
    frame = pd.read_csv(test_csv)[RAW_FEATURE_COLUMNS].head(5)
    probs = model.predict_proba(frame)[:, 1]
    assert ((probs >= 0) & (probs <= 1)).all()


def test_drift_baseline_covers_all_inputs(require_model):
    baseline = json.loads(resolve("models/drift_baseline.json", "DRIFT_BASELINE_PATH").read_text())
    assert set(baseline) == {"numeric", "binary", "categorical"}
    assert len(baseline["numeric"]) == 9 and len(baseline["binary"]) == 7
