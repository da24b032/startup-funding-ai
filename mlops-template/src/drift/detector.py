"""
Data drift detector using the Kolmogorov-Smirnov (KS) test.

Compares the distribution of incoming inference data against the training
baseline. A KS p-value below the configured threshold (default 0.05) signals
drift and triggers an alert.

In production, plug in Evidently AI or WhyLogs for richer dashboards.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.features.engineering import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

BASELINE_PATH = "./data/drift_baseline.json"


def compute_baseline(X: pd.DataFrame, save_path: str = BASELINE_PATH) -> dict:
    """Compute and persist per-feature distribution statistics from training data."""
    baseline: dict[str, dict] = {}
    for col in X.columns:
        vals = X[col].dropna().tolist()
        baseline[col] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "values_sample": vals[:500],   # store a sample for KS test
        }
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(save_path).write_text(json.dumps(baseline))
    logger.info("Drift baseline saved to %s", save_path)
    return baseline


def load_baseline(path: str = BASELINE_PATH) -> dict | None:
    if not Path(path).exists():
        return None
    return json.loads(Path(path).read_text())


def detect_drift(
    X_new: pd.DataFrame,
    threshold: float = 0.05,
    baseline_path: str = BASELINE_PATH,
) -> dict[str, dict]:
    """
    Run KS test for each feature comparing new data against baseline.

    Returns a dict: {feature: {ks_stat, p_value, drifted}}.
    Logs a warning for any drifted feature.
    """
    baseline = load_baseline(baseline_path)
    if baseline is None:
        logger.warning("No drift baseline found at %s — skipping drift check.", baseline_path)
        return {}

    results: dict[str, dict] = {}
    drifted_features: list[str] = []

    for col in X_new.columns:
        if col not in baseline:
            continue
        ref_sample = np.array(baseline[col]["values_sample"])
        new_sample = X_new[col].dropna().to_numpy()

        if len(new_sample) < 10:
            continue

        ks_stat, p_value = stats.ks_2samp(ref_sample, new_sample)
        drifted = p_value < threshold

        results[col] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 4),
            "drifted": bool(drifted),
        }

        if drifted:
            drifted_features.append(col)
            logger.warning("DRIFT DETECTED — feature=%s  KS=%.4f  p=%.4f", col, ks_stat, p_value)

    if not drifted_features:
        logger.info("No drift detected across %d features.", len(results))
    else:
        logger.warning("Drift detected in %d features: %s", len(drifted_features), drifted_features)

    return results


class DriftDetector:
    """Stateful wrapper that accumulates incoming predictions for batch drift checks."""

    def __init__(self, threshold: float = 0.05, window_size: int = 500) -> None:
        self.threshold = threshold
        self.window_size = window_size
        self._buffer: list[dict] = []

    def add(self, features: dict) -> None:
        self._buffer.append(features)
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)

    def check(self) -> dict[str, dict]:
        if len(self._buffer) < 50:
            return {}
        X_new = pd.DataFrame(self._buffer)
        return detect_drift(X_new, threshold=self.threshold)

    def __len__(self) -> int:
        return len(self._buffer)
