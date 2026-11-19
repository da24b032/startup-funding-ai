"""Data drift detection.

Live API requests are kept in a sliding window and compared with a reference
sample stored at training time (``models/drift_baseline.json``):

* numeric inputs  -> two-sample Kolmogorov-Smirnov test (drift if p < threshold)
* categorical     -> total variation distance between the live category
                     frequencies and the training frequencies
                     (drift if distance > threshold)

Drift is *input* drift: it says the incoming startups look different from the
training data, which is a signal to review or retrain, not proof that accuracy
dropped (labels are unknown at prediction time).
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger("drift")

NUMERIC_COLUMNS_WITH_OPTIONAL = ("age_first_milestone_year", "age_last_milestone_year")


class DriftMonitor:
    """Thread-safe sliding-window drift monitor."""

    def __init__(
        self,
        baseline: dict[str, Any],
        p_value_threshold: float = 0.05,
        min_samples: int = 30,
        window_size: int = 500,
        tvd_threshold: float = 0.30,
    ):
        self.baseline = baseline
        self.p_value_threshold = p_value_threshold
        self.min_samples = min_samples
        self.tvd_threshold = tvd_threshold
        self._window: deque[dict[str, Any]] = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._window)

    def add(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._window.append(record)

    def reset(self) -> None:
        with self._lock:
            self._window.clear()

    def check(self) -> dict[str, Any]:
        """Return per-feature drift results for the current window."""
        with self._lock:
            window = list(self._window)

        n = len(window)
        if n < self.min_samples:
            return {
                "window_size": n,
                "min_samples": self.min_samples,
                "ready": False,
                "drifted_features": [],
                "features": {},
                "message": f"Need at least {self.min_samples} predictions "
                f"(have {n}) before drift can be assessed.",
            }

        features: dict[str, dict[str, Any]] = {}

        for col, ref in self.baseline["numeric"].items():
            values = np.array(
                [r[col] for r in window if r.get(col) is not None], dtype=float
            )
            if len(values) < self.min_samples:
                continue
            statistic, p_value = stats.ks_2samp(np.array(ref["sample"]), values)
            features[col] = {
                "type": "numeric",
                "statistic": round(float(statistic), 4),
                "p_value": round(float(p_value), 4),
                "drifted": bool(p_value < self.p_value_threshold),
            }

        for col, ref_freq in self.baseline["categorical"].items():
            live = [str(r[col]).upper() if col == "state_code" else str(r[col]) for r in window if r.get(col)]
            if len(live) < self.min_samples:
                continue
            counts: dict[str, int] = {}
            for value in live:
                counts[value] = counts.get(value, 0) + 1
            live_freq = {k: v / len(live) for k, v in counts.items()}
            keys = set(ref_freq) | set(live_freq)
            tvd = 0.5 * sum(abs(ref_freq.get(k, 0.0) - live_freq.get(k, 0.0)) for k in keys)
            features[col] = {
                "type": "categorical",
                "statistic": round(float(tvd), 4),
                "p_value": None,
                "drifted": bool(tvd > self.tvd_threshold),
            }

        drifted = sorted(f for f, v in features.items() if v["drifted"])
        if drifted:
            logger.warning("Drift detected in %d feature(s): %s", len(drifted), drifted)
        return {
            "window_size": n,
            "min_samples": self.min_samples,
            "ready": True,
            "drifted_features": drifted,
            "features": features,
            "message": (
                f"Drift detected in {len(drifted)} feature(s)."
                if drifted
                else "No drift detected."
            ),
        }
