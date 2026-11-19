"""Unit tests for the sliding-window drift monitor."""

from __future__ import annotations

import numpy as np

from src.monitoring.drift import DriftMonitor


def _baseline() -> dict:
    rng = np.random.default_rng(0)
    return {
        "numeric": {"funding_rounds": {"sample": rng.normal(3, 1, 300).tolist()}},
        "binary": {},
        "categorical": {"state_code": {"CA": 0.6, "NY": 0.4}},
    }


def _record(rounds: float, state: str = "CA") -> dict:
    return {"funding_rounds": rounds, "state_code": state}


def test_not_ready_below_minimum_samples():
    monitor = DriftMonitor(_baseline(), min_samples=30)
    for _ in range(5):
        monitor.add(_record(3))
    result = monitor.check()
    assert result["ready"] is False and result["window_size"] == 5


def test_no_drift_on_same_distribution():
    rng = np.random.default_rng(1)
    monitor = DriftMonitor(_baseline(), min_samples=30)
    for value in rng.normal(3, 1, 200):
        monitor.add(_record(value, "CA" if rng.random() < 0.6 else "NY"))
    assert monitor.check()["drifted_features"] == []


def test_numeric_and_categorical_drift_detected():
    monitor = DriftMonitor(_baseline(), min_samples=30)
    for _ in range(100):
        monitor.add(_record(9.0, "TX"))
    result = monitor.check()
    assert "funding_rounds" in result["drifted_features"]
    assert "state_code" in result["drifted_features"]


def test_reset_clears_window():
    monitor = DriftMonitor(_baseline(), min_samples=30)
    monitor.add(_record(3))
    monitor.reset()
    assert len(monitor) == 0
