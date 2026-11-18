"""Tests for drift detection."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

from src.drift.detector import compute_baseline, detect_drift, DriftDetector


@pytest.fixture
def baseline_file(feature_matrix, tmp_path):
    X, _ = feature_matrix
    path = str(tmp_path / "baseline.json")
    compute_baseline(X, save_path=path)
    return path


def test_no_drift_on_same_distribution(feature_matrix, baseline_file):
    X, _ = feature_matrix
    results = detect_drift(X, threshold=0.05, baseline_path=baseline_file)
    drifted = [f for f, v in results.items() if v["drifted"]]
    # Same distribution — expect few or no drifted features
    assert len(drifted) < len(results) * 0.5


def test_drift_detected_on_shifted_data(feature_matrix, baseline_file):
    X, _ = feature_matrix
    X_shifted = X.copy()
    X_shifted["monthly_charges"] = X_shifted["monthly_charges"] * 5  # massive shift
    results = detect_drift(X_shifted, threshold=0.05, baseline_path=baseline_file)
    assert results.get("monthly_charges", {}).get("drifted") is True


def test_drift_detector_accumulates(feature_matrix):
    X, _ = feature_matrix
    detector = DriftDetector(window_size=100)
    for _, row in X.head(30).iterrows():
        detector.add(row.to_dict())
    assert len(detector) == 30


def test_drift_detector_check_insufficient_data():
    detector = DriftDetector()
    # Buffer < 50 → should return empty results
    for i in range(10):
        detector.add({"monthly_charges": float(i)})
    assert detector.check() == {}
