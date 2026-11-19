"""Unit tests for schema validation and the data-preparation helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.schema import (
    REQUIRED_RAW_COLUMNS,
    DataValidationError,
    validate_raw_dataframe,
)


def _raw_row(**overrides) -> dict:
    row = {c: 0 for c in REQUIRED_RAW_COLUMNS}
    row.update(
        state_code="CA", category_code="web", status="acquired", object_id="c:1",
        age_first_milestone_year=1.0, age_last_milestone_year=2.0,
    )
    row.update(overrides)
    return row


def test_valid_frame_returns_tolerated_issue_report():
    df = pd.DataFrame([_raw_row(), _raw_row(object_id="c:2", age_first_funding_year=-1.0)])
    report = validate_raw_dataframe(df)
    assert report["negative_funding_ages"] == 1
    assert report["duplicate_ids"] == 0


def test_missing_column_is_rejected():
    df = pd.DataFrame([_raw_row()]).drop(columns=["funding_rounds"])
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_raw_dataframe(df)


def test_unknown_status_is_rejected():
    df = pd.DataFrame([_raw_row(status="ipo")])
    with pytest.raises(DataValidationError, match="Unexpected status"):
        validate_raw_dataframe(df)


def test_out_of_range_value_is_rejected():
    df = pd.DataFrame([_raw_row(funding_rounds=999)])
    with pytest.raises(DataValidationError, match="outside plausible ranges"):
        validate_raw_dataframe(df)


def test_pipeline_splits_without_leakage(tmp_path, monkeypatch):
    """End-to-end data stage on the real CSV: disjoint stratified split, no status column."""
    from src.config import resolve
    from src.data import prepare

    raw = resolve("data/raw/startup_data.csv", "RAW_DATA_PATH")
    if not raw.exists():
        pytest.skip("raw dataset not present")
    monkeypatch.setenv("PROCESSED_DIR", str(tmp_path))
    monkeypatch.setenv("RUN_LOG_PATH", str(tmp_path / "runs.jsonl"))
    report = prepare.run_pipeline()

    train = pd.read_csv(tmp_path / "train.csv")
    test = pd.read_csv(tmp_path / "test.csv")
    assert len(train) + len(test) == report["rows_after_cleaning"]
    assert "status" not in train.columns and "target" in train.columns
    assert abs(train["target"].mean() - test["target"].mean()) < 0.02  # stratified
    assert report["pipeline_throughput_rows_per_second"] > 0
