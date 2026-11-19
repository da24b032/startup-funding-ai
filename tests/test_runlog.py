"""Unit tests for the pipeline run history."""

from __future__ import annotations

import pytest

from src.tracking.run_log import logged_stage, read_runs


@pytest.fixture(autouse=True)
def _isolated_log(tmp_path, monkeypatch):
    monkeypatch.setenv("RUN_LOG_PATH", str(tmp_path / "runs.jsonl"))


def test_success_is_recorded_with_throughput():
    logged_stage("demo")(lambda: {"pipeline_throughput_rows_per_second": 10.0})()
    run = read_runs()[0]
    assert run["stage"] == "demo" and run["status"] == "success"
    assert run["throughput_rows_per_second"] == 10.0


def test_failure_is_recorded_and_reraised():
    def boom():
        raise RuntimeError("bad data")

    with pytest.raises(RuntimeError):
        logged_stage("demo")(boom)()
    run = read_runs()[0]
    assert run["status"] == "failed" and "bad data" in run["error"]


def test_newest_first_and_limit():
    for _ in range(3):
        logged_stage("demo")(lambda: {})()
    assert len(read_runs(2)) == 2
    assert read_runs() == sorted(read_runs(), key=lambda r: r["timestamp"], reverse=True)


def test_missing_file_returns_empty_list():
    assert read_runs() == []
