"""Pipeline run history.

Every pipeline stage (``prepare``, ``train``, ...) is wrapped with
:func:`logged_stage`. Each execution appends ONE JSON line to
``reports/pipeline_runs.jsonl`` recording the stage, status, duration and, on
failure, the error. The API exposes this history (``/pipeline/runs``) and the
frontend's Pipeline Console renders it, so failures are visible without
reading server logs.

The file is append-only JSON Lines, which keeps writes atomic enough for a
single-writer pipeline and trivially parseable.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any

from src.config import resolve

logger = logging.getLogger("tracking.run_log")
_LOCK = threading.Lock()


def run_log_path() -> Path:
    """Location of the run-history file (overridable with RUN_LOG_PATH)."""
    return resolve("reports/pipeline_runs.jsonl", "RUN_LOG_PATH")


def append_run(record: dict[str, Any]) -> None:
    """Append one run record. Never raises: logging must not break a pipeline."""
    try:
        path = run_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK, open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    except OSError:  # pragma: no cover - disk problems only
        logger.exception("Could not write pipeline run record")


def read_runs(limit: int = 50) -> list[dict[str, Any]]:
    """Return up to ``limit`` run records, newest first. Missing file -> []."""
    path = run_log_path()
    if not path.exists():
        return []
    runs: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping corrupt run-log line")
    runs.reverse()
    return runs[:limit]


def logged_stage(stage: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: record success/failure and duration of a pipeline stage.

    If the wrapped function returns a dict containing
    ``pipeline_throughput_rows_per_second`` it is stored as
    ``throughput_rows_per_second`` so throughput can be charted over time.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            record: dict[str, Any] = {
                "stage": stage,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            try:
                result = func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - record then re-raise
                record.update(
                    status="failed",
                    duration_seconds=round(time.perf_counter() - start, 3),
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(limit=4),
                )
                append_run(record)
                raise
            record.update(
                status="success", duration_seconds=round(time.perf_counter() - start, 3)
            )
            if isinstance(result, dict):
                if "pipeline_throughput_rows_per_second" in result:
                    record["throughput_rows_per_second"] = result[
                        "pipeline_throughput_rows_per_second"
                    ]
                if "rows_after_cleaning" in result:
                    record["rows"] = result["rows_after_cleaning"]
                if "selected" in result:
                    record["selected_model"] = result["selected"]
            append_run(record)
            return result

        return wrapper

    return decorator
