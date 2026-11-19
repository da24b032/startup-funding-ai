"""Data engineering pipeline (custom, stage-based).

Stages (each is timed so throughput can be reported and monitored):

    ingest -> validate -> clean -> label -> split -> write

Important design choices
------------------------
* The train/test split happens **before** any resampling or fitting, and no
  oversampling (SMOTE) is used, so test metrics are measured on real startups
  only. Class imbalance is handled with ``class_weight`` in the models.
* Cleaning only removes exact duplicate startups. Odd values (e.g. funding
  dated before founding) are kept and encoded as features, not discarded.
* Outputs are plain CSV so DVC can version them.

Run:  python -m src.data.prepare
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_params, resolve, setup_logging
from src.data.schema import (
    POSITIVE_LABEL,
    RAW_FEATURE_COLUMNS,
    RAW_STATUS_COLUMN,
    TARGET,
    validate_raw_dataframe,
)
from src.tracking.run_log import logged_stage

logger = setup_logging("data.prepare")


@contextmanager
def stage(name: str, timings: dict, rows: int):
    """Time a pipeline stage and record seconds and rows/second."""
    start = time.perf_counter()
    yield
    elapsed = max(time.perf_counter() - start, 1e-9)
    timings[name] = {
        "seconds": round(elapsed, 6),
        "rows_per_second": round(rows / elapsed, 1),
    }
    logger.info("stage=%s rows=%d seconds=%.4f", name, rows, elapsed)


def run_pipeline() -> dict:
    params = load_params()["data"]
    raw_path = resolve(params["raw_path"], "RAW_DATA_PATH")
    out_dir = resolve(params["processed_dir"], "PROCESSED_DIR")
    out_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, dict] = {}
    wall_start = time.perf_counter()

    # 1. ingest ---------------------------------------------------------------
    with stage("ingest", timings, rows=1):
        df = pd.read_csv(raw_path)
    rows_in = len(df)
    logger.info("Ingested %d rows x %d columns from %s", *df.shape, raw_path)

    # 2. validate -------------------------------------------------------------
    with stage("validate", timings, rows=rows_in):
        tolerated = validate_raw_dataframe(df)
    logger.info("Validation passed. Tolerated issues: %s", tolerated)

    # 3. clean ----------------------------------------------------------------
    with stage("clean", timings, rows=rows_in):
        df = df.drop_duplicates(subset=params["dedupe_key"], keep="first")
        df = df[RAW_FEATURE_COLUMNS + [RAW_STATUS_COLUMN]].reset_index(drop=True)
    rows_clean = len(df)

    # 4. label ----------------------------------------------------------------
    with stage("label", timings, rows=rows_clean):
        df[TARGET] = (df[RAW_STATUS_COLUMN] == POSITIVE_LABEL).astype(int)
        df = df.drop(columns=[RAW_STATUS_COLUMN])

    # 5. split (stratified, before any fitting) -------------------------------
    with stage("split", timings, rows=rows_clean):
        train_df, test_df = train_test_split(
            df,
            test_size=params["test_size"],
            random_state=params["random_state"],
            stratify=df[TARGET],
        )

    # 6. write ----------------------------------------------------------------
    with stage("write", timings, rows=rows_clean):
        train_df.to_csv(out_dir / "train.csv", index=False)
        test_df.to_csv(out_dir / "test.csv", index=False)

    wall = time.perf_counter() - wall_start
    report = {
        "rows_ingested": rows_in,
        "rows_after_cleaning": rows_clean,
        "duplicates_removed": rows_in - rows_clean,
        "tolerated_issues": tolerated,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_acquired_rate": round(float(train_df[TARGET].mean()), 4),
        "test_acquired_rate": round(float(test_df[TARGET].mean()), 4),
        "feature_columns": RAW_FEATURE_COLUMNS,
        "stages": timings,
        "pipeline_seconds": round(wall, 4),
        "pipeline_throughput_rows_per_second": round(rows_in / max(wall, 1e-9), 1),
    }
    report_path = out_dir / "data_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", report_path)
    return report


if __name__ == "__main__":
    summary = logged_stage("prepare")(run_pipeline)()
    print(json.dumps({k: v for k, v in summary.items() if k != "stages"}, indent=2))
