"""Column definitions and raw-data validation.

The lists below are the single contract between the data pipeline, the
training code and the API. Anything not listed here (names, ids, coordinates,
zip codes, ``closed_at``) is deliberately dropped: identifiers add no
generalisable signal and ``closed_at`` would leak the target.
"""

from __future__ import annotations

import pandas as pd

TARGET = "target"  # 1 = acquired, 0 = closed
RAW_STATUS_COLUMN = "status"
POSITIVE_LABEL = "acquired"

NUMERIC_COLUMNS = [
    "age_first_funding_year",
    "age_last_funding_year",
    "age_first_milestone_year",
    "age_last_milestone_year",
    "relationships",
    "funding_rounds",
    "funding_total_usd",
    "milestones",
    "avg_participants",
]

BINARY_COLUMNS = [
    "has_VC",
    "has_angel",
    "has_roundA",
    "has_roundB",
    "has_roundC",
    "has_roundD",
    "is_top500",
]

CATEGORICAL_COLUMNS = ["state_code", "category_code"]

# Columns a client must send to the API / the model is trained on.
RAW_FEATURE_COLUMNS = NUMERIC_COLUMNS + BINARY_COLUMNS + CATEGORICAL_COLUMNS

# Milestone ages are legitimately missing when a startup has no milestones.
OPTIONAL_COLUMNS = ["age_first_milestone_year", "age_last_milestone_year"]

REQUIRED_RAW_COLUMNS = RAW_FEATURE_COLUMNS + [RAW_STATUS_COLUMN, "object_id"]

# Plausible ranges used for validation (values outside are rejected by the API
# and reported, not silently fixed, by the data pipeline).
VALUE_RANGES = {
    "age_first_funding_year": (-15.0, 40.0),
    "age_last_funding_year": (-15.0, 40.0),
    "age_first_milestone_year": (-20.0, 40.0),
    "age_last_milestone_year": (-20.0, 40.0),
    "relationships": (0, 500),
    "funding_rounds": (0, 30),
    "funding_total_usd": (0, 1e11),
    "milestones": (0, 50),
    "avg_participants": (0, 50),
}

TOP_STATES = ("CA", "NY", "MA")


class DataValidationError(ValueError):
    """Raised when the raw dataset does not match the expected schema."""


def validate_raw_dataframe(df: pd.DataFrame) -> dict[str, int]:
    """Check required columns, label values and numeric ranges.

    Returns a small report of issues that were found but are *tolerated*
    (they are handled explicitly in the cleaning stage). Structural problems
    raise :class:`DataValidationError`.
    """
    missing_cols = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")

    statuses = set(df[RAW_STATUS_COLUMN].dropna().unique())
    if not statuses <= {"acquired", "closed"}:
        raise DataValidationError(f"Unexpected status values: {sorted(statuses)}")
    if df[RAW_STATUS_COLUMN].isna().any():
        raise DataValidationError("Rows with missing status found")

    out_of_range = 0
    for col, (low, high) in VALUE_RANGES.items():
        values = df[col].dropna()
        out_of_range += int(((values < low) | (values > high)).sum())
    if out_of_range:
        raise DataValidationError(f"{out_of_range} values outside plausible ranges")

    return {
        "negative_funding_ages": int((df["age_first_funding_year"] < 0).sum()),
        "missing_milestone_age": int(df["age_first_milestone_year"].isna().sum()),
        "duplicate_ids": int(df["object_id"].duplicated().sum()),
    }
