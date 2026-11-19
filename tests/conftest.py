"""Shared fixtures.

Tests run against the real model artifacts produced by the pipeline
(`dvc repro` or `python -m src.training.train`). If they are missing the
model-dependent tests are skipped with a clear reason instead of failing
confusingly.
"""

from __future__ import annotations

import pytest

from src.config import resolve

SAMPLE_PROFILE = {
    "state_code": "CA",
    "category_code": "software",
    "age_first_funding_year": 1.5,
    "age_last_funding_year": 3.5,
    "funding_rounds": 3,
    "funding_total_usd": 12000000,
    "avg_participants": 3.0,
    "has_VC": 1,
    "has_angel": 0,
    "has_roundA": 1,
    "has_roundB": 1,
    "has_roundC": 0,
    "has_roundD": 0,
    "is_top500": 1,
    "relationships": 8,
    "milestones": 2,
    "age_first_milestone_year": 2.0,
    "age_last_milestone_year": 4.0,
}


@pytest.fixture()
def profile() -> dict:
    """A valid startup profile (fresh copy per test)."""
    return dict(SAMPLE_PROFILE)


@pytest.fixture(scope="session")
def model_ready() -> bool:
    return resolve("models/model.joblib", "MODEL_PATH").exists()


@pytest.fixture()
def require_model(model_ready):
    if not model_ready:
        pytest.skip("models/model.joblib missing - run `python -m src.training.train` first")
