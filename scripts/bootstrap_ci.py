"""Bootstrap percentile confidence interval for held-out ROC-AUC.

Run inside the project's pinned Python environment:
    python scripts/bootstrap_ci.py --n-resamples 5000

This is a descriptive uncertainty interval for the fixed held-out sample, not a
replacement for cross-validation or an independent external validation set.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXPECTED_SKLEARN_VERSION = "1.9.1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models/model.joblib"))
    parser.add_argument("--test", default=str(ROOT / "data/processed/test.csv"))
    parser.add_argument("--n-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(ROOT / "reports/roc_auc_bootstrap_ci.json"))
    args = parser.parse_args()

    if sklearn.__version__ != EXPECTED_SKLEARN_VERSION:
        raise RuntimeError(
            f"This model artifact was serialized with scikit-learn {EXPECTED_SKLEARN_VERSION}; "
            f"the current environment has {sklearn.__version__}. Install the pinned requirements "
            "before loading the model."
        )
    warnings.simplefilter("error", InconsistentVersionWarning)
    model = joblib.load(args.model)
    frame = pd.read_csv(args.test)
    if "target" not in frame:
        raise ValueError("Test CSV must contain the target column named 'target'.")
    X = frame.drop(columns=["target"])
    y = frame["target"].astype(int).to_numpy()
    if np.unique(y).size != 2:
        raise ValueError("ROC-AUC requires both classes in the held-out test set.")

    probabilities = model.predict_proba(X)[:, 1]
    point = float(roc_auc_score(y, probabilities))
    rng = np.random.default_rng(args.seed)
    scores: list[float] = []
    for _ in range(args.n_resamples):
        indices = rng.integers(0, len(y), size=len(y))
        if np.unique(y[indices]).size < 2:
            continue
        scores.append(float(roc_auc_score(y[indices], probabilities[indices])))

    if len(scores) < args.n_resamples * 0.9:
        raise RuntimeError("Too many bootstrap samples contained only one class.")
    lower, upper = np.quantile(scores, [0.025, 0.975])
    result = {
        "metric": "held_out_roc_auc",
        "point_estimate": point,
        "confidence_level": 0.95,
        "interval_method": "percentile bootstrap",
        "lower": float(lower),
        "upper": float(upper),
        "test_rows": int(len(y)),
        "positive_rows": int(y.sum()),
        "bootstrap_resamples_requested": int(args.n_resamples),
        "bootstrap_resamples_used": int(len(scores)),
        "random_seed": int(args.seed),
        "scikit_learn_version": sklearn.__version__,
        "note": "Run with the same scikit-learn version used to train/persist the model.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
