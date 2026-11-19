"""Training pipeline.

Flow
----
1. Load the train/test CSVs produced by ``src.data.prepare``.
2. For every candidate model build ONE scikit-learn Pipeline:
   ``StartupFeatureBuilder -> ColumnTransformer -> estimator``.
3. Estimate performance with stratified cross-validation on the TRAIN set only.
4. Fit on the train set, evaluate once on the untouched TEST set.
5. Log params, metrics, tags and artifacts of every model to MLflow
   (one nested run per model under one parent run).
6. Promote the model with the best mean CV ROC-AUC (never chosen on test data)
   to ``models/model.joblib`` together with metadata and a drift baseline.

Reproducibility: each run is tagged with the git commit hash and the SHA-256
of the training data, and ``params.yaml`` is stored as an artifact. A model can
therefore be traced to exactly one (commit, data version, MLflow run id).

Run:  python -m src.training.train
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from src.config import PROJECT_ROOT, load_params, resolve, setup_logging  # noqa: E402
from src.data.schema import BINARY_COLUMNS, NUMERIC_COLUMNS, RAW_FEATURE_COLUMNS, TARGET  # noqa: E402
from src.features.engineering import (  # noqa: E402
    MODEL_BINARY,
    MODEL_CATEGORICAL,
    MODEL_NUMERIC,
    StartupFeatureBuilder,
)
from src.tracking.run_log import logged_stage  # noqa: E402
from src.tracking.tracker import get_tracker, git_commit  # noqa: E402

logger = setup_logging("training")


# --------------------------------------------------------------------- models
def build_estimator(name: str, cfg: dict[str, Any], seed: int):
    """Create an estimator from its params.yaml section."""
    if name == "logistic_regression":
        return LogisticRegression(random_state=seed, **cfg["logistic_regression"])
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **cfg["random_forest"])
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed, **cfg["gradient_boosting"])
    raise ValueError(f"Unknown model '{name}'")


def build_pipeline(name: str, cfg: dict[str, Any], seed: int) -> Pipeline:
    """Full raw-input -> prediction pipeline (this exact object is served)."""
    encode = ColumnTransformer(
        [
            ("num", StandardScaler(), MODEL_NUMERIC),
            ("bin", "passthrough", MODEL_BINARY),
            ("cat", OneHotEncoder(handle_unknown="ignore"), MODEL_CATEGORICAL),
        ]
    )
    return Pipeline(
        [
            ("features", StartupFeatureBuilder()),
            ("encode", encode),
            ("model", build_estimator(name, cfg, seed)),
        ]
    )


# ------------------------------------------------------------------- metrics
def evaluate(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    pred = pipe.predict(X)
    prob = pipe.predict_proba(X)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred)), 4),
        "recall": round(float(recall_score(y, pred)), 4),
        "f1": round(float(f1_score(y, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y, prob)), 4),
    }


def median_latency_ms(pipe: Pipeline, row: pd.DataFrame, repeats: int = 30) -> float:
    """Median single-row inference latency in milliseconds."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        pipe.predict_proba(row)
        times.append((time.perf_counter() - start) * 1000)
    return round(float(np.median(times)), 3)


# --------------------------------------------------------------------- plots
def save_confusion_matrix(y_true, y_pred, path: Path) -> list[list[int]]:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Closed", "Acquired"])
    ax.set_yticks([0, 1], ["Closed", "Acquired"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center", fontsize=13)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return cm.tolist()


def save_roc_curves(curves: dict[str, tuple], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_importance(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, path: Path, seed: int):
    """Permutation importance on the *raw* inputs (what a user actually enters)."""
    result = permutation_importance(
        pipe, X, y, scoring="roc_auc", n_repeats=15, random_state=seed, n_jobs=1
    )
    order = np.argsort(result.importances_mean)[::-1]
    names = [X.columns[i] for i in order]
    means = [float(result.importances_mean[i]) for i in order]
    stds = [float(result.importances_std[i]) for i in order]
    fig, ax = plt.subplots(figsize=(6, 5))
    top = min(12, len(names))
    ax.barh(names[:top][::-1], means[:top][::-1], xerr=stds[:top][::-1], color="#3b6fb6")
    ax.set_xlabel("Drop in ROC-AUC when shuffled")
    ax.set_title("Permutation importance (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {n: {"mean": round(m, 5), "std": round(s, 5)} for n, m, s in zip(names, means, stds, strict=False)}


# ------------------------------------------------------------- drift baseline
def build_drift_baseline(train_X: pd.DataFrame, max_sample: int = 300, seed: int = 42) -> dict:
    """Reference data: numeric samples, binary prevalences, category frequencies."""
    rng = np.random.default_rng(seed)
    baseline: dict[str, Any] = {"numeric": {}, "binary": {}, "categorical": {}}
    for col in NUMERIC_COLUMNS:
        values = train_X[col].dropna().to_numpy(dtype=float)
        if len(values) > max_sample:
            values = rng.choice(values, size=max_sample, replace=False)
        baseline["numeric"][col] = {
            "sample": [round(float(v), 4) for v in values],
            "mean": round(float(np.mean(values)), 4),
            "std": round(float(np.std(values)), 4),
        }
    for col in BINARY_COLUMNS:
        baseline["binary"][col] = round(float(train_X[col].mean()), 4)
    for col in ("state_code", "category_code"):
        freq = train_X[col].astype(str).value_counts(normalize=True)
        baseline["categorical"][col] = {k: round(float(v), 4) for k, v in freq.items()}
    return baseline


# ----------------------------------------------------------------- utilities
def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flat_params(cfg: dict[str, Any], model_name: str) -> dict[str, Any]:
    return {f"{model_name}.{k}": v for k, v in cfg.get(model_name, {}).items()}


# ----------------------------------------------------------------------- main
def run_training() -> dict[str, Any]:
    all_params = load_params()
    cfg = all_params["train"]
    mlflow_cfg = all_params["mlflow"]
    seed = cfg["random_state"]

    data_dir = resolve(all_params["data"]["processed_dir"], "PROCESSED_DIR")
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    X_train, y_train = train_df[RAW_FEATURE_COLUMNS], train_df[TARGET]
    X_test, y_test = test_df[RAW_FEATURE_COLUMNS], test_df[TARGET]

    models_dir = resolve("models", "MODELS_DIR")
    reports_dir = resolve("reports", "REPORTS_DIR")
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    tracker = get_tracker(
        resolve(mlflow_cfg["tracking_uri"], "MLFLOW_TRACKING_URI").as_posix()
        if "://" not in mlflow_cfg["tracking_uri"]
        else mlflow_cfg["tracking_uri"],
        mlflow_cfg["experiment_name"],
    )
    logger.info("Tracking backend: %s", tracker.backend)

    commit = git_commit()
    data_hash = sha256_of(data_dir / "train.csv")
    cv = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=seed)

    results: dict[str, dict[str, Any]] = {}
    fitted: dict[str, Pipeline] = {}
    run_ids: dict[str, str] = {}
    roc_data: dict[str, tuple] = {}

    with tracker.start_run("training-pipeline") as parent_id:
        tracker.set_tags({"git_commit": commit, "train_data_sha256": data_hash})
        tracker.log_params(
            {
                "cv_folds": cfg["cv_folds"],
                "random_state": seed,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "models_compared": ",".join(cfg["models_to_compare"]),
            }
        )
        tracker.log_artifact(PROJECT_ROOT / "params.yaml")
        report = data_dir / "data_report.json"
        if report.exists():
            tracker.log_artifact(report)

        for name in cfg["models_to_compare"]:
            pipe = build_pipeline(name, cfg, seed)
            with tracker.start_run(name, nested=True) as run_id:
                run_ids[name] = run_id
                tracker.set_tags(
                    {"git_commit": commit, "train_data_sha256": data_hash, "model": name}
                )
                tracker.log_params(flat_params(cfg, name))

                cv_scores = cross_validate(
                    pipe,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=["roc_auc", "accuracy", "f1"],
                    n_jobs=1,
                )
                cv_metrics = {
                    "cv_roc_auc_mean": float(cv_scores["test_roc_auc"].mean()),
                    "cv_roc_auc_std": float(cv_scores["test_roc_auc"].std()),
                    "cv_accuracy_mean": float(cv_scores["test_accuracy"].mean()),
                    "cv_f1_mean": float(cv_scores["test_f1"].mean()),
                }

                start = time.perf_counter()
                pipe.fit(X_train, y_train)
                fit_seconds = time.perf_counter() - start

                test_metrics = {f"test_{k}": v for k, v in evaluate(pipe, X_test, y_test).items()}
                latency = median_latency_ms(pipe, X_test.iloc[[0]])

                extra = {
                    "train_seconds": round(fit_seconds, 3),
                    "latency_ms_single_row": latency,
                }
                tracker.log_metrics(
                    {k: round(v, 4) for k, v in {**cv_metrics, **test_metrics, **extra}.items()}
                )
                tracker.log_model(pipe, "model")

                prob = pipe.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, prob)
                roc_data[name] = (fpr, tpr, test_metrics["test_roc_auc"])
                fitted[name] = pipe
                results[name] = {**cv_metrics, **test_metrics, **extra}
                logger.info(
                    "%s | cv_auc=%.4f | test_auc=%.4f | test_acc=%.4f",
                    name,
                    cv_metrics["cv_roc_auc_mean"],
                    test_metrics["test_roc_auc"],
                    test_metrics["test_accuracy"],
                )

        # ---- model selection on CV only ----------------------------------
        if cfg["selected_model"] == "auto":
            best = max(results, key=lambda m: results[m]["cv_roc_auc_mean"])
        else:
            best = cfg["selected_model"]
        best_pipe = fitted[best]
        logger.info("Selected model: %s (by mean CV ROC-AUC)", best)

        # ---- artifacts ---------------------------------------------------
        cm = save_confusion_matrix(y_test, best_pipe.predict(X_test), reports_dir / "confusion_matrix.png")
        save_roc_curves(roc_data, reports_dir / "roc_curves.png")
        importance = save_importance(
            best_pipe, X_test, y_test, reports_dir / "feature_importance.png", seed
        )
        comparison = pd.DataFrame(results).T.round(4)
        comparison.index.name = "model"
        comparison.to_csv(reports_dir / "model_comparison.csv")

        best_test = {k.removeprefix("test_"): v for k, v in results[best].items() if k.startswith("test_")}
        metrics_payload = {
            "selected_model": best,
            "selection_rule": "highest mean 5-fold CV ROC-AUC on training data",
            **{f"test_{k}": v for k, v in best_test.items()},
            "cv_roc_auc_mean": round(results[best]["cv_roc_auc_mean"], 4),
            "latency_ms_single_row": results[best]["latency_ms_single_row"],
            "confusion_matrix": cm,
        }
        (reports_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
        (reports_dir / "feature_importance.json").write_text(json.dumps(importance, indent=2))

        for artifact in (
            "confusion_matrix.png",
            "roc_curves.png",
            "feature_importance.png",
            "model_comparison.csv",
            "metrics.json",
        ):
            tracker.log_artifact(reports_dir / artifact)
        tracker.log_dict(importance, "feature_importance.json")
        tracker.log_metrics({f"selected_{k}": v for k, v in best_test.items()})
        tracker.set_tags({"selected_model": best})
        tracker.log_model(
            best_pipe, "selected_model", registered_name=mlflow_cfg["registered_model_name"]
        )

        # ---- serving artifacts -------------------------------------------
        joblib.dump(best_pipe, models_dir / "model.joblib")
        metadata = {
            "model_name": best,
            "trained_at": datetime.now(UTC).isoformat(),
            "mlflow_parent_run_id": parent_id,
            "mlflow_model_run_id": run_ids[best],
            "tracking_backend": tracker.backend,
            "git_commit": commit,
            "train_data_sha256": data_hash,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "raw_feature_columns": RAW_FEATURE_COLUMNS,
            "test_metrics": best_test,
            "cv_roc_auc_mean": metrics_payload["cv_roc_auc_mean"],
        }
        (models_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
        baseline = build_drift_baseline(X_train, seed=seed)
        (models_dir / "drift_baseline.json").write_text(json.dumps(baseline))

    return {"selected": best, "results": results, "metadata": metadata}


if __name__ == "__main__":
    summary = logged_stage("train")(run_training)()
    print(json.dumps({"selected": summary["selected"], "test": summary["metadata"]["test_metrics"]}, indent=2))
