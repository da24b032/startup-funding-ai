"""
Training pipeline — trains, evaluates, and persists the churn classifier.

MLflow tracks every run:
  - Parameters: n_estimators, max_depth, random_state, test_size
  - Metrics: accuracy, AUC-ROC, F1, precision, recall
  - Artifacts: trained model (joblib), feature importance plot

Usage:
    python -m src.training.train          # train with settings from .env
    python -m src.training.train --help   # show CLI options
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import settings
from src.data.generate import load_or_generate
from src.features.engineering import prepare_features
from src.training.evaluate import evaluate_model

logger = logging.getLogger(__name__)

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=500, random_state=settings.random_state),
    "random_forest": RandomForestClassifier(
        n_estimators=settings.n_estimators,
        max_depth=settings.max_depth,
        random_state=settings.random_state,
        n_jobs=-1,
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=settings.n_estimators,
        max_depth=settings.max_depth,
        random_state=settings.random_state,
    ),
}


def train(model_type: str = "random_forest") -> dict:
    """
    Run the full training pipeline:
    1. Load / generate data.
    2. Feature engineering.
    3. Train/test split.
    4. Train model + wrap in sklearn Pipeline (scaler → estimator).
    5. Evaluate and log everything to MLflow.
    6. Persist the best model to disk.

    Returns: metrics dict for the trained model.
    """
    logging.basicConfig(level=settings.log_level)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    # 1. Data
    df = load_or_generate(
        settings.data_path,
        n_samples=settings.n_synthetic_samples,
        random_state=settings.random_state,
    )

    # 2. Features
    X, y = prepare_features(df)

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings.test_size, random_state=settings.random_state, stratify=y
    )
    logger.info("Train: %d | Test: %d", len(X_train), len(X_test))

    # 4. Train (inside MLflow run)
    with mlflow.start_run(run_name=model_type) as run:
        mlflow.log_params({
            "model_type": model_type,
            "n_estimators": settings.n_estimators,
            "max_depth": settings.max_depth,
            "random_state": settings.random_state,
            "test_size": settings.test_size,
            "n_train": len(X_train),
            "n_test": len(X_test),
        })

        estimator = MODELS[model_type]
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)

        # 5. Evaluate
        metrics = evaluate_model(pipeline, X_test, y_test)
        mlflow.log_metrics(metrics)
        logger.info("Metrics: %s", metrics)

        # Log sklearn model as MLflow artifact
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        # Log feature importance for tree-based models
        if hasattr(estimator, "feature_importances_"):
            importance = dict(zip(X.columns, estimator.feature_importances_.round(4)))
            mlflow.log_dict(importance, "feature_importance.json")

        run_id = run.info.run_id
        logger.info("MLflow run_id: %s", run_id)

    # 6. Persist model to disk
    Path(settings.model_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(settings.model_dir) / f"{settings.model_name}.joblib"
    joblib.dump(pipeline, model_path)

    # Save metadata alongside model
    meta = {
        "model_type": model_type,
        "mlflow_run_id": run_id,
        "feature_columns": list(X.columns),
        **metrics,
    }
    meta_path = Path(settings.model_dir) / f"{settings.model_name}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info("Model saved to %s | AUC=%.4f", model_path, metrics["roc_auc"])
    return metrics


if __name__ == "__main__":
    import typer

    def _cli(model_type: str = "random_forest"):
        train(model_type=model_type)

    typer.run(_cli)
