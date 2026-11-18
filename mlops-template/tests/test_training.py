"""Tests for training pipeline and evaluation."""

import pytest
from sklearn.pipeline import Pipeline
from src.training.evaluate import evaluate_model
from src.features.engineering import prepare_features


def test_evaluate_model_returns_metrics(churn_df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = prepare_features(churn_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    pipe = Pipeline([("scaler", StandardScaler()), ("model", RandomForestClassifier(n_estimators=10, random_state=42))])
    pipe.fit(X_train, y_train)

    metrics = evaluate_model(pipe, X_test, y_test)
    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    assert "f1" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_evaluate_model_auc_above_random(churn_df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = prepare_features(churn_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    pipe = Pipeline([("scaler", StandardScaler()), ("model", RandomForestClassifier(n_estimators=20, random_state=42))])
    pipe.fit(X_train, y_train)

    metrics = evaluate_model(pipe, X_test, y_test)
    assert metrics["roc_auc"] > 0.55, "Model should beat random on synthetic data"
