# Project Report: Startup Outcome Prediction and Monitoring Prototype

## Abstract

This project combines a startup outcome classification application with a programmatic inference API and a basic monitoring stack. The interactive application uses saved machine-learning artifacts to present startup predictions and analytical views. A separate FastAPI service loads a Random Forest classifier and exposes single-record and batch prediction endpoints. Prometheus scrapes inference metrics, which can be visualized in Grafana.

The implemented classification target is acquisition versus closure. Although the project may be framed broadly around startup success, the supplied model does not directly predict the success of a startup's next funding round. This distinction is important when interpreting the output.

## 1. Introduction

Startups operate in uncertain environments, and historical startup data can be analyzed to identify patterns associated with different outcomes. Machine learning offers a way to learn relationships between startup attributes and a defined outcome label. This project presents an end-to-end demonstration that combines model outputs, explainability and an operational inference interface.

## 2. Problem definition

The supplied startup dataset defines a binary target:
- `acquired` = 1 (positive class);
- `closed` = 0 (negative class).

The classification task is therefore to estimate whether a startup belongs to the acquired or closed class. It is not a model of the next funding-round event. A next-round prediction would require a separately defined target, suitable time-based observations and an evaluation design that avoids future-information leakage.

## 3. Dataset and preparation

The startup project README describes a dataset of 923 startups and 49 raw features, cleaned to 778 rows and 42 features. The project includes preprocessing, missing-value handling, feature engineering, SMOTE, scaling and train/test splitting.

Six engineered features described in the source project README include:
- `funding_efficiency`;
- `investor_strength`;
- `milestone_speed`;
- `network_funding_ratio`;
- `is_top_state`;
- `log_funding`.

The MLOps startup API consumes the saved model's 41 feature columns. Its request contract is based on the exact feature names exposed by the fitted estimator, and callers must supply numeric values in that prepared feature space.

## 4. Model development and results

The source project README reports the following model comparison:

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Random Forest | 81.5% | 0.909 |
| Gradient Boosting | 81.4% | 0.901 |
| XGBoost | 77.6% | 0.882 |
| ANN | 78.5% | 0.880 |
| SVM | 74.8% | 0.844 |
| CNN | 74.3% | 0.840 |
| Logistic Regression | 71.9% | 0.812 |
| LSTM | 71.7% | 0.778 |
| CNN-LSTM | 52.4% | 0.727 |

The selected classical model is Random Forest, reported with ROC-AUC 0.909 in the source README. The API loads the saved `best_classical_model.pkl` artifact.

The combined project root README separately reports an independent test-set evaluation of accuracy 81.15%, precision 78.64%, recall 85.26%, F1-score 81.82% and ROC-AUC 0.9066. These are reported project results; this document does not independently reproduce the evaluation.

## 5. Application and service implementation

### 5.1 Interactive application

The Streamlit application provides prediction and analytical views, including explanations, model comparison and dataset insights. The project contains SHAP/LIME-related artifacts and model outputs to support explainability and analysis.

### 5.2 FastAPI inference service

The custom startup API is implemented in `mlops-template/src/serving/startup_api.py`. It loads the Random Forest model and the training feature CSV. It exposes:
- `GET /health`;
- `GET /features`;
- `POST /predict`;
- `POST /predict/batch`;
- `GET /metrics`.

The API validates that a request contains exactly the model's 41 feature names, constructs a DataFrame in the model's expected column order and returns the predicted class, outcome label, acquired probability, model name and inference latency.

### 5.3 Monitoring

The API exposes a prediction counter, latency histogram and acquisition prediction-rate gauge. Prometheus is configured to scrape `/metrics` every 15 seconds. Grafana is included in the Compose configuration for dashboard visualization.

## 6. Testing and verification

The recorded regression test run for the supplied MLOps template completed with 22 passed and 3 warnings. These tests exercise the template's original churn components and are not equivalent to automated coverage of the custom startup API.

During project execution, the Streamlit dashboard, startup API health endpoint, a valid single prediction and the metrics endpoint were manually checked. Prometheus target availability and Grafana dashboard panels were also manually checked.

A dedicated automated test suite for `startup_api.py` remains a recommended next step. See `04_test_plan_and_cases.md` for the test matrix and the distinction between existing tests and proposed tests.

## 7. Limitations

1. **Target limitation:** The classifier predicts acquired versus closed, not next funding-round success.
2. **API input limitation:** The API requires 41 already engineered/scaled numeric features; raw startup details are not transformed by the API itself.
3. **Containerization limitation:** The existing Dockerfile's API command launches the template's original churn API, not the custom startup API. The startup API was verified as a direct Python process, not as the Compose `api` container.
4. **MLOps integration limitation:** The template's original churn training and drift modules are separate. Startup-specific drift detection and MLflow experiment tracking have not been integrated and verified.
5. **Evaluation limitation:** Reported metrics are tied to the supplied dataset and evaluation setup; they do not guarantee performance on new populations or future time periods.
6. **Explainability limitation:** SHAP/LIME explanations describe model behavior and associations, not causal effects.

## 8. Conclusion

The project demonstrates a startup outcome classification workflow with an interactive Streamlit interface, a separate FastAPI prediction service and a Prometheus/Grafana monitoring path. The Random Forest model is the selected classical estimator in the supplied project results. The service provides health, feature metadata, single and batch inference, and metrics endpoints.

The most important next improvements are to align the target with next-funding-round success if that is the intended business question, encapsulate preprocessing behind the API, add automated tests specifically for the startup service, and adapt the container configuration to launch the startup API before claiming a fully containerized deployment.
