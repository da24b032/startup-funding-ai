# High-Level Design (HLD)

## 1. Objective

The system provides an interactive way to explore startup outcome predictions and a separately accessible HTTP inference service. The trained classification target is `acquired` (1) or `closed` (0). The broader project framing may discuss startup success, but the implemented target is not next-funding-round success.

## 2. Design overview

The solution is organized into two user-facing paths:

- **Interactive analytics path:** Streamlit loads the startup project artifacts and presents prediction, confidence, explanation, model-comparison and dataset-insight views.
- **Programmatic inference path:** FastAPI loads a saved Random Forest model and accepts a strict set of 41 engineered/scaled numeric features. It supports one-record and batch inference.

Monitoring is attached to the API path. Prometheus scrapes the API's `/metrics` endpoint and Grafana is used to view the collected metrics.

## 3. Major design choices and rationale

### 3.1 Reuse of a saved model

The API loads `best_classical_model.pkl` rather than training a model for every request. This keeps inference separate from model training and reduces request-time work. The selected model is a Random Forest classifier.

### 3.2 Strict feature contract

The API derives the required feature names from `model.feature_names_in_`. Incoming feature dictionaries are checked for missing and unexpected keys. This prevents silent feature-order or schema mismatches and ensures the DataFrame columns are created in the model's expected order.

The trade-off is that API callers must already possess the 41 engineered/scaled numeric features. The API does not currently accept a raw startup profile and perform the full preprocessing pipeline itself.

### 3.3 Single and batch inference

`POST /predict` supports one record. `POST /predict/batch` accepts between 1 and 500 records. Batch inference is useful when a client has multiple prepared records and wants to reduce repeated HTTP calls.

### 3.4 Separate interface and service

Streamlit serves the interactive user experience; FastAPI provides a machine-readable interface for other clients. This separation allows the interface and inference service to be used independently.

### 3.5 Metrics-based monitoring

The startup API exposes:
- `startup_predictions_total` — prediction counter labelled by predicted class;
- `startup_prediction_latency_seconds` — latency histogram;
- `startup_acquisition_prediction_rate` — gauge representing the latest single prediction or the acquired fraction of a batch.

Prometheus is configured to scrape `host.docker.internal:8000/metrics` every 15 seconds. Grafana reads the metrics from Prometheus.

## 4. Key components and dependencies

| Component | Technology / artifact | Role |
|---|---|---|
| Interactive UI | Streamlit | User-facing prediction and analysis dashboard |
| API | FastAPI, Pydantic | Request validation and HTTP inference endpoints |
| Data handling | pandas | Build feature DataFrames and load training baseline |
| Model | scikit-learn Random Forest, pickle artifact | Classification and probabilities |
| Metrics | prometheus-client | Counters, histogram, gauge and exposition endpoint |
| Metrics collection | Prometheus | Periodic scraping |
| Monitoring UI | Grafana | Dashboard visualization |
| Supporting artifacts | CSV, pickle, Keras files | Data, model and preprocessing assets |

## 5. Operational assumptions

- The startup API can resolve the model and training-data paths. Environment variables `STARTUP_MODEL_PATH` and `STARTUP_TRAIN_DATA` can override its defaults.
- The model artifact is compatible with the installed scikit-learn and Python environment.
- API input values are numeric and correspond to the exact expected feature definitions and preprocessing.
- Prometheus can reach the API at the configured host and port.
- The Streamlit application has access to its required model and preprocessing artifacts.

## 6. Constraints and future design work

1. Add a raw-input preprocessing layer to the API, or publish a reusable preprocessing package so clients do not have to construct 41 transformed features themselves.
2. Configure the Docker image/Compose service to launch `src.serving.startup_api:app` and mount the correct startup model/data paths before claiming containerized startup inference.
3. Add startup-specific automated API tests.
4. Integrate and validate drift detection against the startup feature baseline.
5. Add MLflow tracking for startup model training and evaluation if experiment tracking is required.
6. Define a target and dataset for next-funding-round success if that is the intended business outcome.
