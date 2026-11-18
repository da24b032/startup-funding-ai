# Startup Success Prediction and MLOps Monitoring System

## 1. Project Overview

An end-to-end machine learning system designed to predict whether a startup will be acquired or closed. The project combines machine learning, explainable AI, interactive prediction, API-based model serving and monitoring.

## 2. System Architecture

The system consists of three major layers:

### A. Machine Learning
- Startup dataset preprocessing and cleaning
- Feature engineering
- Classical machine learning model comparison
- Artificial neural networks and deep learning experiments
- Random Forest as the selected classical model
- SHAP-based model explainability

### B. Application and Model Serving
- Interactive Streamlit dashboard
- FastAPI-based startup prediction service
- Single and batch prediction endpoints
- Input feature validation
- Prediction probabilities and response latency

### C. Monitoring and MLOps
- Prometheus metrics collection
- Grafana monitoring dashboard
- Prediction count monitoring
- Prediction latency monitoring
- Docker Compose configuration for supporting services
- Automated test suite

## 3. Selected Model

The selected model is a Random Forest classifier.

Original project results:
- Accuracy: 81.5%
- ROC-AUC: 0.909

Independent test-set evaluation in the current environment:
- Accuracy: 81.15%
- Precision: 78.64%
- Recall: 85.26%
- F1-score: 81.82%
- ROC-AUC: 0.9066

The target classes are:
- 1: Acquired
- 0: Closed

## 4. Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Deep learning | TensorFlow, Keras |
| Explainability | SHAP, LIME |
| Frontend | Streamlit |
| API | FastAPI, Uvicorn |
| Monitoring | Prometheus, Grafana |
| Containerisation | Docker, Docker Compose |
| Testing | Pytest |

## 5. Running the Applications

### Streamlit dashboard

From the startup project directory:

    cd ~/startup-funding-ai/Success-Prediction-System
    streamlit run app.py

Open http://localhost:8501

### Startup prediction API

From the MLOps template directory, with the project virtual environment activated:

    cd ~/startup-funding-ai/mlops-template
    uvicorn src.serving.startup_api:app --host 0.0.0.0 --port 8000

API documentation: http://localhost:8000/docs

Health check: http://localhost:8000/health

Prometheus metrics: http://localhost:8000/metrics

### Monitoring

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- MLflow service: http://localhost:5000

## 6. Testing

The existing MLOps template test suite was executed using Pytest.

Result: 22 passed.

The custom startup prediction API was also manually verified through its health endpoint, prediction endpoint and Prometheus metrics endpoint.

## 7. Implementation Notes and Limitations

- The startup prediction API loads the existing trained Random Forest model and expects its 41 engineered and scaled features.
- The startup API currently runs directly in the Python virtual environment.
- The existing Dockerfile still launches the original customer-churn API; it has not yet been adapted to launch the startup API.
- The original template's synthetic churn training and feature-engineering pipeline remains separate from the startup prediction pipeline.
- Startup-specific drift detection and MLflow experiment tracking have not yet been integrated and verified.
- The original dataset and model evaluation results are reported separately from the independent test-set evaluation.

## 8. Project Structure

    startup-funding-ai/
    ├── README.md
    ├── Success-Prediction-System/
    │   ├── app.py
    │   ├── models/
    │   └── outputs/
    └── mlops-template/
        ├── src/
        │   └── serving/
        │       └── startup_api.py
        ├── tests/
        ├── monitoring/
        ├── docker-compose.yml
        └── Dockerfile

## 9. Acknowledgements

<small>Built upon the MIT-licensed Startup Success Prediction project by Mohammed Hasnaine and the MIT-licensed MLOps template by Maniswaroop M; original licences and attribution are retained.</small>
