# Test Plan and Test Cases

## 1. Purpose

This document defines verification for the combined startup prediction project. It separates tests that exist in the supplied repository from manual checks reported during project execution and recommended tests that still need to be automated.

## 2. Test scope

- Streamlit application startup and key user workflows.
- Startup FastAPI health, feature metadata, single prediction, batch prediction, validation and metrics.
- Prometheus scraping and Grafana visualization.
- Regression tests for the original MLOps template.

## 3. Test environment

Expected local components:
- Python environment with project dependencies installed.
- Startup project files and model artifacts available.
- Streamlit app run from the startup project.
- Startup FastAPI service run from the MLOps template with valid model/data paths.
- Prometheus and Grafana services available when monitoring tests are performed.

## 4. Test cases

| ID | Area | Test / input | Expected result | Status/evidence |
|---|---|---|---|---|
| TC-01 | Streamlit | Launch the Streamlit application | App loads and dashboard tabs are accessible | Manually verified during project work |
| TC-02 | Streamlit | Open prediction workflow and submit a valid startup profile | Prediction is displayed | Manually verified during project work |
| TC-03 | Streamlit | Inspect explanation/model comparison/dataset insights views | Corresponding analysis views render | Manually verified during project work |
| TC-04 | API health | `GET /health` | HTTP 200 and status `ok`; model and feature count returned | Manually verified; 41 features reported |
| TC-05 | API metadata | `GET /features` | HTTP 200; count and feature-name list returned | Defined by source; verify in deployment |
| TC-06 | API prediction | `POST /predict` with all 41 exact numeric features | HTTP 200; class, outcome, acquired probability, model and latency returned | Manually verified with a row from `X_train.csv` |
| TC-07 | API validation | Omit one required feature | HTTP 422 | Expected by source; automated startup-API test not present |
| TC-08 | API validation | Add an unexpected feature | HTTP 422 | Expected by source; automated startup-API test not present |
| TC-09 | API batch | `POST /predict/batch` with 1 valid record | HTTP 200; aggregate fields and predictions array returned | Defined by source; verify in deployment |
| TC-10 | API batch limit | Submit 0 or more than 500 records | Request rejected by Pydantic validation | Defined by source; verify in deployment |
| TC-11 | Metrics | `GET /metrics` after predictions | Prometheus text response includes startup metrics | Manually verified during project work |
| TC-12 | Prometheus | Check target for startup API | Target reports UP when API is reachable | Manually verified during project work |
| TC-13 | Grafana | Open configured dashboard | Prediction-count and latency panels are visible | Manually verified during project work |
| TC-14 | Template regression | Run `pytest` from `mlops-template` | Existing template tests pass | 22 passed, 3 warnings in the recorded run |

## 5. Existing automated test coverage

The supplied `mlops-template/tests/` suite tests the template's original customer-churn components. The recorded test run completed with **22 passed and 3 warnings**. These tests must not be represented as 22 automated tests of the startup acquisition API: `tests/test_serving.py` imports `src.serving.app`, which is the separate churn API, not `src.serving.startup_api`.

## 6. Recommended automated tests for startup API

Create a dedicated test module that imports `src.serving.startup_api` with controlled model and data fixtures. At minimum automate:
1. Health response structure and feature count.
2. Feature metadata response.
3. Valid single-record prediction and probability range.
4. Missing-feature rejection.
5. Extra-feature rejection.
6. Valid batch response and aggregate arithmetic.
7. Empty batch and batch over 500 rejection.
8. Metrics endpoint content type and metric names.
9. Model/data path failure behavior.
10. API behavior when the model returns unexpected or invalid output.

## 7. Acceptance criteria

The startup API is ready for a basic local demonstration when:
- `/health` returns HTTP 200 and `status: ok`;
- `/features` returns 41 feature names;
- a valid 41-feature request returns a binary outcome and probability in [0, 1];
- malformed feature sets are rejected;
- `/metrics` returns Prometheus exposition text;
- Prometheus can scrape the endpoint.

A full production-readiness claim requires the additional automated tests, robust path/configuration handling, startup-specific monitoring integration and deployment verification.
