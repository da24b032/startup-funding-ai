# Low-Level Design (LLD): Startup Prediction API

## 1. Service details

**Source file:** `mlops-template/src/serving/startup_api.py`  
**Framework:** FastAPI  
**Model:** `RandomForestClassifier`, loaded from `best_classical_model.pkl`  
**Feature count:** 41, determined from `model.feature_names_in_`  
**Default port in the verified local run:** 8000

At import/startup, the module loads the model pickle, obtains the feature names from the model, and reads the training feature CSV into `baseline`.

## 2. Request models

The request models are defined as:

```python
class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ..., description="Exactly 41 model features, using their original names."
    )

class BatchRequest(BaseModel):
    records: list[PredictionRequest] = Field(..., min_length=1, max_length=500)
```

`PredictionRequest.features` must contain exactly the 41 feature names expected by the loaded model. Values must be numeric. `BatchRequest.records` must contain 1–500 records.

## 3. Feature preparation

The internal `prepare_input(features)` function:
1. Computes missing feature names against `FEATURES`.
2. Computes extra feature names not present in `FEATURES`.
3. Raises `ValueError` if either set is non-empty.
4. Builds a one-row pandas DataFrame with columns in the exact model feature order.

This endpoint expects engineered/scaled model inputs. It does not perform raw-data cleaning, feature engineering or scaling for an arbitrary startup profile.

## 4. Endpoint specification

### `GET /health`

**Purpose:** Basic service/model health metadata.

**Request:** No body.

**Successful response example:**

```json
{
  "status": "ok",
  "model": "RandomForestClassifier",
  "features": 41,
  "model_path": "<resolved model path>"
}
```

### `GET /features`

**Purpose:** Return the required feature count and exact feature names.

**Request:** No body.

**Successful response shape:**

```json
{
  "count": 41,
  "feature_names": ["<feature_1>", "<feature_2>", "..."]
}
```

The actual names are taken from the loaded model at runtime; consult this endpoint rather than relying on a manually maintained list.

### `POST /predict`

**Purpose:** Generate a prediction for one prepared startup record.

**Request body:**

```json
{
  "features": {
    "<exact_model_feature_name_1>": 0.0,
    "<exact_model_feature_name_2>": 1.0
  }
}
```

The example is abbreviated: the real request must include all 41 exact feature names.

**Successful response shape:**

```json
{
  "prediction": 1,
  "outcome": "Acquired",
  "acquired_probability": 0.885,
  "model": "RandomForestClassifier",
  "latency_seconds": 0.0364
}
```

Values above are illustrative of the response format, not a guaranteed result for every request. `prediction` is 1 for Acquired and 0 for Closed. `acquired_probability` is the model probability for class 1, rounded to four decimal places. Latency is rounded to six decimal places.

**Error handling:** Feature-schema `ValueError` is returned as HTTP 422. Other exceptions in the single-prediction handler are returned as HTTP 500.

### `POST /predict/batch`

**Purpose:** Predict outcomes for multiple prepared records.

**Request body:**

```json
{
  "records": [
    {"features": {"<all_41_exact_feature_names>": 0.0}}
  ]
}
```

The placeholder above represents a complete feature dictionary, not a literal key. Supply 1–500 records.

**Successful response shape:**

```json
{
  "total": 1,
  "predicted_acquired": 1,
  "predicted_closed": 0,
  "acquired_prediction_rate": 1.0,
  "predictions": [
    {
      "prediction": 1,
      "outcome": "Acquired",
      "acquired_probability": 0.885,
      "model": "RandomForestClassifier",
      "latency_seconds": 0.0364
    }
  ]
}
```

**Error handling:** A feature-schema `ValueError` is returned as HTTP 422. Pydantic validates the request structure and the 1–500 record limit.

### `GET /metrics`

**Purpose:** Expose Prometheus metrics.

**Request:** No body.

**Response:** Prometheus text exposition format (`text/plain; version=0.0.4`).

Metrics declared by the source:
- `startup_predictions_total{predicted_class="..."}` — total predictions by predicted class.
- `startup_prediction_latency_seconds` — inference latency histogram.
- `startup_acquisition_prediction_rate` — gauge updated by single and batch prediction calls.

## 5. Endpoint summary

| Method | Path | Input | Output |
|---|---|---|---|
| GET | `/health` | None | Service/model metadata |
| GET | `/features` | None | Feature count and names |
| POST | `/predict` | One `features` dictionary | Class, outcome, acquired probability, model, latency |
| POST | `/predict/batch` | `records` array (1–500) | Aggregate counts/rate and per-record predictions |
| GET | `/metrics` | None | Prometheus exposition text |

## 6. Model and file configuration

The source defines these environment-variable overrides:

- `STARTUP_MODEL_PATH`; default `../Success-Prediction-System/best_classical_model.pkl`
- `STARTUP_TRAIN_DATA`; default `../Success-Prediction-System/X_train.csv`

Both paths are resolved with `Path.resolve()`. The process must be launched from a working directory where the defaults resolve correctly, or the environment variables must point to the correct files.

## 7. Important implementation notes

- The API returns an acquired probability, not a generic probability of raising the next funding round.
- The API uses the model's `predict()` and `predict_proba()` methods.
- The API loads `baseline` from `X_train.csv`; the current endpoint code does not expose a drift-check endpoint.
- The separate template API in `src/serving/app.py` is a customer-churn service and is not the same service as `startup_api.py`.
