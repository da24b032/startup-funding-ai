# 3. Low-Level Design (LLD): Endpoints and I/O Specification

Base URL: `http://<host>:8000` · Content type: `application/json` · Interactive docs: `/docs` (OpenAPI).
Every response carries an `x-request-id` header (echoed if the client sends one). Logged as `request_id=… method=… path=… status=… duration_ms=…`.

## 1. Input contract — `StartupProfile` (18 fields)

Unknown fields are rejected (`extra="forbid"`). Bounds come from `src/data/schema.py::VALUE_RANGES`, shared with the data pipeline.

| Field | Type | Allowed | Required | Meaning |
|---|---|---|---|---|
| `state_code` | string | exactly 2 letters (upper-cased) | yes | US state |
| `category_code` | string | 1–40 chars (trimmed, lower-cased, spaces → `_`) | yes | Industry |
| `age_first_funding_year` | float | −15 … 40 | yes | Years founding → first funding (negative = funded before founding) |
| `age_last_funding_year` | float | −15 … 40, **≥ first** | yes | Years founding → latest funding |
| `funding_rounds` | int | 0 … 30 | yes | Rounds raised |
| `funding_total_usd` | float | 0 … 1e11 | yes | Total raised (USD) |
| `avg_participants` | float | 0 … 50 | yes | Average investors per round |
| `has_VC`, `has_angel`, `has_roundA`, `has_roundB`, `has_roundC`, `has_roundD`, `is_top500` | int | 0 or 1 | yes | Flags |
| `relationships` | int | 0 … 500 | yes | Recorded relationships |
| `milestones` | int | 0 … 50 | yes | Milestones achieved |
| `age_first_milestone_year` | float or null | −20 … 40 | no | Must be empty when `milestones = 0` |
| `age_last_milestone_year` | float or null | −20 … 40, **≥ first** | no | Must be empty when `milestones = 0` |

**Example request**
```json
{"state_code":"CA","category_code":"software","age_first_funding_year":1.5,"age_last_funding_year":3.5,
 "funding_rounds":3,"funding_total_usd":12000000,"avg_participants":3.0,"has_VC":1,"has_angel":0,
 "has_roundA":1,"has_roundB":1,"has_roundC":0,"has_roundD":0,"is_top500":1,"relationships":8,
 "milestones":2,"age_first_milestone_year":2.0,"age_last_milestone_year":4.0}
```

## 2. Endpoint specifications

### `GET /health`
Liveness + model availability. **200** `{"status":"ok","model_loaded":true,"uptime_seconds":12.3}`; **503** `{"status":"degraded","model_loaded":false,"detail":"<reason>","uptime_seconds":…}` when no model is loaded.

### `GET /model/info`
**200** `{model_name, trained_at, mlflow_run_id, git_commit, train_data_sha256, tracking_backend, n_train, n_test, test_metrics{accuracy,precision,recall,f1,roc_auc}, cv_roc_auc_mean, threshold}` · **503** if no model.

### `GET /schema`
Field definitions for building a form. **200** `{"fields":[{name,label,group,help,type,required, options|min|max}]}` where `type ∈ {category, flag, integer, number}`. Category `options` are the values seen in training, most common first.

### `POST /predict`
**Input:** `StartupProfile`. **200 output (`PredictionResponse`):**
```json
{"prediction":1,"outcome":"Acquired","acquired_probability":0.7228,"closed_probability":0.2772,
 "threshold":0.5,"model_name":"random_forest","model_run_id":"79b4…","latency_ms":43.3}
```
`prediction = 1` iff `acquired_probability ≥ threshold` (from `params.yaml`, default 0.5).

### `POST /predict/explain`
**Input:** `StartupProfile`. **200 output:** all `PredictionResponse` fields plus
`"explanation_method":"occlusion vs typical training startup"` and `"top_factors":[{"field","label","value","impact","direction"}]` (≤ 6 entries; `impact` = change in acquisition probability versus replacing that input with a typical value; `direction ∈ raises|lowers`).

### `POST /predict/batch`
**Input:** `{"records":[StartupProfile, …]}` with 1–500 records. **200 output:**
`{"total","predicted_acquired","predicted_closed","acquired_rate","predictions":[PredictionResponse,…]}`.

### `GET /drift`
**200:** `{"window_size","min_samples","ready","drifted_features":[…],"features":{<field>:{"type","statistic","p_value","drifted"}},"message"}`. `ready=false` until `min_samples` (30) requests are in the window. Numeric: two-sample KS test, drift if `p < 0.05`. Categorical: total-variation distance, drift if `> 0.30`.

### `POST /drift/reset`
Clears the window. **200** `{"status":"reset"}`.

### `GET /pipeline/summary`
Everything the Pipeline Console needs: `{data_report, training_metrics, model_comparison[], dvc_stages[{name,cmd,deps,outs}], feature_importance, recent_runs[≤10], run_counts{success,failed}}`.

### `GET /pipeline/runs?limit=50`
`limit` 1–500. **200** `{"runs":[{stage,timestamp,status,duration_seconds,throughput_rows_per_second?,selected_model?,error?}]}` newest first.

### `GET /metrics`
Prometheus text exposition (`text/plain; version=0.0.4`).

## 3. Error model
All errors share one body: `{"error": "<short title>", "detail": <string|list>, "request_id": "<id>"}`.

| Status | When | `error` |
|---|---|---|
| 422 | Missing/unknown field, out-of-range value, cross-field rule violated, batch > 500 or empty | `Invalid input` |
| 503 | Model artifacts missing/corrupt, or inference fails | `Model unavailable` |
| 500 | Any unexpected exception (full traceback logged with the request id; client sees a generic message) | `Internal server error` |

Validation `detail` lists `{"field","problem"}` pairs.

## 4. Module interfaces (internal)

| Module | Public interface |
|---|---|
| `src.config` | `load_params() -> dict`, `resolve(path, env_var)`, `setup_logging(name)` |
| `src.data.schema` | constants (`RAW_FEATURE_COLUMNS`, `VALUE_RANGES`, …), `validate_raw_dataframe(df) -> dict`, `DataValidationError` |
| `src.data.prepare` | `run_pipeline() -> dict` (CLI: `python -m src.data.prepare`) |
| `src.features.engineering` | `StartupFeatureBuilder` (sklearn transformer: 18 raw → 28 model columns) |
| `src.training.train` | `build_pipeline(name, cfg, seed)`, `evaluate(pipe, X, y)`, `run_training() -> dict` |
| `src.tracking.tracker` | `get_tracker(uri, experiment)` → `MlflowTracker` / offline `LocalTracker` (`start_run`, `log_params`, `log_metrics`, `log_artifact`, `log_model`, `set_tags`) |
| `src.tracking.run_log` | `logged_stage(name)` decorator, `read_runs(limit)` |
| `src.serving.predictor` | `Predictor.from_settings()`, `.predict(records)`, `.explain(record)`, `.info()`; errors `InvalidInputError`, `ModelNotAvailableError` |
| `src.monitoring.drift` | `DriftMonitor(baseline,…)`: `.add(record)`, `.check()`, `.reset()` |
| `src.monitoring.metrics` | `record_prediction`, `set_model_info`, `update_drift`, `update_pipeline` |

## 5. Engineered features (`StartupFeatureBuilder`)
`log_funding`, `funding_efficiency` (log total/(rounds+1)), `investor_strength` (3·VC + 2·angel + A + B + C + D), `milestone_speed`, `network_funding_ratio`, `funding_span_years`, `rounds_per_year`; flags `is_top_state` (CA/NY/MA), `funded_before_founding`, `has_milestone`; groups `state_group` (CA/NY/MA/TX/other) and `category_group` (categories with ≥ 20 training rows, else `other`). Rare-category rules are learned in `fit`; everything else is a deterministic function of the row.

## 6. Prometheus metrics
`startup_http_requests_total{method,endpoint,status}` · `startup_http_request_duration_seconds{endpoint}` · `startup_errors_total{endpoint,error_type}` · `startup_predictions_total{predicted_class}` · `startup_prediction_latency_seconds` · `startup_acquired_probability` · `startup_prediction_batch_size` · `startup_acquired_rate_window` · `startup_model_info{model_name,run_id,git_commit}` · `startup_model_test_metric{metric}` · `startup_feature_drift_statistic{feature}` · `startup_feature_drift_detected{feature}` · `startup_drifted_features` · `startup_drift_window_size` · `startup_pipeline_throughput_rows_per_second` · `startup_pipeline_last_run_timestamp{stage,status}`.

## 7. Configuration (`params.yaml`, overridable by environment)
Paths: `RAW_DATA_PATH`, `PROCESSED_DIR`, `MODELS_DIR`, `REPORTS_DIR`, `MODEL_PATH`, `MODEL_METADATA_PATH`, `DRIFT_BASELINE_PATH`, `RUN_LOG_PATH`, `MLFLOW_TRACKING_URI`, `PARAMS_FILE`, `PROJECT_ROOT`, `LOG_LEVEL`. Frontend: `BACKEND_URL`, `MLFLOW_URL`, `GRAFANA_URL`, `PROMETHEUS_URL`, `BACKEND_TIMEOUT`.
