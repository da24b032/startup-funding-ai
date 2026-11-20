# Startup Outcome AIOps — Viva Evidence Pack

This folder contains the curated screenshots captured from the working project on 20 Nov 2026. Use them as a backup if the viva examiner asks for screenshots instead of (or in addition to) a live demonstration.

## Screenshot index and what each proves

| # | File | What to point out during the viva |
|---|---|---|
| 01 | `01_single_prediction_result_72pct_acquired.png` | Main user-facing prediction screen; example prediction says 72% chance of acquisition and 28% chance of closure. This is one example, not a guarantee. |
| 02 | `02_batch_prediction_30_rows_13_acquired_17_closed.png` | Batch scoring of 30 records; the model predicted 13 acquired and 17 closed. These are predictions, not ground-truth labels. |
| 03 | `03_monitoring_drift_9_features_63_predictions.png` | Monitoring page; 63 predictions in the window and drift detected in 9 input features. Drift is a change in input distributions, not proof that accuracy has fallen. |
| 04 | `04_pipeline_console_dag.png` | In-app pipeline console and stage dependency flow (prepare → train → test / served model). |
| 05 | `05_model_traceability_git_data_mlflow.png` | Traceability metadata tying the served model to an MLflow run ID, Git commit, and training-data SHA-256. |
| 06 | `06_mlflow_training_metrics.png` | MLflow experiment/run metrics; selected ROC-AUC is approximately 0.84. |
| 07 | `07_prometheus_backend_target_up.png` | Prometheus Targets page; backend scrape target is UP (1/1). |
| 08 | `08_grafana_live_dashboard.png` | Grafana dashboard with request rate, latency, prediction classes, rolling acquired share, and inference latency panels. |
| 09 | `09_fastapi_swagger_api_docs.png` | FastAPI Swagger/OpenAPI page showing health, model info, schema, single prediction, explanation, and batch prediction endpoints. |
| 10 | `10_dvc_pipeline_dag_terminal.png` | `dvc dag` output showing the raw-data → prepare → train → test dependency chain. |
| 11 | `11_docker_compose_services_healthy.png` | `docker compose ps` showing backend healthy and frontend, Grafana, MLflow, and Prometheus running with their exposed ports. |


