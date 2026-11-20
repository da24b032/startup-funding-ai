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

## Rubric cross-reference

- **Demonstration / UX:** screenshots 01–04; user manual is in `docs/05_user_manual.md`.
- **Pipeline console and data engineering:** screenshots 04 and 10; design and implementation details are in `docs/01_architecture.md` (or the corresponding architecture document), `docs/02_high_level_design.md`, `docs/03_low_level_design.md`, and the pipeline source under `src/data/`.
- **Experiment tracking / model registry / reproducibility:** screenshots 05, 06, and 10; inspect the MLflow experiment, registered model, `dvc.yaml`, `dvc.lock`, and `reports/` in the repository.
- **Monitoring / AIOps:** screenshots 03, 07, and 08; Prometheus metrics and Grafana provisioning are in `monitoring/` and `src/monitoring/`.
- **API and service separation:** screenshots 09 and 11; API implementation is in `src/serving/`, frontend REST client in `frontend/client.py`, and service definitions in `docker-compose.yml`.
- **Testing / software engineering:** use `docs/04_test_plan_and_cases.md`, `reports/test_report.xml`, and the project test suite. The verified run completed with 40 passing tests; Ruff checks passed.

## Suggested viva sequence (about 4–6 minutes)

1. Start with screenshot 01: explain the prediction and probability output in plain language.
2. Show screenshot 02: explain why batch inference is useful and distinguish predictions from labels.
3. Show screenshot 04 and then 10: explain the stages and how DVC tracks dependencies and avoids rerunning unchanged stages.
4. Show screenshot 06: explain model comparison, cross-validation selection, and the held-out ROC-AUC (~0.838).
5. Show screenshot 05: explain how a prediction can be traced to code, training data, and an MLflow run.
6. Finish with screenshots 03, 07, and 08: explain input drift, Prometheus scraping, and Grafana visualisation.
7. If asked about deployment/API design, show screenshots 09 and 11.

## Important honesty notes

- The 30-row batch result is a model-output demonstration; it does not establish that 17 records are truly closed.
- The single-row 72% output is one illustrative inference, not a calibrated guarantee or business decision.
- Drift is input-distribution monitoring. It does not by itself show that predictive accuracy has degraded.
- A panel with no series in the selected time window is not automatically a system failure.
- Screenshots are point-in-time evidence. If asked, use the live application and repository artifacts as the authoritative demonstration.
- The evidence pack does not replace the source repository, test report, design documents, or user manual.
