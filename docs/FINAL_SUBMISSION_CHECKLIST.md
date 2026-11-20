# Final Submission Checklist

## Included in the project archive
- [x] README and setup instructions
- [x] Source code, frontend and backend
- [x] DVC pipeline files and configuration
- [x] Dataset files, model artifacts and metrics
- [x] MLflow experiment artifacts
- [x] Docker and Docker Compose configuration
- [x] Prometheus configuration and Grafana dashboard
- [x] Architecture and project documentation
- [x] User manual, test plan and operations documentation
- [x] Automated tests and test report
- [x] AI usage disclosure
- [x] Screenshots of the running application and monitoring dashboard

## Verification completed
- [x] `dvc status`: data and pipelines up to date.
- [x] `pytest -q`: 40 passed.
- [x] `ruff check .`: all checks passed.
- [x] `docker compose config -q`: configuration validated.
- [x] Docker stack launched; backend healthy and frontend, MLflow, Prometheus and Grafana running.
- [x] Frontend prediction page returned a prediction.
- [x] Pipeline Console and Monitoring pages opened.
- [x] Prometheus backend target reported UP.
- [x] Grafana Startup Outcome Service dashboard loaded with populated panels.
- [x] MLflow experiment showed a successful parent run and three nested model-comparison runs.
- [x] Reported held-out ROC-AUC: 0.8378.

## Screenshot evidence included
See `evidence/screenshots/`:
- Prediction result
- Monitoring page
- Pipeline Console
- Grafana dashboard

## Packaging note
The archive excludes `.venv`, `.git` and `.dvc/cache` to keep it compact. Confirm whether the instructor expects a DVC remote/cache to be submitted separately.
