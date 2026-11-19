# Demo and local operations runbook

This runbook covers the checks that could not be executed in the original authoring sandbox.
Run commands from the repository root. Do not claim a check passed until its expected result is observed.

## 0. Preflight

```bash
python --version                 # project environment is Python 3.11
 docker --version
 docker compose version
 git status --short
 dvc status
```

The repository's Git history is already authored as `Visaka M <da24b032@smail.iitm.ac.in>`. No author rewrite is needed unless the local Git configuration differs.

## 1. Reproduce the pipeline and tests

```bash
dvc repro
pytest -q
make lint
```

Expected: `prepare`, `train`, and `test` complete; all 40 tests pass; lint reports no issues. Keep the generated `reports/test_report.xml` and `reports/metrics.json` as evidence.

## 2. Build and launch the application stack

The checked-in model and data are mounted into the services. For a clean checkout where outputs are absent, first run `dvc pull` (if the configured local DVC remote is available) or `dvc repro`.

```bash
docker compose config
docker compose up --build -d
docker compose ps
```

Expected services: `backend`, `frontend`, `mlflow`, `prometheus`, and `grafana` are running; backend health becomes `healthy`.

Useful endpoints:

- Frontend: http://localhost:8501
- Backend health: http://localhost:8000/health
- Backend API docs: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- MLflow: http://localhost:5000

Inspect logs if anything is unhealthy:

```bash
docker compose logs --tail=150 backend frontend mlflow prometheus grafana
```

## 3. OPS checklist

### OPS-1 — Compose stack

Run `docker compose up --build -d` and `docker compose ps`. Confirm all five services are running and backend is healthy. Open the frontend and `/health` endpoint.

### OPS-2 — Prometheus scrape

Open http://localhost:9090/targets. Confirm job `startup-outcome-backend` is `UP`. Also query `up{job="startup-outcome-backend"}`; expected value is `1`.

### OPS-3 — Grafana panels

Open http://localhost:3000 and select **Startup Outcome Service**. Submit several predictions through the UI, wait at least one scrape interval (15 seconds), and refresh the dashboard. Confirm request, prediction, latency and drift panels populate. Default local credentials are `admin` / `admin`; anonymous Viewer access is enabled in this compose configuration.

### OPS-4 — Drift alert

Use the UI or API to submit at least 30 deliberately varied but schema-valid requests. Check `/metrics` for `startup_drifted_features` and Prometheus for alert `DataDriftDetected`. Drift is a distribution-change signal, not proof that accuracy has fallen. Do not send invalid values merely to force the alert.

### OPS-5 — MLflow

Open http://localhost:5000. Confirm experiment `startup-outcome-prediction`, one parent run, three nested candidate runs, the registered model, metrics, parameters, tags and artifacts. The optional `mlflow-model` serving container is not part of the default stack; start it only if separately testing the MLflow scoring endpoint:

```bash
docker compose --profile mlflow-serve up --build -d mlflow-model
```

### OPS-6 — Failure-path demonstration

Do this only after backing up the raw CSV and ensuring the working tree is clean. Temporarily remove or rename a required column in a copy of the raw CSV, run `dvc repro`, and confirm the prepare stage fails and the run console records a failed stage. Restore the original file immediately and verify:

```bash
dvc checkout
dvc status
dvc repro
```

Never leave the intentionally corrupted dataset in the submission.

## 4. Browser/UI visual check

Use a desktop browser at http://localhost:8501 and inspect:

1. **Predict:** submit the prefilled example; confirm probability, class and explanation render.
2. **Batch Prediction:** download the template, upload it unchanged, then upload a copy with a required column removed; confirm clear success/error feedback.
3. **Pipeline Console:** inspect the DAG, data engineering, training comparison, run history and traceability tabs.
4. **Monitoring:** confirm live API/Prometheus metrics are visible and links open.
5. Resize to a narrow phone viewport (approximately 390 px wide); check no clipped controls, unreadable tables or horizontal overflow.

Capture screenshots only from the running application. Suggested evidence: Predict result, Pipeline Console DAG, Prometheus target UP, Grafana populated dashboard, MLflow experiment runs, and mobile-width page.

## 5. Shutdown

```bash
docker compose down
```

Use `docker compose down -v` only if you intentionally want to delete named volumes; it is not needed for routine shutdown.

## 6. Optional held-out ROC-AUC confidence interval

The model artifact was serialized with scikit-learn 1.9.1. The project dependency is pinned to that exact version because scikit-learn does not guarantee pickle compatibility across versions. In the project's Python 3.11 environment, run:

```bash
python scripts/bootstrap_ci.py --n-resamples 5000
```

The script checks the scikit-learn version before loading the artifact and writes `reports/roc_auc_bootstrap_ci.json`. Report the interval as a percentile-bootstrap interval for this fixed 185-row test set; it does not replace cross-validation or external validation. Do not generate or quote the interval from a different scikit-learn version.
