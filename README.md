# Startup Outcome Prediction 

Predicts whether a startup will be **acquired** or **closed** from 18 plain-language inputs (funding, investors, milestones, location, industry). Built as an end-to-end AIOps project: reproducible data + training pipeline (DVC), tracked experiments (MLflow), a REST backend (FastAPI), a separate non-technical web UI (Streamlit), and live monitoring with drift detection (Prometheus + Grafana), all containerised with Docker Compose. No cloud services are used.

| | |
|---|---|
| **Target** | `acquired` (1) vs `closed` (0) — approved target. Not a forecast of the next funding round. |
| **Data** | 923 startups → 922 after de-duplication (64.6 % acquired) |
| **Served model** | Random forest, chosen on 5-fold CV ROC-AUC (0.8175) among logistic regression, random forest, gradient boosting |
| **Held-out test** | ROC-AUC **0.838**, accuracy 0.746, precision 0.829, recall 0.767, F1 0.797 (185 startups, touched once) |
| **Latency** | ≈ 24 ms median per single prediction |
| **Tests** | 40 automated, all passing (`reports/test_report.xml`) |

> Metrics are measured on a split made **before** any fitting, with **no oversampling**, so they describe real startups only. With ~185 test rows expect roughly ±0.05 uncertainty on any single metric.

## Quick start (local)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

dvc repro                 # prepare -> train -> test  (writes models/, reports/, mlruns/)
make serve                # backend  http://localhost:8000  (docs at /docs)
make ui                   # frontend http://localhost:8501  (new terminal)
make mlflow-ui            # experiments http://localhost:5000 (optional)
```
Without DVC you can run the stages directly: `make data && make train`.

## Quick start (Docker)
```bash
docker compose --profile train run --rm trainer   # builds data + model once
docker compose up --build
```
| Service | URL |
|---|---|
| Frontend | http://localhost:8501 |
| Backend API / docs | http://localhost:8000 · `/docs` |
| MLflow | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (dashboard *Startup Outcome Service*; admin/admin) |

The frontend reaches the backend **only** through `BACKEND_URL` (REST). The backend image contains no UI code and the frontend image contains no model code.

## Repository layout
```
params.yaml              single source of truth for all pipeline parameters
dvc.yaml / dvc.lock      pipeline DAG: prepare -> train -> test
MLproject, python_env.yaml   MLflow Project (identical dev/test environments)
data/raw/startup_data.csv.dvc   raw data, versioned by DVC (local remote .dvc-storage)
src/
  config.py              params + paths + logging
  data/                  schema.py (contract + validation), prepare.py (data pipeline)
  features/              engineering.py (StartupFeatureBuilder, shared by train + serve)
  training/              train.py (CV, comparison, selection, MLflow logging, artifacts)
  tracking/              tracker.py (MLflow wrapper), run_log.py (pipeline run history)
  serving/               api.py, predictor.py, schemas.py, field_meta.py
  monitoring/            drift.py, metrics.py
frontend/                Streamlit app: client.py (REST only), ui.py, app.py, pages/
monitoring/              prometheus.yml, alerts.yml, Grafana provisioning + dashboard
tests/                   40 tests (unit, API, pipeline)
docs/                    the five mandatory documents + diagrams
Dockerfile.backend, Dockerfile.frontend, docker-compose.yml, Makefile
```

## Mandatory documents
1. [Architecture diagram + block explanation](docs/01_architecture.md)
2. [High-level design](docs/02_high_level_design.md)
3. [Low-level design (endpoints, I/O)](docs/03_low_level_design.md)
4. [Test plan, cases and report](docs/04_test_plan_and_cases.md)
5. [User manual (non-technical)](docs/05_user_manual.md)

## Reproducibility
Every experiment is tied to **(git commit, training-data SHA-256, MLflow run id)**, stored as MLflow tags and in `models/model_metadata.json`, and shown by `GET /model/info` and the Pipeline Console. To reproduce: `git checkout <commit> && dvc pull && dvc repro`.

## Data source
The dataset is a public startup dataset (923 companies, Crunchbase-derived fields) published on Kaggle by Mohammed Hasnaine, used here for educational purposes under its original terms.

## Limitations
Small dataset; US-centric and historical; label is acquired vs closed; explanations describe the model, not causes; input-drift alerts indicate change, not accuracy loss. Not suitable as the sole basis for investment, lending or hiring decisions.

## Optional uncertainty estimate

After installing the pinned project dependencies, run `python scripts/bootstrap_ci.py --n-resamples 5000` to calculate a seeded percentile-bootstrap 95% confidence interval for held-out ROC-AUC. The script requires scikit-learn 1.9.1, matching the serialized model artifact.

