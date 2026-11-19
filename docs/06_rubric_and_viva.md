# 6. Evaluation-rubric checklist and viva preparation

## A. Rubric → evidence

### Demonstration (5)
| Rubric item | Where it is satisfied |
|---|---|
| Intuitive UX for non-technical users, foolproof, no UI errors | Form generated from `/schema`; drop-downs, toggles, bounded numbers; examples; plain-language results; friendly errors (`frontend/app.py`, `client.py`) |
| Colours, look-and-feel, responsive | `frontend/ui.py` CSS (result cards, mobile breakpoint), `.streamlit/config.toml` theme |
| User manual | `docs/05_user_manual.md` |
| Separate screen for the ML pipeline incl. data engineering | **Pipeline Console** page (flow DAG, data-engineering tab, training tab) |
| Pipeline management console: errors, failures, successes | **Run console** tab ← `reports/pipeline_runs.jsonl` ← `logged_stage` decorator; failures show the error message |
| Speed and throughput of the pipeline | Per-stage seconds and rows/s in `data_report.json`; headline throughput metric; Prometheus gauge |
| Orchestrating several tools into one UX | Console aggregates data from the API and links to MLflow, Grafana and Prometheus |

### Software engineering (5)
| Rubric item | Evidence |
|---|---|
| Design doc, HLD, LLD with endpoint I/O, architecture diagram | `docs/01`–`03`, `architecture.png` |
| **Strict loose coupling frontend ↔ backend via configurable REST** | `frontend/client.py` is the only network code; `BACKEND_URL` env var; separate Dockerfiles; the frontend image has no `src/` |
| Python style, logging, exception handling, comments, design adherence | `ruff` clean (`make lint`); `setup_logging`, request-id access logs; typed error classes and central handlers; docstrings in every module |
| APIs follow the spec | `tests/test_serving.py` checks the contract; the LLD mirrors `schemas.py` |
| Unit tests | 40 tests in `tests/` |
| Test plan, case list, report with pass/fail counts, acceptance criteria | `docs/04_test_plan_and_cases.md`, `reports/test_report.xml` |

### AIOps implementation (10)
| Rubric item | Evidence | Honest note |
|---|---|---|
| **Data engineering** pipeline and throughput | `src/data/prepare.py`: 6 timed stages, ~20 k rows/s; DVC stage `prepare` | **Custom pipeline** (the rubric allows "or Custom"). Ray/Spark/Kubernetes are not used: with 922 rows they would be pure overhead. |
| **CI with DVC**, DVC DAG | `dvc.yaml`, `dvc.lock`, `docs/dvc_dag.png`; `dvc repro` runs prepare → train → test | |
| Git / Git LFS / DVC versioning of code, data, models | Git for code; DVC for raw data (`.dvc`), processed data, model and reports; local DVC remote | Git LFS is not used; DVC covers data and model versioning. |
| **Experiment tracking**: metrics, params, artifacts, beyond autolog | Parent run + 3 nested runs; per-model params, CV and test metrics, latency, train seconds; tags `git_commit`, `train_data_sha256`; artifacts (plots, `params.yaml`, `data_report.json`, `feature_importance.json`); registered model | Everything is logged explicitly (no autolog). |
| **Prometheus instrumentation**, what is monitored, **Grafana** NRT | `src/monitoring/metrics.py` (16 metric families: HTTP, errors, predictions, latency, score distribution, drift, pipeline, model info); `monitoring/` (scrape config, 4 alert rules, provisioned 12-panel dashboard, 10 s refresh) | The Docker stack was not executed in the authoring sandbox: run checklist OPS-1…OPS-4 before the demo. |
| **MLflow APIfication** | Model logged in MLflow format and registered; `make serve-mlflow` or compose profile `mlflow-serve` expose `POST /invocations` | Verified locally (returned a prediction). The **UI-facing** API is FastAPI, as the rubric asks. |
| **MLproject** for identical dev/test environments | `MLproject` + `python_env.yaml` (`mlflow run .`) | Not executed here (it builds a fresh environment); run it once before the demo. |
| **FastAPI** for the UI-facing APIs | `src/serving/api.py` | |
| **Dockerised** backend and frontend, **docker compose** as two separate services | `Dockerfile.backend`, `Dockerfile.frontend`, `docker-compose.yml` | Not built in the sandbox (no Docker available). |

### Viva (5): see section B.

## B. Likely viva questions and crisp answers

**1. Why did you not use SMOTE?** Oversampling before the split leaks synthetic copies of test-set neighbours into training and inflates metrics. I split first, never resample, and use `class_weight="balanced"`. My test numbers are lower than an oversampled pipeline would show, but they are honest.

**2. How did you pick the model, and did you look at the test set?** Selection uses mean 5-fold CV ROC-AUC on the training set only. The test set is evaluated once per candidate for reporting. Gradient boosting actually scored higher on test (0.855 vs 0.838); I deliberately did not switch, because choosing by test score would make the test estimate optimistic. The CV gap between random forest (0.8175) and gradient boosting (0.8157) is far inside one standard deviation (~0.03), so they are statistically tied.

**3. What is train/serve skew and how do you avoid it?** Different code or fitted state in training and inference. The saved artifact is one scikit-learn Pipeline (feature builder → encoder → model) loaded by the API, so both paths run the same object.

**4. What would have leaked in the original data?** `closed_at` exists only for closed startups and would reveal the label. It is dropped, along with ids, names and coordinates.

**5. Explain the DVC DAG and what `dvc repro` does.** Stages declare deps, params and outs; DVC hashes them (`dvc.lock`) and re-runs only stages whose inputs changed. Changing `random_forest.n_estimators` re-runs `train` and `test` but not `prepare`.

**6. How do you reproduce the model behind a prediction?** `/model/info` returns the git commit, data SHA-256 and MLflow run id. Then `git checkout <commit> && dvc pull && dvc repro`.

**7. How is drift detected, and does drift mean accuracy fell?** Live requests fill a sliding window that is compared with a baseline saved at training: KS test (p < 0.05) for numeric inputs, total-variation distance (> 0.30) for categories. It is *input* drift; labels are unknown at serving time, so it signals "review or retrain", not "accuracy dropped". Testing many features raises false-alarm risk; the minimum-sample guard (30) and the alert's `for: 2m` mitigate it.

**8. Why is the frontend separate, and how do you prove it?** Loose coupling: independent deploy, scale and replace. The frontend image contains only `frontend/`, talks HTTP via `BACKEND_URL`, and shows a friendly message when the backend is down (UI-2).

**9. Why a custom data pipeline instead of Spark or Ray?** About 1 k rows: a distributed engine's start-up cost exceeds the whole job. The pipeline is stage-based and timed, so it could be ported stage by stage (for example to Ray Data) if the data grew.

**10. What about the small test set?** Report CV mean ± std next to test metrics, avoid over-interpreting 1–2 point differences, and (future work) use repeated CV or a bootstrap confidence interval.

**11. How are explanations computed, and what are their limits?** Occlusion: replace one input at a time with a typical training value and measure the change in predicted probability. It is cheap and model-agnostic, but it ignores interactions and is not causal.

**12. What happens when the model file is missing?** The API still starts; `/health` returns 503 `degraded`; prediction endpoints return 503 with a clear message; nothing crashes.

**13. Why a 0.5 threshold?** It is the default in `params.yaml`; with precision 0.83 and recall 0.77 it is reasonable, and it can be moved without retraining.

**14. Biggest limitations?** Small, historical, US-centric data; the label is acquired vs closed; wide metric uncertainty; not for sole decision-making.

## C. Before the demo
1. `git status` clean; `dvc status` says up to date; `dvc dag` ready to show.
2. Run `dvc repro` once live (or show it already green) to display the three stages.
3. Start the services, make ~40 varied predictions, then show Grafana, the Monitoring page and the Pipeline Console.
4. Break the CSV (OPS-6) to show a failed run in the console, then restore it with `dvc checkout`.
5. Open MLflow: parent run, nested runs, registered model, tags.
6. Run checklist items OPS-1…OPS-6 and `mlflow run .` once; they were not executable in the authoring sandbox.
