# 2. High-Level Design (HLD)

## 1. Problem and objective
**Problem.** Investors and founders want an early, data-driven signal of whether a startup is on a path to a successful exit.
**Target (approved).** Binary classification: **`acquired` (1) vs `closed` (0)**.
**Objective.** Build an end-to-end AIOps application: a reproducible data and training pipeline, tracked experiments, a served model behind a REST API, a non-technical web UI, and live monitoring including data drift.
**Success metrics.** Primary: ROC-AUC on a held-out test set (target ≥ 0.80). Secondary: F1, accuracy; operational: inference latency, pipeline throughput, zero unhandled errors.

## 2. Data
- **Source:** public startup dataset (923 rows, 49 raw columns): company profile, funding history, investors, milestones, location, industry and final status. It is used here for educational purposes.
- **After cleaning:** 922 unique startups (1 duplicate id removed); 64.6 % acquired / 35.4 % closed.
- **Inputs used by the model (18):** funding timing, rounds and totals, investor flags (VC/angel/Series A–D), relationships, milestones, state, industry.
- **Deliberately dropped:** names, ids, coordinates, zip codes, and `closed_at` (it would leak the label).

## 3. Design principles and rationale

| # | Decision | Rationale |
|---|---|---|
| D1 | **Frontend and backend are separate services linked only by configurable REST** | Required loose coupling; independent deployment/scaling; UI cannot depend on model internals |
| D2 | **One scikit-learn Pipeline = features + encoding + model, served as-is** | Eliminates train/serve skew; API takes raw human inputs |
| D3 | **Split before any fitting; no SMOTE/oversampling; class imbalance handled by `class_weight`** | Test metrics reflect real startups only; avoids optimistic leakage |
| D4 | **Model selected on cross-validated ROC-AUC of the training set, never on the test set** | The test set is used once, for an honest estimate |
| D5 | **Three transparent candidate models** (logistic regression, random forest, gradient boosting) | Small tabular dataset; interpretable, fast, strong baselines; comparison is itself an experiment |
| D6 | **`params.yaml` is the single source of truth** | DVC re-runs only affected stages; every experiment is reproducible from commit + params |
| D7 | **Custom stage-based data pipeline** (ingest → validate → clean → label → split → write), each stage timed | Dataset is ~1 k rows; Ray/Spark/Kubernetes would add overhead without benefit. Stage timings and rows/s are measured and exposed |
| D8 | **MLflow behind a thin tracker wrapper** | Training code is independent of tracking back-end; real MLflow is used whenever installed |
| D9 | **Validation at the API boundary** (ranges, cross-field rules, unknown fields rejected) with uniform error bodies and request ids | Foolproof inputs; debuggable failures |
| D10 | **Occlusion-based explanations** (replace each input by a typical value, observe probability change) | Model-agnostic, dependency-free, understandable to non-technical users |
| D11 | **Drift monitoring on live inputs** (KS test + total-variation distance) | Labels are not available at prediction time; input drift is the practical early-warning signal |
| D12 | **Prometheus + Grafana, no cloud services** | Course constraint; open-source standard stack |

## 4. Technology stack
Python 3.11 · pandas/NumPy · scikit-learn · **DVC** · **Git** · **MLflow** · FastAPI + Pydantic + Uvicorn · Streamlit + Plotly · prometheus-client · **Prometheus** · **Grafana** · Docker + Docker Compose · pytest.

## 5. Quality attributes
| Attribute | How it is addressed |
|---|---|
| Reproducibility | DVC DAG + `dvc.lock`, fixed seeds, commit hash + data SHA-256 + MLflow run id on every model |
| Reliability | Model failure → `503` with message, process stays up; `/health` reports `degraded` |
| Observability | Structured access logs with request ids, 15+ Prometheus metrics, Grafana dashboard, alert rules |
| Usability | Form generated from API schema, examples, plain-language results, friendly error messages |
| Testability | Predictor/drift/features have no web dependency; 40 automated tests |
| Security | Strict input validation, `extra="forbid"`, no secrets in code, no cloud services |

## 6. Constraints and limitations
- Small dataset (922 rows) → wide confidence intervals; metrics are estimates, not guarantees.
- The label is **acquired vs closed**; it is not a forecast of the next funding round.
- Historical data may not represent current markets; geographic coverage is US-centric.
- Explanations describe the model, not causes. Not for sole use in investment or lending decisions.
