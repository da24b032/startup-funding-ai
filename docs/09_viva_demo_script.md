# Viva demonstration script (about 7–9 minutes)

Use this as a speaking guide, not a memorised monologue. Keep the application, API docs, MLflow, Prometheus and Grafana open in separate browser tabs before the demo.

## 1. Problem and application (45 seconds)

"This project predicts whether a startup in the historical dataset is more likely to be acquired or closed. It is a binary classification task. The application accepts startup funding, investor, milestone, location and industry attributes, then returns a prediction with a probability and an explanation. It is a decision-support demonstration, not investment advice."

## 2. Architecture and loose coupling (60 seconds)

Show `docs/architecture.png`.

"The Streamlit frontend and FastAPI inference backend are separate services. The frontend contains only REST client code and uses the configurable `BACKEND_URL`; it does not load the model or import the training pipeline. The backend owns validation, inference, explanations, batch prediction, drift checks and metrics. Docker Compose runs the frontend and backend independently alongside MLflow, Prometheus and Grafana."

## 3. Live prediction (60 seconds)

Open Predict, use the supplied example, and submit.

"The form is generated from the API schema and sends a JSON request to FastAPI. The backend validates the fields, applies the same saved preprocessing/model pipeline used during training, and returns the predicted class, class probabilities and explanation."

If asked about probability: "It is the model's estimated class probability, not a calibrated guarantee or a causal explanation."

## 4. Data pipeline and DVC (75 seconds)

Show Pipeline Console and the DVC DAG; if time allows run `dvc repro` in a terminal.

"DVC manages the prepare → train → test DAG. The prepare stage is a six-stage custom pipeline: ingest, validate, clean, label, split and write. It records stage duration and rows per second. For this 923-row dataset, preparation takes about 0.0385 seconds and reaches about 24,000 rows per second. I retained a custom pipeline rather than Ray/Spark because distributed startup overhead would dominate this small workload; the rubric accepts a custom pipeline, and the stages remain modular for future scaling."

## 5. Model selection and evaluation (75 seconds)

Show model comparison and `reports/metrics.json`.

"I compared logistic regression, random forest and gradient boosting. Model selection used the highest mean five-fold cross-validation ROC-AUC on training data only. Random forest was selected with CV ROC-AUC 0.8175. The held-out test ROC-AUC is 0.8378 on 185 records. Gradient boosting had a slightly higher test score, but I did not select it using the test set because that would leak test information into model selection."

"The older 0.909 result came from a different, oversampled evaluation setup. The current result is more defensible because the split occurs before fitting and no oversampling is applied to the test data."

## 6. Experiment tracking and reproducibility (60 seconds)

Show MLflow experiment, parent run, nested candidate runs, registered model and tags.

"MLflow records parameters, CV/test metrics, timing, plots and metadata. Every model run is linked to the Git commit, training-data SHA-256 and MLflow run ID. DVC tracks pipeline inputs and outputs, so a specific experiment can be reconstructed from its code and data versions."

## 7. Monitoring and drift (75 seconds)

Show Prometheus targets, Grafana dashboard and Monitoring page after submitting several predictions.

"The API exports request, error, prediction, latency, model and drift metrics. Prometheus scrapes the backend every 15 seconds; Grafana visualises them and evaluates alert rules. Drift detection compares a recent input window with a training baseline using KS tests for numeric features and total-variation distance for categorical features. Drift means the input distribution changed; it does not prove model accuracy dropped because ground-truth labels are not immediately available at inference time."

## 8. Quality, limitations and close (45 seconds)

Show test report and documents.

"The project has 40 automated tests covering data validation, feature engineering, drift, run logging, API contracts and training artifacts. The main limitations are the small historical dataset, US-centric coverage and the simplified acquired-versus-closed target. The system demonstrates a reproducible local AI application lifecycle, but should not be used as the sole basis for a real investment decision."

## Likely challenge questions

- **Why is the test score lower than 0.909?** The old result used a different/oversampled evaluation. This version avoids resampling leakage and evaluates on a held-out set.
- **Why not select gradient boosting?** It had a higher test score, but selection was based on training-only CV to avoid tuning to the test set.
- **Does drift mean accuracy has fallen?** No. It signals feature-distribution change; labelled outcomes are needed to measure performance degradation.
- **Why custom pipeline instead of Ray/Spark?** 923 rows; ~0.0385 seconds; ~24k rows/s; distributed startup cost is not justified, and the rubric allows a custom pipeline.
- **How do you reproduce a model?** Check out its Git commit, restore the matching DVC data, then reproduce the DVC stages; MLflow run ID and data hash provide traceability.
