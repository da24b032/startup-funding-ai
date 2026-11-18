# User Manual (Non-Technical)

## 1. What the application does

The application demonstrates machine-learning analysis of startup outcomes. It classifies a startup as **Acquired** or **Closed** and presents supporting prediction and analysis information.

**Important:** The model's actual target is acquisition versus closure. It should not be interpreted as a direct forecast of whether a startup will raise its next funding round.

## 2. Opening the interactive application

1. Start the project in the configured Python environment.
2. Open a terminal in the `Success-Prediction-System` folder.
3. Run:

   ```bash
   streamlit run app.py
   ```

4. Open the local web address printed by Streamlit (usually `http://localhost:8501`).
5. Wait for the dashboard to finish loading.

If the page does not open, check that the terminal process is still running and that the required Python packages and model files are present.

## 3. Using the dashboard

The application includes views for the following tasks:

- **Prediction:** Enter the startup information requested by the interface and submit it to obtain an outcome prediction.
- **Explanation:** Review the explanation view to understand which input factors contributed to the model's output. Explanations describe model behavior; they do not prove that a factor causes success or failure.
- **Model Comparison:** Review the reported performance of the evaluated models.
- **Dataset Insights:** Explore descriptive patterns and visualizations in the dataset.

The exact input fields and controls are defined by the running application. Use the labels and guidance shown on screen; do not substitute arbitrary values or assume that every field is optional.

## 4. Understanding the result

- **Acquired:** The model predicts class 1 (acquired).
- **Closed:** The model predicts class 0 (closed).
- **Probability:** The displayed probability is a model estimate, not a guarantee or a calibrated promise of a real-world outcome.
- **Explanation:** Feature attribution can help describe why the model produced a result, but it should be treated as decision support only.

The reported model evaluation is based on the dataset and evaluation procedure documented in the project. Performance on future startups may differ.

## 5. Using the API (for technical users)

The FastAPI service is separate from the Streamlit dashboard. When it is running on port 8000, open:

- `http://localhost:8000/docs` for interactive API documentation.
- `GET /health` to check service status.
- `GET /features` to retrieve the exact 41 required feature names.
- `POST /predict` for one prepared feature record.
- `POST /predict/batch` for 1–500 prepared records.
- `GET /metrics` for monitoring metrics.

The API requires all 41 numeric, engineered/scaled model features. It does not accept an arbitrary raw startup profile. See `03_low_level_design.md` for request and response structures.

## 6. Monitoring

Prometheus collects API metrics and Grafana displays them. In the supplied configuration, Prometheus scrapes the startup API at `host.docker.internal:8000/metrics`. The monitoring dashboards are intended for technical operators, not ordinary end users.

## 7. Troubleshooting

| Problem | What to check |
|---|---|
| Streamlit page does not load | Confirm the Streamlit process is running and use the local URL printed in the terminal |
| Missing model/file error | Confirm the project model and data artifacts are present at the configured paths |
| API health check fails | Confirm the startup API process is running and port 8000 is available |
| API returns HTTP 422 | Check that all 41 exact feature names are present, with no extra keys and numeric values |
| API returns HTTP 500 | Check the server terminal for the underlying model or runtime error |
| Prometheus target is DOWN | Confirm the API is reachable from Prometheus and the scrape target/port are correct |
| Prediction seems surprising | Review the model explanation and remember that predictions are estimates, not guarantees |

## 8. Responsible use

Use the application as an educational prototype and decision-support demonstration. Do not use its output as the sole basis for investment, lending, hiring or other high-impact decisions. The model may reflect limitations and biases in its source data.
