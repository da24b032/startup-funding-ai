.PHONY: serve-mlflow install data train pipeline serve ui test lint dag up down mlflow-ui clean

install:
	pip install -r requirements-dev.txt

data:            ## data engineering stage
	python -m src.data.prepare

train:           ## train + track + select model
	python -m src.training.train

pipeline:        ## full reproducible pipeline (DVC DAG)
	dvc repro

dag:
	dvc dag

serve:           ## backend API on :8000
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

serve-mlflow:    ## MLflow-native model server on :5001 (uses the registered model)
	MLFLOW_TRACKING_URI=file://$(PWD)/mlruns mlflow models serve -m models:/startup-outcome-classifier/latest --env-manager local --port 5001

ui:              ## frontend on :8501 (backend must be running)
	cd frontend && BACKEND_URL=http://localhost:8000 streamlit run app.py

mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns --port 5000

test:
	pytest --junitxml=reports/test_report.xml -q

lint:
	ruff check src tests frontend

up:
	docker compose up --build

down:
	docker compose down

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ htmlcov
