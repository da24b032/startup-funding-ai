.PHONY: install generate train train-all serve test lint format docker-up clean mlflow-ui

install:
	pip install -e ".[dev]"

generate:
	python -m src.cli generate

train:
	python -m src.cli train --model random_forest

train-all:
	python -m src.cli train --all

serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns --port 5000

test:
	pytest

lint:
	ruff check src/ tests/

format:
	black src/ tests/
	ruff check --fix src/ tests/

docker-up:
	docker-compose up --build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/ .pytest_cache/ .ruff_cache/
	rm -f models/*.joblib models/*_meta.json data/raw/*.csv data/drift_baseline.json
