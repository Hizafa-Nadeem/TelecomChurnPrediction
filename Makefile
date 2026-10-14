PYTHON ?= .venv/Scripts/python

.PHONY: setup data split train tune calibrate evaluate gate explain package serve docker-build docker-run loadtest lint typecheck test check

setup:
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e . --no-deps

data:
	$(PYTHON) -m churn.data --config configs/params.yaml --raw-dir data/raw --out-dir data/interim

split:
	$(PYTHON) -m churn.split --config configs/params.yaml --in-dir data/interim --out-dir data/processed

train:
	$(PYTHON) -m churn.train --config configs/params.yaml --data-dir data/processed

tune:
	$(PYTHON) -m churn.tune --config configs/params.yaml --data-dir data/processed --out-dir reports/tuning

calibrate:
	$(PYTHON) -m churn.calibrate --config configs/params.yaml --data-dir data/processed --tuning-dir reports/tuning --out-dir reports/calibration --model-dir models

evaluate:
	$(PYTHON) -m churn.evaluate --config configs/params.yaml --data-dir data/processed --model-dir models --calibration-dir reports/calibration --out-dir reports/evaluation

gate:
	$(PYTHON) -m churn.gate --config configs/params.yaml --result reports/evaluation/test_evaluation.json

explain:
	$(PYTHON) -m churn.explain --config configs/params.yaml --data-dir data/processed --model-dir models --out-dir reports/explainability

package:
	$(PYTHON) -m churn.package --config configs/params.yaml --data-dir data/processed --model-dir models --calibration-dir reports/calibration --evaluation-dir reports/evaluation --export-dir build/model

serve:
	$(PYTHON) -m uvicorn api.main:app --host 127.0.0.1 --port 8000

docker-build:
	docker build -t churn-api:local .

docker-run:
	docker run --rm -p 8000:8000 churn-api:local

loadtest:
	$(PYTHON) -m locust -f tests/load/locustfile.py --headless -u 10 -r 5 -t 60s --host http://127.0.0.1:8000 --csv reports/load/locust

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check .

typecheck:
	$(PYTHON) -m mypy src api tests

test:
	$(PYTHON) -m pytest

check: lint typecheck test
