PYTHON ?= .venv/Scripts/python

.PHONY: setup data split train tune calibrate evaluate gate explain lint typecheck test check

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

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check .

typecheck:
	$(PYTHON) -m mypy src tests

test:
	$(PYTHON) -m pytest

check: lint typecheck test
