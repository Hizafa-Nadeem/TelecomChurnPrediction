PYTHON ?= .venv/Scripts/python

.PHONY: setup data split repro lint typecheck test check

setup:
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e . --no-deps

data:
	$(PYTHON) -m churn.data --config configs/params.yaml --raw-dir data/raw --out-dir data/interim

split:
	$(PYTHON) -m churn.split --config configs/params.yaml --in-dir data/interim --out-dir data/processed

# DVC runs `python` from PATH, so put the venv first.
repro:
	PATH="$(CURDIR)/.venv/Scripts:$$PATH" $(PYTHON) -m dvc repro

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check .

typecheck:
	$(PYTHON) -m mypy src tests

test:
	$(PYTHON) -m pytest

check: lint typecheck test
