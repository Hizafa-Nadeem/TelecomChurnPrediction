PYTHON ?= .venv/Scripts/python

.PHONY: setup lint typecheck test check

setup:
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e . --no-deps

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check .

typecheck:
	$(PYTHON) -m mypy src tests

test:
	$(PYTHON) -m pytest

check: lint typecheck test
