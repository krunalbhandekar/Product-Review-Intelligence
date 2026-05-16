.PHONY: venv install dev test lint typecheck format clean

PY := .venv/bin/python
PIP := .venv/bin/pip

venv:
	python3 -m venv .venv
	$(PIP) install -U pip

install: venv
	$(PIP) install -e ".[dev]"

dev:
	$(PY) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

format:
	$(PY) -m ruff format .

typecheck:
	$(PY) -m mypy app

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache
