.PHONY: setup dev test lint format typecheck build docker-build docs clean release

PYTHON ?= python3
VENV ?= .venv

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -U pip
	$(VENV)/bin/pip install -e ".[gpu,eval,dev]"
	$(VENV)/bin/pre-commit install
	ollama pull qwen2.5:14b-instruct || true
	ollama pull bge-m3 || true

dev:
	$(VENV)/bin/uvicorn nexusmind.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload

test:
	$(VENV)/bin/pytest

lint:
	$(VENV)/bin/ruff check .
	$(VENV)/bin/ruff format --check .

typecheck:
	$(VENV)/bin/mypy

format:
	$(VENV)/bin/ruff check --fix .
	$(VENV)/bin/ruff format .

build:
	$(VENV)/bin/python -m build

docker-build:
	docker build -t nexusmind:latest .

docs:
	$(VENV)/bin/mkdocs serve

clean:
	rm -rf $(VENV) build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

release:
	$(VENV)/bin/python scripts/release.sh