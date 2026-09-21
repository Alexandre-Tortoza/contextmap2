PYTHON ?= python

SMOKE_VENV ?= .smoke-venv

.PHONY: install lint format format-check typecheck test check build smoke-install clean

install:
	$(PYTHON) -m pip install -e '.[dev]'

lint:
	ruff check src tests

format:
	ruff format src tests

format-check:
	ruff format --check src tests

typecheck:
	mypy src tests

test:
	pytest

check: lint format-check typecheck test

build:
	$(PYTHON) -m build

# Reproduz localmente o smoke de instalação da CI: constrói a wheel e a testa em um
# venv novo, só com as dependências base, nunca no ambiente de desenvolvimento.
smoke-install:
	rm -rf dist $(SMOKE_VENV)
	$(PYTHON) -m build
	$(PYTHON) -m venv $(SMOKE_VENV)
	$(SMOKE_VENV)/bin/python -m pip install --no-cache-dir dist/*.whl
	$(SMOKE_VENV)/bin/python tests/packaging/smoke_installed_package.py

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache build dist $(SMOKE_VENV) *.egg-info src/*.egg-info
