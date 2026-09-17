PYTHON ?= python

.PHONY: install lint format format-check typecheck test check build clean

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

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache build dist *.egg-info src/*.egg-info
