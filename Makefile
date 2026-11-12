.PHONY: install check lint format typecheck test test-all models bench clean

install:
	uv sync

check: lint typecheck test

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

format:
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

typecheck:
	uv run mypy

test:
	uv run pytest

test-all:
	uv run pytest -m ''

models:
	uv run --group bench python scripts/export_models.py

bench:
	uv run --group bench tessera bench all

clean:
	rm -rf build dist .mypy_cache .pytest_cache .ruff_cache
