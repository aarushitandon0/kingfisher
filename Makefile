# Kingfisher — developer entry points.
# Requires: docker compose, and a Python 3.11 env with `pip install -e ".[dev]"`.

.DEFAULT_GOAL := help
.PHONY: help dirs db-up db-down db-migrate db-revision test lint format api

PY ?= python

help:  ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

dirs:  ## Recreate the gitignored data directories (fresh clone)
	mkdir -p data/raw data/interim data/processed

db-up:  ## Start PostGIS and wait until it is accepting connections
	docker compose up -d db
	@echo "waiting for postgis..."
	@until docker compose exec -T db pg_isready -U kingfisher -d kingfisher >/dev/null 2>&1; do sleep 1; done
	@echo "kingfisher db ready on port $${POSTGRES_PORT:-5432}"

db-down:  ## Stop PostGIS (volume is preserved)
	docker compose down

db-migrate: db-up  ## Apply all migrations
	alembic upgrade head

db-revision:  ## New migration: make db-revision m="add foo"
	alembic revision -m "$(m)"

test:  ## Run the test suite
	pytest

lint:  ## Ruff + mypy, no writes
	ruff check .
	ruff format --check .
	mypy .

format:  ## Apply ruff formatting and safe fixes
	ruff check --fix .
	ruff format .

api:  ## Run the API locally
	uvicorn api.main:app --reload --port 8000
