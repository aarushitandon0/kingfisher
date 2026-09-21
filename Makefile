# Kingfisher - developer entry points.
# Requires: docker compose, and a Python 3.11 env with `pip install -e ".[dev]"`.

.DEFAULT_GOAL := help
.PHONY: help dirs data data-list smoke smoke-sat smoke-weather l0 observability l1 l1-estimate l2 static dataset train evaluate db-up db-down db-migrate db-revision test lint format api

PY ?= python

help:  ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

dirs:  ## Recreate the gitignored data directories (fresh clone)
	mkdir -p data/raw data/interim data/processed

data-list:  ## Show the dataset manifest and what is already on disk
	$(PY) scripts/fetch_datasets.py --list

data:  ## Download the open-access datasets into data/raw/ (resumable, ~800 MB)
	$(PY) scripts/fetch_datasets.py

smoke: smoke-sat smoke-weather  ## Run both Day-0 gates

smoke-sat:  ## GATE: Sentinel Hub Statistical API returns numbers
	$(PY) scripts/smoke_test_sentinelhub.py

smoke-weather:  ## GATE: Open-Meteo returns data and the cache hits on replay
	$(PY) scripts/smoke_test_weather.py

l0: db-migrate  ## L0 backbone: OSM -> reaches -> catchments -> PostGIS + GeoJSON
	$(PY) -m pipeline.l0_network --city $(or $(city),coimbra)

observability:  ## GATE: how many reaches Sentinel-2 can actually see (Day 1)
	$(PY) scripts/check_observability.py --city $(or $(city),coimbra)

l1-estimate:  ## L1: print the Sentinel Hub PU plan for a full run (spends nothing)
	$(PY) -m pipeline.l1_satellite --city $(or $(city),coimbra) --estimate

l1:  ## L1: Sentinel-2 observations (make l1 max_pu=5000; observable reaches first)
	$(PY) -m pipeline.l1_satellite --city $(or $(city),coimbra) $(if $(max_pu),--max-pu $(max_pu),)

l2:  ## L2: Open-Meteo drivers at catchment centroids (run after l1 for upstream state)
	$(PY) -m pipeline.l2_drivers --city $(or $(city),coimbra)

static:  ## L2: static catchment attributes (land-cover adapter chain)
	$(PY) -m pipeline.l2_static --city $(or $(city),coimbra)

dataset:  ## Modelling frame + walk-forward splits + data summary
	$(PY) -m pipeline.build_dataset --city $(or $(city),coimbra)

train:  ## L3: LightGBM quantile baseline - walk-forward fits + production fit + SHAP
	$(PY) -m models.baseline_gbm --city $(or $(city),coimbra)

evaluate:  ## L3: walk-forward metrics vs baselines -> results/metrics.json + figures
	$(PY) -m models.evaluate --city $(or $(city),coimbra)

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
