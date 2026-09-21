# Kingfisher - developer entry points.
# Requires: docker compose, and a Python 3.11 env with `pip install -e ".[dev]"`.

.DEFAULT_GOAL := help
.PHONY: help dirs data data-list smoke smoke-sat smoke-weather l0 observability l1 l1-estimate l1-probe l2 static asissued exposure s2-shift dataset train evaluate nh-export ealstm-smoke ealstm-train ealstm-status colab-bundle hindcast h2h anomaly db-up db-down db-migrate db-revision test lint format api

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

l1:  ## L1: Sentinel-2 plan in config/sentinel2.yaml (make l1 max_pu=3000)
	$(PY) -m pipeline.l1_satellite --city $(or $(city),coimbra) $(if $(max_pu),--max-pu $(max_pu),)

l1-probe:  ## L1: fetch 2016-2017 for ONE reach and report usable rows (no DB write)
	$(PY) -m pipeline.l1_satellite --city $(or $(city),coimbra) --probe-pre-2018

s2-shift:  ## L1 QA: level shift in the indices at the Sentinel-2C platform change
	$(PY) scripts/check_s2_platform_shift.py --city $(or $(city),coimbra)

l2:  ## L2: Open-Meteo drivers at catchment centroids (run after l1 for upstream state)
	$(PY) -m pipeline.l2_drivers --city $(or $(city),coimbra)

static:  ## L2: static catchment attributes (land-cover adapter chain)
	$(PY) -m pipeline.l2_static --city $(or $(city),coimbra)

asissued:  ## L2: as-issued ECMWF IFS 00 UTC runs (budgeted, resumable) -> future-driver table
	$(PY) -m pipeline.asissued_weather --city $(or $(city),coimbra) --fetch --build

exposure:  ## L4: OSM + GHS-POP exposure within the buffer of every reach -> exposure_features
	$(PY) -m pipeline.exposure_features --city $(or $(city),coimbra)

dataset:  ## Modelling frame + walk-forward splits + data summary
	$(PY) -m pipeline.build_dataset --city $(or $(city),coimbra)

train:  ## L3: LightGBM quantile baseline - walk-forward fits + production fit + SHAP
	$(PY) -m models.baseline_gbm --city $(or $(city),coimbra)

evaluate:  ## L3: walk-forward metrics (A/B x oracle/as-issued) -> metrics.json + forecasts table
	$(PY) -m models.evaluate --city $(or $(city),coimbra)

nh-export:  ## P5.1: modelling frame -> NeuralHydrology GenericDataset (data/processed/nh)
	$(PY) -m pipeline.export_neuralhydrology --city $(or $(city),coimbra)

ealstm-smoke:  ## P5.1: 1-epoch EA-LSTM end-to-end check (CPU, ~1.5 min)
	$(PY) -m models.ealstm smoke

ealstm-train:  ## P5.1: 2 targets x 5 seeds on this machine (CPU: hours - prefer Colab)
	$(PY) -m models.ealstm train --target all --skip-trained

ealstm-status:  ## P5.1: which EA-LSTM members are trained
	$(PY) -m models.ealstm status

colab-bundle:  ## P5.1: dist/kingfisher_colab.zip for notebooks/ealstm_colab.ipynb
	$(PY) scripts/make_colab_bundle.py

hindcast:  ## P5.1: EA-LSTM simulations at every OK obs date >= 2024 (needs trained members)
	$(PY) -m models.ealstm hindcast --city $(or $(city),coimbra)

h2h:  ## P5.4: head-to-head, assimilation + calibration, production gate, sensitivity
	$(PY) -m models.head_to_head --city $(or $(city),coimbra)

anomaly:  ## P5.3: weather-explained anomaly detector (after h2h)
	$(PY) -m models.anomaly --city $(or $(city),coimbra)

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
