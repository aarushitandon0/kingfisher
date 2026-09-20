# Kingfisher

**Kingfisher watches the water continuously, warns before the water turns, and shows what
to change so it turns less often.**

An early-warning and resilience-planning system for urban streams, built for the
OneAquaHealth IEEE Global Hackathon 2026, Track 6: Resilience Informatics.

Three pillars: **Predict** (reach-level 10-day forecast with prediction intervals),
**Warn** (calibrated exceedance-probability alerts with driver attribution and exposure
context), **Adapt** (intervention and climate scenarios from cited literature
coefficients).

Design: [MASTERSPEC.md](masterspec.md). Data acquisition: [DATA_SOURCES.md](data_sources.md).
Working rules: [CLAUDE.md](claude.md).

---

## Status

Scaffolding only. No pipeline logic yet — see the day plan in MASTERSPEC.md §16.

## Quick start

```bash
cp .env.example .env          # fill in SH_CLIENT_ID / SH_CLIENT_SECRET
python -m venv .venv && . .venv/Scripts/activate   # or: uv venv
pip install -e ".[dev]"

make db-up                    # PostGIS 16 / 3.4 on localhost:5432, database "kingfisher"
make db-migrate               # apply migrations (creates every MASTERSPEC §5 table)
make test
make lint
make api                      # http://localhost:8000/health
```

## Data

Nothing in `data/raw/` is committed - it is gitignored, and re-acquired with one command.
`DATA_INVENTORY.md` records the provenance of every file (source URL, access date,
licence) and is committed.

```bash
python scripts/fetch_datasets.py --list          # manifest + what is already on disk
python scripts/fetch_datasets.py                 # ~800 MB, resumable, idempotent
python scripts/fetch_datasets.py --only osm_coimbra copdem
```

Open access, fetched automatically:

| Dataset | Use |
|---------|-----|
| OSM Coimbra bbox (Overpass JSON) | reaches, roads, exposure features - usable immediately |
| OSM Portugal extract (Geofabrik `.pbf`) | full-country source for pyrosm/osmium |
| Copernicus DEM GLO-30, 4 tiles | catchment delineation fallback while MERIT Hydro access is pending |
| ESA WorldCover 10 m | land cover; CLMS substitute and the Pune adapter's source |
| GHS-POP 3 arcsec | exposure denominator |
| Open-Meteo archive + forecast | driver history and the live path, cached per request |

Needs an account or a manual step (listed by `--list`, never silently skipped):
Sentinel-2 via CDSE, MERIT Hydro (password by request), CLMS, VIIRS, SNIRH stations.

## Gates

Both Day-0 go/no-go checks are scripts, and both print an unambiguous PASSED/FAILED line.

```bash
python scripts/smoke_test_weather.py       # Open-Meteo + cache replay
python scripts/smoke_test_sentinelhub.py   # Sentinel Hub Statistical API
```

The Sentinel Hub gate needs `SH_CLIENT_ID` / `SH_CLIENT_SECRET` in `.env`; without them
it prints exactly what to do and exits non-zero. If that gate does not print numbers,
fix the credentials before anything else - nothing downstream works without it.

## Layout

```
config/      cities/*.yaml, thresholds.yaml, intervention_coefficients.yaml
core/        settings, structured logging, disk cache, db session
scripts/     dataset fetcher and the two Day-0 smoke-test gates
pipeline/    L0 network · L1 satellite · L2 drivers/static · dataset assembly
models/      LightGBM baseline · SAGE-TS hybrid · anomaly · evaluation
engine/      alerts · scenarios · priorities · exposure   (pure functions, no I/O)
api/         FastAPI app, routes, Pydantic schemas, FHIR export
frontend/    React + MapLibre (Day 7)
migrations/  Alembic
data/        gitignored except reference/
```

`core/` is the one addition to the MASTERSPEC §15 layout: the structured logger, settings
and DB session are needed by `pipeline/`, `models/` and `api/` alike, and `engine/` stays
free of all three.

## Logging

Every pipeline module uses the same logger, and every stage accounts for its rows:

```python
from core.logging import get_logger, stage

log = get_logger(__name__)

with stage(log, "l1_satellite", city="coimbra") as s:
    s.record(rows_in=4012, rows_out=3788)
    s.drop(224, "NO_WATER_PIXELS")
```

`LOG_FORMAT=json` for machine-readable output; `console` for development.

## What Kingfisher does not do

- No disease or health-outcome prediction. Exposure pathways and proximity only.
- No declaration that water is safe or unsafe.
- No pollution-source attribution to a named polluter.
- No chemical concentration retrieval — optical proxies, not lab values.
- No replacement for professional monitoring. Kingfisher prioritises it.

Scenario outputs are planning estimates, not predictions, and not causal claims.

## Attribution

```
Contains modified Copernicus Sentinel data (2015–2026), processed by Kingfisher.
Contains Copernicus Land Monitoring Service information (Imperviousness Density,
  Riparian Zones, Urban Atlas).
Weather data © Open-Meteo.com (CC-BY-4.0), derived from ECMWF ERA5/ERA5-Land.
Map data © OpenStreetMap contributors, available under the Open Database Licence.
MERIT Hydro © Dai Yamazaki (University of Tokyo) — CC-BY-NC 4.0 / ODbL dual licensed.
Water quality reference data: SNIRH / Agência Portuguesa do Ambiente; EEA Waterbase.
VIIRS nighttime lights: Earth Observation Group, Colorado School of Mines.
Population: GHSL, European Commission Joint Research Centre.
```
