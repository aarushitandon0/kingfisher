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
models/      LightGBM baseline · EA-LSTM (NeuralHydrology) · assimilation · calibration ·
             anomaly · evaluation · head-to-head gate
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

## Head-to-head and the production gate (P5.4)

Decided on the 2024 validation fold before the test fold was scored (`results/gate.json`,
`results/head_to_head.json`). All numbers are **hindcast with observed weather - upper
bound on live skill**. Both models are assimilated and conformally calibrated identically.

| Validation CRPS (lower is better) | ealstm+assim | lightgbm+assim | winner |
|---|---:|---:|---|
| NDCI, 1-3 d | 0.0630 | 0.0625 | LightGBM |
| NDCI, 4-7 d | 0.0652 | 0.0668 | EA-LSTM |
| NDCI, 8-10 d | 0.0679 | 0.0701 | EA-LSTM |
| Turbidity index, 1-3 d | 5.039 | 4.603 | LightGBM |
| Turbidity index, 4-7 d | 4.844 | 4.684 | LightGBM |
| Turbidity index, 8-10 d | 4.833 | 4.817 | LightGBM |

**LightGBM stays production.** The EA-LSTM needed to win 2 of 3 buckets for both targets
with coverage no worse; it won 2 of 3 for NDCI (and missed the coverage test by 0.00006,
which is noise) and 0 of 3 for turbidity, so it fails on turbidity regardless. On the
spatial-holdout reaches (never seen by the EA-LSTM) it is also worse on validation
(turbidity CRPS 5.26 vs 4.93; 80% coverage 0.76 vs 0.81), although LightGBM variant A was
trained on those reaches and the comparison is not like-for-like. On test, LightGBM has one
more year of training data than the EA-LSTM (trained on 2016-2023 only). The EA-LSTM has
only 6 usable static attributes to tell reaches apart (3 of 9 were dropped: mostly empty
or a land-cover proxy), which limits it. Assimilation helps the EA-LSTM (CRPS 5.14 to 4.94
on turbidity) and does nothing for LightGBM.

## Scenario sensitivity check (P5.4)

The scenario engine perturbs catchment attributes and re-runs the model, so the model's
response to a perturbed attribute has to point the right way. Check: +10 percentage
points `imperviousness_pct` on every reach, re-infer the P50 on the first day of each
month in 2024, and compare (`make h2h`, `results/head_to_head.json` →
`scenario_sensitivity`). More impervious cover is expected to **raise** turbidity.

| Model | Target | Reaches up | Reaches down | No change | Median change |
|-------|--------|-----------:|-------------:|----------:|--------------:|
| LightGBM B (no reach identity, used for scenarios) | turbidity index | 25% | **62%** | 13% | -0.11 (-1.5%) |
| LightGBM A (production) | turbidity index | 8% | **75%** | 17% | -0.08 (-0.8%) |
| EA-LSTM (5-seed ensemble) | turbidity index | 36% | **64%** | 0% | -0.02 (-0.3%) |

**Both models move the wrong way on most reaches.** Adding impervious cover lowers the
turbidity forecast on 62% (LightGBM B), 75% (LightGBM A) and 64% (EA-LSTM) of the 353
reaches, by about 1% or less. The EA-LSTM is closer to flat than LightGBM but still points
the wrong way. The model has
learned a between-reach association, not the physical response. Only 41-51 observable
reaches carry targets, most of them on the main river (37 of the 41 training reaches are
Strahler order 4-5; mean catchment ~1,500 km²) with mostly low imperviousness (mean 8.6%,
range 3-51% across the training reaches), so imperviousness is confounded
with everything else that differs between those reaches. Neither model passes this check, so
model re-inference must not be used to size the imperviousness effect of an
intervention. Scenario effect sizes come from `config/intervention_coefficients.yaml`
(cited literature), as the design requires, and not from the model.

### The formal response check (MASTERSPEC 9.6)

`make response-check`: each static attribute a lever perturbs is moved -1 SD and +1 SD on
every held-out reach-day of the 2025-26 test fold and re-inferred with LightGBM B trained
through 2024. Effect = change in mean P50 per 2-SD swing, in target SDs; negligible below
0.02 (fixed in config before the check was first run).
`results/scenario_response_check_coimbra.json`.

| Attribute | Target | Effect (target SD) | Sign vs literature | Verdict |
|-----------|--------|-------------------:|--------------------|---------|
| imperviousness_pct | turbidity index | -0.033 | wrong | WRONG_SIGN |
| imperviousness_pct | NDCI | -0.315 | wrong | WRONG_SIGN |
| riparian_width_m | turbidity index | +0.014 | wrong | NEGLIGIBLE |
| riparian_width_m | NDCI | +0.099 | wrong | WRONG_SIGN |

The model is not trusted for any static lever. The scenario engine therefore reports
permeable paving, green roofs and riparian buffers as **NOT_ESTIMABLE** (with this check
shown beside the result), because the coefficient table has no cited effect on the stream
state itself that could be applied directly. Detention basins and street sweeping act on
weather drivers and go through the model. Both have honest near-zero central effects in the
literature, and the results show exactly that.

## The live run: forecast -> alerts

```bash
make live-weather       # the ECMWF IFS 00 UTC run issued on the frame's last day (1 request)
make forecasts-latest   # re-issue the production forecast on that run -> forecasts (LIVE)
make alerts             # guardrails + SHAP + exposure -> alerts + alert_runs
```

The production forecast is issued on the last day of the archive. The target-day weather
for that forecast has not happened yet, so it comes from the ECMWF run issued that
morning. The live system would have exactly this run. The rows are labelled `LIVE`. If
the run has not been fetched, the forecast is still issued, labelled `ORACLE`, and every
row counts its missing weather features (`future_drivers_missing`). The forecast
endpoint then shows a note.

**Run of 2026-09-20 (Coimbra):** 706 candidates (353 reaches x 2 variables).

| Outcome | Count | Why |
|---|---:|---|
| ALERT | 1 | CMB-0145, turbidity index, peak P(exceed) 0.62 on 2026-09-30 |
| WATCH | 0 | |
| INSUFFICIENT_EVIDENCE | 604 | every driver-only reach: no seasonal threshold exists (never filled from neighbours) |
| suppressed: confidence | 58 | peak P(exceed) below 0.6 |
| suppressed: drift | 43 | recent residuals outside the calibration band |

Suppressed candidates have no severity, so they are not alert rows. `alert_runs` counts
them per guardrail, and `/api/alerts` returns those counts. A run that alerts on nothing
is still recorded, so the API never passes "no run" off as "no alerts".

**Drift residuals come from the walk-forward test-fold model.** The production model is
fitted up to the issue date, so its residuals on recent dates are in-sample and would
flatter it. Until it has issued forecasts of its own, the only out-of-sample residuals
come from the test-fold model (trained through 2024, 1-day lead, archive weather). In
Jul-Sep 2026 its turbidity observations fall outside P10-P90 more than half the time on
28 of 47 reaches. So the guardrail fires often. That errs toward suppression, and every
alert's `basis.drift_residual_source` says where its residuals came from.

## Values chosen by the team (not by the literature)

These are planning choices. They are fixed in config and do not come from a data fit:

- **Scenario interval widening factor 1.5** (`config/intervention_coefficients.yaml`): scenario interval
  width = 1.5 x max(its own width, the baseline width). This makes it strictly wider than
  the baseline interval.
- **Negligible-response cutoff 0.02 target SD** (`config/modelling.yaml`): fixed before
  the response check was first run.
- **Exposure weights for `/api/priorities`** (`config/priorities.yaml`): they rank
  where monitoring is worth most. They are labelled as a planning choice, not a risk score.

Scenario requests send `extent` (how much of a lever), not `magnitude`. In
`config/intervention_coefficients.yaml`, `magnitude` is the cited effect size, and a
request can never set it. A request that sends `magnitude` gets a 422 that says why.

**Spatial scope of scenarios.** A lever changes only the catchments of the selected
reaches. The treated area also lies inside the catchments of the reaches downstream, and
that effect is not propagated to them. Every scenario result states this.

## Attribution

```
Contains modified Copernicus Sentinel data (2015–2026), processed by Kingfisher.
Contains Copernicus Land Monitoring Service information (Imperviousness Density,
  Riparian Zones, Urban Atlas).
Weather data © Open-Meteo.com (CC-BY-4.0), derived from ECMWF ERA5/ERA5-Land and
  ECMWF IFS HRES forecasts.
Map data © OpenStreetMap contributors, available under the Open Database Licence.
MERIT Hydro © Dai Yamazaki (University of Tokyo) — CC-BY-NC 4.0 / ODbL dual licensed.
Water quality reference data: SNIRH / Agência Portuguesa do Ambiente; EEA Waterbase.
VIIRS nighttime lights: Earth Observation Group, Colorado School of Mines.
Population: GHSL, European Commission Joint Research Centre.
```
