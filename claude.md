# CLAUDE.md — Kingfisher

Working context for Claude Code. Read this before touching anything. And pls dont commit anything on github yourself... i will do that manually

---

## What this is

**Kingfisher** — an early-warning and resilience-planning system for urban streams.
Built for the OneAquaHealth IEEE Global Hackathon 2026, **Track 6: Resilience Informatics**.
Deadline **1 Oct 2026, 09:30 IST**. This is a ten-day build.

Three pillars, and every piece of work belongs to one of them:

1. **Predict** — reach-level 10-day forecast of turbidity and chlorophyll proxies with
   prediction intervals
2. **Warn** — calibrated exceedance-probability alerts with driver attribution, guardrails
   and exposure context
3. **Adapt** — intervention and climate scenario engine using cited literature coefficients

Full design: `MASTERSPEC.md`. Data acquisition: `DATA_SOURCES.md`. Neither is optional
reading.

---

## Non-negotiable invariants

Violating any of these breaks the project's credibility with a judging panel that
includes a working freshwater ecologist.

1. **No LLM computes a number.** Ever. Models and deterministic code produce every value.
   An LLM may only phrase text around numbers that already exist. If you find yourself
   asking a model to estimate, aggregate or rank, stop.

2. **Never silently impute.** A missing reach-date is `NULL` with a `quality_flag`
   (`CLOUD`, `NO_WATER_PIXELS`, `OUT_OF_RANGE`). Never zero. Never an unlabelled
   forward-fill. If you interpolate, the interpolation is a flagged column.

3. **Fail loudly.** If a data source returns nothing, raise. A pipeline that produces a
   plausible-looking empty result is how you end up demoing a model trained on nothing.

4. **Cache every network response to disk** before it reaches a DataFrame, keyed by
   request parameters. We will re-run this pipeline fifty times and should hit the
   network once.

5. **`INSUFFICIENT_EVIDENCE` is a first-class output.** The system is allowed to refuse to
   alert. That refusal is displayed in the UI and counted in the metrics. Never coerce it
   into a low-confidence alert.

6. **Scenario effect sizes come from `config/intervention_coefficients.yaml`, never from
   the model.** Every coefficient carries a citation. An uncited coefficient must not be
   merged.

7. **No causal language on scenario outputs.** "Planning estimate", not "prediction".
   Intervals are widened on scenario runs and the caveat is shown, not hidden in a tooltip.

8. **No health outcome prediction.** We map exposure pathways — schools, playgrounds,
   footpaths, access points, population. We do not predict disease, and we do not declare
   water safe or unsafe.

9. **Driver aggregation is over the upstream contributing catchment polygon**, never a
   buffer around the reach. This is the hydrologically correct unit.

10. **Honest metrics only.** If the model loses to seasonal-naive, that goes in the
    results table. No cherry-picked splits, no quietly dropped reaches.

---

## Architecture in one breath

```
OSM waterways → reaches → MERIT Hydro upstream catchments
   ├── Sentinel-2 (Sentinel Hub Statistical API) → turbidity/NDCI/MNDWI + water_pixel_count
   └── Open-Meteo (archive for training, forecast for live) → API, antecedent dry days,
       first-flush index, temp×low-flow
        → pooled model with reach embedding → P10/P50/P90 + CDF
             ├── alert engine (probability + guardrails + SHAP attribution + exposure)
             ├── scenario engine (cited coefficients perturb drivers → re-infer)
             └── prioritisation (uncertainty × risk × exposure)
        → FastAPI → React + MapLibre
```

The model is **driver → state**, not pure time-series extrapolation. This is forced by
the mixed-pixel problem and it is what makes the scenario engine possible: catchment
attributes are *inputs*, so they can be perturbed.

---

## The constraint that shapes everything

**Sentinel-2 is 10m. Urban streams are often 3–8m wide.** Many reaches will have zero
clean water pixels.

Consequences you must respect in code:
- `water_pixel_count` is stored on every reach-date and is a first-class field
- `reaches.observable` distinguishes optically observable from driver-only reaches
- Driver-only reaches are still forecast, but are capped at `WATCH` severity — never `ALERT`
- The observable/unobservable split is **reported**, not hidden. It is a genuine finding
  about the limits of satellite monitoring for urban streams.

---

## Stack

**Backend:** Python 3.11, FastAPI, PostGIS, GeoPandas, Shapely, `sentinelhub-py`,
`pysheds`, LightGBM, PyTorch, `neuralhydrology`, `captum`, SHAP
**Frontend:** React + TypeScript + Vite, MapLibre GL, Zustand, Tailwind, Recharts
**Run:** `docker compose up` for Postgres/PostGIS; `uv` or `venv` for Python

Frontend scaffolding is lifted from the Heat Surgeon project (MapLibre + Zustand + Vite).
Do not rebuild a basemap from scratch.

---

## Conventions

- **Type hints everywhere.** Pydantic models at every API boundary.
- **Pure functions in `engine/`.** No I/O, no DB, no network. Alerts, scenarios and
  prioritisation must be testable without a server — the same separation used in
  ConceptGraph's graph engine and SwasthyaSetu's `ml/` boundary.
- **Config over constants.** Cities in `config/cities/*.yaml`, thresholds in
  `config/thresholds.yaml`, coefficients in `config/intervention_coefficients.yaml`.
  Adding Pune should require a YAML file and a land-cover adapter, nothing else.
- **`reach_id` is the universal key.** Stable, string, prefixed by city (`CMB-0041`,
  `PUN-0007`).
- **All timestamps UTC.** Dates are dates, not datetimes, for daily aggregates.
- **Structured logging.** Every pipeline stage logs rows in, rows out, rows dropped and why.

---

## Testing

Priority order — if time is short, the top three are non-negotiable:

1. **Guardrail tests.** Break each guardrail deliberately and assert the suite catches it.
   (This is the Razorpay pattern: 30 tests, every guardrail broken on purpose.)
2. **Leakage tests.** Assert no future information enters any training fold. Walk-forward
   splits verified programmatically.
3. **Quality-flag tests.** Assert missing data never becomes zero and never becomes an
   unlabelled fill.
4. Feature-engineering unit tests (API decay, antecedent dry days, first-flush index)
   against hand-computed fixtures.
5. Scenario tests: every coefficient applied has a citation; scenario CIs are wider than
   baseline CIs.
6. API contract tests against the Pydantic schemas.

---

## Do not

- Do not download Sentinel-2 granules. Use the Statistical API. (~1GB per granule ×
  hundreds = your whole timeline.)
- Do not derive flow direction from a raw DEM. MERIT Hydro ships it precomputed.
- Do not train one model per reach. Pool across reaches with a learned embedding —
  ~500 timesteps per reach is far too thin otherwise.
- Do not add features after Day 8 18:00. Feature freeze is binding.
- Do not commit `.env`, anything under `data/raw/`, or any credential.
- Do not let the scenario engine ask the model for intervention effects.
- Do not write new visual design language. Match the Heat Surgeon scaffolding.

---

## Current state

Update this section as you go so a fresh session knows where things stand.

```
Day 0  — [ ] accounts, smoke tests
Day 1  — [ ] L0 network + catchments + observability gate
Day 2  — [x] L1 satellite + L2 drivers
           L1 plan in config/sentinel2.yaml: water indices for the 51 observable reaches,
           2018-2026 (14,141 OK obs); riparian NDVI = one July window per reach-year, all
           353 reaches -> riparian_ndvi_window. Turbidity = ACOLITE S2 MSI B4 Nechad coef
           (cited in sentinel2.yaml), displayed "turbidity index", no unit. 2016-17 probed
           on CMB-0056 only (20 + 37 OK) - not fetched for the rest. Live forecast pinned
           to ecmwf_ifs; as-issued 00 UTC runs via `make asissued` (Open-Meteo budgeted:
           resumes where it stopped). S2C shift check: scripts/check_s2_platform_shift.py.
Day 3  — [x] LightGBM baseline + walk-forward eval        ← first shippable system
           Variants A (reach_id) / B (no reach identity), 7 quantiles, urban_fraction
           dropped while PROXY_*, evaluated ORACLE vs ASISSUED weather; `make evaluate`
           writes metrics.json + forecasts table. Thresholds: per-reach seasonal (engine/
           thresholds.py). After L1 riparian finishes: `make static dataset train evaluate`.
Day 4  — [x] alert engine + guardrails + exposure
           engine/alerts.py (5 guardrails, INSUFFICIENT_EVIDENCE first-class), tests/
           test_guardrails.py breaks each one; engine/exposure.py + `make exposure`.
           Wired (Day 6): `make live-weather forecasts-latest alerts` - engine/alert_run.py
           (pure) + models/alert_run.py -> alerts + alert_runs (migration 0005).
Day 5  — [x] EA-LSTM + assimilation + calibration + anomaly + gate   (SAGE-TS DROPPED)
           GATE (decided on 2024 val): LIGHTGBM STAYS PRODUCTION. EA-LSTM won 2/3 buckets for
           NDCI (and missed coverage-no-worse by 0.00006) but 0/3 for turbidity -> fails.
           Both models are reported in results/head_to_head.json and the README. P5.1 path:
           NeuralHydrology 1.13 (Colab T4, 5 seeds x 2 targets), not the PyTorch fallback.
           models/: ealstm, cmal, assimilation, calibration, anomaly (weather-explained),
           head_to_head. Run order: make hindcast h2h anomaly. Found: imperviousness response
           has the WRONG SIGN for BOTH models on 62-75% of reaches (README) - scenario effects
           must come only from config/intervention_coefficients.yaml. Statics used: 6 of 9.
           Anomaly scored against the 95th-pct PROXY (incidents.csv is empty), labelled so.
           Assimilation helps EA-LSTM, adds nothing to LightGBM A. Attribution (IG) not run:
           EA-LSTM is not production. Do not spend Day 6 rescuing the EA-LSTM.
Day 6  — [x] scenario engine + API
           engine/scenarios.py (pure): variant B only; per lever x variable the path is
           MODEL_PERTURBATION | LITERATURE_DIRECT | NOT_ESTIMABLE from the 9.6 response check
           (`make response-check` -> results/scenario_response_check_<city>.json, tied to the
           serving model version; negligible threshold fixed in modelling.yaml first).
           RESULT ON COIMBRA: every static lever FAILS (imperviousness WRONG_SIGN both targets;
           riparian_width NEGLIGIBLE turbidity / WRONG_SIGN ndci) -> paving, green roofs,
           buffer are NOT_ESTIMABLE until a CITED `direct_effect` is added to the YAML (schema
           exists; none invented). Detention + sweeping (driver levers) go through the model.
           Exceedance days = sum of daily P(exceed), Poisson-binomial interval; scenario
           interval = 1.5 x max(own, baseline) width, asserted strictly wider. Central value at
           the cited magnitude (sweeping stays 0). No downstream propagation (stated on result).
           API: all 11 MASTERSPEC 12 routes, api/repository.py (PostGIS) behind a Protocol,
           migration 0004, `make forecasts-latest` writes production forecasts to the DB.
           Tests: test_scenarios, test_api_contract, test_api_db (marked db, cleans up),
           test_priorities, test_alert_run.
           [x] live run: production forecast now uses the ECMWF run of its issue date
           (weather=LIVE, 0 missing future drivers; `make live-weather` = 1 request).
           Alert run 2026-09-20: 1 ALERT (CMB-0145 turbidity, 0.62), 604 INSUFFICIENT_EVIDENCE
           (all driver-only: no threshold), suppressed 58 confidence + 43 drift. Drift
           residuals come from the wf-test model (production is in-sample) - labelled in
           basis. alert_runs records suppressed counts; /api/alerts returns them.
           Speed: /api/reaches ~0.3-0.9 s (vectorised thresholds + partial index on
           production forecasts); scenario model warmed at API startup (SCENARIO_WARMUP).
           Scenarios still do NOT propagate downstream (stated on every result).
           [x] config/intervention_coefficients.yaml: 5 cited levers (buffer, permeable
           paving, detention, street sweeping, green roofs); daylighting is not_quantified
           (no effect size in the literature). Load ONLY via core.config.
           load_intervention_coefficients() (engine/coefficients.py validates; load_config
           refuses the file). Every number needs a reference whose `supports` claims it.
           Detention (0.3 %, Emerson 2005) and street sweeping (central 0, Selbig 2007) are
           honest near-zero effects - do not "fix" them. Costs are in source currency
           (street sweeping is USD); never convert silently. hybrid_sage.py deleted.
Day 7  — [x] frontend: map, detail, alerts
           frontend/ (Vite 8, React 19, MapLibre 6, Zustand, Tailwind 4, Recharts 3): `make web`.
           Visual language is design.md (chart paper, sediment ramp, hatched INSUFFICIENT_EVIDENCE).
           Heat Surgeon had no MapLibre code to lift (built SVG bundle only) - basemap is
           OpenFreeMap Positron recoloured in src/map/basemap.ts. Hydrograph rail scrubs the map
           through past observations (observed / seasonal threshold) and forecast days.
           New read-only API routes for it: /api/timeline, /api/reaches/{id}/catchment,
           /api/reaches/{id}/attribution (STORED production TreeSHAP, nothing recomputed),
           /api/exposure, /api/scenarios/interventions; alerts list now carries stored exposure;
           GZip on. /api/validation/metrics takes ?city= (results/<city>/metrics.json for any
           city but the primary - core.settings.results_dir_for).
Day 8  — [~] frontend: scenarios, validation, Pune        ← feature freeze 18:00
           [x] scenario workbench (click/lasso, cited levers + extents, swipe before/after,
           per-reach days + intervals, costs in source currency, citations as text) and
           validation page (reliability, skill table incl. losses, anomaly P/R/F1 vs PROXY,
           lead time = not computed, observable vs driver-only).
           Pune: data paths moved into config/cities/*.yaml `data:`; fetch_datasets.py --only
           pune; Pune riparian window 15 Oct-15 Nov (monsoon). L0 on COPDEM fallback: 312
           reaches, 57 monotonicity violations (18%, REVIEW - bad snaps on the 30 m DEM).
           Gate: 59/312 observable. Full as-issued backfill NOT run for Pune (quota) - only
           the live run for the production forecast, so Pune's ASISSUED evaluation is absent.
           CHECKED 2026-09-24: Pune has reaches + observations + riparian NDVI in the DB but
           NO drivers_daily, catchment_attributes, exposure_features, frame, model, forecasts
           or metrics. So Pune's scenarios/validation/priorities 503/404 and its map has no
           forecast. Needs: make l2 static exposure dataset train evaluate response-check
           live-weather forecasts-latest alerts city=pune.
Day 9  — [ ] docs, FHIR, deploy
Day 10 — [ ] video, SUBMIT
```

---

## If you are behind

Cut in this order. The scenario workbench is the **last** thing to go, not the first —
it is the Track 6 requirement most other entries will miss.

1. Climate scenarios → cut
2. FHIR export → cut
3. Pune transfer demo → cut
4. EA-LSTM → fall back to LightGBM, report the comparison
5. Prioritisation panel → cut
6. Scenario workbench → only if the alternative is not shipping

Never cut: honest validation metrics, guardrails, the limitations section of the README.