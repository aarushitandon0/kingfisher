# Kingfisher — Master Specification

**OneAquaHealth IEEE Global Hackathon 2026 · Track 6: Resilience Informatics**
**Submission deadline:** 1 Oct 2026, 09:30 IST
**Status:** v1.0 design freeze

---

## 1. Problem statement

Urban stream monitoring is episodic. Urban stream failure is episodic. The two episodes
do not line up.

Professional sampling runs on a calendar. Citizen observation runs on whoever felt like
walking. But the events that actually degrade urban freshwater ecosystems run on
*weather's* schedule: combined sewer overflows during intense rain, first-flush runoff
after a dry spell, illicit discharges, summer eutrophication crashes during low flow and
high temperature. A stream can fail on a Tuesday and be sampled a month later, by which
point the water has moved on and the evidence with it.

The OneAquaHealth project's own findings show the shape of this gap: across 100 urban
stream sites in five European cities, pharmaceuticals were detected at 91% of monitored
sites, and diatom deformities were needed as an early-warning indicator precisely because
standard monitoring does not catch what happens between visits.

**The gap is not data volume. It is detection latency — and the absence of tools that
say what to change.**

Track 6 asks for three things. Kingfisher delivers three things:

| Track 6 asks for | Kingfisher pillar |
|------------------|-------------------|
| Predictive dashboards | **Predict** — reach-level 10-day forecast with uncertainty |
| Alerts | **Warn** — calibrated exceedance probability with driver attribution and exposure context |
| Resilience tools | **Adapt** — intervention and climate scenario engine |

---

## 2. Core insight

Two data regimes exist and neither suffices alone:

- **Satellite is continuous but shallow.** Sentinel-2 revisits every 5 days, free, back to
  2015 — but it sees only turbidity, chlorophyll and water extent. It will never see a
  pharmaceutical or a diatom.
- **In-situ and citizen sampling is sparse but deep.** It sees everything, rarely.

Kingfisher uses the continuous shallow signal, driven by catchment-scale weather forcing,
to predict reach condition ahead of time, flag anomalies in the gaps, and tell a Local
Alliance where the next field visit is worth most.

The architecture is **driver → state**, not pure time-series extrapolation:

```
catchment forcing (weather, antecedent conditions)
        × static catchment attributes (imperviousness, riparian, light)
        → predicted reach state (turbidity, chlorophyll)
        ← assimilated from satellite where the channel is observable
```

This choice is forced by the mixed-pixel problem (§6.1) and it pays off twice: narrow
unobservable reaches are still predicted, and because catchment attributes are *inputs*,
they can be perturbed — which is what makes Pillar 3 possible at all.

---

## 3. Scope

### In scope (v1)
- Coimbra, Portugal — ~40 reaches on Mondego tributaries
- Pune, India — ~15 reaches on the Mula-Mutha, as a transfer demo
- 10-day-ahead forecast of turbidity and chlorophyll proxies with prediction intervals
- Calibrated threshold-exceedance alerts with driver attribution and guardrails
- Exposure-pathway mapping (schools, playgrounds, footpaths, access points, population)
- Intervention scenario engine with literature-cited coefficients
- Walk-forward validation against held-out years and real station data
- FHIR export of alerts (stretch — see §14)

### Explicitly NOT in scope — state these in the README
- Disease prediction. We map exposure pathways; we do not predict health outcomes.
- Water safety certification. Kingfisher does not declare water safe or unsafe.
- Pollution source attribution to a named polluter.
- Replacing professional monitoring. Kingfisher prioritises it.
- Chemical concentration retrieval. We model optical proxies, not lab values.

Declaring non-goals is not weakness. On a panel containing an actual freshwater ecologist,
it is the difference between a credible system and an overclaiming demo.

---

## 4. System architecture

```
┌─────────────────────────────────────────────────────────────┐
│ L0  SPATIAL BACKBONE                                        │
│     OSM waterways → reach segmentation (200–500m)           │
│     MERIT Hydro → upstream contributing catchment per reach │
│     Reach topology graph (upstream/downstream adjacency)    │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│ L1  OBSERVABLES        │  L2  DRIVERS                       │
│  Sentinel-2 via        │   Open-Meteo archive (train)       │
│  Statistical API       │   Open-Meteo forecast (live)       │
│  → MNDWI, NDCI,        │   → API, ADD, intensity, temp      │
│    turbidity, NDVI     │   Static: imperviousness,          │
│  + water_pixel_count   │     riparian, ALAN, roads, pop     │
└────────────────────────┴───────────┬────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────┐
│ L3  MODEL                                                   │
│     Baseline: gradient boosting, quantile loss              │
│     Upgrade: SAGE-TS hybrid — wavelet burstiness routes     │
│       smooth regimes → Mamba, bursty regimes → Transformer  │
│     Pooled across reaches + learned reach embedding         │
│     Output: 10-day forecast, P10/P50/P90                    │
│     Burstiness score doubles as anomaly detector            │
└────────────────────────────────────┬────────────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────┐
        │                            │                        │
┌───────▼─────────┐  ┌───────────────▼────────┐  ┌────────────▼──────────┐
│ L4 ALERT ENGINE │  │ L5 SCENARIO ENGINE     │  │ L6 PRIORITISATION     │
│  P(exceedance)  │  │  cited coefficients →  │  │  expected info value  │
│  guardrails     │  │  perturb drivers →     │  │  = uncertainty        │
│  attribution    │  │  re-run inference      │  │    × risk × exposure  │
│  exposure ctx   │  │  Δ exceedance days     │  │  → next field visit   │
└───────┬─────────┘  └───────────────┬────────┘  └────────────┬──────────┘
        └────────────────────────────┼────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────┐
│ L7  API (FastAPI) → FRONTEND (React + MapLibre + Zustand)   │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Data model (PostGIS)

```sql
-- L0
reaches(
  reach_id TEXT PK, city TEXT, name TEXT,
  geom GEOMETRY(LineString,4326),
  length_m FLOAT, strahler_order INT,
  catchment_geom GEOMETRY(Polygon,4326), catchment_area_km2 FLOAT,
  observable BOOLEAN,            -- enough water pixels for satellite assimilation
  median_water_pixels FLOAT
)
reach_topology(upstream_id TEXT, downstream_id TEXT)

-- L1
observations(
  reach_id TEXT, obs_date DATE, source TEXT,     -- 'S2' | 'SNIRH' | 'EEA' | 'CITIZEN'
  turbidity_proxy FLOAT, ndci FLOAT, mndwi FLOAT, riparian_ndvi FLOAT,
  water_pixel_count INT, cloud_fraction FLOAT,
  quality_flag TEXT,                             -- 'OK'|'CLOUD'|'NO_WATER_PIXELS'|'OUT_OF_RANGE'
  PRIMARY KEY (reach_id, obs_date, source)
)

-- L2
drivers_daily(
  reach_id TEXT, date DATE,
  precip_mm FLOAT, precip_max_hourly FLOAT, temp_mean_c FLOAT, temp_max_c FLOAT,
  soil_moisture FLOAT, et0 FLOAT,
  api_7 FLOAT, api_14 FLOAT, api_30 FLOAT,       -- antecedent precipitation index
  antecedent_dry_days INT,
  first_flush_index FLOAT,
  PRIMARY KEY (reach_id, date)
)
catchment_attributes(
  reach_id TEXT PK, year INT,
  imperviousness_pct FLOAT, riparian_ndvi_mean FLOAT, riparian_width_m FLOAT,
  road_density_km_km2 FLOAT, alan_radiance FLOAT,
  population INT, urban_fraction FLOAT
)

-- L3/L4
forecasts(
  reach_id TEXT, issued_date DATE, target_date DATE, variable TEXT,
  p10 FLOAT, p50 FLOAT, p90 FLOAT, burstiness FLOAT, model_version TEXT,
  PRIMARY KEY (reach_id, issued_date, target_date, variable)
)
alerts(
  alert_id UUID PK, reach_id TEXT, issued_date DATE, target_window DATERANGE,
  variable TEXT, exceedance_prob FLOAT, threshold_value FLOAT,
  severity TEXT,                                  -- 'WATCH'|'ALERT'|'INSUFFICIENT_EVIDENCE'
  attribution JSONB,                              -- driver contributions
  exposure JSONB,                                 -- schools/paths/population within buffer
  suppressed_reason TEXT                          -- populated when a guardrail fired
)

-- L5
scenarios(scenario_id UUID PK, name TEXT, city TEXT, config JSONB, created_at TIMESTAMP)
scenario_results(scenario_id UUID, reach_id TEXT, baseline_exceedance_days FLOAT,
                 scenario_exceedance_days FLOAT, delta FLOAT, ci_low FLOAT, ci_high FLOAT)

-- exposure
exposure_features(reach_id TEXT, feature_type TEXT, count INT,
                  nearest_distance_m FLOAT, geom GEOMETRY)
```

---

## 6. Feature engineering

### 6.1 Observability handling — read this before writing L1

Sentinel-2 is 10m. Urban streams are often 3–8m wide. Expect a substantial fraction of
reaches to return **zero clean water pixels**.

Rules:
- `water_pixel_count` is stored on every reach-date. It is a first-class field.
- `observable = median_water_pixels >= N` (start N=5, tune on Day 1).
- Unobservable reach-dates are `NULL` with `quality_flag`, never zero, never silently
  interpolated.
- Unobservable reaches are still forecast — from drivers and upstream state. They simply
  contribute no assimilation signal.
- **Report the observable/unobservable split in the README and the video.** "31 of 47
  reaches are optically observable at 10m; the remaining 16 are driver-predicted" is a
  genuine finding about the limits of satellite monitoring for urban streams. Hiding it
  is what a weak entry does.

### 6.2 Driver features — the physics goes here

| Feature | Definition | Mechanism it encodes |
|---------|-----------|----------------------|
| `api_k` | Σ precip(t−i)·λ^i, λ≈0.9, k ∈ {7,14,30} | catchment wetness / runoff readiness |
| `antecedent_dry_days` | consecutive days with precip < 1mm before an event | **first flush** — the single strongest predictor in urban stormwater literature |
| `first_flush_index` | `antecedent_dry_days × precip_max_hourly` | accumulated surface pollutant load × mobilising energy |
| `precip_max_hourly` | peak hourly intensity in the window | intensity drives scour; totals alone do not |
| `precip_duration_h` | hours with precip > 0.5mm | sustained vs flashy events |
| `temp_low_flow_index` | `temp_mean × (1 − api_14/max_api)` | eutrophication risk: warm water + low flow |
| `doy_sin`, `doy_cos` | seasonal harmonics | annual cycle |
| `upstream_state_lag1` | previous state of upstream reach | longitudinal propagation via topology graph |

All driver aggregation is over the **upstream contributing catchment polygon**, not a
buffer around the reach. This is the hydrologically correct unit and it is a visible
quality signal to a domain judge.

### 6.3 Static attributes

`imperviousness_pct`, `riparian_ndvi_mean`, `riparian_width_m`, `road_density`,
`alan_radiance`, `population`, `catchment_area_km2`, `strahler_order`.

These are the **scenario levers**. Anything you want the user to be able to change in
Pillar 3 must be a feature here.

---

## 7. Model

### 7.1 Baseline — build this FIRST, Day 3

Gradient boosting (LightGBM) with quantile objective at α ∈ {0.1, 0.5, 0.9}, one model
per horizon bucket (1–3d, 4–7d, 8–10d), pooled across all reaches with `reach_id` as a
categorical.

**Why baseline first:** it trains in minutes, gives you SHAP attribution for free (which
feeds the alert engine's driver attribution directly), and guarantees you have a working
system before the ambitious part. If the hybrid fails on Day 6, you still ship.

### 7.2 Upgrade — SAGE-TS hybrid

Adapt the existing SAGE-TS architecture rather than reinventing it:

- Wavelet decomposition of the reach state series produces a **burstiness score** per
  timestep.
- Smooth regimes route to the **Mamba** expert — seasonal cycles, slow eutrophication
  buildup, baseflow condition.
- Bursty regimes route to the **Transformer** expert — storm response, discharge events,
  sharp transitions.
- Driver features enter both experts as exogenous inputs.

**The structural gift:** in SAGE-TS the burstiness signal already doubles as an anomaly
detector. Here, that means **the early-warning alert falls out of the routing decision**.
You are not bolting a detector onto a forecaster — the detector is load-bearing.

**Data thinness — critical.** ~10 years × ~50 usable acquisitions ≈ 500 timesteps per
reach. That is thin for a Transformer. Therefore:
- Train **one pooled model across all reaches** with a learned reach embedding, not one
  model per reach. This multiplies effective sample size by the reach count and is
  standard practice in hydrological ML.
- Keep model capacity small. Regularise hard.
- If the hybrid does not beat LightGBM on held-out years, **ship LightGBM and report the
  comparison**. A documented negative result on a novel architecture is more credible
  than an undocumented win.

### 7.3 Anomaly detection

Two signals, reported separately:
- **Burstiness spike** — routing-derived, from the hybrid
- **Forecast residual** — observed falls outside P10–P90 by a margin

An anomaly requires either. Scored against `data/reference/incidents.csv` (§1.6 of
DATA_SOURCES.md).

---

## 8. Alert engine

An alert is **a calibrated probability of threshold exceedance**, never a raw prediction.

```
exceedance_prob = P(state_t+h > threshold)  from the predictive distribution
```

Thresholds derived from EEA/SNIRH water quality classes where available, otherwise from
the reach's own seasonal historical distribution — **and the derivation is stated on the
alert itself**.

### Guardrails (pattern lifted from the Razorpay policy engine)

| Guardrail | Rule |
|-----------|------|
| Minimum confidence | no alert below `exceedance_prob = 0.6` |
| Cooldown | max one alert per reach per 72h |
| Staleness suppression | if last usable observation > 21 days old, downgrade to `INSUFFICIENT_EVIDENCE` |
| Unobservable reach cap | driver-only reaches can reach `WATCH` but not `ALERT` |
| Model drift | if recent residuals exceed a calibration band, suppress and flag |

**`INSUFFICIENT_EVIDENCE` is a first-class output, displayed in the UI, and counted in
the metrics.** Refusing to alert is a decision the system makes and reports — the same
stance as `refused` being a first-class outcome in the Razorpay engine and
`audit_passed = false` being published in PrismRank.

### Attribution

Every alert carries driver contributions from SHAP (baseline) or attention/routing
weights (hybrid):

> **ALERT — Reach R-041, Ribeira de Coselhas**
> Turbidity exceedance probability 0.78 for 23–25 Sep
> Drivers: 18 antecedent dry days (+0.31), forecast 34mm in 6h (+0.28),
> upstream imperviousness 61% (+0.19)
> Exposure: 3 public access points, riverside footpath, primary school 180m downstream
> Basis: 6 usable observations in last 30 days · model v0.3.1

---

## 9. Scenario engine (Pillar 3 — the Track 6 requirement most entries will miss)

### 9.1 The honest problem

Perturbing a statistical model's features and re-running inference is **correlational
extrapolation, not causal inference**. Presented as "our AI says buffers cut turbidity
39%", a freshwater ecologist will dismantle it.

### 9.2 The fix — separate the model from the intervention physics

```
model  →  baseline risk + uncertainty            (learned from data)
literature coefficients  →  Δ driver features    (cited, auditable, in a table)
re-run inference on perturbed features  →  scenario risk
```

The intervention effect sizes come from **published stormwater and riparian literature in
a transparent, cited coefficient table** — not from the model. The model does only what it
was trained to do. Every scenario output can cite its source.

This is the same discipline as PhantomField's published Claim Strength Score formula and
SwasthyaSetu's hard `ml/` boundary: the numbers come from a defensible place, and the
place is visible.

### 9.3 Coefficient table (`config/intervention_coefficients.yaml`)

Each entry carries `effect`, `magnitude`, `uncertainty_range`, `citation`, `applies_to`.

| Intervention | Perturbs | Direction |
|--------------|----------|-----------|
| Riparian buffer restoration | `riparian_ndvi_mean`, `riparian_width_m` ↑ | sediment + nutrient retention |
| Permeable paving | `imperviousness_pct` ↓ | runoff volume + first-flush load |
| Detention/retention basin | `precip_max_hourly` effective ↓ | attenuates peak intensity |
| Daylighting a culverted reach | `observable` → true, `riparian_*` ↑ | habitat + self-purification |
| Street sweeping programme | `first_flush_index` ↓ | reduces accumulated surface load |
| Green roofs | `imperviousness_pct` ↓ (partial) | runoff volume |

Populate magnitudes from literature on Day 6 and **cite every row**. An uncited row is
worse than an absent one.

### 9.4 Climate scenarios (stretch)

Re-force the model with perturbed weather: +2°C summer mean, +20% rainfall intensity,
+30% longest dry spell. Produces the single most quotable line in the whole submission:

> *This stream survives today's weather and fails under 2040's.*

### 9.5 Output contract

```
Baseline:  23.4 exceedance days/year  [CI 18.1 – 29.7]
Scenario:  14.1 exceedance days/year  [CI  9.2 – 21.3]
Δ         −9.3 days/year (−40%)
Cost estimate: €X
⚠ Planning estimate. Intervention effect sizes from cited literature applied to a
  statistical model; not a causal experiment. Intervals widened accordingly.
```

**Always widen intervals on scenario runs, and always show the caveat.** Do not hide it
in a tooltip.

---

## 10. Prioritisation (demoted from a pillar — a dashboard panel)

```
information_value(reach) = forecast_uncertainty × predicted_risk × exposure_weight
```

Ranked list: "Next field visit best spent at R-041, R-017, R-023." Links Kingfisher to the
citizen science app and to Local Alliances. One panel, not a subsystem.

---

## 11. Validation protocol — this is what wins Architecture

**Split:** train ≤ 2023, validate 2024, test 2025–2026. Walk-forward. No shuffling, no
leakage across the reach-time boundary.

**Report all of the following, favourable or not:**

| Metric | Against |
|--------|---------|
| MAE, RMSE, CRPS by horizon | seasonal-naive **and** climatology baselines |
| Reliability diagram | alert probability calibration |
| Precision / recall / F1 | anomaly detection vs `incidents.csv` |
| Lead-time distribution | days of warning actually delivered |
| False alarm rate at operating threshold | — |
| Skill by reach observability | observable vs driver-only |
| Transfer performance | Coimbra-trained model applied to Pune |

**If the forecast loses to seasonal-naive on some reaches, publish that.** You have done
exactly this before — the WHESTBench `--runner local` vs `--runner subprocess` discrepancy,
`audit_passed = false` with three warnings in PrismRank. This panel contains real
researchers. Honest negative results read as rigour.

**The reliability diagram is the single most valuable chart in the submission.** Almost no
hackathon entry shows calibrated probabilities. Put it in the video.

---

## 12. API surface (FastAPI)

```
GET  /api/cities
GET  /api/reaches?city=coimbra              → GeoJSON + current status
GET  /api/reaches/{id}                      → detail, history, catchment, exposure
GET  /api/reaches/{id}/forecast?horizon=10  → P10/P50/P90 series
GET  /api/alerts?city=&active=true          → active alerts incl. INSUFFICIENT_EVIDENCE
GET  /api/alerts/{id}                       → attribution + exposure + basis
POST /api/scenarios                         → {interventions:[{reach_ids, type, magnitude}]}
GET  /api/scenarios/{id}                    → per-reach deltas + CIs + citations
GET  /api/priorities?city=                  → ranked next-visit list
GET  /api/validation/metrics                → all §11 metrics, served live to the UI
GET  /api/export/fhir/{alert_id}            → FHIR Observation + Location bundle
```

`/api/validation/metrics` being a **live endpoint rather than a static image** is a strong
credibility signal. The judges can see you did not hand-pick the numbers.

---

## 13. Frontend

Stack: React + TypeScript + Vite + MapLibre GL + Zustand + Tailwind + Recharts.
**Lift the scaffolding from Heat Surgeon.** Do not write a basemap again.

### Screens

**1. Map (default)**
Stream network coloured by current exceedance probability. Catchment polygon on reach
hover. Alert pins. Exposure features toggleable. City switcher (Coimbra / Pune).

**2. Reach detail (drawer)**
Observed history + 10-day forecast fan chart (P10/P50/P90). Driver attribution bars.
Catchment attribute card. Exposure list. Observability badge — "optically observable" vs
"driver-predicted", stated plainly.

**3. Alerts**
Active alerts, severity-sorted. `INSUFFICIENT_EVIDENCE` shown, not hidden. Each expands to
attribution, exposure, basis and guardrail status.

**4. Scenario workbench** ← *the Track 6 differentiator, do not cut this*
Select reaches on the map → pick interventions → Run → split-view before/after with
Δ exceedance days, cost, CIs, and the citation for every coefficient used.

**5. Validation**
Reliability diagram, skill-vs-baseline table, precision/recall, lead-time distribution.
Served from the live endpoint.

**Design note:** UX is one fifth of the rubric and is your weakest axis. Budget two full
days. Resist adding features on Day 9.

---

## 14. FHIR export (stretch, half a day)

HL7 Europe is an actual OneAquaHealth consortium partner, there is a dedicated FHIR
sandbox for this hackathon, and Gora Datta (FHL7) sits on the panel.

Map an alert to a FHIR `Observation` with `Location`, environmental exposure coding, and
`Device` for the model provenance. One endpoint, one screenshot in the video, one
paragraph in the README. Disproportionate return for the effort. Cut only if Day 9 is
on fire.

---

## 15. Repository layout

```
kingfisher/
├── CLAUDE.md
├── README.md
├── DATA_SOURCES.md
├── MASTERSPEC.md
├── .env.example
├── config/
│   ├── cities/coimbra.yaml
│   ├── cities/pune.yaml
│   ├── intervention_coefficients.yaml
│   └── thresholds.yaml
├── data/                       # gitignored except reference/
│   ├── raw/ interim/ processed/
│   └── reference/incidents.csv
├── pipeline/
│   ├── l0_network.py           # OSM → reaches → catchments (MERIT Hydro)
│   ├── l1_satellite.py         # Sentinel Hub Statistical API + indices
│   ├── l2_drivers.py           # Open-Meteo + feature engineering
│   ├── l2_static.py            # CLMS, VIIRS, OSM attributes
│   └── build_dataset.py
├── models/
│   ├── baseline_gbm.py
│   ├── hybrid_sage.py
│   ├── anomaly.py
│   └── evaluate.py
├── engine/
│   ├── alerts.py               # probability + guardrails + attribution
│   ├── scenarios.py            # cited coefficients → perturb → re-infer
│   ├── priorities.py
│   └── exposure.py
├── api/
│   ├── main.py routes/ schemas.py
│   └── fhir.py
├── frontend/
├── tests/
└── notebooks/
```

---

## 16. Ten-day plan

| Day | Date | Deliverable | Gate |
|-----|------|-------------|------|
| 0 | Sep 20 | All accounts. Sentinel Hub smoke test green. SNIRH stations identified. | **Statistical API returns numbers, or stop and fix.** |
| 1 | Sep 21 | L0 complete: reaches + catchments + topology. Observability check run. | **≥15 observable reaches, or widen the study area now.** |
| 2 | Sep 22 | L1 + L2: satellite series and driver series in PostGIS. | Dataset assembled, gaps labelled with reason codes. |
| 3 | Sep 23 | LightGBM baseline trained, walk-forward evaluated, SHAP working. | **A working end-to-end system exists from here on.** |
| 4 | Sep 24 | Alert engine + guardrails + exposure layer. | Alerts firing with attribution. |
| 5 | Sep 25 | SAGE-TS hybrid trained and compared to baseline. | **Hard gate: if it doesn't beat baseline by EOD, ship baseline and move on.** |
| 6 | Sep 26 | Scenario engine + cited coefficient table. API complete. | Scenarios return deltas with CIs. |
| 7 | Sep 27 | Frontend: map, reach detail, alerts. | Clickable end to end. |
| 8 | Sep 28 | Frontend: scenario workbench, validation page. Pune transfer. | Feature freeze, 18:00. |
| 9 | Sep 29 | README, architecture doc, limitations, FHIR, deploy. | Deployed and reachable. |
| 10 | Sep 30 | Demo video. **Submit.** | Submitted — not on the morning of the 1st. |

---

## 17. Risk register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Too few observable reaches | **High** | Severe | Day 1 gate; widen area downstream; driver-only reaches still forecast |
| Sentinel Hub PU quota exhausted | Medium | Severe | Cache aggressively; monthly aggregates first; second account as backup |
| Hybrid underperforms baseline | Medium | Low | Baseline already shipped Day 3; report comparison honestly |
| Ground truth too sparse for anomaly scoring | Medium | Medium | Fall back to percentile-based statistical anomalies, labelled as a proxy |
| CLMS approval delays | Medium | Medium | Use ESA WorldCover / GHSL built-up as substitute |
| Catchment delineation eats a day | Medium | Medium | MERIT Hydro's precomputed accumulation avoids this — do not use a raw DEM |
| Frontend runs out of time | Medium | **Severe** (UX = 20%) | Heat Surgeon scaffolding; hard feature freeze Day 8 18:00 |
| Scope creep | **High** | Severe | §3 non-goals are binding. Scenario workbench is the last thing cut, not the first. |

---

## 18. Judging alignment

| Criterion | Where Kingfisher earns it |
|-----------|---------------------------|
| **Impact & Alignment** | Track 6 *is* the consortium's stated mission — an AI-based Environmental Surveillance System built on early warning indicators. Exposure pathways make the One Health link explicit without overclaiming. |
| **Innovation** | Wavelet-routed hybrid forecaster where the routing signal *is* the anomaly detector. Upstream catchment delineation. ALAN at reach scale. Cited-coefficient scenario engine. |
| **Architecture** | Driver→state design forced by a real constraint. Walk-forward validation with baselines. Guardrails with `INSUFFICIENT_EVIDENCE`. Live metrics endpoint. |
| **UX** | Two dedicated days. Scenario workbench is the memorable interaction. |
| **Scale** | Pune transfer on a different continent with a config change. Global data sources throughout. FHIR interoperability. |

---

## 19. The narrative

Open with it, close with it, put it on the title card:

> **Kingfisher watches the water continuously, warns before the water turns, and shows
> what to change so it turns less often.**