# Kingfisher — Data Sources & Pre-Collection Guide

**Project:** Kingfisher — Resilience Informatics for urban streams
**Hackathon:** OneAquaHealth IEEE Global Hackathon 2026, Track 6
**Primary site:** Coimbra, Portugal (Mondego tributaries)
**Transfer site:** Pune, India (Mula-Mutha) — proves global scalability

---

## 0. Pre-collection: do these BEFORE writing any code

Every one of these is free. Some take hours to activate, so start them all on Day 0
in parallel rather than discovering a blocker on Day 4.

| # | Account | URL | What you get | Activation time | Blocking? |
|---|---------|-----|--------------|-----------------|-----------|
| 1 | Copernicus Data Space Ecosystem (CDSE) | dataspace.copernicus.eu | Sentinel-2 access + Sentinel Hub OAuth client | Instant | **YES — hard blocker** |
| 2 | CDSE Sentinel Hub OAuth client | shapps.dataspace.copernicus.eu/dashboard | `client_id` + `client_secret` for Statistical API | Instant | **YES** |
| 3 | Copernicus Land Monitoring Service (CLMS) | land.copernicus.eu | Imperviousness, Urban Atlas, Riparian Zones | Minutes–hours (manual approval possible) | Medium |
| 4 | Climate Data Store (CDS) | cds.climate.copernicus.eu | ERA5-Land (optional, citation-grade) | Instant + per-dataset ToU acceptance | No (Open-Meteo covers it) |
| 5 | Earth Observation Group | eogdata.mines.edu | VIIRS nighttime lights | Instant | No |
| 6 | SNIRH (Portugal) | snirh.apambiente.pt | Ground-truth water quality for Coimbra | None — open web | No |

**No account required:** Open-Meteo, OSM Overpass, Geofabrik, MERIT Hydro, EEA Waterbase,
GHSL population.

### Pre-collection checklist (Day 0, ~3 hours)

- [ ] Register CDSE. Confirm email.
- [ ] Create Sentinel Hub OAuth client. Store `client_id` / `client_secret` in `.env`.
- [ ] **Smoke-test it immediately**: run one Statistical API request for a single
      polygon over a one-month window. If this fails, everything downstream fails.
      Do not proceed until you see numbers come back.
- [ ] Check your Sentinel Hub **processing unit (PU) quota** on the dashboard. The free
      tier is finite. Budget it (see §3).
- [ ] Register CLMS. Request/download Imperviousness Density and Riparian Zones for
      the Coimbra tile. These are large downloads — start them and let them run.
- [ ] Hit Open-Meteo archive API once for Coimbra. No key needed; confirm JSON returns.
- [ ] Download the Portugal OSM extract from Geofabrik (~400MB).
- [ ] Download MERIT Hydro tiles covering Coimbra.
- [ ] Browse SNIRH manually. Find 2–5 monitoring stations on Mondego tributaries near
      Coimbra and note their station IDs. **This is your ground truth — if you can't
      find usable stations, the validation story weakens badly, so check on Day 0.**
- [ ] Create `data/raw/`, `data/interim/`, `data/processed/` and a `DATA_INVENTORY.md`
      logging every file you pull with its date, source URL and licence.

---

## 1. The datasets

### 1.1 Stream network — the spatial backbone

**Primary: OpenStreetMap waterways**
- Source: Geofabrik regional extract (`portugal-latest.osm.pbf`) or Overpass API
- Filter: `waterway in (river, stream, canal)` plus `waterway=riverbank` / `natural=water` polygons
- Tools: `osmium` or `pyrosm` to extract, `geopandas` to handle
- Licence: ODbL — **attribution required in your submission**
- Why: global coverage, so your Pune transfer uses identical code

**Better for Europe: EU-Hydro River Network Database (CLMS)**
- Photo-interpreted river network with drainage catchments already delineated, 1:5000-ish
  accuracy, EEA-39 coverage
- If it covers Coimbra cleanly, use it for the primary site and fall back to OSM for Pune.
  Saves you catchment delineation work.

**Reach segmentation:** split the network into 200–500m reaches at confluences and at
fixed intervals. Each reach gets a stable `reach_id`. This is the unit of everything.

---

### 1.2 Catchment delineation — the part most entries skip

**MERIT Hydro** (Yamazaki et al.) — use this, not a raw DEM.
- Source: hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/
- 3 arc-second (~90m) global
- **Ships with flow direction, flow accumulation and upstream drainage area already
  computed.** Deriving these yourself from a raw DEM is a full day of pit-filling and
  debugging. Don't.
- Tools: `pysheds` or `whitebox` for the upstream trace
- Licence: CC-BY-NC 4.0 / ODbL dual — non-commercial is fine for a hackathon, but
  **state it in your README**

**Alternative if MERIT gives trouble:** Copernicus DEM GLO-30 (30m, free, via CDSE) plus
`pysheds` fill → flow direction → accumulation → upstream trace. Higher resolution,
more work.

**Output:** for each `reach_id`, a polygon of its upstream contributing catchment. Every
driver feature is aggregated over *this polygon*, not over a circle around the reach.
This is the hydrologically correct move and a freshwater ecologist will notice.

---

### 1.3 Sentinel-2 — the observable

**Source:** Copernicus Data Space Ecosystem
**Collection:** `sentinel-2-l2a` (Level-2A, atmospherically corrected, surface reflectance)
**Coverage:** 2015–present, 5-day revisit with both satellites, 10m for visible/NIR
**Cost:** free

**ACCESS METHOD — this decision matters more than any other in the project.**

❌ **Do not** download granules. STAC search at `https://stac.dataspace.copernicus.eu/v1/search`
   works and you can filter `eo:cloud_cover`, but each granule is ~1GB and you'd need
   hundreds. That is days of bandwidth and disk.

✅ **Do** use the **Sentinel Hub Statistical API** via `sentinelhub-py`. You post a reach
   polygon, a date range and an evalscript; the server computes and returns mean / std /
   percentiles per polygon per acquisition date. You receive kilobytes of JSON instead of
   gigabytes of raster.

```python
from sentinelhub import SHConfig, SentinelHubStatistical, DataCollection

config = SHConfig()
config.sh_client_id = os.environ["SH_CLIENT_ID"]
config.sh_client_secret = os.environ["SH_CLIENT_SECRET"]
config.sh_base_url = "https://sh.dataspace.copernicus.eu"
config.sh_token_url = (
    "https://identity.dataspace.copernicus.eu/auth/realms/"
    "CDSE/protocol/openid-connect/token"
)
```

**Bands to request:** B02 (blue, 490nm), B03 (green, 560nm), B04 (red, 665nm),
B05 (red edge, 705nm), B08 (NIR, 842nm), B11 (SWIR, 1610nm), plus SCL (scene
classification) for cloud and water masking.

**Indices computed in the evalscript:**

| Index | Formula | Proxy for |
|-------|---------|-----------|
| MNDWI | (B03 − B11) / (B03 + B11) | water extent — mask pixels where MNDWI ≤ 0 |
| NDCI | (B05 − B04) / (B05 + B04) | chlorophyll-a / eutrophication |
| Turbidity (Nechad) | a·B04 / (1 − B04/C) | suspended sediment |
| NDVI (riparian buffer) | (B08 − B04) / (B08 + B04) | riparian vegetation condition |

**⚠️ THE BIGGEST TECHNICAL RISK IN THE PROJECT**

Sentinel-2 is 10m. Many urban streams are 3–8m wide. A large share of your reaches will
have **zero clean water pixels**, and those that have some will suffer mixed-pixel
contamination from banks, bridges and shadow.

Mitigations, in order:
1. Compute a **`water_pixel_count`** per reach per date and store it. Treat any reach-date
   with fewer than N clean water pixels as *missing*, not as zero.
2. Classify reaches into **observable** (wide enough, stable water pixels) and
   **unobservable** (too narrow). Report the split honestly — it is a real finding about
   urban stream monitoring, not a failure.
3. Architect around it: the model is **driver → state**, so unobservable reaches are
   predicted from catchment forcing and upstream state. Satellite observations are
   *assimilated where available*. The system degrades gracefully instead of breaking.
4. Include a few wider reaches (main Mondego channel) so you always have well-observed
   examples to demo.

**Run this check on Day 1.** If fewer than ~15 reaches are observable in Coimbra, widen
the study area downstream before you build anything on top.

**Processing units:** the free tier is finite. Budget: ~40 reaches × ~10 years of
5-day composites is manageable, but request monthly aggregates first, verify the pipeline,
*then* go to full temporal resolution. Cache every response to disk keyed by
`(reach_id, date_range, evalscript_hash)` and never re-request.

---

### 1.4 Weather — the drivers

**Primary: Open-Meteo. Free, no API key, no queue.**

Two endpoints, and you need both:

| Endpoint | Use | Coverage |
|----------|-----|----------|
| `archive-api.open-meteo.com/v1/archive` | training history | 1940–present (ERA5/ERA5-Land derived) |
| `api.open-meteo.com/v1/forecast` | live 10-day forecast | now + 16 days |

**Why this matters:** ERA5-Land from the Climate Data Store lags real time by about three
months. It is excellent for training and citation, and **useless for a live early-warning
system**. If you build only on CDS, your demo cannot forecast the present. Open-Meteo's
archive is ERA5-derived so it is scientifically equivalent for training, and its forecast
endpoint gives you the live path. One source, both jobs, no queue.

**Variables:** `precipitation`, `temperature_2m`, `soil_moisture_0_to_7cm`,
`et0_fao_evapotranspiration`, `relative_humidity_2m`, `wind_speed_10m`. Hourly for
training, then aggregate to daily.

**Query point:** the centroid of each reach's *upstream catchment*, not the reach itself.

**Optional — CDS ERA5-Land.** Worth 30 minutes for the README's credibility
("validated against ERA5-Land reanalysis"). `pip install "cdsapi>=0.7.7"`, put your
personal access token in `~/.cdsapirc` with `url: https://cds.climate.copernicus.eu/api`,
accept the dataset terms of use on the `reanalysis-era5-land` page (per-dataset, not
per-account), then retrieve. Do this only if the core pipeline is already green.

---

### 1.5 Static catchment attributes — the pressures

These are the levers your scenario engine perturbs. All aggregated over the upstream
catchment polygon.

| Attribute | Source | Resolution | Notes |
|-----------|--------|-----------|-------|
| Imperviousness density | CLMS Imperviousness Density | 10m/20m, 2006/2009/2012/2015/2018/2021 | The dominant urban stream pressure. Multi-year → you get change too. |
| Land use | CLMS Urban Atlas 2018/2021 | vector, functional urban areas | Coimbra is covered. |
| Land cover | CORINE Land Cover | 100m | Fallback / Pune-equivalent reasoning. |
| Riparian condition | CLMS Riparian Zones | vector+raster, EU river buffers | Purpose-built product for exactly this. Use it. |
| Artificial light at night | VIIRS DNB monthly composites (EOG) | ~500m, 2012–present | **A named OneAquaHealth pressure that almost no one monitors at reach scale.** Cheap differentiator. |
| Road density | OSM `highway=*` | vector | Proxy for traffic-derived pollutant loading. |
| Population | GHSL GHS-POP | 100m, global | Exposure denominator; works for Pune too. |

---

### 1.6 Ground truth — the thing that makes validation real

**SNIRH (Portugal)** — `snirh.apambiente.pt`
National water resources information system. River monitoring stations with historical
series: nutrients, BOD/COD, suspended solids, dissolved oxygen, and in some cases
turbidity and flow. Free, downloadable.
**This is your validation gold.** Find stations on Mondego tributaries near Coimbra,
match them to reaches spatially, and you have real measured values to score forecasts
against.

**EEA Waterbase — Water Quality ICM** — `eea.europa.eu/data-and-maps/data/waterbase-water-quality-icm`
EU-wide aggregated station measurements. Broader coverage, coarser temporal resolution
than SNIRH. Use for cross-check and for threshold definitions.

**Incident records for anomaly-detection scoring:**
- Local/regional news archives for fish kills, sewage discharges, algal blooms
- Portuguese Environment Agency (APA) incident reporting
- EEA exceedance records
Even 5–15 documented events in 10 years is enough to report precision and recall with
honest confidence intervals. Hand-curate them into `data/reference/incidents.csv` with
date, location, type and source URL.

**If ground truth turns out to be thin:** fall back to *statistical* anomaly definition —
exceedance of the reach's own historical 95th percentile for the season — and say
explicitly in the README that this is a proxy, not verified incidents. Honest and still
scoreable.

---

### 1.7 Exposure layer — the One Health link

All from OSM, buffered around each reach:

| Feature | OSM tag |
|---------|---------|
| Schools | `amenity=school`, `amenity=kindergarten` |
| Playgrounds | `leisure=playground` |
| Parks | `leisure=park`, `leisure=garden` |
| Footpaths / cycleways along the stream | `highway=footway`, `highway=cycleway` |
| Water access points | `leisure=swimming_area`, fords, slipways, steps to water |
| Healthcare | `amenity=hospital`, `amenity=clinic` |

Plus GHS-POP population within the exposure buffer.

**Do not predict disease.** Map pathways and proximity only. An alert reads:
*"elevated exceedance probability on reach R-041, which has 3 public access points, a
riverside footpath, and a primary school 180m downstream."* That is a defensible One
Health statement. "Predicted gastroenteritis risk 12%" is not, and a researcher on the
panel will say so.

---

### 1.8 Pune transfer demo

Same code, different config. This is what proves Scale.

| Need | Source |
|------|--------|
| Stream network | OSM (Mula, Mutha, Pavana, Indrayani) |
| Catchments | MERIT Hydro (global) |
| Satellite | Sentinel-2 (global) — the Mula-Mutha is wide, so observability is *better* than Coimbra |
| Weather | Open-Meteo (global) |
| Imperviousness | Not CLMS (EU only) → use ESA WorldCover 10m or GHSL built-up |
| Ground truth | CPCB / MPCB water quality data for Mula-Mutha |

The only code change should be a config file and one land-cover adapter. If you find
yourself rewriting the pipeline, your abstractions are wrong.

---

## 2. Data volume and time budget

| Stage | Volume | Wall-clock |
|-------|--------|-----------|
| OSM Portugal extract | ~400 MB | 20 min |
| MERIT Hydro tiles | ~200 MB | 20 min |
| Catchment delineation, 40 reaches | ~50 MB | 1–2 h compute |
| Sentinel-2 via Statistical API, 40 reaches × 10 y | ~50 MB JSON | 3–6 h incl. retries |
| Open-Meteo archive, 40 catchments × 10 y | ~200 MB | 1 h |
| CLMS static layers | ~2 GB | 1–2 h |
| VIIRS monthly | ~500 MB | 30 min |
| **Total** | **~3.5 GB** | **~1.5 days** |

Budget Day 1 and half of Day 2 for acquisition. It always takes longer than this table says.

---

## 3. Hard rules

1. **Cache everything.** Every API response hits disk before it hits a DataFrame. You will
   re-run the pipeline fifty times; you should hit the network once.
2. **Log provenance.** Every file in `DATA_INVENTORY.md` with source URL, access date,
   licence.
3. **Never silently impute.** A missing reach-date is `NULL` with a reason code
   (`CLOUD`, `NO_WATER_PIXELS`, `OUT_OF_RANGE`), never a zero and never a forward-fill
   that isn't labelled as one.
4. **Fail loudly.** If a source returns nothing, raise. Do not let the pipeline produce a
   plausible-looking empty result — that is how you demo a model trained on nothing.
5. **Commit the config, never the keys.** `.env` in `.gitignore`, `.env.example` committed.

---

## 4. Attribution block for your submission

```
Contains modified Copernicus Sentinel data (2015–2026), processed by Kingfisher.
Contains Copernicus Land Monitoring Service information (Imperviousness Density,
  Riparian Zones, Urban Atlas).
Weather data © Open-Meteo.com (CC-BY-4.0), derived from ECMWF ERA5/ERA5-Land.
Map data © OpenStreetMap contributors, available under the Open Database Licence.
MERIT Hydro © Dai Yamazaki (University of Tokyo).
Water quality reference data: SNIRH / Agência Portuguesa do Ambiente; EEA Waterbase.
VIIRS nighttime lights: Earth Observation Group, Colorado School of Mines.
Population: GHSL, European Commission Joint Research Centre.
```

Put this in the README **and** on a slide in the demo video. Judges from an EU research
consortium notice correct data attribution, and most hackathon entries get it wrong.