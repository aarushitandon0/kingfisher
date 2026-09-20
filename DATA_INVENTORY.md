# Data inventory

Every file pulled into `data/` gets a row here: what it is, where it came from,
when it was accessed, and under what licence. Required by DATA_SOURCES.md 3.2.
Regenerate with `python scripts/fetch_datasets.py`.

Nothing in `data/raw/` is committed - see `.gitignore`. Only `data/reference/` is tracked.

| File | Dataset | Source URL | Accessed (UTC) | Size | sha256[:16] | Licence |
|------|---------|-----------|----------------|------|-------------|---------|
| `data/raw/dem/Copernicus_DSM_COG_10_N39_00_W008_00_DEM.tif` | Copernicus DEM GLO-30 tile N39W008 (30 m) | https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N39_00_W008_00_DEM/Copernicus_DSM_COG_10_N39_00_W008_00_DEM.tif | 2026-09-20 17:25 | 44.5 MB | `9e598aca06f93456` | Copernicus DEM, free and open (ESA/Airbus) - attribution required |
| `data/raw/dem/Copernicus_DSM_COG_10_N39_00_W009_00_DEM.tif` | Copernicus DEM GLO-30 tile N39W009 (30 m) | https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N39_00_W009_00_DEM/Copernicus_DSM_COG_10_N39_00_W009_00_DEM.tif | 2026-09-20 17:25 | 47.8 MB | `1d57c2d490d68ff6` | Copernicus DEM, free and open (ESA/Airbus) - attribution required |
| `data/raw/dem/Copernicus_DSM_COG_10_N40_00_W008_00_DEM.tif` | Copernicus DEM GLO-30 tile N40W008 (30 m) | https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N40_00_W008_00_DEM/Copernicus_DSM_COG_10_N40_00_W008_00_DEM.tif | 2026-09-20 17:25 | 43.3 MB | `bfee8bbb9a2db078` | Copernicus DEM, free and open (ESA/Airbus) - attribution required |
| `data/raw/dem/Copernicus_DSM_COG_10_N40_00_W009_00_DEM.tif` | Copernicus DEM GLO-30 tile N40W009 (30 m) | https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N40_00_W009_00_DEM/Copernicus_DSM_COG_10_N40_00_W009_00_DEM.tif | 2026-09-20 17:25 | 38.4 MB | `cd2751fbf2ec30fb` | Copernicus DEM, free and open (ESA/Airbus) - attribution required |
| `data/raw/landcover/ESA_WorldCover_10m_2021_v200_N39W009_Map.tif` | ESA WorldCover 10 m 2021 v200, tile N39W009 | https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N39W009_Map.tif | 2026-09-20 17:27 | 129.1 MB | `56f4c518cd028837` | CC-BY-4.0 - ESA WorldCover project / Contains modified Copernicus data |
| `data/raw/osm/coimbra_overpass.json` | OSM waterways, roads and exposure features for the Coimbra bbox (Overpass) | https://overpass-api.de/api/interpreter | 2026-09-20 17:25 | 11.8 MB | `1513a39ba2f18238` | ODbL - OpenStreetMap contributors (attribution required) |
| `data/raw/osm/portugal-latest.osm.pbf` | OSM Portugal extract (Geofabrik, portugal-latest.osm.pbf) | https://download.geofabrik.de/europe/portugal-latest.osm.pbf | 2026-09-20 17:26 | 423.5 MB | `6b3095ed59025f17` | ODbL - OpenStreetMap contributors (attribution required) |
| `data/raw/population/GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0_R5_C18.zip` | GHS-POP 2020, 3 arcsec, tile R5_C18 (Iberia) | https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E2020_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0_R5_C18.zip | 2026-09-20 17:27 | 102.7 MB | `11dd3bdf3c1b6c6d` | CC-BY-4.0 - European Commission, Joint Research Centre (GHSL) |

## Pending - needs an account or a manual step

| Dataset | What to do | Licence |
|---------|-----------|---------|
| Sentinel-2 L2A optical series (turbidity, NDCI, MNDWI, NDVI) | Create a CDSE OAuth client, put SH_CLIENT_ID/SH_CLIENT_SECRET in .env, then run scripts/smoke_test_sentinelhub.py. Pulled per reach polygon by the Statistical API - granules are never downloaded. | Copernicus Sentinel data - free, attribution required |
| MERIT Hydro flow direction / accumulation / upstream area (90 m) | The download is password-protected. Request access on the MERIT Hydro page; credentials arrive by email, usually within a day. Tiles needed for Coimbra: dir/upa/elv n30w010. Until then, the Copernicus DEM GLO-30 tiles fetched here are the documented fallback (pysheds fill -> flow direction -> accumulation). | CC-BY-NC 4.0 / ODbL dual - non-commercial use, state it in the README |
| CLMS Imperviousness Density, Riparian Zones, Urban Atlas | Register at land.copernicus.eu and request the Coimbra tiles. Approval can take hours. Substitute in the meantime: the WorldCover tile fetched here. | Copernicus Land Monitoring Service - free, attribution required |
| VIIRS DNB monthly nighttime lights (ALAN) | Register at eogdata.mines.edu, then pull the monthly composites for Iberia. | Earth Observation Group, Colorado School of Mines - free, attribution |
| SNIRH water quality stations (validation ground truth) | Browse manually, pick 2-5 stations on Mondego tributaries near Coimbra, and record the station IDs in config/cities/coimbra.yaml -> ground_truth.station_ids. This is the validation gold; do it on Day 0. | APA / SNIRH - open data |

## Weather

Open-Meteo responses are cached per request under `data/raw/weather/` with a
`.meta.json` sidecar carrying the URL, parameters and fetch time - that directory
is its own inventory. Weather data (c) Open-Meteo.com, CC-BY-4.0, ERA5-derived.

