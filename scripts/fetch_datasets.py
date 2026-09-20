"""Download the open datasets Kingfisher needs into data/raw/, and log provenance.

    python scripts/fetch_datasets.py --list
    python scripts/fetch_datasets.py                 # everything open-access
    python scripts/fetch_datasets.py --only osm_coimbra copdem
    python scripts/fetch_datasets.py --skip osm_portugal

Everything here is free and needs no credentials. Sources that DO need an account
(Sentinel-2 via CDSE, CLMS, VIIRS, MERIT Hydro) are listed by `--list` as MANUAL with
what to do about them; they are not silently skipped.

Downloads are resumable (HTTP Range) and idempotent: a file already present with the
expected size is left alone, so re-running costs nothing. Every completed file is
written to DATA_INVENTORY.md with its source URL, access date and licence.

Nothing downloaded here is committed. data/raw/ is gitignored; only data/reference/
(hand-curated, small) is tracked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests  # noqa: E402

from core.logging import get_logger  # noqa: E402

log = get_logger(__name__)

RAW = REPO_ROOT / "data" / "raw"
INVENTORY = REPO_ROOT / "DATA_INVENTORY.md"
CHUNK = 1 << 20

# Coimbra study area (config/cities/coimbra.yaml). Kept here as literals so this script
# stays runnable before the config loader exists.
BBOX = (-8.5200, 40.1500, -8.3400, 40.2800)  # min_lon, min_lat, max_lon, max_lat


@dataclass
class Dataset:
    key: str
    description: str
    url: str
    dest: Path
    licence: str
    approx_mb: float = 0.0
    manual: str = ""  # non-empty => needs credentials, cannot be auto-fetched
    notes: str = ""
    tags: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Open-access datasets
# --------------------------------------------------------------------------
COPDEM_TILES = [
    ("N40", "W009"),  # Coimbra and the lower Mondego
    ("N40", "W008"),  # upstream, east of the city
    ("N39", "W009"),  # south
    ("N39", "W008"),
]


def _copdem(lat: str, lon: str) -> Dataset:
    name = f"Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM"
    return Dataset(
        key=f"copdem_{lat}{lon}".lower(),
        description=f"Copernicus DEM GLO-30 tile {lat}{lon} (30 m)",
        url=f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif",
        dest=RAW / "dem" / f"{name}.tif",
        licence="Copernicus DEM, free and open (ESA/Airbus) - attribution required",
        approx_mb=37,
        notes="Fallback for MERIT Hydro: fill -> flow direction -> accumulation via pysheds.",
        tags=["copdem", "dem"],
    )


DATASETS: list[Dataset] = [
    Dataset(
        key="osm_coimbra",
        description="OSM waterways, roads and exposure features for the Coimbra bbox (Overpass)",
        url="https://overpass-api.de/api/interpreter",
        dest=RAW / "osm" / "coimbra_overpass.json",
        licence="ODbL - OpenStreetMap contributors (attribution required)",
        approx_mb=25,
        notes="Immediate L0/exposure input; does not wait on the 400 MB country extract.",
        tags=["osm"],
    ),
    Dataset(
        key="osm_portugal",
        description="OSM Portugal extract (Geofabrik, portugal-latest.osm.pbf)",
        url="https://download.geofabrik.de/europe/portugal-latest.osm.pbf",
        dest=RAW / "osm" / "portugal-latest.osm.pbf",
        licence="ODbL - OpenStreetMap contributors (attribution required)",
        approx_mb=404,
        notes="Full-country extract for pyrosm/osmium; avoids repeated Overpass load.",
        tags=["osm"],
    ),
    *[_copdem(lat, lon) for lat, lon in COPDEM_TILES],
    Dataset(
        key="worldcover",
        description="ESA WorldCover 10 m 2021 v200, tile N39W009",
        url=(
            "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
            "ESA_WorldCover_10m_2021_v200_N39W009_Map.tif"
        ),
        dest=RAW / "landcover" / "ESA_WorldCover_10m_2021_v200_N39W009_Map.tif",
        licence="CC-BY-4.0 - ESA WorldCover project / Contains modified Copernicus data",
        approx_mb=123,
        notes="Global land cover: CLMS substitute while approval is pending, and the "
        "Pune adapter's primary source.",
        tags=["landcover"],
    ),
    Dataset(
        key="ghs_pop",
        description="GHS-POP 2020, 3 arcsec, tile R5_C18 (Iberia)",
        url=(
            "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/"
            "GHS_POP_E2020_GLOBE_R2023A_4326_3ss/V1-0/tiles/"
            "GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0_R5_C18.zip"
        ),
        dest=RAW / "population" / "GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0_R5_C18.zip",
        licence="CC-BY-4.0 - European Commission, Joint Research Centre (GHSL)",
        approx_mb=98,
        notes="Exposure denominator. Tile bounds are verified after download.",
        tags=["population"],
    ),
]

# --------------------------------------------------------------------------
# Sources that need an account or a manual step. Listed, never silently skipped.
# --------------------------------------------------------------------------
MANUAL: list[Dataset] = [
    Dataset(
        key="sentinel2",
        description="Sentinel-2 L2A optical series (turbidity, NDCI, MNDWI, NDVI)",
        url="https://sh.dataspace.copernicus.eu",
        dest=RAW / "sentinelhub",
        licence="Copernicus Sentinel data - free, attribution required",
        manual=(
            "Create a CDSE OAuth client, put SH_CLIENT_ID/SH_CLIENT_SECRET in .env, then "
            "run scripts/smoke_test_sentinelhub.py. Pulled per reach polygon by the "
            "Statistical API - granules are never downloaded."
        ),
        tags=["satellite"],
    ),
    Dataset(
        key="merit_hydro",
        description="MERIT Hydro flow direction / accumulation / upstream area (90 m)",
        url="http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/",
        dest=RAW / "merit_hydro",
        licence="CC-BY-NC 4.0 / ODbL dual - non-commercial use, state it in the README",
        manual=(
            "The download is password-protected. Request access on the MERIT Hydro page; "
            "credentials arrive by email, usually within a day. Tiles needed for Coimbra: "
            "dir/upa/elv n30w010. Until then, the Copernicus DEM GLO-30 tiles fetched here "
            "are the documented fallback (pysheds fill -> flow direction -> accumulation)."
        ),
        tags=["hydrology"],
    ),
    Dataset(
        key="clms",
        description="CLMS Imperviousness Density, Riparian Zones, Urban Atlas",
        url="https://land.copernicus.eu",
        dest=RAW / "clms",
        licence="Copernicus Land Monitoring Service - free, attribution required",
        manual=(
            "Register at land.copernicus.eu and request the Coimbra tiles. Approval can "
            "take hours. Substitute in the meantime: the WorldCover tile fetched here."
        ),
        tags=["landcover"],
    ),
    Dataset(
        key="viirs",
        description="VIIRS DNB monthly nighttime lights (ALAN)",
        url="https://eogdata.mines.edu/nighttime_light/monthly/v10/",
        dest=RAW / "viirs",
        licence="Earth Observation Group, Colorado School of Mines - free, attribution",
        manual="Register at eogdata.mines.edu, then pull the monthly composites for Iberia.",
        tags=["alan"],
    ),
    Dataset(
        key="snirh",
        description="SNIRH water quality stations (validation ground truth)",
        url="https://snirh.apambiente.pt",
        dest=RAW / "snirh",
        licence="APA / SNIRH - open data",
        manual=(
            "Browse manually, pick 2-5 stations on Mondego tributaries near Coimbra, and "
            "record the station IDs in config/cities/coimbra.yaml -> ground_truth.station_ids. "
            "This is the validation gold; do it on Day 0."
        ),
        tags=["ground_truth"],
    ),
]

# Waterways, roads and the exposure features (DATA_SOURCES.md 1.7) in one Overpass call.
OVERPASS_QUERY = f"""
[out:json][timeout:180];
(
  way["waterway"~"^(river|stream|canal|ditch|drain)$"]({BBOX[1]},{BBOX[0]},{BBOX[3]},{BBOX[2]});
  way["natural"="water"]({BBOX[1]},{BBOX[0]},{BBOX[3]},{BBOX[2]});
  way["highway"]({BBOX[1]},{BBOX[0]},{BBOX[3]},{BBOX[2]});
  node["amenity"~"^(school|kindergarten|hospital|clinic)$"]({BBOX[1]},{BBOX[0]},{BBOX[3]},{BBOX[2]});
  way["amenity"~"^(school|kindergarten|hospital|clinic)$"]({BBOX[1]},{BBOX[0]},{BBOX[3]},{BBOX[2]});
  node["leisure"~"^(playground|park|garden|swimming_area)$"]({BBOX[1]},{BBOX[0]},{BBOX[3]},{BBOX[2]});
  way["leisure"~"^(playground|park|garden|swimming_area)$"]({BBOX[1]},{BBOX[0]},{BBOX[3]},{BBOX[2]});
);
out body geom;
"""


def human(mb: float) -> str:
    return f"{mb / 1024:.2f} GB" if mb >= 1024 else f"{mb:.1f} MB"


def fetch_overpass(dataset: Dataset) -> tuple[bool, str]:
    dataset.dest.parent.mkdir(parents=True, exist_ok=True)
    if dataset.dest.exists() and dataset.dest.stat().st_size > 1_000_000:
        return True, f"already present ({dataset.dest.stat().st_size / 1e6:.1f} MB)"
    started = time.perf_counter()
    response = requests.post(
        dataset.url, data={"data": OVERPASS_QUERY}, timeout=300,
        headers={"User-Agent": "kingfisher-hackathon/0.1 (urban stream monitoring)"},
    )
    if response.status_code != 200:
        raise RuntimeError(f"Overpass returned HTTP {response.status_code}: {response.text[:300]}")
    payload = response.json()
    elements = payload.get("elements", [])
    if not elements:
        raise RuntimeError("Overpass returned zero elements for the Coimbra bbox - "
                           "an empty result is a failure, not an empty extract.")
    dataset.dest.write_text(json.dumps(payload), encoding="utf-8")
    waterways = sum(1 for e in elements if e.get("tags", {}).get("waterway"))
    return True, (
        f"{len(elements):,} elements ({waterways:,} waterway ways), "
        f"{dataset.dest.stat().st_size / 1e6:.1f} MB in {time.perf_counter() - started:.1f}s"
    )


def fetch_http(dataset: Dataset) -> tuple[bool, str]:
    """Resumable streaming download. Returns (ok, message)."""
    dataset.dest.parent.mkdir(parents=True, exist_ok=True)
    part = dataset.dest.with_suffix(dataset.dest.suffix + ".part")

    head = requests.head(dataset.url, allow_redirects=True, timeout=60)
    total = int(head.headers.get("content-length", 0))

    if dataset.dest.exists():
        have = dataset.dest.stat().st_size
        if total == 0 or have == total:
            return True, f"already present ({have / 1e6:.1f} MB)"
        print(f"      size mismatch (have {have:,}, expect {total:,}) - refetching")
        dataset.dest.unlink()

    resume_from = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    if resume_from:
        print(f"      resuming at {resume_from / 1e6:.1f} MB")

    started = time.perf_counter()
    with requests.get(dataset.url, stream=True, timeout=300, headers=headers) as response:
        if response.status_code not in (200, 206):
            raise RuntimeError(f"HTTP {response.status_code} for {dataset.url}")
        mode = "ab" if response.status_code == 206 else "wb"
        written = resume_from if mode == "ab" else 0
        last_print = 0.0
        with part.open(mode) as handle:
            for chunk in response.iter_content(CHUNK):
                handle.write(chunk)
                written += len(chunk)
                now = time.perf_counter()
                if total and now - last_print > 5:
                    pct = 100.0 * written / total
                    rate = written / 1e6 / max(now - started, 0.001)
                    print(f"      {pct:5.1f}%  {written / 1e6:8.1f} / {total / 1e6:.1f} MB"
                          f"  ({rate:.1f} MB/s)")
                    last_print = now

    if total and part.stat().st_size != total:
        raise RuntimeError(
            f"incomplete download: got {part.stat().st_size:,} of {total:,} bytes. "
            "Re-run to resume."
        )
    part.replace(dataset.dest)
    size_mb = dataset.dest.stat().st_size / 1e6
    return True, f"{size_mb:.1f} MB in {time.perf_counter() - started:.0f}s"


def sha256_head(path: Path, limit: int = 64 << 20) -> str:
    """Checksum of the first `limit` bytes - enough to detect a truncated or swapped
    file without re-reading gigabytes."""
    digest = hashlib.sha256()
    read = 0
    with path.open("rb") as handle:
        while read < limit:
            chunk = handle.read(min(CHUNK, limit - read))
            if not chunk:
                break
            digest.update(chunk)
            read += len(chunk)
    return digest.hexdigest()[:16]


def write_inventory(records: list[dict]) -> None:
    """Rewrite DATA_INVENTORY.md. Provenance for every acquired file (DATA_SOURCES 3.2)."""
    lines = [
        "# Data inventory",
        "",
        "Every file pulled into `data/` gets a row here: what it is, where it came from,",
        "when it was accessed, and under what licence. Required by DATA_SOURCES.md 3.2.",
        "Regenerate with `python scripts/fetch_datasets.py`.",
        "",
        "Nothing in `data/raw/` is committed - see `.gitignore`. Only `data/reference/`",
        "is tracked.",
        "",
        "| File | Dataset | Source URL | Accessed (UTC) | Size | sha256[:16] | Licence |",
        "|------|---------|-----------|----------------|------|-------------|---------|",
    ]
    for record in sorted(records, key=lambda r: r["path"]):
        lines.append(
            f"| `{record['path']}` | {record['description']} | {record['url']} | "
            f"{record['accessed']} | {record['size_mb']:.1f} MB | `{record['sha256']}` | "
            f"{record['licence']} |"
        )
    lines += [
        "",
        "## Pending - needs an account or a manual step",
        "",
        "| Dataset | What to do | Licence |",
        "|---------|-----------|---------|",
    ]
    for dataset in MANUAL:
        lines.append(f"| {dataset.description} | {dataset.manual} | {dataset.licence} |")
    lines += [
        "",
        "## Weather",
        "",
        "Open-Meteo responses are cached per request under `data/raw/weather/` with a",
        "`.meta.json` sidecar carrying the URL, parameters and fetch time - that directory",
        "is its own inventory. Weather data (c) Open-Meteo.com, CC-BY-4.0, ERA5-derived.",
        "",
    ]
    INVENTORY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_raster_bounds(path: Path) -> str:
    """Report a raster's bounds so a wrong tile is caught immediately, not on Day 2."""
    try:
        import rasterio
    except ImportError:
        return ""
    try:
        target = path
        opener = f"zip://{path}!/" if path.suffix == ".zip" else None
        if opener:
            import zipfile

            with zipfile.ZipFile(path) as archive:
                tifs = [n for n in archive.namelist() if n.lower().endswith(".tif")]
            if not tifs:
                return "  (zip contains no .tif)"
            target = f"zip://{path}!/{tifs[0]}"
        with rasterio.open(target) as src:
            b = src.bounds
            covers = (
                b.left <= BBOX[0] and b.right >= BBOX[2]
                and b.bottom <= BBOX[1] and b.top >= BBOX[3]
            )
            intersects = (
                b.left < BBOX[2] and b.right > BBOX[0]
                and b.bottom < BBOX[3] and b.top > BBOX[1]
            )
            if covers:
                verdict = "covers the Coimbra bbox"
            elif intersects:
                verdict = "partially covers the Coimbra bbox"
            else:
                # Neighbouring tiles are deliberate: upstream tracing leaves the bbox.
                verdict = "adjacent tile (no bbox overlap) - kept for upstream tracing"
            return (f"  bounds ({b.left:.2f}, {b.bottom:.2f}, {b.right:.2f}, {b.top:.2f}) "
                    f"{src.crs} - {verdict}")
    except Exception as exc:  # noqa: BLE001 - verification must never fail the download
        return f"  (bounds check skipped: {type(exc).__name__}: {exc})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="show the manifest and exit")
    parser.add_argument("--only", nargs="+", default=None, help="dataset keys or tags to fetch")
    parser.add_argument("--skip", nargs="+", default=[], help="dataset keys or tags to skip")
    args = parser.parse_args()

    if args.list:
        print("OPEN ACCESS - fetched by this script")
        for dataset in DATASETS:
            state = "present" if dataset.dest.exists() else "missing"
            print(f"  {dataset.key:<18} {human(dataset.approx_mb):>9}  [{state:>7}]  "
                  f"{dataset.description}")
        print("\nMANUAL - needs an account or a manual step")
        for dataset in MANUAL:
            print(f"  {dataset.key:<18} {'':>9}  [{'manual':>7}]  {dataset.description}")
            print(f"      -> {dataset.manual}")
        total = sum(d.approx_mb for d in DATASETS)
        print(f"\n  total open-access download: ~{human(total)}")
        return 0

    selected = [
        d for d in DATASETS
        if (args.only is None or d.key in args.only or set(d.tags) & set(args.only))
        and not (d.key in args.skip or set(d.tags) & set(args.skip))
    ]
    if not selected:
        print("No datasets matched. Try --list.")
        return 1

    print("=" * 78)
    print(f"KINGFISHER - DATASET FETCH  ({len(selected)} datasets, "
          f"~{human(sum(d.approx_mb for d in selected))})")
    print("=" * 78)

    records: list[dict] = []
    failures: list[tuple[str, str]] = []

    for index, dataset in enumerate(selected, 1):
        print(f"\n[{index}/{len(selected)}] {dataset.key} - {dataset.description}")
        print(f"      {dataset.url}")
        try:
            if dataset.key == "osm_coimbra":
                _, message = fetch_overpass(dataset)
            else:
                _, message = fetch_http(dataset)
            print(f"      OK: {message}")
            if dataset.dest.suffix in {".tif", ".zip"}:
                detail = verify_raster_bounds(dataset.dest)
                if detail:
                    print(f"    {detail}")
            records.append(
                {
                    "path": dataset.dest.relative_to(REPO_ROOT).as_posix(),
                    "description": dataset.description,
                    "url": dataset.url,
                    "accessed": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
                    "size_mb": dataset.dest.stat().st_size / 1e6,
                    "sha256": sha256_head(dataset.dest),
                    "licence": dataset.licence,
                }
            )
        except Exception as exc:  # noqa: BLE001 - one bad source must not stop the rest
            print(f"      FAILED: {type(exc).__name__}: {exc}")
            failures.append((dataset.key, str(exc)))

    # Keep rows for files fetched in earlier runs.
    for dataset in DATASETS:
        if dataset.dest.exists() and not any(
            r["path"] == dataset.dest.relative_to(REPO_ROOT).as_posix() for r in records
        ):
            records.append(
                {
                    "path": dataset.dest.relative_to(REPO_ROOT).as_posix(),
                    "description": dataset.description,
                    "url": dataset.url,
                    "accessed": datetime.fromtimestamp(
                        dataset.dest.stat().st_mtime, UTC
                    ).strftime("%Y-%m-%d %H:%M"),
                    "size_mb": dataset.dest.stat().st_size / 1e6,
                    "sha256": sha256_head(dataset.dest),
                    "licence": dataset.licence,
                }
            )

    write_inventory(records)

    print("\n" + "=" * 78)
    print(f"  {len(records)} file(s) on disk, "
          f"{sum(r['size_mb'] for r in records) / 1000:.2f} GB total")
    print(f"  inventory written to {INVENTORY.relative_to(REPO_ROOT)}")
    if failures:
        print(f"  {len(failures)} FAILED:")
        for key, error in failures:
            print(f"    {key}: {error}")
    print("  reminder: data/raw/ is gitignored. Nothing here gets committed.")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
