"""GATE (Day 1): how many reaches can Sentinel-2 actually see?

    python scripts/check_observability.py --city coimbra
    python scripts/check_observability.py --city coimbra --limit 10   # verify plumbing first

Sentinel-2 is 10 m. Urban streams are 3-8 m wide. This script measures, per reach, how
many pixels in a 12-month year are simultaneously water (MNDWI > 0 and SCL = WATER) and
cloud-free - and therefore which reaches carry any optical signal at all.

One Statistical API request per reach, one year, aggregated per acquisition date. Every
response is cached under data/raw/sentinelhub/ keyed by (reach_id, dates, buffer,
evalscript hash), so a re-run costs zero processing units (DATA_SOURCES.md 3.1).

Outputs:
    data/interim/observability_<city>.json        per-reach statistics, machine readable
    data/interim/observability_<city>.geojson     reaches coloured observable/unobservable
    data/interim/observability_<city>_hist.png    histogram of median water pixel counts
    PostGIS: reaches.observable, reaches.median_water_pixels

THE VERDICT LINE IS THE POINT. "N of M reaches are optically observable at 10m" is both
the go/no-go for the study area and a genuine finding to report (MASTERSPEC 6.1) - the
unobservable reaches are not failures, they are driver-predicted reaches capped at WATCH.

WHAT COUNTS AS A WATER PIXEL
----------------------------
Strict, as specified: MNDWI > 0 AND SCL = 6 (WATER) AND the pixel is not cloud, shadow,
cirrus, saturated or no-data. The MNDWI-only count is also reported on every date,
because when the strict count collapses it matters whether the cause is "no water signal
at all" or "SCL will not call a 3 m stream water" - those imply different fixes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Windows consoles default to cp1252; gate output must never come back as mojibake.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from shapely.geometry import LineString, mapping, shape  # noqa: E402
from shapely.ops import transform as shapely_transform  # noqa: E402

from core.cache import DiskCache, cache_key  # noqa: E402
from core.logging import get_logger, stage  # noqa: E402
from core.settings import DATA_DIR, get_settings  # noqa: E402
from pipeline.l0_network import load_city_config  # noqa: E402

log = get_logger(__name__)

SH_BASE_URL = "https://sh.dataspace.copernicus.eu"
SH_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)

INTERIM_DIR = DATA_DIR / "interim"

DEFAULT_START = "2023-01-01"
DEFAULT_END = "2023-12-31"

# Half-width of the corridor sampled around the reach centreline, in metres. 15 m gives a
# 30 m corridor: three 10 m pixels across, enough to contain a channel and its immediate
# water edge without dragging in a whole street of bank.
DEFAULT_BUFFER_M = 15.0

RULE = "=" * 78

# SCL: 0 no-data, 1 saturated, 3 cloud shadow, 6 water, 8/9 cloud medium/high, 10 cirrus.
EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B03", "B11", "SCL", "dataMask"] }],
    output: [
      { id: "water", bands: 2, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}

function isUsable(scl) {
  return !(scl === 0 || scl === 1 || scl === 3 || scl === 8 || scl === 9 || scl === 10);
}

function evaluatePixel(s) {
  var mndwi = (s.B03 - s.B11) / (s.B03 + s.B11);
  var usable = s.dataMask === 1 && isUsable(s.SCL) ? 1 : 0;
  var strict = usable === 1 && mndwi > 0 && s.SCL === 6 ? 1 : 0;
  var mndwiOnly = usable === 1 && mndwi > 0 ? 1 : 0;
  return { water: [strict, mndwiOnly], dataMask: [usable] };
}
"""


# ---------------------------------------------------------------------------
# reaches in
# ---------------------------------------------------------------------------
def load_reaches_from_db(city: str) -> list[dict[str, Any]]:
    from sqlalchemy import text

    from core.db import session_scope

    with session_scope() as session:
        rows = session.execute(
            text(
                "SELECT reach_id, name, strahler_order, length_m, ST_AsGeoJSON(geom) AS geom "
                "FROM reaches WHERE city = :city ORDER BY reach_id"
            ),
            {"city": city},
        ).mappings()
        return [
            {
                "reach_id": r["reach_id"],
                "name": r["name"],
                "strahler_order": r["strahler_order"],
                "length_m": r["length_m"],
                "geom": shape(json.loads(r["geom"])),
            }
            for r in rows
        ]


def load_reaches_from_geojson(city: str) -> list[dict[str, Any]]:
    path = INTERIM_DIR / f"reaches_{city}.geojson"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m pipeline.l0_network --city {city}` first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "reach_id": f["properties"]["reach_id"],
            "name": f["properties"].get("name"),
            "strahler_order": f["properties"].get("strahler_order"),
            "length_m": f["properties"].get("length_m"),
            "geom": shape(f["geometry"]),
        }
        for f in payload["features"]
    ]


def load_reaches(city: str, source: str) -> tuple[list[dict[str, Any]], str]:
    """Reaches from PostGIS (the source of truth) or the L0 GeoJSON export."""
    if source in ("db", "auto"):
        try:
            reaches = load_reaches_from_db(city)
            if reaches:
                return reaches, "postgis"
            if source == "db":
                raise RuntimeError(f"PostGIS has no reaches for city={city}. Run L0 first.")
            log.warning("observability.db_empty", city=city, fallback="geojson")
        except Exception as exc:  # noqa: BLE001 - --source auto must survive a stopped db
            if source == "db":
                raise
            log.warning(
                "observability.db_unavailable",
                error=f"{type(exc).__name__}: {exc}",
                fallback="geojson",
            )
    reaches = load_reaches_from_geojson(city)
    if not reaches:
        raise RuntimeError("L0 GeoJSON contains no reaches - refusing to run an empty gate.")
    return reaches, "geojson"


# ---------------------------------------------------------------------------
# one request per reach
# ---------------------------------------------------------------------------
def reach_geometry(line: LineString, buffer_m: float) -> tuple[Any, float]:
    """Buffer the centreline into a sampling corridor, in the local UTM zone.

    The Statistical API reads `resolution` in the units of the request CRS - in WGS84 that
    would be degrees, and the service would collapse the reach to one ~2.5 km pixel. UTM
    metres are what make resolution=(10, 10) mean Sentinel-2's native grid (the same trap
    documented in scripts/smoke_test_sentinelhub.py).
    """
    from pyproj import Transformer
    from sentinelhub import CRS, Geometry  # type: ignore[attr-defined]

    centroid = line.centroid
    utm = CRS.get_utm_from_wgs84(centroid.x, centroid.y)
    to_utm = Transformer.from_crs("EPSG:4326", utm.ogc_string(), always_xy=True).transform
    corridor = shapely_transform(to_utm, line).buffer(buffer_m, cap_style=2)
    return Geometry(corridor, crs=utm), corridor.area


def build_request(geometry: Any, start: str, end: str, config: Any) -> Any:
    from sentinelhub import (  # type: ignore[attr-defined]
        DataCollection,
        SentinelHubStatistical,
    )

    return SentinelHubStatistical(
        aggregation=SentinelHubStatistical.aggregation(
            evalscript=EVALSCRIPT,
            time_interval=(start, end),
            aggregation_interval="P1D",
            resolution=(10, 10),
        ),
        input_data=[
            SentinelHubStatistical.input_data(
                DataCollection.SENTINEL2_L2A.define_from("s2l2a_cdse", service_url=SH_BASE_URL)
            )
        ],
        geometry=geometry,
        config=config,
    )


def sh_config(client_id: str, client_secret: str) -> Any:
    from sentinelhub import SHConfig  # type: ignore[attr-defined]

    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    config.sh_base_url = SH_BASE_URL
    config.sh_token_url = SH_TOKEN_URL
    return config


# ---------------------------------------------------------------------------
# response -> per-date rows -> per-reach statistics (pure, unit-testable)
# ---------------------------------------------------------------------------
def dates_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per acquisition date. Every number here comes from the API response.

    A date with no usable pixel is recorded with water_pixels = None and a quality_flag,
    never as a zero (CLAUDE.md #2): "the satellite could not see" and "the satellite saw
    no water" are different facts and the model must be able to tell them apart.
    """
    rows: list[dict[str, Any]] = []
    for interval in payload.get("data", []):
        day = interval["interval"]["from"][:10]
        outputs = interval.get("outputs", {})
        if not outputs:
            rows.append({"date": day, "quality_flag": "CLOUD", "water_pixels": None})
            continue
        bands = outputs["water"]["bands"]
        strict, mndwi_only = bands["B0"]["stats"], bands["B1"]["stats"]
        usable = int(strict["sampleCount"]) - int(strict["noDataCount"])
        if usable <= 0:
            rows.append({"date": day, "quality_flag": "CLOUD", "water_pixels": None})
            continue
        water = int(round(float(strict["mean"]) * usable))
        rows.append(
            {
                "date": day,
                "quality_flag": "OK" if water > 0 else "NO_WATER_PIXELS",
                "usable_pixels": usable,
                "water_pixels": water,
                "water_pixels_mndwi_only": int(round(float(mndwi_only["mean"]) * usable)),
            }
        )
    for failed in payload.get("failedIntervals", []):
        rows.append(
            {
                "date": failed.get("interval", {}).get("from", "?")[:10],
                "quality_flag": "OUT_OF_RANGE",
                "water_pixels": None,
            }
        )
    return sorted(rows, key=lambda r: r["date"])


def summarise_reach(
    reach: dict[str, Any], rows: list[dict[str, Any]], threshold: int
) -> dict[str, Any]:
    """Median / min / max water pixel count over the dates the satellite could see.

    Cloudy dates are excluded from the statistics rather than counted as zero. A reach
    with no usable date at all gets median None and observable False with the reason
    recorded - it is not silently treated as dry.
    """
    usable_rows = [r for r in rows if r["water_pixels"] is not None]
    counts = [r["water_pixels"] for r in usable_rows]
    mndwi_counts = [
        r["water_pixels_mndwi_only"] for r in usable_rows if "water_pixels_mndwi_only" in r
    ]
    median = statistics.median(counts) if counts else None
    return {
        "reach_id": reach["reach_id"],
        "name": reach["name"],
        "strahler_order": reach["strahler_order"],
        "length_m": reach["length_m"],
        "dates_returned": len(rows),
        "dates_usable": len(usable_rows),
        "dates_cloud": sum(1 for r in rows if r["quality_flag"] == "CLOUD"),
        "dates_failed": sum(1 for r in rows if r["quality_flag"] == "OUT_OF_RANGE"),
        "dates_with_water": sum(1 for c in counts if c > 0),
        "median_water_pixels": median,
        "min_water_pixels": min(counts) if counts else None,
        "max_water_pixels": max(counts) if counts else None,
        "median_water_pixels_mndwi_only": (
            statistics.median(mndwi_counts) if mndwi_counts else None
        ),
        "observable": bool(median is not None and median >= threshold),
        "reason": None if counts else "NO_USABLE_DATE",
    }


def ascii_histogram(values: list[float], bins: tuple[int, ...] = (0, 1, 2, 5, 10, 20, 50)) -> str:
    """Histogram of median water pixel counts, for a terminal that has no matplotlib."""
    edges = list(bins) + [float("inf")]
    labels, counts = [], []
    for low, high in zip(edges[:-1], edges[1:], strict=False):
        if high == float("inf"):
            labels.append(f">={low}")
            counts.append(sum(1 for v in values if v >= low))
        elif high == low + 1:
            labels.append(f"{low}")
            counts.append(sum(1 for v in values if low <= v < high))
        else:
            labels.append(f"{low}-{high - 1}")
            counts.append(sum(1 for v in values if low <= v < high))
    widest = max(counts) if counts else 0
    lines = []
    for label, count in zip(labels, counts, strict=False):
        bar = "#" * int(round(40 * count / widest)) if widest else ""
        lines.append(f"    {label:>8} px | {bar:<40} {count}")
    return "\n".join(lines)


def write_histogram_png(values: list[float], path: Path, city: str, threshold: int) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("observability.histogram_skipped", reason="matplotlib not installed")
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=range(0, int(max(values, default=0)) + 2))
    ax.axvline(threshold, linestyle="--", color="crimson", label=f"observable >= {threshold}")
    ax.set_xlabel("median water pixels per acquisition date")
    ax.set_ylabel("reaches")
    ax.set_title(f"{city}: optical observability at 10 m")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
def write_geojson(
    reaches: list[dict[str, Any]], results: dict[str, dict[str, Any]], city: str
) -> Path:
    features = []
    for reach in reaches:
        result = results.get(reach["reach_id"])
        if result is None:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(reach["geom"]),
                "properties": {
                    "reach_id": result["reach_id"],
                    "name": result["name"],
                    "strahler_order": result["strahler_order"],
                    "observable": result["observable"],
                    # For a quick QGIS/geojson.io categorised style.
                    "colour": "#1f78b4" if result["observable"] else "#b0b0b0",
                    "median_water_pixels": result["median_water_pixels"],
                    "min_water_pixels": result["min_water_pixels"],
                    "max_water_pixels": result["max_water_pixels"],
                    "median_water_pixels_mndwi_only": result["median_water_pixels_mndwi_only"],
                    "dates_usable": result["dates_usable"],
                    "dates_cloud": result["dates_cloud"],
                    "reason": result["reason"],
                },
            }
        )
    path = INTERIM_DIR / f"observability_{city}.geojson"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8"
    )
    return path


def write_postgis(results: dict[str, dict[str, Any]], city: str) -> int:
    from sqlalchemy import text

    from core.db import session_scope

    with session_scope() as session:
        updated = 0
        for result in results.values():
            updated += session.execute(
                text(
                    "UPDATE reaches SET observable = :observable, "
                    "median_water_pixels = :median WHERE reach_id = :reach_id AND city = :city"
                ),
                {
                    "observable": result["observable"],
                    "median": result["median_water_pixels"],
                    "reach_id": result["reach_id"],
                    "city": city,
                },
            ).rowcount  # type: ignore[attr-defined]
    log.info("observability.postgis.written", city=city, rows_updated=updated)
    return updated


def _or_dash(value: Any) -> str:
    """A missing count prints as '-', never as 0 - the same rule as the data model."""
    return "-" if value is None else str(value)


def credentials_error(detail: str) -> str:
    return (
        "\n  SENTINEL HUB AUTHENTICATION FAILED\n"
        f"  {detail}\n\n"
        "  1. Register at https://dataspace.copernicus.eu and confirm the email.\n"
        "  2. Create an OAuth client at\n"
        "     https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings\n"
        "  3. cp .env.example .env and set SH_CLIENT_ID / SH_CLIENT_SECRET.\n"
        "  4. Re-run scripts/smoke_test_sentinelhub.py, then this script.\n"
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Day-1 optical observability gate")
    parser.add_argument("--city", default="coimbra")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--buffer-m", type=float, default=None, help="corridor half-width")
    parser.add_argument("--limit", type=int, default=None, help="first N reaches (plumbing test)")
    parser.add_argument("--source", choices=("auto", "db", "geojson"), default="auto")
    parser.add_argument("--no-db", action="store_true", help="do not write back to PostGIS")
    parser.add_argument("--refresh", action="store_true", help="ignore the cache (spends PUs)")
    args = parser.parse_args(argv)

    cfg = load_city_config(args.city)
    threshold = int(cfg.get("observability", {}).get("min_water_pixels", 5))
    buffer_m = args.buffer_m or float(
        cfg.get("observability", {}).get("buffer_m", DEFAULT_BUFFER_M)
    )

    print(RULE)
    print(f"KINGFISHER - OPTICAL OBSERVABILITY GATE  ({args.city})")
    print(RULE)
    print(f"  window        : {args.start} .. {args.end}  (aggregation P1D)")
    print(f"  corridor      : reach centreline buffered {buffer_m:g} m each side")
    print("  water pixel   : MNDWI > 0 AND SCL = 6 (WATER) AND not cloud/shadow/cirrus")
    print(f"  observable    : median water pixels >= {threshold}")

    reaches, source = load_reaches(args.city, args.source)
    if args.limit:
        reaches = reaches[: args.limit]
    print(f"  reaches       : {len(reaches)} (from {source})")

    settings = get_settings()
    try:
        client_id, client_secret = settings.require_sentinel_hub()
    except RuntimeError as exc:
        print(credentials_error(str(exc)))
        print(f"{RULE}\n  GATE: BLOCKED - no CDSE credentials, nothing was measured.\n{RULE}")
        return 1

    config = sh_config(client_id, client_secret)
    cache = DiskCache("sentinelhub")
    evalscript_hash = cache_key({"evalscript": EVALSCRIPT})

    results: dict[str, dict[str, Any]] = {}
    network_calls = 0
    failures: Counter[str] = Counter()

    with stage(log, "check_observability", city=args.city) as counters:
        for index, reach in enumerate(reaches, start=1):
            params = {
                "reach_id": reach["reach_id"],
                "city": args.city,
                "start": args.start,
                "end": args.end,
                "buffer_m": buffer_m,
                "evalscript_hash": evalscript_hash,
                "collection": "sentinel-2-l2a",
                "aggregation": "P1D",
            }
            entry = cache.get(params, slug=f"obs_{reach['reach_id']}") if not args.refresh else None
            if entry is None:
                try:
                    geometry, _ = reach_geometry(reach["geom"], buffer_m)
                    payload = build_request(geometry, args.start, args.end, config).get_data()[0]
                except Exception as exc:  # noqa: BLE001 - one reach must not kill the gate
                    text = f"{type(exc).__name__}: {exc}"
                    failures[type(exc).__name__] += 1
                    log.error(
                        "observability.request_failed",
                        reach_id=reach["reach_id"],
                        error=text,
                    )
                    if any(t in text.lower() for t in ("401", "invalid_client", "unauthorized")):
                        print(credentials_error(text))
                        print(f"{RULE}\n  GATE: BLOCKED - authentication failed.\n{RULE}")
                        return 1
                    continue
                cache.put(
                    params,
                    payload,
                    slug=f"obs_{reach['reach_id']}",
                    url=f"{SH_BASE_URL}/api/v1/statistics",
                    source="Copernicus Sentinel-2 L2A via Sentinel Hub Statistical API (CDSE)",
                )
                network_calls += 1
            else:
                payload = entry.payload

            rows = dates_from_payload(payload)
            results[reach["reach_id"]] = summarise_reach(reach, rows, threshold)
            if index % 25 == 0 or index == len(reaches):
                log.info(
                    "observability.progress",
                    done=index,
                    total=len(reaches),
                    network_calls=network_calls,
                    observable_so_far=sum(1 for r in results.values() if r["observable"]),
                )

        if not results:
            raise RuntimeError(
                "Every Statistical API request failed - refusing to report an observability "
                f"verdict on zero measurements. Failures: {dict(failures)}"
            )

        observable = [r for r in results.values() if r["observable"]]
        medians = [
            r["median_water_pixels"]
            for r in results.values()
            if r["median_water_pixels"] is not None
        ]
        counters.record(
            rows_in=len(reaches),
            rows_out=len(results),
            rows_dropped=len(reaches) - len(results),
            reason="REQUEST_FAILED" if failures else None,
            observable=len(observable),
            network_calls=network_calls,
        )

    # ---- report -----------------------------------------------------------
    report = {
        "city": args.city,
        "window": {"start": args.start, "end": args.end},
        "buffer_m": buffer_m,
        "threshold_water_pixels": threshold,
        "reaches_measured": len(results),
        "reaches_observable": len(observable),
        "reaches_failed": sum(failures.values()),
        "failures": dict(failures),
        "network_calls": network_calls,
        "reaches": sorted(results.values(), key=lambda r: r["reach_id"]),
    }
    report_path = INTERIM_DIR / f"observability_{args.city}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    geojson_path = write_geojson(reaches, results, args.city)
    hist_path = write_histogram_png(
        medians, INTERIM_DIR / f"observability_{args.city}_hist.png", args.city, threshold
    )

    ranked = sorted(
        results.values(),
        key=lambda r: (-(r["median_water_pixels"] or -1), r["reach_id"]),
    )
    print("\n  TOP 15 REACHES BY MEDIAN WATER PIXEL COUNT")
    print(
        f"  {'reach':<10} {'name':<26} {'ord':>3} {'median':>7} {'min':>5} {'max':>5}"
        f" {'dates':>6} {'mndwi-only':>11}"
    )
    print("  " + "-" * 76)
    for row in ranked[:15]:
        print(
            f"  {row['reach_id']:<10} {(row['name'] or '-')[:26]:<26}"
            f" {row['strahler_order'] or 0:>3}"
            f" {_or_dash(row['median_water_pixels']):>7}"
            f" {_or_dash(row['min_water_pixels']):>5}"
            f" {_or_dash(row['max_water_pixels']):>5}"
            f" {row['dates_usable']:>6}"
            f" {_or_dash(row['median_water_pixels_mndwi_only']):>11}"
        )

    print("\n  HISTOGRAM OF MEDIAN WATER PIXEL COUNTS")
    print(ascii_histogram([float(m) for m in medians]))

    if not args.no_db:
        try:
            write_postgis(results, args.city)
            db_note = "reaches.observable / median_water_pixels updated"
        except Exception as exc:  # noqa: BLE001 - the measurement still stands
            db_note = f"PostGIS write FAILED ({type(exc).__name__}: {exc}) - files still written"
            log.error("observability.postgis_failed", error=str(exc))
    else:
        db_note = "--no-db: PostGIS not updated"

    total = len(results)
    n_observable = len(observable)
    print(f"\n{RULE}")
    print(f"  {n_observable} of {total} reaches are optically observable at 10m.")
    print(RULE)
    print(
        f"  driver-only reaches : {total - n_observable}  (forecast from drivers, "
        f"capped at WATCH severity)"
    )
    print(f"  no usable date      : {sum(1 for r in results.values() if r['reason'])}")
    print(f"  requests to CDSE    : {network_calls} ({total - network_calls} served from cache)")
    print(f"  report              : {report_path}")
    print(f"  geojson             : {geojson_path}")
    print(f"  histogram           : {hist_path}")
    print(f"  postgis             : {db_note}")
    print(RULE)
    if n_observable >= 15:
        print(f"  GATE: PASSED - {n_observable} observable reaches (>= 15). Build on this.")
        print(RULE)
        return 0
    print(f"  GATE: FAILED - only {n_observable} observable reaches (< 15).")
    print("  Widen the study area DOWNSTREAM to wider channel and re-run TODAY:")
    print(f"    1. edit config/cities/{args.city}.yaml bbox (extend along the main channel)")
    print(f"    2. python -m pipeline.l0_network --city {args.city}")
    print(f"    3. python scripts/check_observability.py --city {args.city}")
    print("  Do not build on a study area you cannot see (DATA_SOURCES.md 1.3).")
    print(RULE)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
