"""GATE: Open-Meteo returns data, and the second run hits the cache.

    python scripts/smoke_test_weather.py

Fetches, for one Coimbra coordinate:
  1. archive-api.open-meteo.com/v1/archive   2015-01-01 .. 2024-12-31 (training history)
  2. api.open-meteo.com/v1/forecast          next 10 days (the live path)

Both responses are cached on disk under data/raw/weather/ keyed by
(lat, lon, start, end, variables_hash). Every later module uses that same cache - see
pipeline/openmeteo.py, which is the reusable client; this script only drives it and
prints the gate result.

WHY OPEN-METEO AND NOT CDS ERA5-LAND: ERA5-Land from the Climate Data Store lags real
time by ~3 months, so it cannot drive a live early-warning system. Open-Meteo's archive
is ERA5-derived (scientifically equivalent for training) and its forecast endpoint
covers the live path. One source, both jobs.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Windows consoles default to cp1252; gate output must never come back as mojibake.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.cache import DiskCache  # noqa: E402
from pipeline.openmeteo import (  # noqa: E402
    ARCHIVE_URL,
    FORECAST_URL,
    HOURLY_VARIABLES,
    OpenMeteoError,
    fetch_archive,
    fetch_forecast,
    summarise_hourly,
)

# Mondego at Coimbra. Stand-in for a catchment centroid; the real pipeline queries the
# centroid of each reach's upstream catchment polygon, never the reach itself.
LAT, LON = 40.2075, -8.4300

ARCHIVE_START = "2015-01-01"
ARCHIVE_END = "2024-12-31"
FORECAST_DAYS = 10

RULE = "=" * 78


def _report(title: str, payload: dict, hit: bool, elapsed: float) -> dict:
    summary = summarise_hourly(payload)
    total_nulls = sum(summary["nulls"].values())
    print(f"\n{title}")
    print("-" * len(title))
    print(f"  source        : {'CACHE HIT' if hit else 'NETWORK FETCH'}  ({elapsed:.2f}s)")
    print(f"  date range    : {summary['start']}  ..  {summary['end']}")
    print(f"  rows (hourly) : {summary['rows']:,}")
    print("  nulls by variable:")
    for variable, nulls in summary["nulls"].items():
        unit = summary["units"].get(variable, "")
        pct = 100.0 * nulls / summary["rows"] if summary["rows"] else 0.0
        print(f"      {variable:<32} {nulls:>7,} null ({pct:5.2f}%)  [{unit}]")
    print(f"  total nulls   : {total_nulls:,}")
    print("  note          : nulls stay null. Nothing here is filled or zeroed.")
    return summary


def main() -> int:
    print(RULE)
    print("KINGFISHER - OPEN-METEO SMOKE TEST")
    print(RULE)
    print(f"  location  : {LAT}, {LON}  (Mondego at Coimbra)")
    print(f"  variables : {', '.join(HOURLY_VARIABLES)}")
    print(f"  archive   : {ARCHIVE_URL}")
    print(f"  forecast  : {FORECAST_URL}")

    try:
        t0 = time.perf_counter()
        archive, archive_hit = fetch_archive(LAT, LON, ARCHIVE_START, ARCHIVE_END)
        archive_elapsed = time.perf_counter() - t0

        t0 = time.perf_counter()
        forecast, forecast_hit = fetch_forecast(LAT, LON, days=FORECAST_DAYS)
        forecast_elapsed = time.perf_counter() - t0
    except OpenMeteoError as exc:
        print(f"\n  FAILED: {exc}")
        print("\n  GATE: FAILED - no weather data. Check network access to open-meteo.com.")
        print(RULE)
        return 1

    archive_summary = _report(
        f"1. ARCHIVE  {ARCHIVE_START} .. {ARCHIVE_END}", archive, archive_hit, archive_elapsed
    )
    forecast_summary = _report(
        f"2. FORECAST  next {FORECAST_DAYS} days", forecast, forecast_hit, forecast_elapsed
    )

    # Second call, same parameters: must be served from disk.
    t0 = time.perf_counter()
    _, archive_hit_2 = fetch_archive(LAT, LON, ARCHIVE_START, ARCHIVE_END)
    _, forecast_hit_2 = fetch_forecast(LAT, LON, days=FORECAST_DAYS)
    replay_elapsed = time.perf_counter() - t0

    cache = DiskCache("weather")
    cached_files = sorted(cache.root.glob("*.json"))
    cached_bytes = sum(f.stat().st_size for f in cached_files)

    print("\n3. CACHE")
    print("--------")
    print(f"  directory     : {cache.root.relative_to(REPO_ROOT)}")
    print(f"  files         : {len(cached_files)} ({cached_bytes / 1024:.0f} KB)")
    print(f"  replay        : archive_hit={archive_hit_2}  forecast_hit={forecast_hit_2}"
          f"  ({replay_elapsed:.3f}s)")

    expect_rows = 10 * 365 * 24  # ~10 years hourly, leap days aside
    ok = (
        archive_summary["rows"] > expect_rows * 0.95
        and forecast_summary["rows"] >= FORECAST_DAYS * 24
        and archive_hit_2
        and forecast_hit_2
    )

    print(f"\n{RULE}")
    if ok:
        print("  GATE: PASSED")
        print(f"    archive  {archive_summary['rows']:,} hourly rows "
              f"({archive_summary['start'][:10]} .. {archive_summary['end'][:10]})")
        print(f"    forecast {forecast_summary['rows']:,} hourly rows "
              f"({forecast_summary['start'][:10]} .. {forecast_summary['end'][:10]})")
        print("    cache hit on replay for both endpoints")
    else:
        print("  GATE: FAILED")
        if archive_summary["rows"] <= expect_rows * 0.95:
            print(f"    archive returned {archive_summary['rows']:,} rows, "
                  f"expected ~{expect_rows:,}")
        if forecast_summary["rows"] < FORECAST_DAYS * 24:
            print(f"    forecast returned {forecast_summary['rows']:,} rows, "
                  f"expected >= {FORECAST_DAYS * 24}")
        if not (archive_hit_2 and forecast_hit_2):
            print("    second run did NOT hit the cache - the cache key is unstable")
    print(RULE)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
