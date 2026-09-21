"""Open-Meteo client with an on-disk cache under `data/raw/weather/`.

WHY OPEN-METEO AND NOT CDS ERA5-LAND
------------------------------------
ERA5-Land from the Climate Data Store lags real time by roughly three months. It is
excellent for training and for citation, and useless for a live early-warning system:
a demo built only on CDS cannot forecast the present. Open-Meteo's archive endpoint is
ERA5/ERA5-Land derived, so it is scientifically equivalent for training, and its
forecast endpoint covers the live path. One source, both jobs, no API key, no queue.
CDS ERA5-Land stays on the list as an optional cross-check for the README, not as the
driver of the operational path.

Cache key is (lat, lon, start, end, variables_hash) — see `core.cache`. Query points
are the centroid of each reach's upstream catchment, not the reach itself; that is the
caller's job (pipeline/l2_drivers.py).
"""

from __future__ import annotations

import random
import time
from datetime import date
from typing import Any

import requests

from core.cache import DiskCache, hash_variables
from core.logging import get_logger

log = get_logger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# The driver variables (DATA_SOURCES.md §1.4). Hourly for training, aggregated to daily
# in l2_drivers — precip_max_hourly in particular needs the hourly series.
HOURLY_VARIABLES = [
    "precipitation",
    "temperature_2m",
    "soil_moisture_0_to_7cm",
    "et0_fao_evapotranspiration",
]

TIMEOUT_S = 120
MAX_ATTEMPTS = 6
BACKOFF_BASE_S = 5.0
BACKOFF_MAX_S = 120.0
_cache = DiskCache("weather")


class OpenMeteoError(RuntimeError):
    """Raised when Open-Meteo fails or returns nothing usable. We fail loudly."""


def _get(url: str, params: dict[str, Any]) -> requests.Response:
    """GET with exponential backoff on 429 / 5xx / connection errors.

    Open-Meteo's minutely and hourly limits clear by waiting; the DAILY limit does not,
    so a 429 that says "daily" is raised at once rather than slept on.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(url, params=params, timeout=TIMEOUT_S)
        except requests.RequestException as exc:
            if attempt == MAX_ATTEMPTS:
                raise OpenMeteoError(f"Open-Meteo request to {url} failed: {exc}") from exc
            error = f"{type(exc).__name__}: {exc}"
        else:
            retryable = response.status_code == 429 or response.status_code >= 500
            if not retryable:
                return response
            if response.status_code == 429 and "daily" in response.text.lower():
                raise OpenMeteoError(
                    f"Open-Meteo daily request limit reached: {response.text[:300]}. "
                    "Cached responses are kept; re-run tomorrow to continue."
                )
            if attempt == MAX_ATTEMPTS:
                return response
            error = f"HTTP {response.status_code}: {response.text[:200]}"
        wait = min(BACKOFF_MAX_S, BACKOFF_BASE_S * 2 ** (attempt - 1)) * (
            1 + 0.25 * random.random()
        )
        log.warning("openmeteo.retry", url=url, attempt=attempt, wait_s=round(wait, 1), error=error)
        time.sleep(wait)
    raise AssertionError("unreachable")


def _request(url: str, params: dict[str, Any]) -> dict[str, Any]:
    response = _get(url, params)

    if response.status_code != 200:
        # Open-Meteo returns a JSON body with `reason` on 4xx.
        detail = response.text[:400]
        raise OpenMeteoError(
            f"Open-Meteo returned HTTP {response.status_code} for {url}\n  {detail}"
        )

    payload = response.json()
    if payload.get("error"):
        raise OpenMeteoError(f"Open-Meteo error: {payload.get('reason', payload)}")

    hourly = payload.get("hourly") or {}
    if not hourly.get("time"):
        raise OpenMeteoError(
            f"Open-Meteo returned no hourly timesteps for {url} with params {params}. "
            "An empty result is a failure, not an empty DataFrame."
        )
    return payload


def fetch_archive(
    lat: float,
    lon: float,
    start: date | str,
    end: date | str,
    variables: list[str] | None = None,
    *,
    refresh: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Hourly ERA5-derived history. Returns (payload, cache_hit)."""
    variables = variables or HOURLY_VARIABLES
    start_s, end_s = str(start), str(end)
    key = {
        "endpoint": "archive",
        "lat": round(lat, 5),
        "lon": round(lon, 5),
        "start": start_s,
        "end": end_s,
        "variables_hash": hash_variables(variables),
    }
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_s,
        "end_date": end_s,
        "hourly": ",".join(variables),
        "timezone": "UTC",
    }
    entry = _cache.get_or_fetch(
        key,
        lambda: _request(ARCHIVE_URL, params),
        slug=f"archive_{lat:.4f}_{lon:.4f}_{start_s}_{end_s}_{hash_variables(variables)}",
        url=ARCHIVE_URL,
        source="Open-Meteo archive (ERA5/ERA5-Land derived), CC-BY-4.0",
        refresh=refresh,
    )
    return entry.payload, entry.hit


def fetch_forecast(
    lat: float,
    lon: float,
    days: int = 10,
    variables: list[str] | None = None,
    *,
    issued: date | str | None = None,
    past_days: int = 0,
    refresh: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Hourly forecast for the live path. Returns (payload, cache_hit).

    `past_days` (<= 92) prepends recent days from the forecast model, which is how the
    live series bridges the few days the archive has not caught up on yet.

    The cache key carries the issue date: a forecast fetched today is a different
    object from the same horizon fetched tomorrow, and conflating them would be a
    leak. Re-running on the same day is a hit; running tomorrow re-fetches.
    """
    variables = variables or HOURLY_VARIABLES
    issued_s = str(issued or date.today())
    key = {
        "endpoint": "forecast",
        "lat": round(lat, 5),
        "lon": round(lon, 5),
        "start": issued_s,
        "end": f"{issued_s}+{days}d",
        "variables_hash": hash_variables(variables),
        **({"past_days": past_days} if past_days else {}),
    }
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(variables),
        "forecast_days": days,
        "timezone": "UTC",
        **({"past_days": past_days} if past_days else {}),
    }
    entry = _cache.get_or_fetch(
        key,
        lambda: _request(FORECAST_URL, params),
        slug=(
            f"forecast_{lat:.4f}_{lon:.4f}_{issued_s}_{days}d"
            f"{f'_p{past_days}' if past_days else ''}_{hash_variables(variables)}"
        ),
        url=FORECAST_URL,
        source="Open-Meteo forecast, CC-BY-4.0",
        refresh=refresh,
    )
    return entry.payload, entry.hit


def summarise_hourly(payload: dict[str, Any]) -> dict[str, Any]:
    """Row counts, date range and per-variable null counts. No imputation happens here
    or anywhere else: a null stays a null and is carried into the quality flags."""
    hourly = payload["hourly"]
    times = hourly["time"]
    variables = [k for k in hourly if k != "time"]
    return {
        "rows": len(times),
        "start": times[0],
        "end": times[-1],
        "nulls": {v: sum(1 for x in hourly[v] if x is None) for v in variables},
        "units": payload.get("hourly_units", {}),
    }
