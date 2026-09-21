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

LIVE MODEL AND AS-ISSUED RUNS
-----------------------------
The live forecast is pinned to one NWP model (`models=ecmwf_ifs`, ECMWF IFS HRES 9 km)
rather than Open-Meteo's best-match blend, so the live path is the same model whose past
00 UTC runs are replayed from the Single Runs API (`fetch_single_run`) to measure
as-issued forecast skill. Skill measured on one model and served from another would be
a number about nothing.

CALL BUDGET
-----------
Open-Meteo's free tier is 10,000 calls/day, 5,000/hour, 600/minute, where one request
counts as n_locations * ceil(days / 14) * ceil(variables / 10) calls. Every cached meta
records its weighted `calls`, so the cache directory is the call ledger and
`weighted_calls_since` answers "how much of today's budget is gone".
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

import requests

from core.cache import DiskCache, hash_variables
from core.logging import get_logger

log = get_logger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"

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


class ModelRunUnavailable(OpenMeteoError):
    """The Single Runs archive has no such run (e.g. ecmwf_ifs 2025-08-05 00 UTC). A
    fact about the archive, not a transient error: recorded and skipped by the caller."""


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


def call_weight(n_locations: int, days: int, n_variables: int) -> int:
    """Open-Meteo's accounting: locations x ceil(days/14) x ceil(variables/10)."""
    return max(1, n_locations) * max(1, math.ceil(days / 14)) * max(1, math.ceil(n_variables / 10))


def _meta_calls(meta: dict[str, Any]) -> int:
    """Weighted calls of one cached response. Metas written before the ledger existed
    carry no `calls`; theirs is reconstructed from the request parameters."""
    if meta.get("calls") is not None:
        return int(meta["calls"])
    params = meta.get("params", {})
    if params.get("endpoint") == "archive":
        days = (date.fromisoformat(params["end"]) - date.fromisoformat(params["start"])).days + 1
        return call_weight(1, days, len(HOURLY_VARIABLES))
    if params.get("endpoint") == "forecast":
        return call_weight(1, 10 + int(params.get("past_days", 0)), len(HOURLY_VARIABLES))
    return 1


def weighted_calls_since(since: datetime) -> int:
    """Weighted Open-Meteo calls this cache fetched from the network after `since`."""
    return sum(
        _meta_calls(meta)
        for meta in _cache.iter_meta()
        if datetime.fromisoformat(meta["fetched_at"]) >= since
    )


def _request(url: str, params: dict[str, Any]) -> Any:
    response = _get(url, params)

    if response.status_code != 200:
        # Open-Meteo returns a JSON body with `reason` on 4xx.
        detail = response.text[:400]
        raise OpenMeteoError(
            f"Open-Meteo returned HTTP {response.status_code} for {url}\n  {detail}"
        )

    try:
        payload = response.json()
    except ValueError as exc:  # a 200 with a non-JSON body - seen 2026-09-21
        if "modelRunUnavailable" in response.text:
            raise ModelRunUnavailable(response.text[:300]) from exc
        raise OpenMeteoError(
            f"Open-Meteo returned a non-JSON body for {url}: {response.text[:200]!r}"
        ) from exc
    # A multi-location request returns a list, one payload per location, in order.
    for item in payload if isinstance(payload, list) else [payload]:
        if item.get("error"):
            raise OpenMeteoError(f"Open-Meteo error: {item.get('reason', item)}")
        hourly = item.get("hourly") or {}
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
    model: str | None = None,
    refresh: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Hourly forecast for the live path. Returns (payload, cache_hit).

    `model` pins one NWP model (e.g. "ecmwf_ifs"); None is Open-Meteo's best-match
    blend. The live pipeline always pins it - see the module docstring.

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
        **({"model": model} if model else {}),
    }
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(variables),
        "forecast_days": days,
        "timezone": "UTC",
        **({"past_days": past_days} if past_days else {}),
        **({"models": model} if model else {}),
    }
    entry = _cache.get_or_fetch(
        key,
        lambda: _request(FORECAST_URL, params),
        slug=(
            f"forecast_{lat:.4f}_{lon:.4f}_{issued_s}_{days}d"
            f"{f'_p{past_days}' if past_days else ''}{f'_{model}' if model else ''}"
            f"_{hash_variables(variables)}"
        ),
        url=FORECAST_URL,
        source="Open-Meteo forecast, CC-BY-4.0",
        refresh=refresh,
    )
    return entry.payload, entry.hit


class CallBudgetExhausted(RuntimeError):
    """The next request would take a rolling weighted call count past its cap."""


def single_run_key(
    points: Sequence[tuple[float, float]],
    run: datetime,
    variables: list[str],
    model: str,
    days: int,
) -> dict[str, Any]:
    return {
        "endpoint": "single_run",
        "model": model,
        "run": run.strftime("%Y-%m-%dT%H:%M"),
        "points": [[round(lat, 5), round(lon, 5)] for lat, lon in points],
        "forecast_days": days,
        "variables_hash": hash_variables(variables),
    }


def _run_slug(run: datetime, model: str) -> str:
    return f"single-run_{model}_{run.strftime('%Y-%m-%dT%H')}"


def cached_single_run(
    points: Sequence[tuple[float, float]],
    run: datetime,
    variables: list[str],
    model: str,
    days: int,
) -> list[dict[str, Any]] | None:
    entry = _cache.get(single_run_key(points, run, variables, model, days), _run_slug(run, model))
    return None if entry is None else list(entry.payload)


def fetch_single_run(
    points: Sequence[tuple[float, float]],
    run: datetime,
    variables: list[str],
    *,
    model: str = "ecmwf_ifs",
    days: int = 11,
    daily_cap: int = 9000,
    hourly_cap: int = 4500,
    refresh: bool = False,
) -> tuple[list[dict[str, Any]], bool]:
    """One archived NWP run exactly as issued, for every point in ONE request (the
    Single Runs API accepts coordinate lists). Returns (payload per point, cache_hit).

    Before a network call the rolling 24 h and 1 h weighted call counts are checked
    against the caps; exceeding either raises CallBudgetExhausted. The caller stops, and
    every run already fetched stays cached.
    """
    key = single_run_key(points, run, variables, model, days)
    slug = _run_slug(run, model)
    if not refresh:
        entry = _cache.get(key, slug)
        if entry is not None:
            return list(entry.payload), True
    calls = call_weight(len(points), days, len(variables))
    now = datetime.now(UTC)
    day_used = weighted_calls_since(now - timedelta(hours=24))
    hour_used = weighted_calls_since(now - timedelta(hours=1))
    if day_used + calls > daily_cap or hour_used + calls > hourly_cap:
        raise CallBudgetExhausted(
            f"Open-Meteo budget: {day_used} weighted calls in the last 24 h (cap {daily_cap}), "
            f"{hour_used} in the last hour (cap {hourly_cap}); the next request costs {calls}."
        )
    params = {
        "latitude": ",".join(str(lat) for lat, _ in points),
        "longitude": ",".join(str(lon) for _, lon in points),
        "models": model,
        "run": key["run"],
        "hourly": ",".join(variables),
        "forecast_days": days,
        "timezone": "UTC",
    }
    payload = _request(SINGLE_RUNS_URL, params)
    payloads = payload if isinstance(payload, list) else [payload]
    if len(payloads) != len(points):
        raise OpenMeteoError(
            f"run {key['run']}: asked for {len(points)} points, got {len(payloads)}"
        )
    _cache.put(
        key,
        payloads,
        slug=slug,
        url=SINGLE_RUNS_URL,
        source=f"Open-Meteo Single Runs API, {model} run {key['run']}, CC-BY-4.0",
        extra_meta={"calls": calls, "request": json.dumps(params)},
    )
    return payloads, False


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
