"""As-issued weather: the future drivers the live system would actually have had.

Run:

    python -m pipeline.asissued_weather --city coimbra --fetch      # resumable, budgeted
    python -m pipeline.asissued_weather --city coimbra --build      # -> parquet

WHY
---
Training and the "oracle" evaluation use archive weather (ERA5) for the t+h driver
columns - perfect prognosis. The live system has a forecast instead. This module replays
every 00 UTC run of the pinned live model (config weather.forecast_model, ECMWF IFS HRES
9 km, archived by Open-Meteo's Single Runs API from 2024-03-14) and rebuilds the same
future-driver features from it, so skill can be reported both ways on the same rows.

FETCH
-----
One request per run carries every weather grid cell (the API takes coordinate lists).
Each response is cached before it is parsed (CLAUDE.md #4). Open-Meteo weights a request
as n_locations x ceil(days/14) x ceil(vars/10) calls, and the free tier is 10,000/day and
5,000/hour; the fetcher checks the rolling weighted totals in the cache ledger before
every request and stops with CallBudgetExhausted when a cap would be crossed - rerun
later and it resumes from the cache. Requests are also paced to stay under 600/minute.

THE SPLICE - what an issue at 00 UTC on day t knows
---------------------------------------------------
  days <= t-1           archive (ERA5). Honest caveat: in real time ERA5 lags ~5 days,
                        so antecedent terms here are still slightly optimistic.
  day t, hour 00        archive. Open-Meteo labels hourly precipitation with the END of
                        the hour, so 00 UTC is rain that fell 23-24 UTC on day t-1 -
                        before the run, and the run itself returns null for it.
  day t 01..23, t+1..   the run.
A day of the run with fewer than 24 hourly values is NULL, never a partial sum; the
240 h HRES horizon leaves day t+10 NULL in many runs, and those rows say so.

Features rebuilt per (cell, t, h), with the definitions of pipeline/l2_drivers.py:
precip_mm, precip_max_hourly, temp_mean_c, api_7, first_flush_index at t+h, and
precip_fut_cum_mm = rain over t+1..t+h.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from core.config import as_date
from core.logging import get_logger, stage
from core.settings import DATA_DIR
from pipeline import openmeteo
from pipeline.build_dataset import PROCESSED_DIR
from pipeline.l0_network import load_city_config
from pipeline.l2_drivers import (
    FeatureSettings,
    daily_aggregate,
    engineer_features,
    fetch_cell_series,
)

log = get_logger(__name__)

FUTURE_COLUMNS = ["precip_mm", "precip_max_hourly", "first_flush_index", "api_7", "temp_mean_c"]
MIN_SECONDS_BETWEEN_REQUESTS = 1.2  # 10 locations/request -> <= 500 calls/minute
# Runs the archive says do not exist - recorded so they are never re-requested, and
# reported as missing (their issue dates get no as-issued forecast, never a fill).
UNAVAILABLE_PATH = DATA_DIR / "raw" / "weather" / "single-run_unavailable.json"


# ---------------------------------------------------------------------------
# pure
# ---------------------------------------------------------------------------
def run_hourly(payload: dict[str, Any]) -> pd.DataFrame:
    """One point of a single-run payload -> hourly frame (precip, temp), UTC index."""
    hourly = payload["hourly"]
    frame = pd.DataFrame(
        {
            "precip": pd.to_numeric(pd.Series(hourly["precipitation"]), errors="coerce"),
            "temp": pd.to_numeric(pd.Series(hourly["temperature_2m"]), errors="coerce"),
        }
    )
    frame.index = pd.to_datetime(pd.Series(hourly["time"]), utc=True)
    return frame


def run_daily(
    hourly: pd.DataFrame, issued: date, archive_hour00_precip: float, dry_threshold_mm: float
) -> pd.DataFrame:
    """Daily precip_mm / precip_max_hourly / temp_mean_c from one run, days t..t+n.

    Day t's 00 UTC precipitation comes from the archive (see module docstring); every
    other value is the run's. A day missing any hour of a variable is NULL for it.
    """
    h = hourly.copy()
    t0 = pd.Timestamp(issued, tz="UTC")
    if t0 in h.index and math.isnan(h.at[t0, "precip"]):
        h.at[t0, "precip"] = archive_hour00_precip
    day = h.index.normalize().tz_localize(None)
    g = h.groupby(day)
    out = pd.DataFrame(
        {
            "precip_mm": g["precip"].sum(min_count=1),
            "precip_max_hourly": g["precip"].max(),
            "temp_mean_c": g["temp"].mean(),
        }
    )
    complete = g.count() == 24
    out.loc[~complete["precip"], ["precip_mm", "precip_max_hourly"]] = np.nan
    out.loc[~complete["temp"], "temp_mean_c"] = np.nan
    out.index = pd.DatetimeIndex(out.index, name="date")
    return out[out.index >= pd.Timestamp(issued)]


def asissued_features(
    archive_daily: pd.DataFrame,
    run: pd.DataFrame,
    issued: date,
    horizons: list[int],
    fs: FeatureSettings,
) -> dict[int, dict[str, float]]:
    """Future-driver features for each horizon, as issued at 00 UTC on `issued`.

    archive_daily: date-indexed, contiguous, with precip_mm and antecedent_dry_days
                   (pipeline.l2_drivers definitions) - only days < issued are read, plus
                   antecedent_dry_days AT issued, which counts days before it.
    run:           run_daily() output, days issued..issued+n.
    """
    t = pd.Timestamp(issued)
    past = archive_daily.loc[archive_daily.index < t, "precip_mm"]
    # Spliced daily precip: archive for < t, run for >= t.
    precip = pd.concat([past, run["precip_mm"]])
    add_at_t = archive_daily["antecedent_dry_days"].get(t, np.nan)
    out: dict[int, dict[str, float]] = {}
    for h in horizons:
        d = t + pd.Timedelta(days=h)
        row: dict[str, float] = {c: np.nan for c in [*FUTURE_COLUMNS, "precip_fut_cum_mm"]}
        if d in run.index:
            row["precip_mm"] = float(run.at[d, "precip_mm"])
            row["precip_max_hourly"] = float(run.at[d, "precip_max_hourly"])
            row["temp_mean_c"] = float(run.at[d, "temp_mean_c"])
        # api_7(d) = sum_{i=1..7} precip(d-i) lam^i; any NULL day -> NULL.
        window = [precip.get(d - pd.Timedelta(days=i), np.nan) for i in range(1, 8)]
        weights = [fs.api_lambda**i for i in range(1, 8)]
        row["api_7"] = float(np.dot(window, weights))  # NaN propagates
        # antecedent dry days: walk back from d-1 over run days, then hand over to the
        # archive's count at t if every run day in between was dry.
        run_len = 0.0
        known = True
        for k in range(1, h + 1):
            p = precip.get(d - pd.Timedelta(days=k), np.nan)
            if math.isnan(p):
                known = False
                break
            if p >= fs.dry_day_threshold_mm:
                break
            run_len += 1
        else:
            run_len = run_len + add_at_t if not math.isnan(add_at_t) else math.nan
        add = run_len if known else math.nan
        row["first_flush_index"] = add * row["precip_max_hourly"]
        cum = [precip.get(t + pd.Timedelta(days=k), np.nan) for k in range(1, h + 1)]
        row["precip_fut_cum_mm"] = float(np.sum(cum)) if not np.isnan(cum).any() else np.nan
        out[h] = row
    return out


def run_times(start: date, end: date, hour: int) -> list[datetime]:
    days = (end - start).days
    return [
        datetime(start.year, start.month, start.day, hour) + timedelta(days=i)
        for i in range(days + 1)
    ]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_cells(city: str) -> list[tuple[float, float]]:
    from sqlalchemy import text

    from core.db import session_scope

    with session_scope() as session:
        rows = session.execute(
            text(
                "SELECT DISTINCT q.query_lat, q.query_lon FROM driver_query_points q "
                "JOIN reaches r USING (reach_id) WHERE r.city = :city ORDER BY 1, 2"
            ),
            {"city": city},
        ).all()
    if not rows:
        raise RuntimeError(f"no driver_query_points for {city} - run pipeline.l2_drivers first")
    return [(float(a), float(b)) for a, b in rows]


def _settings(city: str) -> dict[str, Any]:
    weather = load_city_config(city)["weather"]
    a = weather.get("asissued")
    if not a:
        raise ValueError(f"config weather.asissued is not set for {city}")
    return {
        "model": str(weather["forecast_model"]),
        "start": as_date(a["start"]),
        "hour": int(a.get("run_hour_utc", 0)),
        "days": int(a["forecast_days"]),
        "variables": list(a["variables"]),
        "daily_cap": int(a["daily_call_cap"]),
        "hourly_cap": int(a["hourly_call_cap"]),
    }


def fetch(city: str, *, start: date | None = None, end: date | None = None) -> dict[str, Any]:
    """Fetch every run from `start` (default: config asissued.start) to `end` (default:
    yesterday). start = end = an issue date fetches that one run - what the production
    forecast needs - without replaying the backlog."""
    cfg = _settings(city)
    cells = load_cells(city)
    end = end or datetime.now(UTC).date() - timedelta(days=1)
    start = start or cfg["start"]
    if start < cfg["start"]:
        raise ValueError(f"{start} is before the first archived run ({cfg['start']})")
    runs = run_times(start, end, cfg["hour"])
    counts: Counter[str] = Counter()
    stopped: str | None = None
    last_request = 0.0
    unavailable: dict[str, str] = (
        json.loads(UNAVAILABLE_PATH.read_text(encoding="utf-8"))
        if UNAVAILABLE_PATH.exists()
        else {}
    )
    with stage(log, "asissued_fetch", city=city) as counters:
        for i, run in enumerate(runs):
            if openmeteo.cached_single_run(cells, run, cfg["variables"], cfg["model"], cfg["days"]):
                counts["cached"] += 1
                continue
            if run.strftime("%Y-%m-%dT%H:%M") in unavailable:
                counts["unavailable"] += 1
                continue
            wait = MIN_SECONDS_BETWEEN_REQUESTS - (time.monotonic() - last_request)
            if wait > 0:
                time.sleep(wait)
            try:
                for attempt in range(1, 4):
                    try:
                        openmeteo.fetch_single_run(
                            cells,
                            run,
                            cfg["variables"],
                            model=cfg["model"],
                            days=cfg["days"],
                            daily_cap=cfg["daily_cap"],
                            hourly_cap=cfg["hourly_cap"],
                        )
                        break
                    except (openmeteo.CallBudgetExhausted, openmeteo.ModelRunUnavailable):
                        raise
                    except openmeteo.OpenMeteoError as exc:
                        # transient (e.g. a non-JSON 200); nothing was cached, so a retry
                        # cannot double-count. Third failure propagates - fail loudly.
                        if attempt == 3:
                            raise
                        log.warning(
                            "asissued.retry", run=str(run), attempt=attempt, error=str(exc)[:200]
                        )
                        time.sleep(30 * attempt)
            except openmeteo.ModelRunUnavailable as exc:
                unavailable[run.strftime("%Y-%m-%dT%H:%M")] = str(exc)[:200]
                UNAVAILABLE_PATH.write_text(json.dumps(unavailable, indent=2), encoding="utf-8")
                counts["unavailable"] += 1
                log.warning("asissued.run_unavailable", run=str(run))
                continue
            except openmeteo.CallBudgetExhausted as exc:
                stopped = str(exc)
                log.warning("asissued.budget_stop", at_run=str(run), done=i, error=stopped)
                break
            last_request = time.monotonic()
            counts["fetched"] += 1
            if counts["fetched"] % 50 == 0:
                log.info("asissued.progress", run=str(run), **counts, of=len(runs))
        counters.record(
            rows_in=len(runs),
            rows_out=counts["cached"] + counts["fetched"],
            cells=len(cells),
            requests=dict(counts),
        )
    remaining = len(runs) - counts["cached"] - counts["fetched"] - counts["unavailable"]
    return {
        "runs_total": len(runs),
        "cells": len(cells),
        **counts,
        "remaining": remaining,
        "stopped": stopped,
        "weighted_calls_last_24h": openmeteo.weighted_calls_since(
            datetime.now(UTC) - timedelta(hours=24)
        ),
    }


def build(city: str, *, horizons: list[int] | None = None) -> dict[str, Any]:
    """Every cached run -> data/processed/asissued_<city>.parquet, one row per
    (query_lat, query_lon, issued_date, horizon). Runs not yet fetched are absent, and
    the summary says how many."""
    cfg = _settings(city)
    fs = FeatureSettings.from_config()
    horizons = horizons or list(range(1, 11))
    city_cfg = load_city_config(city)
    weather = city_cfg["weather"]
    cells = load_cells(city)
    end = datetime.now(UTC).date() - timedelta(days=1)
    runs = run_times(cfg["start"], end, cfg["hour"])
    fetch_start = as_date(weather["archive_start"]) - timedelta(days=int(weather["spinup_days"]))
    rows: list[dict[str, Any]] = []
    missing_runs = 0
    with stage(log, "asissued_build", city=city) as counters:
        archive: dict[tuple[float, float], tuple[pd.DataFrame, pd.Series]] = {}
        for lat, lon in cells:
            hourly, _, _ = fetch_cell_series(
                lat, lon, fetch_start, end, list(weather["variables"]), forecast_days=None
            )
            hourly = hourly.drop(columns="source")
            daily = engineer_features(daily_aggregate(hourly, fs.duration_threshold_mm), fs)
            hour00 = hourly["precip"][hourly.index.hour == 0]
            hour00.index = hour00.index.normalize().tz_localize(None)
            archive[(lat, lon)] = (daily, hour00)
        for run in runs:
            payloads = openmeteo.cached_single_run(
                cells, run, cfg["variables"], cfg["model"], cfg["days"]
            )
            if payloads is None:
                missing_runs += 1
                continue
            issued = run.date()
            for (lat, lon), payload in zip(cells, payloads, strict=True):
                daily, hour00 = archive[(lat, lon)]
                h00 = float(hour00.get(pd.Timestamp(issued), np.nan))
                rd = run_daily(run_hourly(payload), issued, h00, fs.dry_day_threshold_mm)
                for h, feats in asissued_features(daily, rd, issued, horizons, fs).items():
                    rows.append(
                        {"query_lat": lat, "query_lon": lon, "issued_date": issued, "horizon": h}
                        | feats
                    )
        if not rows:
            raise RuntimeError("no cached as-issued runs - run with --fetch first")
        out = pd.DataFrame(rows)
        path = PROCESSED_DIR / f"asissued_{city}.parquet"
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, index=False)
        nulls = {c: int(out[c].isna().sum()) for c in [*FUTURE_COLUMNS, "precip_fut_cum_mm"]}
        null_by_h = out.groupby("horizon")["precip_mm"].apply(lambda s: int(s.isna().sum()))
        counters.record(
            rows_in=len(runs),
            rows_out=len(out),
            missing_runs=missing_runs,
            null_counts=nulls,
            precip_null_by_horizon=null_by_h.to_dict(),
        )
    return {
        "runs_total": len(runs),
        "runs_missing": missing_runs,
        "rows": len(out),
        "path": str(path),
        "null_counts": nulls,
        "precip_null_by_horizon": {int(k): int(v) for k, v in null_by_h.items()},
    }


def _issue_date(city: str, value: str | None) -> date | None:
    if value is None:
        return None
    if value != "latest":
        return date.fromisoformat(value)
    frame = pd.read_parquet(PROCESSED_DIR / f"frame_{city}.parquet", columns=["date"])
    if frame.empty:
        raise RuntimeError(f"frame_{city}.parquet is empty - run `make dataset`")
    last: pd.Timestamp = pd.to_datetime(frame["date"]).max()
    return date(last.year, last.month, last.day)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="As-issued ECMWF IFS weather (Single Runs API)")
    parser.add_argument("--city", default="coimbra")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument(
        "--issue-date",
        help="fetch only the run issued on this date (YYYY-MM-DD), or 'latest' for the "
        "modelling frame's last day - the production forecast's issue date",
    )
    args = parser.parse_args(argv)
    if not (args.fetch or args.build):
        parser.error("pass --fetch and/or --build")
    if args.fetch:
        d = _issue_date(args.city, args.issue_date)
        print(json.dumps(fetch(args.city, start=d, end=d), indent=2, default=str))
    if args.build:
        print(json.dumps(build(args.city), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
