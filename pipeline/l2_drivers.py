"""L2 - drivers: Open-Meteo weather at each reach's UPSTREAM CATCHMENT centroid, hourly,
aggregated to UTC days, then engineered into exactly the MASTERSPEC 6.2 feature set.
Writes drivers_daily and driver_query_points.

Run:

    python -m pipeline.l2_drivers --city coimbra
    python -m pipeline.l2_drivers --city coimbra --no-forecast     # archive only

Run it AFTER L1: upstream_state_lag1 is built from S2 observations. Re-running is cheap
(every weather response is cached), so re-run after each L1 pass.

WHERE THE WEATHER COMES FROM
----------------------------
The query point is the centroid of the reach's upstream contributing catchment, not the
reach (MASTERSPEC 6.2) - the weather that drives a reach falls on its catchment. The
centroid is then snapped to the `weather.query_grid_deg` grid (0.1 deg, ERA5-Land's
native spacing). Open-Meteo serves one grid cell per query anyway; snapping just stops us
asking for the same cell 35 times. Every reach's centroid, snapped query point and the
grid point Open-Meteo actually served are written to driver_query_points.

Training history is the archive endpoint (ERA5/ERA5-Land derived; source='ARCHIVE').
Days after the archive's last complete day come from the forecast endpoint with
past_days, and are written with source='FORECAST' so they can never enter a training fold.

THE FEATURES (MASTERSPEC 6.2) - precise definitions, because the tests pin them
--------------------------------------------------------------------------------
Daily aggregates, per UTC date, from 24 hourly values (fewer than 24 -> NULL):
  precip_mm            sum of hourly precipitation
  precip_max_hourly    max hourly precipitation (intensity drives scour)
  precip_duration_h    hours with precipitation > duration_threshold_mm (0.5)
  temp_mean_c / temp_max_c, soil_moisture (mean), et0 (sum)

  api_k(t)             sum_{i=1..k} precip_mm(t-i) * lambda^i, lambda = 0.9, k = 7/14/30.
                       Antecedent: day t itself is excluded. NULL if any of the k days is
                       NULL - a missing day is never read as a dry one.
  antecedent_dry_days  consecutive days immediately before t with precip_mm < 1.0.
                       0 if t-1 was wet. NULL when the run length cannot be known (it runs
                       back into a NULL day or the start of the series).
  first_flush_index    antecedent_dry_days(t) * precip_max_hourly(t)
  temp_low_flow_index  temp_mean_c * (1 - api_14 / max_api_14), the ratio clipped to [0,1].
                       max_api_14 is taken over dates <= splits.train_end ONLY. Taking it
                       over the whole series would leak the test years' wettest spell into
                       every training row.
  doy_sin, doy_cos     sin/cos(2*pi*doy/365.25)
  upstream_state_lag1  catchment-area-weighted mean over direct upstream reaches of their
                       latest OK S2 observation at or before t-1, if no older than
                       upstream_max_age_days. Two columns (turbidity, NDCI) because the
                       state has two variables; upstream_state_flag says why it is NULL.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform

from core.config import as_date, load_config
from core.logging import get_logger, stage
from pipeline import openmeteo
from pipeline.l0_network import load_city_config

log = get_logger(__name__)

HOURLY_RENAME = {
    "precipitation": "precip",
    "temperature_2m": "temp",
    "soil_moisture_0_to_7cm": "soil_moisture",
    "et0_fao_evapotranspiration": "et0",
}
REQUIRED_HOURLY = list(HOURLY_RENAME)

DRIVER_COLUMNS = [
    "reach_id",
    "date",
    "source",
    "precip_mm",
    "precip_max_hourly",
    "precip_duration_h",
    "temp_mean_c",
    "temp_max_c",
    "soil_moisture",
    "et0",
    "api_7",
    "api_14",
    "api_30",
    "antecedent_dry_days",
    "first_flush_index",
    "temp_low_flow_index",
    "doy_sin",
    "doy_cos",
    "upstream_state_lag1_turbidity",
    "upstream_state_lag1_ndci",
    "upstream_state_flag",
]


@dataclass(frozen=True)
class FeatureSettings:
    api_lambda: float
    api_windows: tuple[int, ...]
    dry_day_threshold_mm: float
    duration_threshold_mm: float
    upstream_max_age_days: int
    train_end: date

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None = None) -> FeatureSettings:
        cfg = cfg or load_config("modelling")
        f = cfg["features"]
        return cls(
            api_lambda=float(f["api_lambda"]),
            api_windows=tuple(int(k) for k in f["api_windows"]),
            dry_day_threshold_mm=float(f["dry_day_threshold_mm"]),
            duration_threshold_mm=float(f["duration_threshold_mm"]),
            upstream_max_age_days=int(f["upstream_max_age_days"]),
            train_end=as_date(cfg["splits"]["train_end"]),
        )


# ---------------------------------------------------------------------------
# pure: hourly -> daily
# ---------------------------------------------------------------------------
def hourly_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Open-Meteo payload -> hourly frame indexed by UTC timestamp. Nulls stay NaN."""
    hourly = payload["hourly"]
    missing = [v for v in REQUIRED_HOURLY if v not in hourly]
    if missing:
        raise ValueError(f"Open-Meteo payload lacks hourly variables {missing}")
    frame = pd.DataFrame(
        {
            HOURLY_RENAME[v]: pd.to_numeric(pd.Series(hourly[v]), errors="coerce")
            for v in REQUIRED_HOURLY
        }
    )
    frame.index = pd.to_datetime(pd.Series(hourly["time"]), utc=True)
    frame.index.name = "time"
    return frame


def daily_aggregate(hourly: pd.DataFrame, duration_threshold_mm: float) -> pd.DataFrame:
    """UTC-day aggregates. A variable's daily value is NULL unless all 24 hours of it are
    present - a day with a gap is not a day with less rain."""
    hourly = hourly[~hourly.index.duplicated(keep="first")].sort_index()
    day = hourly.index.tz_convert("UTC").normalize().tz_localize(None)
    grouped = hourly.groupby(day)
    complete = grouped.count() == 24

    precip = hourly["precip"]
    out = pd.DataFrame(
        {
            "precip_mm": grouped["precip"].sum(min_count=1),
            "precip_max_hourly": grouped["precip"].max(),
            "precip_duration_h": (precip > duration_threshold_mm)
            .astype(float)
            .where(precip.notna())
            .groupby(day)
            .sum(min_count=1),
            "temp_mean_c": grouped["temp"].mean(),
            "temp_max_c": grouped["temp"].max(),
            "soil_moisture": grouped["soil_moisture"].mean(),
            "et0": grouped["et0"].sum(min_count=1),
        }
    )
    source_of = {
        "precip_mm": "precip",
        "precip_max_hourly": "precip",
        "precip_duration_h": "precip",
        "temp_mean_c": "temp",
        "temp_max_c": "temp",
        "soil_moisture": "soil_moisture",
        "et0": "et0",
    }
    for column, variable in source_of.items():
        out.loc[~complete[variable], column] = np.nan
    out.index = pd.DatetimeIndex(out.index, name="date")
    return out


# ---------------------------------------------------------------------------
# pure: the load-bearing features
# ---------------------------------------------------------------------------
def _require_daily_contiguous(series: pd.Series) -> None:
    index = pd.DatetimeIndex(series.index)
    if len(index) > 1 and not (np.diff(index.values) == np.timedelta64(1, "D")).all():
        raise ValueError("feature input must be a contiguous daily series (reindex first)")


def api_k(precip: pd.Series, k: int, lam: float) -> pd.Series:
    """api_k(t) = sum_{i=1..k} precip(t-i) * lam^i. NULL for the first k days and for any
    window containing a NULL (NaN propagates through the dot product by design)."""
    _require_daily_contiguous(precip)
    values = precip.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) > k:
        windows = sliding_window_view(values, k)[:-1]  # row j = days j .. j+k-1, for t = j+k
        weights = lam ** np.arange(k, 0, -1)  # oldest day (t-k) gets lam^k, t-1 gets lam^1
        out[k:] = windows @ weights
    return pd.Series(out, index=precip.index, name=f"api_{k}")


def antecedent_dry_days(precip: pd.Series, threshold_mm: float) -> pd.Series:
    """Consecutive dry days (precip < threshold) immediately before t.

    NULL while the run length is unknown: at the series start, and after any NULL day
    until the next wet day resets the count.
    """
    _require_daily_contiguous(precip)
    values = precip.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    run: int | None = None  # dry-run length ending at the previous day; None = unknown
    for t in range(1, len(values)):
        p = values[t - 1]
        if math.isnan(p):
            run = None
        elif p >= threshold_mm:
            run = 0
        elif run is not None:
            run += 1
        out[t] = np.nan if run is None else run
    return pd.Series(out, index=precip.index, name="antecedent_dry_days")


def first_flush_index(dry_days: pd.Series, precip_max_hourly: pd.Series) -> pd.Series:
    """antecedent_dry_days * precip_max_hourly. NULL if either is NULL."""
    return (dry_days * precip_max_hourly).rename("first_flush_index")


def doy_harmonics(index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    angle = 2 * np.pi * index.dayofyear.to_numpy() / 365.25
    return (
        pd.Series(np.sin(angle), index=index, name="doy_sin"),
        pd.Series(np.cos(angle), index=index, name="doy_cos"),
    )


def max_api_training(api_14: pd.Series, train_end: date) -> float:
    """The temp_low_flow_index normaliser, from the training period only."""
    train = api_14[pd.DatetimeIndex(api_14.index) <= pd.Timestamp(train_end)]
    value = float(train.max()) if train.notna().any() else math.nan
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"max api_14 over the training period (<= {train_end}) is {value}; cannot "
            "normalise temp_low_flow_index. Is the archive empty before train_end?"
        )
    return value


def temp_low_flow_index(temp_mean: pd.Series, api_14: pd.Series, max_api: float) -> pd.Series:
    """temp_mean * (1 - api_14/max_api), wetness ratio clipped to [0, 1]: a test-period
    spell wetter than anything in training reads as 'fully wet', not as negative dryness."""
    dryness = (1.0 - api_14 / max_api).clip(lower=0.0, upper=1.0)
    return (temp_mean * dryness).rename("temp_low_flow_index")


def engineer_features(daily: pd.DataFrame, fs: FeatureSettings) -> pd.DataFrame:
    """Daily aggregates (one weather cell) -> the 6.2 driver features, except the upstream
    state, which depends on the reach rather than the cell."""
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D", name="date")
    df = daily.reindex(full)  # a missing day becomes an explicit NULL row, never a gap
    for k in fs.api_windows:
        df[f"api_{k}"] = api_k(df["precip_mm"], k, fs.api_lambda)
    df["antecedent_dry_days"] = antecedent_dry_days(df["precip_mm"], fs.dry_day_threshold_mm)
    df["first_flush_index"] = first_flush_index(df["antecedent_dry_days"], df["precip_max_hourly"])
    max_api = max_api_training(df["api_14"], fs.train_end)
    df["temp_low_flow_index"] = temp_low_flow_index(df["temp_mean_c"], df["api_14"], max_api)
    df["doy_sin"], df["doy_cos"] = doy_harmonics(full)
    df.attrs["max_api_14_train"] = max_api
    return df


# ---------------------------------------------------------------------------
# pure: upstream state
# ---------------------------------------------------------------------------
def upstream_state_lag1(
    observations: pd.DataFrame,
    topology: Sequence[tuple[str, str]],
    catchment_area: dict[str, float | None],
    reach_ids: Sequence[str],
    dates: pd.DatetimeIndex,
    max_age_days: int,
) -> pd.DataFrame:
    """Per (reach, date): area-weighted mean of direct upstream reaches' latest OK
    observation dated in [t - max_age_days, t - 1].

    `observations` columns: reach_id, obs_date, quality_flag, turbidity_proxy, ndci.
    Returns reach_id, date, upstream_state_lag1_turbidity, upstream_state_lag1_ndci,
    upstream_state_flag.
    """
    upstream_of: dict[str, list[str]] = {r: [] for r in reach_ids}
    for up, down in topology:
        if down in upstream_of:
            upstream_of[down].append(up)

    ok = observations[observations["quality_flag"] == "OK"]
    asof: dict[tuple[str, str], pd.Series] = {}
    needed = {u for ups in upstream_of.values() for u in ups}
    for rid, group in ok[ok["reach_id"].isin(needed)].groupby("reach_id"):
        g = group.assign(obs_date=pd.to_datetime(group["obs_date"])).set_index("obs_date")
        for var in ("turbidity_proxy", "ndci"):
            daily = g[var].groupby(level=0).first().reindex(dates)
            # ffill over max_age-1 days, then shift: an obs dated d serves t in [d+1, d+max_age].
            asof[(str(rid), var)] = daily.ffill(limit=max_age_days - 1).shift(1)

    frames = []
    for rid in reach_ids:
        ups = upstream_of[rid]
        out = pd.DataFrame({"reach_id": rid, "date": dates})
        if not ups:
            out["upstream_state_lag1_turbidity"] = np.nan
            out["upstream_state_lag1_ndci"] = np.nan
            out["upstream_state_flag"] = "NO_UPSTREAM_REACH"
            frames.append(out)
            continue
        areas = [catchment_area.get(u) for u in ups]
        weights = (
            np.array(areas, dtype=float)
            if all(a is not None and a > 0 for a in areas)
            else np.ones(len(ups))
        )
        present = np.zeros(len(dates), dtype=bool)
        for var, column in (
            ("turbidity_proxy", "upstream_state_lag1_turbidity"),
            ("ndci", "upstream_state_lag1_ndci"),
        ):
            matrix = np.column_stack(
                [
                    asof[(u, var)].to_numpy(dtype=float)
                    if (u, var) in asof
                    else np.full(len(dates), np.nan)
                    for u in ups
                ]
            )
            mask = ~np.isnan(matrix)
            wsum = (mask * weights).sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                value = np.where(wsum > 0, np.nansum(matrix * weights, axis=1) / wsum, np.nan)
            out[column] = value
            present |= wsum > 0
        out["upstream_state_flag"] = np.where(present, "OK", "NO_RECENT_UPSTREAM_OBS")
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# pure: where to ask for weather
# ---------------------------------------------------------------------------
def snap(value: float, grid_deg: float) -> float:
    return round(round(value / grid_deg) * grid_deg, 6)


def catchment_centroid(catchment: Any, metric_crs: str) -> tuple[float, float]:
    """Centroid computed in a metric CRS (a WGS84 centroid is skewed by latitude)."""
    to_metric = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True).transform
    to_wgs = Transformer.from_crs(metric_crs, "EPSG:4326", always_xy=True).transform
    point = shapely_transform(to_metric, catchment).centroid
    lon, lat = to_wgs(point.x, point.y)
    return float(lon), float(lat)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_reach_context(city: str) -> dict[str, Any]:
    from sqlalchemy import text

    from core.db import session_scope

    with session_scope() as session:
        reaches = (
            session.execute(
                text(
                    "SELECT reach_id, catchment_area_km2, "
                    "ST_AsGeoJSON(catchment_geom) AS catchment "
                    "FROM reaches WHERE city = :city ORDER BY reach_id"
                ),
                {"city": city},
            )
            .mappings()
            .all()
        )
        topology = session.execute(
            text(
                "SELECT t.upstream_id, t.downstream_id FROM reach_topology t "
                "JOIN reaches r ON r.reach_id = t.downstream_id WHERE r.city = :city"
            ),
            {"city": city},
        ).all()
        obs = pd.read_sql(
            text(
                "SELECT o.reach_id, o.obs_date, o.quality_flag, o.turbidity_proxy, o.ndci "
                "FROM observations o JOIN reaches r USING (reach_id) "
                "WHERE r.city = :city AND o.source = 'S2'"
            ),
            session.connection(),
            params={"city": city},
        )
    if not reaches:
        raise RuntimeError(f"No reaches in PostGIS for city={city}. Run L0 first.")
    return {"reaches": reaches, "topology": [(u, d) for u, d in topology], "observations": obs}


def fetch_cell_series(
    lat: float,
    lon: float,
    start: date,
    end: date,
    variables: list[str],
    *,
    forecast_days: int | None,
) -> tuple[pd.DataFrame, dict[str, Any], Counter[str]]:
    """Archive (year chunks, cached) + optional forecast bridge -> hourly frame with a
    `source` column. Returns (hourly, served grid point, cache counters)."""
    counts: Counter[str] = Counter()
    parts: list[pd.DataFrame] = []
    served: dict[str, Any] = {}
    year = start.year
    while year <= end.year:
        lo, hi = max(start, date(year, 1, 1)), min(end, date(year, 12, 31))
        payload, hit = openmeteo.fetch_archive(lat, lon, lo, hi, variables)
        counts["archive_hit" if hit else "archive_fetch"] += 1
        served = {k: payload.get(k) for k in ("latitude", "longitude", "elevation")}
        parts.append(hourly_frame(payload).assign(source="ARCHIVE"))
        year += 1
    hourly = pd.concat(parts)
    hourly = hourly[~hourly.index.duplicated(keep="first")].sort_index()

    if forecast_days:
        complete_days = hourly["precip"].notna().groupby(hourly.index.normalize()).sum()
        full_days = complete_days[complete_days == 24]
        if full_days.empty:
            raise RuntimeError(f"Open-Meteo archive has no complete day at {lat},{lon}")
        last_archive = full_days.index.max().tz_localize(None).date()
        today = datetime.now(UTC).date()
        past_days = min(92, max(0, (today - last_archive).days))
        payload, hit = openmeteo.fetch_forecast(
            lat, lon, days=forecast_days, variables=variables, issued=today, past_days=past_days
        )
        counts["forecast_hit" if hit else "forecast_fetch"] += 1
        forecast = hourly_frame(payload).assign(source="FORECAST")
        cutoff = pd.Timestamp(last_archive + timedelta(days=1), tz="UTC")
        hourly = pd.concat([hourly[hourly.index < cutoff], forecast[forecast.index >= cutoff]])
    return hourly, served, counts


def build_drivers(
    city: str,
    *,
    end: date | None = None,
    forecast: bool = True,
    write_db: bool = True,
) -> dict[str, Any]:
    city_cfg = load_city_config(city)
    weather = city_cfg["weather"]
    fs = FeatureSettings.from_config()
    archive_start = as_date(weather["archive_start"])
    fetch_start = archive_start - timedelta(days=int(weather.get("spinup_days", 365)))
    end = end or (datetime.now(UTC).date() - timedelta(days=1))
    grid = float(weather.get("query_grid_deg", 0.1))
    variables = list(weather["variables"])
    for required in REQUIRED_HOURLY:
        if required not in variables:
            raise ValueError(f"config weather.variables must include {required}")
    metric_crs = city_cfg["crs"]["metric"]

    ctx = load_reach_context(city)
    with stage(log, "l2_drivers", city=city) as counters:
        # ---- where each reach's weather comes from --------------------------------
        points: list[dict[str, Any]] = []
        for r in ctx["reaches"]:
            if r["catchment"] is None:
                # Never fall back to the reach's own location: that is the wrong
                # hydrological unit, and doing it silently would hide the gap.
                counters.drop(1, "NO_CATCHMENT")
                continue
            lon, lat = catchment_centroid(shape(json.loads(r["catchment"])), metric_crs)
            points.append(
                {
                    "reach_id": r["reach_id"],
                    "centroid_lon": lon,
                    "centroid_lat": lat,
                    "query_lon": snap(lon, grid),
                    "query_lat": snap(lat, grid),
                    "grid_deg": grid,
                }
            )
        if not points:
            raise RuntimeError("No reach has a catchment polygon - nothing to query.")
        cells = sorted({(p["query_lat"], p["query_lon"]) for p in points})
        log.info("l2.query_points", reaches=len(points), unique_cells=len(cells), grid_deg=grid)

        # ---- per cell: fetch, aggregate, engineer ---------------------------------
        cell_features: dict[tuple[float, float], pd.DataFrame] = {}
        served_by_cell: dict[tuple[float, float], dict[str, Any]] = {}
        cache_counts: Counter[str] = Counter()
        hourly_rows = 0
        for lat, lon in cells:
            hourly, served, counts = fetch_cell_series(
                lat,
                lon,
                fetch_start,
                end,
                variables,
                forecast_days=int(weather.get("forecast_days", 10)) if forecast else None,
            )
            cache_counts.update(counts)
            hourly_rows += len(hourly)
            daily = daily_aggregate(hourly.drop(columns="source"), fs.duration_threshold_mm)
            day_index = hourly.index.normalize().tz_localize(None)
            daily["source"] = (
                (hourly["source"] == "FORECAST")
                .groupby(day_index)
                .any()
                .map({True: "FORECAST", False: "ARCHIVE"})
            )
            features = engineer_features(daily.drop(columns="source"), fs)
            features["source"] = daily["source"].reindex(features.index)
            features = features[features.index >= pd.Timestamp(archive_start)]
            null_days = int(features["precip_mm"].isna().sum())
            log.info(
                "l2.cell",
                lat=lat,
                lon=lon,
                served_lat=served.get("latitude"),
                served_lon=served.get("longitude"),
                days=len(features),
                forecast_days=int((features["source"] == "FORECAST").sum()),
                null_precip_days=null_days,
                max_api_14_train=round(features.attrs["max_api_14_train"], 2),
                **counts,
            )
            cell_features[(lat, lon)] = features
            served_by_cell[(lat, lon)] = served

        # ---- broadcast to reaches + upstream state --------------------------------
        all_dates = pd.date_range(
            min(f.index.min() for f in cell_features.values()),
            max(f.index.max() for f in cell_features.values()),
            freq="D",
        )
        area = {r["reach_id"]: r["catchment_area_km2"] for r in ctx["reaches"]}
        upstream = upstream_state_lag1(
            ctx["observations"],
            ctx["topology"],
            area,
            [p["reach_id"] for p in points],
            all_dates,
            fs.upstream_max_age_days,
        )
        frames = []
        for p in points:
            f = cell_features[(p["query_lat"], p["query_lon"])]
            frames.append(f.rename_axis("date").reset_index().assign(reach_id=p["reach_id"]))
            served = served_by_cell[(p["query_lat"], p["query_lon"])]
            p.update(
                served_lon=served.get("longitude"),
                served_lat=served.get("latitude"),
                served_elevation_m=served.get("elevation"),
                source="Open-Meteo archive (ERA5/ERA5-Land derived), CC-BY-4.0",
            )
        drivers = pd.concat(frames, ignore_index=True).merge(
            upstream, on=["reach_id", "date"], how="left", validate="one_to_one"
        )
        drivers["date"] = drivers["date"].dt.date
        drivers = drivers[DRIVER_COLUMNS]

        null_counts = {
            c: int(drivers[c].isna().sum())
            for c in DRIVER_COLUMNS
            if c not in ("reach_id", "date", "source") and drivers[c].isna().any()
        }
        counters.record(
            rows_in=hourly_rows,
            rows_out=len(drivers),
            reaches=len(points),
            unique_cells=len(cells),
            date_range=f"{drivers['date'].min()}..{drivers['date'].max()}",
            forecast_rows=int((drivers["source"] == "FORECAST").sum()),
            null_counts=null_counts,
            upstream_flags=dict(Counter(drivers["upstream_state_flag"])),
            weather_requests=dict(cache_counts),
        )
        if write_db:
            write_drivers(drivers, pd.DataFrame(points))

    return {
        "rows": len(drivers),
        "reaches": len(points),
        "unique_cells": len(cells),
        "null_counts": null_counts,
        "forecast_rows": int((drivers["source"] == "FORECAST").sum()),
    }


def write_drivers(drivers: pd.DataFrame, points: pd.DataFrame) -> None:
    from sqlalchemy import text

    from core.db import bulk_upsert, session_scope

    reach_ids = list(points["reach_id"])
    with session_scope() as session:
        # Stale forecast rows are replaced wholesale; archive rows are upserted.
        session.execute(
            text("DELETE FROM drivers_daily WHERE source = 'FORECAST' AND reach_id = ANY(:ids)"),
            {"ids": reach_ids},
        )
    written = bulk_upsert(drivers, "drivers_daily", ["reach_id", "date"])
    bulk_upsert(points, "driver_query_points", ["reach_id"])
    log.info("l2.postgis.written", drivers_rows=written, query_points=len(points))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="L2 weather drivers")
    parser.add_argument("--city", default="coimbra")
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--no-forecast", action="store_true")
    parser.add_argument("--no-db", action="store_true")
    args = parser.parse_args(argv)
    summary = build_drivers(
        args.city, end=args.end, forecast=not args.no_forecast, write_db=not args.no_db
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
