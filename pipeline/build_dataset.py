"""Modelling frame: observations + drivers_daily + catchment_attributes, keyed by
(reach_id, date), with targets for horizons 1-10 and walk-forward splits.

Run:

    python -m pipeline.build_dataset --city coimbra

Writes data/processed/frame_<city>.parquet and prints the data summary.

ONE ROW = ONE ISSUE DATE
------------------------
A row is what the system knows on its issue date t, plus what it will be asked to predict:

  features at t      drivers_daily at t (ARCHIVE rows only; FORECAST rows never enter the
                     frame), static catchment attributes, reach attributes, and
                     <var>_asof / <var>_asof_age_days: the latest OK S2 observation dated
                     <= t, LABELLED with its age and NULL beyond asof_max_age_days. That
                     is the only carry-forward in the frame, and it says so in its name.
  future drivers     <col>_fut_h<h> = driver <col> at t+h (archive weather standing in for
                     the forecast the live system will have - perfect prognosis; report it)
  targets            target_<var>_h<h> = OK S2 observation at exactly t+h, else NULL

WALK-FORWARD SPLITS (config/modelling.yaml)
-------------------------------------------
Rows are assigned to a fold by issue date: train <= 2023, validate 2024, test 2025-2026.
No shuffling. Within a fold, any target or future-driver column that points past the
fold's end is masked (embargo): a 28 Dec 2023 training row cannot carry a 2 Jan 2024
target. `split_frame` and `walk_forward` apply this; the frame on disk is unmasked so
both walk-forward folds (train<=2023 -> 2024, train<=2024 -> 2025+) can be cut from it.

tests/test_build_dataset.py perturbs every observation and every weather value after a
fold boundary and asserts that the earlier folds come out bit-identical.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from core.config import as_date, load_config
from core.logging import get_logger, stage
from core.settings import DATA_DIR

log = get_logger(__name__)

PROCESSED_DIR = DATA_DIR / "processed"

DRIVER_FEATURES = [
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
]
STATIC_FEATURES = [
    "imperviousness_pct",
    "riparian_ndvi_mean",
    "riparian_width_m",
    "road_density_km_km2",
    "alan_radiance",
    "population",
    "urban_fraction",
]
REACH_FEATURES = ["catchment_area_km2", "strahler_order", "observable", "median_water_pixels"]
FOLDS = ("train", "val", "test")


@dataclass(frozen=True)
class FrameSettings:
    variables: tuple[str, ...]
    horizons: tuple[int, ...]
    asof_max_age_days: int
    future_driver_columns: tuple[str, ...]
    train_end: date
    val_start: date
    val_end: date
    test_start: date
    test_end: date | None

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None = None) -> FrameSettings:
        cfg = cfg or load_config("modelling")
        s, t, f = cfg["splits"], cfg["targets"], cfg["frame"]
        return cls(
            variables=tuple(t["variables"]),
            horizons=tuple(int(h) for h in t["horizons"]),
            asof_max_age_days=int(f["asof_max_age_days"]),
            future_driver_columns=tuple(f["future_driver_columns"]),
            train_end=as_date(s["train_end"]),
            val_start=as_date(s["val_start"]),
            val_end=as_date(s["val_end"]),
            test_start=as_date(s["test_start"]),
            test_end=as_date(s["test_end"]) if s.get("test_end") else None,
        )

    def fold_bounds(self, last_date: date) -> dict[str, tuple[date | None, date]]:
        return {
            "train": (None, self.train_end),
            "val": (self.val_start, self.val_end),
            "test": (self.test_start, self.test_end or last_date),
        }


def target_column(var: str, h: int) -> str:
    return f"target_{var}_h{h}"


def future_column(col: str, h: int) -> str:
    return f"{col}_fut_h{h}"


# ---------------------------------------------------------------------------
# frame construction (pure)
# ---------------------------------------------------------------------------
def _reach_frame(
    rid: str,
    drivers: pd.DataFrame,
    obs: pd.DataFrame | None,
    fs: FrameSettings,
) -> pd.DataFrame:
    """One reach. `drivers` is that reach's contiguous daily driver rows."""
    d = drivers.sort_values("date").reset_index(drop=True)
    dates = pd.DatetimeIndex(pd.to_datetime(d["date"]))
    if len(dates) > 1 and not (np.diff(dates.values) == np.timedelta64(1, "D")).all():
        raise ValueError(f"{rid}: drivers_daily is not a contiguous daily series")
    cols: dict[str, Any] = {"reach_id": rid, "date": dates}
    for col in DRIVER_FEATURES:
        cols[col] = d[col].to_numpy(dtype="float64") if col in d else np.full(len(d), np.nan)

    # Observations on the same daily axis. Only OK rows are state; everything else is a
    # labelled absence and stays NaN here.
    if obs is not None and not obs.empty:
        o = obs.assign(date=pd.to_datetime(obs["obs_date"])).groupby("date").first()
        o = o.reindex(dates)
    else:
        o = pd.DataFrame(index=dates, columns=list(fs.variables), dtype="float64")

    def ahead(values: np.ndarray, h: int) -> np.ndarray:
        shifted = np.full(len(values), np.nan)
        shifted[: len(values) - h] = values[h:]
        return shifted

    has_obs = o[list(fs.variables)].notna().any(axis=1).to_numpy()
    obs_dates = pd.Series(np.where(has_obs, dates.values, np.datetime64("NaT")), index=dates)
    age = (dates.to_series() - obs_dates.ffill()).dt.days
    fresh = (age <= fs.asof_max_age_days).to_numpy()
    for var in fs.variables:
        values = o[var].to_numpy(dtype="float64")
        # As-of value: labelled carry-forward with its age, NULL once stale.
        carried = pd.Series(values, index=dates).ffill().to_numpy()
        cols[f"{var}_asof"] = np.where(fresh, carried, np.nan)
        for h in fs.horizons:
            cols[target_column(var, h)] = ahead(values, h)
    cols["obs_asof_age_days"] = np.where(fresh, age.to_numpy(dtype="float64"), np.nan)

    for col in fs.future_driver_columns:
        for h in fs.horizons:
            cols[future_column(col, h)] = ahead(cols[col], h)
    return pd.DataFrame(cols)


def build_frame(
    drivers: pd.DataFrame,
    observations: pd.DataFrame,
    static: pd.DataFrame,
    reaches: pd.DataFrame,
    fs: FrameSettings,
) -> pd.DataFrame:
    """Assemble the (reach_id, date) modelling frame. Pure: DataFrames in, DataFrame out.

    drivers       drivers_daily rows (must be source == ARCHIVE)
    observations  observations rows; only source == S2 and quality_flag == OK are used
    static        catchment_attributes rows
    reaches       reach_id, catchment_area_km2, strahler_order, observable, median_water_pixels
    """
    if "source" in drivers and (drivers["source"] != "ARCHIVE").any():
        raise ValueError("build_frame received FORECAST driver rows - filter them first")
    ok = observations[(observations["source"] == "S2") & (observations["quality_flag"] == "OK")]
    obs_by_reach = {rid: g for rid, g in ok.groupby("reach_id")}
    frames = [
        _reach_frame(rid, g, obs_by_reach.get(rid), fs) for rid, g in drivers.groupby("reach_id")
    ]
    if not frames:
        raise RuntimeError("no driver rows - refusing to build an empty frame")
    frame = pd.concat(frames, ignore_index=True)

    static_cols = ["reach_id", *[c for c in STATIC_FEATURES if c in static]]
    frame = frame.merge(static[static_cols], on="reach_id", how="left", validate="many_to_one")
    reach_cols = ["reach_id", *[c for c in REACH_FEATURES if c in reaches]]
    frame = frame.merge(reaches[reach_cols], on="reach_id", how="left", validate="many_to_one")
    for c in STATIC_FEATURES + REACH_FEATURES:
        if c not in frame:
            frame[c] = np.nan
    frame["observable"] = frame["observable"].astype("boolean")

    last = frame["date"].max().date()
    frame["fold"] = assign_fold(frame["date"], fs, last)
    frame["reach_id"] = frame["reach_id"].astype("category")
    float_cols = frame.select_dtypes("float64").columns
    frame[float_cols] = frame[float_cols].astype("float32")
    return frame.copy()  # defragment after the column-wise edits above


def assign_fold(dates: pd.Series, fs: FrameSettings, last_date: date) -> pd.Series:
    d = pd.to_datetime(dates).dt.date
    fold = pd.Series(pd.NA, index=dates.index, dtype="string")
    for name, (lo, hi) in fs.fold_bounds(last_date).items():
        mask = (d <= hi) if lo is None else ((d >= lo) & (d <= hi))
        fold[mask] = name
    return fold


# ---------------------------------------------------------------------------
# splits (pure) - the leakage boundary
# ---------------------------------------------------------------------------
def horizon_columns(frame: pd.DataFrame, fs: FrameSettings) -> dict[int, list[str]]:
    """Every column that refers to date t+h, by h."""
    out: dict[int, list[str]] = {}
    for h in fs.horizons:
        cols = [target_column(v, h) for v in fs.variables]
        cols += [future_column(c, h) for c in fs.future_driver_columns]
        out[h] = [c for c in cols if c in frame]
    return out


def embargo(rows: pd.DataFrame, end: date, fs: FrameSettings) -> pd.DataFrame:
    """Mask every t+h column whose date t+h falls after `end`."""
    rows = rows.copy()
    d = pd.to_datetime(rows["date"])
    for h, cols in horizon_columns(rows, fs).items():
        beyond = (d + pd.Timedelta(days=h)) > pd.Timestamp(end)
        if beyond.any():
            rows.loc[beyond, cols] = np.nan
    return rows


def split_frame(frame: pd.DataFrame, fs: FrameSettings) -> dict[str, pd.DataFrame]:
    """train / val / test, each embargoed at its own end date."""
    last = pd.to_datetime(frame["date"]).max().date()
    bounds = fs.fold_bounds(last)
    return {name: embargo(frame[frame["fold"] == name], bounds[name][1], fs) for name in FOLDS}


def walk_forward(frame: pd.DataFrame, fs: FrameSettings) -> list[dict[str, Any]]:
    """Two expanding-window folds: train<=train_end -> val; train<=val_end -> test."""
    last = pd.to_datetime(frame["date"]).max().date()
    d = pd.to_datetime(frame["date"]).dt.date
    folds = []
    for name, train_end, (lo, hi) in (
        ("val", fs.train_end, (fs.val_start, fs.val_end)),
        ("test", fs.val_end, (fs.test_start, fs.test_end or last)),
    ):
        folds.append(
            {
                "name": name,
                "train_end": train_end,
                "train": embargo(frame[d <= train_end], train_end, fs),
                "eval": embargo(frame[(d >= lo) & (d <= hi)], hi, fs),
            }
        )
    return folds


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def summarise(frame: pd.DataFrame, fs: FrameSettings) -> str:
    lines: list[str] = []
    rule = "=" * 78
    targets = [target_column(v, 1) for v in fs.variables]
    any_target = (
        frame[[target_column(v, h) for v in fs.variables for h in fs.horizons]].notna().any(axis=1)
    )

    lines += [rule, "KINGFISHER - MODELLING FRAME SUMMARY", rule]
    lines.append(
        f"  rows {len(frame):,} | reaches {frame['reach_id'].nunique()} | columns "
        f"{frame.shape[1]} | dates {frame['date'].min().date()} .. {frame['date'].max().date()}"
    )
    lines.append(f"  rows with >= 1 target (trainable): {int(any_target.sum()):,}")

    lines.append("\n  DATE COVERAGE BY FOLD (after embargo)")
    lines.append(
        f"  {'fold':<6} {'from':<11} {'to':<11} {'rows':>10} {'reaches':>8} "
        + " ".join(f"{t.replace('target_', '')[:18]:>18}" for t in targets)
    )
    for name, rows in split_frame(frame, fs).items():
        if rows.empty:
            lines.append(f"  {name:<6} (empty)")
            continue
        counts = " ".join(f"{int(rows[t].notna().sum()):>18,}" for t in targets)
        lines.append(
            f"  {name:<6} {str(rows['date'].min().date()):<11} {str(rows['date'].max().date()):<11}"
            f" {len(rows):>10,} {rows['reach_id'].nunique():>8} {counts}"
        )

    lines.append("\n  OBSERVABLE vs DRIVER-ONLY (MASTERSPEC 6.1 - report, do not hide)")
    # An OK observation on day t is exactly an as-of age of 0 on day t.
    by_reach = (
        frame.assign(ok_today=frame["obs_asof_age_days"] == 0)
        .groupby("reach_id", observed=True)
        .agg(
            observable=("observable", "first"),
            rows=("date", "size"),
            ok_obs=("ok_today", "sum"),
        )
    )
    for label, mask in (
        ("observable", by_reach["observable"] == True),  # noqa: E712 - nullable boolean
        ("driver-only", by_reach["observable"] == False),  # noqa: E712
        ("unassessed", by_reach["observable"].isna()),
    ):
        sub = by_reach[mask.fillna(False)] if label != "unassessed" else by_reach[mask]
        if sub.empty and label == "unassessed":
            continue
        lines.append(
            f"  {label:<12} reaches {len(sub):>4} | rows {int(sub['rows'].sum()):>10,} | "
            f"OK S2 obs {int(sub['ok_obs'].sum()):>7,} | reaches with any OK obs "
            f"{int((sub['ok_obs'] > 0).sum()):>4}"
        )

    lines.append("\n  ROWS PER REACH")
    rows = by_reach["rows"]
    lines.append(
        f"  min {rows.min()} | median {int(rows.median())} | max {rows.max()} | "
        f"OK obs per reach: min {by_reach['ok_obs'].min()} "
        f"median {int(by_reach['ok_obs'].median())} "
        f"max {by_reach['ok_obs'].max()}"
    )
    top = by_reach.sort_values("ok_obs", ascending=False).head(10)
    lines.append(
        "  top reaches by OK observations: "
        + ", ".join(f"{rid} ({int(r.ok_obs)})" for rid, r in top.iterrows())
    )

    lines.append("\n  MISSING FRACTION PER COLUMN (all rows)")
    skip = {"reach_id", "date", "fold"}
    cols = [
        c
        for c in frame.columns
        if c not in skip and "_fut_h" not in c and not c.startswith("target_")
    ]
    missing = frame[cols].isna().mean().sort_values(ascending=False)
    for col, frac in missing.items():
        bar = "#" * int(round(frac * 30))
        lines.append(f"  {col:<32} {frac:>7.1%} {bar}")
    fut = [c for c in frame.columns if "_fut_h" in c]
    tgt = [c for c in frame.columns if c.startswith("target_")]
    if fut:
        lines.append(
            f"  {'<future drivers, ' + str(len(fut)) + ' cols>':<32} "
            f"{frame[fut].isna().mean().mean():>7.1%}  (mean)"
        )
    for var in fs.variables:
        cols_v = [c for c in tgt if f"_{var}_h" in c]
        lines.append(
            f"  {'<targets ' + var + '>':<32} {frame[cols_v].isna().mean().mean():>7.1%}  "
            "(mean over horizons - satellite revisit + cloud + observability)"
        )
    lines.append(rule)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_inputs(city: str) -> dict[str, pd.DataFrame]:
    from sqlalchemy import text

    from core.db import session_scope

    with session_scope() as session:
        conn = session.connection()
        params = {"city": city}
        reaches = pd.read_sql(
            text(
                "SELECT reach_id, catchment_area_km2, strahler_order, observable, "
                "median_water_pixels FROM reaches WHERE city = :city"
            ),
            conn,
            params=params,
        )
        drivers = pd.read_sql(
            text(
                "SELECT d.* FROM drivers_daily d JOIN reaches r USING (reach_id) "
                "WHERE r.city = :city"
            ),
            conn,
            params=params,
        )
        observations = pd.read_sql(
            text(
                "SELECT o.reach_id, o.obs_date, o.source, o.quality_flag, o.turbidity_proxy, "
                "o.ndci FROM observations o JOIN reaches r USING (reach_id) WHERE r.city = :city"
            ),
            conn,
            params=params,
        )
        static = pd.read_sql(
            text(
                "SELECT c.* FROM catchment_attributes c JOIN reaches r USING (reach_id) "
                "WHERE r.city = :city"
            ),
            conn,
            params=params,
        )
    return {"reaches": reaches, "drivers": drivers, "observations": observations, "static": static}


def build_dataset(city: str, *, write: bool = True) -> dict[str, Any]:
    fs = FrameSettings.from_config()
    inputs = load_inputs(city)
    with stage(log, "build_dataset", city=city) as counters:
        drivers = inputs["drivers"]
        if drivers.empty:
            raise RuntimeError(f"drivers_daily is empty for {city} - run pipeline.l2_drivers")
        forecast = drivers["source"] != "ARCHIVE"
        if forecast.any():
            counters.drop(int(forecast.sum()), "FORECAST_SOURCE")
        drivers = drivers[~forecast]
        if inputs["observations"].empty:
            log.warning("build_dataset.no_observations", note="targets will be all NULL - run L1")
        if inputs["static"].empty:
            log.warning("build_dataset.no_static", note="static features all NULL - run L2 static")
        frame = build_frame(
            drivers, inputs["observations"], inputs["static"], inputs["reaches"], fs
        )
        counters.record(
            rows_in=len(inputs["drivers"]),
            rows_out=len(frame),
            reaches=int(frame["reach_id"].nunique()),
            obs_ok=int(
                (
                    (inputs["observations"]["quality_flag"] == "OK")
                    & (inputs["observations"]["source"] == "S2")
                ).sum()
            ),
        )
        if write:
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            path = PROCESSED_DIR / f"frame_{city}.parquet"
            frame.to_parquet(path, index=False)
            log.info(
                "build_dataset.written", path=str(path), rows=len(frame), columns=frame.shape[1]
            )
    print(summarise(frame, fs))
    return {"rows": len(frame), "columns": frame.shape[1]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assemble the modelling frame")
    parser.add_argument("--city", default="coimbra")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    summary = build_dataset(args.city, write=not args.no_write)
    print(json.dumps(summary, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
