"""Per-reach seasonal exceedance thresholds - pure, no I/O.

config/thresholds.yaml defines them: for each (reach, variable, season), the configured
percentile of that reach's OK observations in that season, over a fitting period that
ends before the period being judged. Too few observations -> no threshold (a missing
row), and the caller must treat that as INSUFFICIENT_EVIDENCE, never as a low bar.

The same table feeds models/evaluate.py (reliability diagram) and engine/alerts.py.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

TABLE_COLUMNS = [
    "reach_id",
    "variable",
    "season",
    "threshold",
    "n_obs",
    "clim_exceed_freq",
    "percentile",
    "fit_end",
]


def month_to_season(seasons: dict[str, list[int]]) -> dict[int, str]:
    out: dict[int, str] = {}
    for name, months in seasons.items():
        for m in months:
            if m in out:
                raise ValueError(f"month {m} is in two seasons ({out[m]}, {name})")
            out[int(m)] = name
    if sorted(out) != list(range(1, 13)):
        raise ValueError(f"seasons must cover months 1-12 exactly once, got {sorted(out)}")
    return out


def season_of(dates: pd.Series, seasons: dict[str, list[int]]) -> pd.Series:
    lookup = month_to_season(seasons)
    return pd.to_datetime(dates).dt.month.map(lookup)


def seasonal_thresholds(obs: pd.DataFrame, fit_end: date, cfg: dict[str, Any]) -> pd.DataFrame:
    """obs: reach_id, date, variable, value (OK observations only).
    cfg: the whole thresholds config. Returns one row per (reach, variable, season) that
    has >= min_obs fitting-period observations; reach-seasons below that are absent."""
    seasons = cfg["seasons"]
    rows: list[dict[str, Any]] = []
    fit = obs[pd.to_datetime(obs["date"]) <= pd.Timestamp(fit_end)]
    fit = fit.assign(season=season_of(fit["date"], seasons))
    for var, vcfg in cfg["variables"].items():
        if vcfg["derivation"] != "per_reach_seasonal_percentile":
            raise ValueError(f"{var}: unsupported derivation {vcfg['derivation']!r}")
        pct, min_obs = float(vcfg["percentile"]), int(vcfg["min_obs"])
        for (rid, season), g in fit[fit["variable"] == var].groupby(["reach_id", "season"]):
            values = g["value"].to_numpy(dtype="float64")
            values = values[~np.isnan(values)]
            if len(values) < min_obs:
                continue
            thr = float(np.quantile(values, pct))
            rows.append(
                {
                    "reach_id": str(rid),
                    "variable": var,
                    "season": season,
                    "threshold": thr,
                    "n_obs": len(values),
                    "clim_exceed_freq": float((values > thr).mean()),
                    "percentile": pct,
                    "fit_end": fit_end,
                }
            )
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def lookup(
    targets: pd.DataFrame, table: pd.DataFrame, seasons: dict[str, list[int]]
) -> pd.DataFrame:
    """targets: reach_id, variable, target_date. Returns threshold, clim_exceed_freq and
    n_obs aligned to targets' index (NaN where the reach-season has no threshold)."""
    t = targets[["reach_id", "variable"]].astype(str).copy()
    t["season"] = season_of(targets["target_date"], seasons).to_numpy()
    merged = t.merge(
        table[["reach_id", "variable", "season", "threshold", "clim_exceed_freq", "n_obs"]],
        on=["reach_id", "variable", "season"],
        how="left",
        validate="many_to_one",
    )
    merged.index = targets.index
    return merged[["threshold", "clim_exceed_freq", "n_obs"]]
