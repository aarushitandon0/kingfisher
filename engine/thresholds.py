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


def derivation_text(cfg: dict[str, Any], variable: str) -> str:
    """How a variable's threshold is derived, in words - shown on every alert."""
    v = cfg["variables"][variable]
    return (
        f"{v['derivation']}: P{round(float(v['percentile']) * 100)} of the "
        f"{' '.join(str(v['source']).split())} (min {v['min_obs']} observations)"
    )


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
    parts: list[pd.DataFrame] = []
    fit = obs[pd.to_datetime(obs["date"]) <= pd.Timestamp(fit_end)]
    fit = pd.DataFrame(
        {
            "reach_id": fit["reach_id"].astype(str).to_numpy(dtype=object),
            "variable": fit["variable"].astype(str).to_numpy(dtype=object),
            "season": season_of(fit["date"], seasons).to_numpy(dtype=object),
            "value": pd.to_numeric(fit["value"]).to_numpy(dtype="float64"),
        }
    ).dropna(subset=["value"])
    for var, vcfg in cfg["variables"].items():
        if vcfg["derivation"] != "per_reach_seasonal_percentile":
            raise ValueError(f"{var}: unsupported derivation {vcfg['derivation']!r}")
        pct, min_obs = float(vcfg["percentile"]), int(vcfg["min_obs"])
        v = fit[fit["variable"] == var]
        if v.empty:
            continue
        # linear interpolation, as np.quantile's default
        g = v.groupby(["reach_id", "season"], sort=True)["value"]
        agg = pd.DataFrame({"threshold": g.quantile(pct), "n_obs": g.size()})
        agg = agg[agg["n_obs"] >= min_obs]
        if agg.empty:
            continue
        above = v.join(agg["threshold"], on=["reach_id", "season"], how="inner")
        freq = (
            (above["value"] > above["threshold"])
            .groupby([above["reach_id"], above["season"]])
            .mean()
        )
        agg = agg.assign(clim_exceed_freq=freq).reset_index()
        parts.append(
            agg.assign(
                variable=var,
                n_obs=agg["n_obs"].astype(int),
                clim_exceed_freq=agg["clim_exceed_freq"].astype(float),
                percentile=pct,
                fit_end=fit_end,
            )[TABLE_COLUMNS]
        )
    if not parts:
        return pd.DataFrame(columns=TABLE_COLUMNS)
    return pd.concat(parts, ignore_index=True)


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
