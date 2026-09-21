"""Residual assimilation (P5.2) - pure functions, no I/O. Applied IDENTICALLY to the
LightGBM and EA-LSTM forecasts, so the comparison between them is fair.

    residuals(forecasts)                              -> r0 at every OK-observation date
    assimilate(forecasts, residual_table, tau, ...)   -> forecasts, shifted, with provenance
    fit_tau(val_forecasts, residual_table, grid, ...) -> tau minimising CRPS on 2024

THE CORRECTION
--------------
For a forecast issued on day t for target date t+h:

  s   = the reach's last OK observation dated ON OR BEFORE t (never after - asserted)
  r0  = T(observed_s) - T(P50 of the model's un-assimilated forecast for date s)
  age = (t + h) - s, in days
  every quantile q at t+h becomes  T^-1( T(q) + r0 * exp(-age / tau) )

T is the target's modelling transform (log1p for the turbidity index, identity for NDCI;
config/modelling.yaml ealstm.targets), so the shift is a location shift of the whole
predictive distribution in the space the models are trained in, and every quantile moves
by the same amount there. tau is fitted on the 2024 validation fold only.

"The model's forecast for date s" is the shortest-horizon forecast for target s in the
same fold - for the EA-LSTM under hindcast weather every horizon gives the same number
(the simulation for s); for LightGBM it is the 1-day-ahead forecast issued s-1. Only
out-of-sample forecasts are used: a residual date must lie inside the fold (a fold's
model has trained on everything before it). Early in a fold that means no correction.

NO CORRECTION - `assimilated` is false and `assim_reason` says why:
  UNOBSERVABLE_REACH    reach_observable is false or NULL (driver-only reaches)
  NO_OBSERVATION        no OK observation with a residual on or before t in this fold
  STALE_OBSERVATION     t - s > max_obs_age_days (21, the alert engine's staleness
                        guardrail - the same rule, applied at the issue date)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pipeline.export_neuralhydrology import INVERSE_TRANSFORMS, TRANSFORMS

QCOLS = ("p05", "p10", "p25", "p50", "p75", "p90", "p95")
KEY = ["fold", "variable", "reach_id"]

UNOBSERVABLE = "UNOBSERVABLE_REACH"
NO_OBS = "NO_OBSERVATION"
STALE = "STALE_OBSERVATION"


class LeakageError(AssertionError):
    """An observation dated after the issue date reached a forecast."""


@dataclass(frozen=True)
class AssimilationSettings:
    tau_grid: tuple[float, ...]
    max_obs_age_days: int

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> AssimilationSettings:
        a = cfg["assimilation"]
        return cls(tuple(float(t) for t in a["tau_grid_days"]), int(a["max_obs_age_days"]))


def transforms_from_config(cfg: Mapping[str, Any]) -> dict[str, str]:
    """{variable: transform} from config/modelling.yaml ealstm.targets."""
    return {v: str(t["transform"]) for v, t in cfg["ealstm"]["targets"].items()}


def _fwd(values: np.ndarray, transform: str) -> np.ndarray:
    return np.asarray(TRANSFORMS[transform](values))


def _inv(values: np.ndarray, transform: str) -> np.ndarray:
    return np.asarray(INVERSE_TRANSFORMS[transform](values))


def qcols(df: pd.DataFrame) -> list[str]:
    return [c for c in QCOLS if c in df.columns]


# ---------------------------------------------------------------------------
# residuals
# ---------------------------------------------------------------------------
def residuals(forecasts: pd.DataFrame, transforms: Mapping[str, str]) -> pd.DataFrame:
    """r0 at every observed target date: fold, variable, reach_id, obs_date, r0 (in the
    transformed space), source_horizon. One row per (fold, variable, reach, date).

    forecasts: the un-assimilated forecast rows (fold, variable, reach_id, issued_date,
    horizon, target_date, target, p50). Rows without an observed target are ignored."""
    f = forecasts[forecasts["target"].notna() & forecasts["p50"].notna()]
    if f.empty:
        return pd.DataFrame(columns=[*KEY, "obs_date", "r0", "source_horizon"])
    f = f.sort_values("horizon").drop_duplicates([*KEY, "target_date"], keep="first")
    parts = []
    for var, g in f.groupby("variable"):
        tr = transforms[str(var)]
        r0 = _fwd(g["target"].to_numpy(dtype="float64"), tr) - _fwd(
            g["p50"].to_numpy(dtype="float64"), tr
        )
        parts.append(
            pd.DataFrame(
                {
                    "fold": g["fold"].to_numpy(),
                    "variable": var,
                    "reach_id": g["reach_id"].astype(str).to_numpy(),
                    "obs_date": pd.to_datetime(g["target_date"]).to_numpy(),
                    "r0": r0,
                    "source_horizon": g["horizon"].to_numpy(dtype="int64"),
                }
            )
        )
    out = pd.concat(parts, ignore_index=True)
    return out[np.isfinite(out["r0"])].reset_index(drop=True)


# ---------------------------------------------------------------------------
# the correction
# ---------------------------------------------------------------------------
def _observable(rows: pd.DataFrame) -> np.ndarray:
    flag = rows["reach_observable"].astype("boolean")
    return np.asarray(flag.fillna(False).to_numpy(dtype=bool))


def attach_last_residual(forecasts: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    """For every forecast row, the last residual dated ON OR BEFORE its issue date, same
    fold / variable / reach. Adds obs_date (NaT if none) and r0."""
    left = forecasts.assign(
        _row=np.arange(len(forecasts)),
        _t=pd.to_datetime(forecasts["issued_date"]),
        reach_id=forecasts["reach_id"].astype(str),
    )
    if res.empty:
        return left.assign(obs_date=pd.NaT, r0=np.nan).drop(columns=["_t"])
    right = res[[*KEY, "obs_date", "r0"]].assign(_t=pd.to_datetime(res["obs_date"]))
    out = []
    # merge_asof needs one sort key; do it per (fold, variable) with `by` on reach_id
    for (fold, var), g in left.groupby(["fold", "variable"], sort=False):
        r = right[(right["fold"] == fold) & (right["variable"] == var)]
        g = g.sort_values("_t")
        if r.empty:
            out.append(g.assign(obs_date=pd.NaT, r0=np.nan))
            continue
        m = pd.merge_asof(
            g,
            r.drop(columns=["fold", "variable"]).sort_values("_t"),
            on="_t",
            by="reach_id",
            direction="backward",  # obs_date <= issued_date, never after
            allow_exact_matches=True,
        )
        out.append(m)
    merged = pd.concat(out, ignore_index=True).sort_values("_row").drop(columns=["_t"])
    used = merged["obs_date"].notna()
    late = pd.to_datetime(merged.loc[used, "obs_date"]) > pd.to_datetime(
        merged.loc[used, "issued_date"]
    )
    if late.any():
        raise LeakageError(f"{int(late.sum())} forecasts would use an observation after issue")
    return merged.reset_index(drop=True)


def assimilate(
    forecasts: pd.DataFrame,
    res: pd.DataFrame,
    tau: float | Mapping[str, float],
    *,
    max_obs_age_days: int,
    transforms: Mapping[str, str],
) -> pd.DataFrame:
    """Shift every quantile of every eligible forecast row; see the module docstring.
    tau: one value, or {variable: tau}. Returns a copy with the quantiles replaced and
    provenance columns: assimilated, assim_reason, assim_obs_date, assim_obs_age_days,
    assim_age_days, assim_shift (transformed space)."""
    if not (isinstance(tau, Mapping) or tau > 0):
        raise ValueError(f"tau must be > 0, got {tau}")
    m = attach_last_residual(forecasts, res)
    issued = pd.to_datetime(m["issued_date"])
    target = pd.to_datetime(m["target_date"])
    obs = pd.to_datetime(m["obs_date"])
    obs_age = (issued - obs).dt.days.to_numpy(dtype="float64")
    age = (target - obs).dt.days.to_numpy(dtype="float64")
    observable = _observable(m)
    has = m["r0"].notna().to_numpy()
    fresh = has & (obs_age <= max_obs_age_days)
    ok = observable & fresh

    reason = np.full(len(m), None, dtype=object)
    reason[~observable] = UNOBSERVABLE
    reason[observable & ~has] = NO_OBS
    reason[observable & has & ~fresh] = STALE

    taus = (
        m["variable"].map(lambda v: float(tau[str(v)])).to_numpy(dtype="float64")
        if isinstance(tau, Mapping)
        else np.full(len(m), float(tau))
    )
    shift = np.where(ok, m["r0"].to_numpy(dtype="float64") * np.exp(-age / taus), 0.0)
    out = forecasts.copy().reset_index(drop=True)
    for var in out["variable"].unique():
        rows = (out["variable"] == var).to_numpy() & ok
        if not rows.any():
            continue
        tr = transforms[str(var)]
        for c in qcols(out):
            v = out.loc[rows, c].to_numpy(dtype="float64")
            out.loc[rows, c] = _inv(_fwd(v, tr) + shift[rows], tr)
    out["assimilated"] = ok
    out["assim_reason"] = reason
    out["assim_obs_date"] = np.where(ok, obs.dt.date, None)
    out["assim_obs_age_days"] = np.where(ok, obs_age, np.nan)
    out["assim_age_days"] = np.where(ok, age, np.nan)
    out["assim_shift"] = shift
    return out


# ---------------------------------------------------------------------------
# tau
# ---------------------------------------------------------------------------
def crps_rows(
    rows: pd.DataFrame, levels: Sequence[float] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
) -> float:
    from models.evaluate import crps_quantile

    y = rows["target"].to_numpy(dtype="float64")
    names = [f"p{round(a * 100):02d}" for a in levels]
    return float(np.mean(crps_quantile(y, [rows[n].to_numpy() for n in names], levels)))


def fit_tau(
    val_forecasts: pd.DataFrame,
    res: pd.DataFrame,
    grid: Sequence[float],
    *,
    max_obs_age_days: int,
    transforms: Mapping[str, str],
    val_end: Any,
) -> dict[str, Any]:
    """Per variable: CRPS on the validation fold for every tau in the grid (and with no
    correction), and the argmin. Refuses any row outside the validation fold."""
    from models.calibration import assert_validation_only

    assert_validation_only(val_forecasts, val_end)
    assert_validation_only(
        res.rename(columns={"obs_date": "target_date"}).assign(issued_date=res["obs_date"]), val_end
    )
    out: dict[str, Any] = {}
    for var, g in val_forecasts.groupby("variable"):
        g = g[g["target"].notna()]
        r = res[res["variable"] == var]
        base = crps_rows(g)
        scores = {}
        for tau in grid:
            a = assimilate(
                g, r, float(tau), max_obs_age_days=max_obs_age_days, transforms=transforms
            )
            scores[str(tau)] = crps_rows(a)
        best = min(scores, key=lambda k: scores[k])
        n_corr = int(
            assimilate(g, r, float(best), max_obs_age_days=max_obs_age_days, transforms=transforms)[
                "assimilated"
            ].sum()
        )
        out[str(var)] = {
            "tau_days": float(best),
            "crps_by_tau": scores,
            "crps_no_assimilation": base,
            "crps_best": scores[best],
            "rows": int(len(g)),
            "rows_corrected": n_corr,
            "at_grid_edge": best in (str(grid[0]), str(grid[-1])),
        }
    return out
