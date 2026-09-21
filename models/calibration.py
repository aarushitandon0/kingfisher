"""Calibration (P5.2) - pure functions, no I/O. Applied IDENTICALLY to LightGBM and
EA-LSTM (and their +assimilation variants), fitted on the 2024 validation fold ONLY.
Every fit function refuses a row outside that fold (assert_validation_only), so the
2025-2026 test fold cannot touch a fitted number.

INTERVAL - split-conformal quantile regression (CQR)
----------------------------------------------------
Romano, Patterson & Candes (2019), "Conformalized Quantile Regression", NeurIPS 32.
Per cell = (variable, horizon bucket, observability class), on the validation rows:

    E_i = max(q10_i - y_i, y_i - q90_i)                  (transformed target space)
    Q   = the ceil((n + 1)(1 - alpha)) / n empirical quantile of E,  alpha = 1 - 0.80

and the calibrated interval is [q10 - Q, q90 + Q] - wider if the model under-covered,
narrower if it over-covered (Q < 0). The outer quantiles p05 / p95 move by the same Q so
the tails stay outside the interval; the seven quantiles are then re-sorted, and the
number of rows that needed it is recorded. A cell with fewer than min_conformal_rows
validation rows is NOT adjusted and says so - it is never pooled silently from another
cell. Driver-only reaches have no targets, so their cell is always "no calibration rows".

EXCEEDANCE PROBABILITY - isotonic recalibration
-----------------------------------------------
Per (variable, observability class): isotonic regression of the event (observed >
threshold) on the forecast probability, fitted on the validation fold ONLY IF it holds
>= min_exceedance_events events for that class. Otherwise the class is left
uncalibrated and the record says "uncalibrated: insufficient events" (metrics.json).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from core.config import as_date
from models.assimilation import _fwd, _inv, qcols

OBS_CLASSES = ("observable", "driver_only", "unassessed")
INSUFFICIENT_EVENTS = "uncalibrated: insufficient events"


class TestFoldTouched(AssertionError):
    """A calibration fit saw a row outside the validation fold."""


def assert_validation_only(rows: pd.DataFrame, val_end: Any) -> None:
    """Every row must be a validation-fold row whose issue AND target date are on or
    before the validation fold's end."""
    end = pd.Timestamp(as_date(val_end))
    if rows.empty:
        return
    if "fold" in rows and (rows["fold"] != "val").any():
        bad = sorted(set(rows.loc[rows["fold"] != "val", "fold"].astype(str)))
        raise TestFoldTouched(f"calibration fit given non-validation folds {bad}")
    for col in ("issued_date", "target_date"):
        if col in rows and (pd.to_datetime(rows[col]) > end).any():
            raise TestFoldTouched(f"calibration fit given {col} after {end.date()}")


def obs_class(rows: pd.DataFrame) -> pd.Series:
    flag = rows["reach_observable"].astype("boolean")
    out = pd.Series("unassessed", index=rows.index, dtype=object)
    out[(flag == True).fillna(False)] = "observable"  # noqa: E712
    out[(flag == False).fillna(False)] = "driver_only"  # noqa: E712
    return out


def bucket_of(horizon: int, buckets: Mapping[str, Sequence[int]]) -> str:
    for name, hs in buckets.items():
        if int(horizon) in hs:
            return name
    raise ValueError(f"horizon {horizon} in no bucket {dict(buckets)}")


# ---------------------------------------------------------------------------
# conformal
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConformalCell:
    variable: str
    bucket: str
    obs_class: str
    n: int
    q_adjust: float | None  # transformed units; None = not adjusted
    reason: str | None = None
    coverage_before: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "q_adjust": self.q_adjust,
            "adjusted": self.q_adjust is not None,
            "reason": self.reason,
            "coverage_80_before": self.coverage_before,
        }


@dataclass
class ConformalFit:
    target_coverage: float
    cells: dict[tuple[str, str, str], ConformalCell] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "target_coverage": self.target_coverage,
            "method": "CQR (Romano et al. 2019)",
        }
        for (v, b, c), cell in sorted(self.cells.items()):
            out.setdefault(v, {}).setdefault(b, {})[c] = cell.as_dict()
        return out


def conformity_scores(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.asarray(np.maximum(lo - y, y - hi))


def conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    """The ceil((n+1) * coverage) / n empirical quantile (finite-sample valid). If that
    exceeds n the interval would have to be infinite; return +inf - never clip it."""
    n = len(scores)
    k = math.ceil((n + 1) * coverage)
    if k > n:
        return math.inf
    return float(np.sort(scores)[k - 1])


def fit_conformal(
    val_rows: pd.DataFrame,
    *,
    transforms: Mapping[str, str],
    buckets: Mapping[str, Sequence[int]],
    val_end: Any,
    target_coverage: float = 0.80,
    min_rows: int = 30,
) -> ConformalFit:
    assert_validation_only(val_rows, val_end)
    fit = ConformalFit(target_coverage)
    rows = val_rows[val_rows["target"].notna()]
    cls = obs_class(rows)
    bk = rows["horizon"].map(lambda h: bucket_of(int(h), buckets))
    for var in sorted(val_rows["variable"].unique()):
        tr = transforms[str(var)]
        for b in buckets:
            for c in OBS_CLASSES:
                m = ((rows["variable"] == var) & (bk == b) & (cls == c)).to_numpy()
                n = int(m.sum())
                if n == 0:
                    fit.cells[(str(var), b, c)] = ConformalCell(
                        str(var), b, c, 0, None, "no calibration rows"
                    )
                    continue
                g = rows[m]
                y = _fwd(g["target"].to_numpy(dtype="float64"), tr)
                lo = _fwd(g["p10"].to_numpy(dtype="float64"), tr)
                hi = _fwd(g["p90"].to_numpy(dtype="float64"), tr)
                cov = float(np.mean((y >= lo) & (y <= hi)))
                if n < min_rows:
                    fit.cells[(str(var), b, c)] = ConformalCell(
                        str(var), b, c, n, None, f"fewer than {min_rows} calibration rows", cov
                    )
                    continue
                q = conformal_quantile(conformity_scores(y, lo, hi), target_coverage)
                if not math.isfinite(q):
                    fit.cells[(str(var), b, c)] = ConformalCell(
                        str(var), b, c, n, None, "coverage unattainable with n rows", cov
                    )
                    continue
                fit.cells[(str(var), b, c)] = ConformalCell(str(var), b, c, n, q, None, cov)
    return fit


def apply_conformal(
    rows: pd.DataFrame,
    fit: ConformalFit,
    *,
    transforms: Mapping[str, str],
    buckets: Mapping[str, Sequence[int]],
) -> pd.DataFrame:
    """Adjusted copy. Adds conformal_q (NaN where the cell was not adjusted),
    conformal_adjusted, quantiles_resorted."""
    out = rows.copy().reset_index(drop=True)
    cls = obs_class(out)
    bk = out["horizon"].map(lambda h: bucket_of(int(h), buckets))
    q_adj = np.full(len(out), np.nan)
    for (v, b, c), cell in fit.cells.items():
        if cell.q_adjust is None:
            continue
        m = ((out["variable"] == v) & (bk == b) & (cls == c)).to_numpy()
        q_adj[m] = cell.q_adjust
    cols = qcols(out)
    resorted = np.zeros(len(out), dtype=bool)
    for var in out["variable"].unique():
        m = (out["variable"] == var).to_numpy() & np.isfinite(q_adj)
        if not m.any():
            continue
        tr = transforms[str(var)]
        qt = np.column_stack([_fwd(out.loc[m, c].to_numpy(dtype="float64"), tr) for c in cols])
        for j, c in enumerate(cols):
            if c in ("p05", "p10"):
                qt[:, j] -= q_adj[m]
            elif c in ("p90", "p95"):
                qt[:, j] += q_adj[m]
        crossed = np.any(np.diff(qt, axis=1) < 0, axis=1)
        qt = np.sort(qt, axis=1)
        resorted[np.flatnonzero(m)[crossed]] = True
        for j, c in enumerate(cols):
            out.loc[m, c] = _inv(qt[:, j], tr)
    out["conformal_q"] = q_adj
    out["conformal_adjusted"] = np.isfinite(q_adj)
    out["quantiles_resorted"] = resorted
    return out


# ---------------------------------------------------------------------------
# isotonic
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IsotonicMap:
    x: tuple[float, ...]
    y: tuple[float, ...]

    def __call__(self, p: np.ndarray) -> np.ndarray:
        out = np.interp(p, self.x, self.y)
        return np.asarray(np.where(np.isnan(p), np.nan, out))


def fit_isotonic(
    prob: np.ndarray, event: np.ndarray, *, min_events: int
) -> tuple[IsotonicMap | None, dict[str, Any]]:
    """(map or None, record). Caller must pass validation-fold rows only."""
    from sklearn.isotonic import IsotonicRegression

    ok = ~np.isnan(prob)
    p, e = prob[ok], event[ok].astype(float)
    n_events = int(e.sum())
    rec: dict[str, Any] = {"n": int(ok.sum()), "events": n_events, "min_events": min_events}
    if n_events < min_events:
        rec["status"] = INSUFFICIENT_EVENTS
        return None, rec
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
    iso.fit(p, e)
    rec["status"] = "calibrated: isotonic"
    return IsotonicMap(
        tuple(map(float, iso.X_thresholds_)), tuple(map(float, iso.y_thresholds_))
    ), rec


def fit_isotonic_by_class(
    val_rows: pd.DataFrame, prob: np.ndarray, *, min_events: int, val_end: Any
) -> tuple[dict[tuple[str, str], IsotonicMap], dict[str, Any]]:
    """val_rows carries variable, reach_observable, target, threshold; prob aligned."""
    assert_validation_only(val_rows, val_end)
    maps: dict[tuple[str, str], IsotonicMap] = {}
    record: dict[str, Any] = {}
    cls = obs_class(val_rows).to_numpy()
    thr = val_rows["threshold"].to_numpy(dtype="float64")
    y = val_rows["target"].to_numpy(dtype="float64")
    for var in sorted(val_rows["variable"].unique()):
        for c in OBS_CLASSES:
            m = (
                (val_rows["variable"] == var).to_numpy()
                & (cls == c)
                & ~np.isnan(thr)
                & ~np.isnan(y)
            )
            if not m.any():
                record.setdefault(str(var), {})[c] = {
                    "n": 0,
                    "events": 0,
                    "status": INSUFFICIENT_EVENTS,
                    "min_events": min_events,
                }
                continue
            mp, rec = fit_isotonic(prob[m], y[m] > thr[m], min_events=min_events)
            record.setdefault(str(var), {})[c] = rec
            if mp is not None:
                maps[(str(var), c)] = mp
    return maps, record


def apply_isotonic(
    rows: pd.DataFrame, prob: np.ndarray, maps: Mapping[tuple[str, str], IsotonicMap]
) -> tuple[np.ndarray, np.ndarray]:
    """(probabilities, calibrated flag). Rows of an uncalibrated class keep the raw value."""
    out = prob.copy()
    flag = np.zeros(len(prob), dtype=bool)
    cls = obs_class(rows).to_numpy()
    var = rows["variable"].to_numpy()
    for (v, c), mp in maps.items():
        m = (var == v) & (cls == c)
        out[m] = mp(prob[m])
        flag[m] = True
    return out, flag


def val_end_from_config(cfg: Mapping[str, Any]) -> date:
    return as_date(cfg["splits"]["val_end"])
