"""Prioritisation - pure functions only (MASTERSPEC 10). No I/O, no DB, no network.

information_value(reach) = forecast_uncertainty x predicted_risk x exposure_weight
Produces the ranked next-field-visit list. One panel, not a subsystem. Definitions and the
exposure weights are in config/priorities.yaml; the weights are a stated planning choice.

A reach that cannot be scored is listed in `not_ranked` with the reason - NO_THRESHOLD
(driver-only reaches have no threshold, so no risk), NO_FORECAST, NO_EXPOSURE - and is
never given a value of zero.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

METHOD = (
    "information value = forecast uncertainty x predicted risk x exposure weight. Risk: peak "
    "daily P(exceed) over the latest forecast window. Uncertainty: mean P90-P10 over the "
    "window / the largest such mean among ranked reaches (per variable). Exposure weight: "
    "log1p(E) / log1p(max E), E = weighted count of exposure features in the reach buffer + "
    "population / population_unit. The reach's value uses its highest-scoring variable."
)


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PriorityComponents(_Frozen):
    variable: str
    predicted_risk: float
    forecast_uncertainty: float
    exposure_weight: float
    exposure_score: float


class RankedReach(_Frozen):
    rank: int
    reach_id: str
    information_value: float
    components: PriorityComponents
    flags: tuple[str, ...] = ()


class NotRanked(_Frozen):
    reach_id: str
    reason: Literal["NO_THRESHOLD", "NO_FORECAST", "NO_EXPOSURE"]


class Priorities(_Frozen):
    method: str = METHOD
    weights: dict[str, float]
    population_unit: float
    ranked: tuple[RankedReach, ...]
    not_ranked: tuple[NotRanked, ...]


def exposure_score(
    rows: pd.DataFrame, weights: Mapping[str, float], population_unit: float
) -> tuple[float, tuple[str, ...]]:
    """One reach's exposure rows (feature_type, count) -> (E, flags). A NULL population is
    left out of E and flagged POPULATION_UNAVAILABLE; a NULL feature count likewise."""
    e = 0.0
    flags: list[str] = []
    for ftype, count in zip(rows["feature_type"], rows["count"], strict=True):
        missing = count is None or (isinstance(count, float) and math.isnan(count))
        if ftype == "population":
            if missing:
                flags.append("POPULATION_UNAVAILABLE")
            else:
                e += float(count) / population_unit
        elif ftype in weights:
            if missing:
                flags.append(f"{ftype.upper()}_COUNT_UNAVAILABLE")
            else:
                e += float(weights[ftype]) * float(count)
    return e, tuple(flags)


def rank_reaches(
    reach_ids: Sequence[str],
    forecasts: pd.DataFrame,
    exposure: pd.DataFrame,
    weights: Mapping[str, float],
    population_unit: float,
) -> Priorities:
    """forecasts: reach_id, variable, p10, p90, exceedance_prob (NaN where the reach-day
    has no threshold), one row per target day of the latest window.
    exposure: reach_id, feature_type, count (engine.exposure rows)."""
    if population_unit <= 0:
        raise ValueError("population_unit must be positive")
    if any(w < 0 for w in weights.values()):
        raise ValueError("exposure weights must be non-negative")
    not_ranked: list[NotRanked] = []
    per_var: dict[str, dict[str, tuple[float, float]]] = {}  # var -> reach -> (risk, width)
    candidates: list[str] = []
    for rid in dict.fromkeys(reach_ids):
        f = forecasts[forecasts["reach_id"] == rid]
        if f.empty:
            not_ranked.append(NotRanked(reach_id=rid, reason="NO_FORECAST"))
            continue
        scored = False
        for var, g in f.groupby("variable"):
            p = g["exceedance_prob"].to_numpy(dtype="float64")
            width = (g["p90"] - g["p10"]).to_numpy(dtype="float64")
            if np.isfinite(p).any() and np.isfinite(width).any():
                per_var.setdefault(str(var), {})[rid] = (
                    float(np.nanmax(p)),
                    float(np.nanmean(width)),
                )
                scored = True
        if not scored:
            not_ranked.append(NotRanked(reach_id=rid, reason="NO_THRESHOLD"))
        elif (exposure["reach_id"] == rid).any():
            candidates.append(rid)
        else:
            not_ranked.append(NotRanked(reach_id=rid, reason="NO_EXPOSURE"))

    exp = {
        rid: exposure_score(exposure[exposure["reach_id"] == rid], weights, population_unit)
        for rid in candidates
    }
    max_e = max((e for e, _ in exp.values()), default=0.0)
    max_w = {
        var: max((w for r, (_, w) in d.items() if r in exp), default=0.0)
        for var, d in per_var.items()
    }
    rows: list[tuple[float, str, PriorityComponents, tuple[str, ...]]] = []
    for rid in candidates:
        e, flags = exp[rid]
        ew = math.log1p(e) / math.log1p(max_e) if max_e > 0 else 0.0
        best: PriorityComponents | None = None
        for var, d in per_var.items():
            if rid not in d:
                continue
            risk, width = d[rid]
            u = width / max_w[var] if max_w[var] > 0 else 0.0
            comp = PriorityComponents(
                variable=var,
                predicted_risk=risk,
                forecast_uncertainty=u,
                exposure_weight=ew,
                exposure_score=e,
            )
            if best is None or u * risk > best.forecast_uncertainty * best.predicted_risk:
                best = comp
        assert best is not None  # a candidate has at least one scored variable
        value = best.forecast_uncertainty * best.predicted_risk * best.exposure_weight
        rows.append((value, rid, best, flags))
    rows.sort(key=lambda t: (-t[0], t[1]))
    return Priorities(
        weights={k: float(v) for k, v in weights.items()},
        population_unit=float(population_unit),
        ranked=tuple(
            RankedReach(rank=i + 1, reach_id=rid, information_value=v, components=c, flags=fl)
            for i, (v, rid, c, fl) in enumerate(rows)
        ),
        not_ranked=tuple(not_ranked),
    )
