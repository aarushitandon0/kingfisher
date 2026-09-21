"""Alert engine - pure functions only. No I/O, no DB, no network, so every guardrail is
testable without a server (tests/test_guardrails.py breaks each one on purpose).

An alert is a calibrated probability of threshold exceedance, never a raw prediction
(MASTERSPEC 8). The probability comes from engine.probability - the same CDF the
reliability diagram in models/evaluate.py is scored with.

    compute_exceedance_probability(forecast_distribution, threshold) -> float
    apply_guardrails(candidate_alert, reach_state, history)          -> Alert | Suppressed
    build_attribution(shap_values, feature_names)                    -> list[DriverContribution]
    compose_alert(...)                                                -> Alert | Suppressed

THREE OUTCOMES, ALL RETURNED VALUES, NONE AN EXCEPTION
------------------------------------------------------
  Alert(severity=ALERT | WATCH)          issue it
  Alert(severity=INSUFFICIENT_EVIDENCE)  the system refuses to judge, says why, and the
                                         refusal is shown in the UI and counted in the
                                         metrics. It carries NO exceedance probability -
                                         the number it would have had is kept apart in
                                         `withheld_exceedance_prob`, so it can never be
                                         read as a weak alert.
  Suppressed                             a guardrail stopped a candidate; the reason is
                                         recorded (alerts.suppressed_reason)

GUARDRAILS (config/thresholds.yaml -> guardrails; MASTERSPEC 8), in evaluation order
------------------------------------------------------------------------------------
  staleness     last usable observation older than staleness_days (21), or none at all
                -> INSUFFICIENT_EVIDENCE. For a driver-only reach the evidence is the
                latest usable observation of the upstream reaches its forecast leans on;
                the caller supplies it in ReachState.last_usable_observation.
  drift         recent residuals (observed vs the forecast issued for that date) leave
                the calibration band -> Suppressed(DRIFT)
  confidence    exceedance_prob < min_exceedance_prob (0.6) -> Suppressed
  cooldown      an ALERT or WATCH already issued for the reach < cooldown_hours (72) ago
                -> Suppressed(COOLDOWN). INSUFFICIENT_EVIDENCE records do not start one.
  observability a reach not proven optically observable (observable is False or NULL)
                is capped at WATCH, never ALERT.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import Any

import numpy as np

from engine.probability import DEFAULT_LEVELS, exceedance_probability_multi

# z_0.9 - z_0.1 for a normal distribution: (p90 - p10) / Z_SPREAD estimates one sigma.
Z_SPREAD = 2.5631031310892007

# Features that are not drivers and are never presented as one. reach_id is the
# reach's own learned offset (variant A only); horizon is lead time.
NON_DRIVER_FEATURES = frozenset({"reach_id", "horizon"})
STATE_FEATURES = frozenset(
    {
        "turbidity_proxy_asof",
        "ndci_asof",
        "obs_asof_age_days",
        "upstream_state_lag1_turbidity",
        "upstream_state_lag1_ndci",
    }
)
STATIC_FEATURES = frozenset(
    {
        "imperviousness_pct",
        "riparian_ndvi_mean",
        "riparian_width_m",
        "road_density_km_km2",
        "alan_radiance",
        "population",
        "urban_fraction",
        "catchment_area_km2",
        "strahler_order",
        "observable",
        "median_water_pixels",
    }
)


class Severity(StrEnum):
    WATCH = "WATCH"
    ALERT = "ALERT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


# ---------------------------------------------------------------------------
# value types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ForecastDistribution:
    """Quantile forecast for one reach, variable and target date."""

    quantiles: tuple[float, ...]  # values, aligned with levels
    levels: tuple[float, ...] = DEFAULT_LEVELS
    target_date: date | None = None

    def __post_init__(self) -> None:
        if len(self.quantiles) != len(self.levels):
            raise ValueError(f"{len(self.quantiles)} quantiles for {len(self.levels)} levels")

    def at(self, level: float) -> float:
        return float(self.quantiles[self.levels.index(level)])


@dataclass(frozen=True)
class DriverContribution:
    feature: str
    contribution: float  # SHAP value, in the target variable's units
    value: float | None = None
    kind: str = "weather_driver"  # weather_driver | catchment_attribute | recent_state


@dataclass(frozen=True)
class Residual:
    """A past forecast for a date that was later observed."""

    obs_date: date
    observed: float
    p10: float
    p50: float
    p90: float


@dataclass(frozen=True)
class ReachState:
    reach_id: str
    observable: bool | None  # None = never assessed; treated as not proven observable
    last_usable_observation: date | None
    evidence_source: str = "reach"  # "reach" | "upstream" (driver-only reaches)
    usable_observations_30d: int = 0
    recent_residuals: tuple[Residual, ...] = ()


@dataclass(frozen=True)
class HistoryEntry:
    reach_id: str
    issued_at: datetime
    severity: Severity
    variable: str = ""


@dataclass(frozen=True)
class GuardrailConfig:
    min_exceedance_prob: float = 0.60
    cooldown_hours: float = 72.0
    staleness_days: int = 21
    unobservable_max_severity: Severity = Severity.WATCH
    drift_window_days: int = 60
    drift_max_abs_z: float = 2.5
    drift_max_outside_fraction: float = 0.50
    drift_min_residuals: int = 3

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> GuardrailConfig:
        g = cfg["guardrails"]
        d = g["drift"]
        return cls(
            min_exceedance_prob=float(g["min_exceedance_prob"]),
            cooldown_hours=float(g["cooldown_hours"]),
            staleness_days=int(g["staleness_days"]),
            unobservable_max_severity=Severity(g["unobservable_max_severity"]),
            drift_window_days=int(d["residual_window_days"]),
            drift_max_abs_z=float(d["max_abs_standardised_residual"]),
            drift_max_outside_fraction=float(d["max_outside_fraction"]),
            drift_min_residuals=int(d["min_residuals"]),
        )


@dataclass(frozen=True)
class Alert:
    reach_id: str
    variable: str
    issued_at: datetime
    severity: Severity
    target_window: tuple[date, date]
    exceedance_prob: float | None  # None for INSUFFICIENT_EVIDENCE, always
    threshold_value: float | None
    threshold_derivation: str | None
    peak_target_date: date | None = None
    withheld_exceedance_prob: float | None = None  # INSUFFICIENT_EVIDENCE only
    suppressed_reason: str | None = None  # why INSUFFICIENT_EVIDENCE / why capped
    attribution: tuple[DriverContribution, ...] = ()
    exposure: Mapping[str, Any] | None = None
    basis: Mapping[str, Any] = field(default_factory=dict)
    guardrails: Mapping[str, str] = field(default_factory=dict)
    model_version: str | None = None

    def __post_init__(self) -> None:
        if self.severity is Severity.INSUFFICIENT_EVIDENCE:
            if self.exceedance_prob is not None:
                raise ValueError("INSUFFICIENT_EVIDENCE must not carry an exceedance probability")
            if not self.suppressed_reason:
                raise ValueError("INSUFFICIENT_EVIDENCE must say why")
        elif self.exceedance_prob is None or not math.isfinite(self.exceedance_prob):
            raise ValueError(f"{self.severity} needs a finite exceedance probability")


@dataclass(frozen=True)
class Suppressed:
    reach_id: str
    variable: str
    issued_at: datetime
    guardrail: str
    reason: str
    exceedance_prob: float | None
    guardrails: Mapping[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# probability + attribution
# ---------------------------------------------------------------------------
def compute_exceedance_probability(
    forecast_distribution: ForecastDistribution, threshold: float
) -> float:
    """P(state > threshold) under engine.probability's piecewise-linear CDF through all of
    the forecast's quantiles. NaN if the threshold or any quantile is missing - the
    caller turns that into INSUFFICIENT_EVIDENCE, never into 0."""
    p = exceedance_probability_multi(
        np.asarray(forecast_distribution.quantiles, dtype="float64"),
        forecast_distribution.levels,
        np.float64(threshold if threshold is not None else np.nan),
    )
    return float(p)


def _kind(feature: str) -> str:
    if feature in STATE_FEATURES:
        return "recent_state"
    if feature in STATIC_FEATURES:
        return "catchment_attribute"
    return "weather_driver"


def build_attribution(
    shap_values: Sequence[float],
    feature_names: Sequence[str],
    feature_values: Sequence[float | None] | None = None,
    *,
    top_k: int | None = None,
) -> list[DriverContribution]:
    """SHAP row -> driver contributions, largest magnitude first.

    reach_id and horizon are dropped: the reach's learned offset and the lead time are
    not drivers and must not be shown as one. Exact-zero and non-finite contributions
    are dropped (LightGBM gives 0 to a feature that did not split for this row). A bias
    term, if the caller passes one, must be removed before calling.
    """
    if len(shap_values) != len(feature_names):
        raise ValueError(f"{len(shap_values)} SHAP values for {len(feature_names)} features")
    if feature_values is not None and len(feature_values) != len(feature_names):
        raise ValueError("feature_values must align with feature_names")
    out = []
    for i, (name, c) in enumerate(zip(feature_names, shap_values, strict=True)):
        c = float(c)
        if name in NON_DRIVER_FEATURES or c == 0.0 or not math.isfinite(c):
            continue
        v = None if feature_values is None else feature_values[i]
        v = None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
        out.append(DriverContribution(feature=name, contribution=c, value=v, kind=_kind(name)))
    out.sort(key=lambda d: -abs(d.contribution))
    return out[:top_k] if top_k else out


# ---------------------------------------------------------------------------
# guardrails - each (candidate, state, history, cfg) -> outcome or None (= pass)
# ---------------------------------------------------------------------------
Outcome = Alert | Suppressed
Guardrail = Callable[[Alert, ReachState, Sequence[HistoryEntry], GuardrailConfig], Outcome | None]


def _insufficient(candidate: Alert, reason: str) -> Alert:
    return replace(
        candidate,
        severity=Severity.INSUFFICIENT_EVIDENCE,
        withheld_exceedance_prob=candidate.exceedance_prob,
        exceedance_prob=None,
        suppressed_reason=reason,
    )


def _suppress(candidate: Alert, guardrail: str, reason: str) -> Suppressed:
    return Suppressed(
        reach_id=candidate.reach_id,
        variable=candidate.variable,
        issued_at=candidate.issued_at,
        guardrail=guardrail,
        reason=reason,
        exceedance_prob=candidate.exceedance_prob,
    )


def guard_staleness(
    c: Alert, s: ReachState, h: Sequence[HistoryEntry], cfg: GuardrailConfig
) -> Outcome | None:
    if s.last_usable_observation is None:
        return _insufficient(c, "STALE_EVIDENCE: no usable observation on record")
    age = (c.issued_at.date() - s.last_usable_observation).days
    if age > cfg.staleness_days:
        return _insufficient(
            c,
            f"STALE_EVIDENCE: last usable {s.evidence_source} observation {age} days old "
            f"(> {cfg.staleness_days})",
        )
    return None


def drift_statistics(
    residuals: Sequence[Residual], issued: date, window_days: int
) -> dict[str, float | int]:
    recent = [r for r in residuals if 0 < (issued - r.obs_date).days <= window_days]
    z, outside = [], 0
    for r in recent:
        sigma = (r.p90 - r.p10) / Z_SPREAD
        z.append(abs(r.observed - r.p50) / sigma if sigma > 0 else math.inf)
        outside += int(r.observed < r.p10 or r.observed > r.p90)
    return {
        "n": len(recent),
        "mean_abs_z": float(np.mean(z)) if z else math.nan,
        "outside_fraction": outside / len(recent) if recent else math.nan,
    }


def guard_drift(
    c: Alert, s: ReachState, h: Sequence[HistoryEntry], cfg: GuardrailConfig
) -> Outcome | None:
    st = drift_statistics(s.recent_residuals, c.issued_at.date(), cfg.drift_window_days)
    if st["n"] < cfg.drift_min_residuals:
        return None  # too few residuals to judge drift; the status says so
    if (
        st["mean_abs_z"] > cfg.drift_max_abs_z
        or st["outside_fraction"] > cfg.drift_max_outside_fraction
    ):
        return _suppress(
            c,
            "drift",
            f"DRIFT: {st['n']} residuals in {cfg.drift_window_days} d, mean |z| "
            f"{st['mean_abs_z']:.2f} (max {cfg.drift_max_abs_z}), outside P10-P90 "
            f"{st['outside_fraction']:.0%} (max {cfg.drift_max_outside_fraction:.0%})",
        )
    return None


def guard_confidence(
    c: Alert, s: ReachState, h: Sequence[HistoryEntry], cfg: GuardrailConfig
) -> Outcome | None:
    p = c.exceedance_prob
    if p is None or not math.isfinite(p):
        return _suppress(c, "confidence", "BELOW_MIN_CONFIDENCE: no probability")
    if p < cfg.min_exceedance_prob:
        return _suppress(
            c, "confidence", f"BELOW_MIN_CONFIDENCE: {p:.3f} < {cfg.min_exceedance_prob}"
        )
    return None


def guard_cooldown(
    c: Alert, s: ReachState, h: Sequence[HistoryEntry], cfg: GuardrailConfig
) -> Outcome | None:
    window = timedelta(hours=cfg.cooldown_hours)
    for e in h:
        if (
            e.reach_id == c.reach_id
            and e.severity in (Severity.ALERT, Severity.WATCH)
            and timedelta(0) <= c.issued_at - e.issued_at < window
        ):
            hours = (c.issued_at - e.issued_at).total_seconds() / 3600
            return _suppress(
                c,
                "cooldown",
                f"COOLDOWN: {e.severity} issued for {e.reach_id} {hours:.0f} h ago "
                f"(< {cfg.cooldown_hours:.0f} h)",
            )
    return None


def guard_observability(
    c: Alert, s: ReachState, h: Sequence[HistoryEntry], cfg: GuardrailConfig
) -> Outcome | None:
    if s.observable is not True and c.severity is Severity.ALERT:
        why = "driver-only reach" if s.observable is False else "observability never assessed"
        return replace(
            c,
            severity=cfg.unobservable_max_severity,
            suppressed_reason=f"CAPPED_AT_{cfg.unobservable_max_severity}: {why}",
        )
    return None


# Order matters: evidence first (no judgement without it), then model health, then the
# decision rules. Tests assert every entry here has a deliberate-break test.
GUARDRAILS: dict[str, Guardrail] = {
    "staleness": guard_staleness,
    "drift": guard_drift,
    "confidence": guard_confidence,
    "cooldown": guard_cooldown,
    "observability": guard_observability,
}
# Guardrails that modify the candidate and let it continue, rather than end it.
_MODIFIERS = frozenset({"observability"})


def apply_guardrails(
    candidate_alert: Alert,
    reach_state: ReachState,
    history: Sequence[HistoryEntry],
    config: GuardrailConfig | None = None,
) -> Alert | Suppressed:
    """Run every guardrail in order. The first that ends the candidate decides the
    outcome; each guardrail's status is recorded on what is returned."""
    cfg = config or GuardrailConfig()
    if reach_state.reach_id != candidate_alert.reach_id:
        raise ValueError(
            f"state is for {reach_state.reach_id}, alert for {candidate_alert.reach_id}"
        )
    status: dict[str, str] = {}
    current = candidate_alert
    for name, guard in GUARDRAILS.items():
        outcome = guard(current, reach_state, history, cfg)
        if outcome is None:
            status[name] = "pass"
            continue
        status[name] = "fired"
        if name in _MODIFIERS and isinstance(outcome, Alert):
            current = outcome
            continue
        for rest in list(GUARDRAILS)[list(GUARDRAILS).index(name) + 1 :]:
            status[rest] = "not_evaluated"
        return replace(outcome, guardrails=status)
    return replace(current, guardrails=status)


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------
def compose_alert(
    *,
    reach_id: str,
    variable: str,
    issued_at: datetime,
    forecasts: Sequence[ForecastDistribution],
    thresholds: Sequence[float | None],
    threshold_derivation: str,
    reach_state: ReachState,
    history: Sequence[HistoryEntry],
    shap_values: Sequence[float] | None = None,
    feature_names: Sequence[str] | None = None,
    feature_values: Sequence[float | None] | None = None,
    exposure: Mapping[str, Any] | None = None,
    model_version: str | None = None,
    config: GuardrailConfig | None = None,
    top_k_drivers: int = 5,
) -> Alert | Suppressed:
    """One reach, one variable, one target window (the `forecasts`' target dates, each
    with its own threshold - a window can straddle a season boundary).

    The alert probability is the PEAK daily exceedance probability over the window.
    Missing thresholds or forecasts are INSUFFICIENT_EVIDENCE, returned, not raised.
    """
    if len(forecasts) != len(thresholds) or not forecasts:
        raise ValueError("forecasts and thresholds must be non-empty and aligned")
    dates = [f.target_date for f in forecasts]
    if any(d is None for d in dates):
        raise ValueError("every forecast needs a target_date")
    window = (min(d for d in dates if d), max(d for d in dates if d))
    basis = {
        "usable_observations_30d": reach_state.usable_observations_30d,
        "last_usable_observation": reach_state.last_usable_observation,
        "evidence_source": reach_state.evidence_source,
        "probability": "peak daily P(exceed) over the window, piecewise-linear CDF "
        f"through {len(forecasts[0].levels)} forecast quantiles",
    }
    attribution: tuple[DriverContribution, ...] = ()
    if shap_values is not None and feature_names is not None:
        attribution = tuple(
            build_attribution(shap_values, feature_names, feature_values, top_k=top_k_drivers)
        )
    base: dict[str, Any] = {
        "reach_id": reach_id,
        "variable": variable,
        "issued_at": issued_at,
        "target_window": window,
        "threshold_derivation": threshold_derivation,
        "attribution": attribution,
        "exposure": exposure,
        "basis": basis,
        "model_version": model_version,
    }

    usable = [
        (f, t)
        for f, t in zip(forecasts, thresholds, strict=True)
        if t is not None and math.isfinite(t)
    ]
    if not usable:
        return Alert(
            **base,
            severity=Severity.INSUFFICIENT_EVIDENCE,
            exceedance_prob=None,
            threshold_value=None,
            suppressed_reason="NO_THRESHOLD: too few past observations in this reach-season",
            guardrails={name: "not_evaluated" for name in GUARDRAILS},
        )
    probs = [(compute_exceedance_probability(f, t), f, t) for f, t in usable]
    finite = [x for x in probs if math.isfinite(x[0])]
    if not finite:
        return Alert(
            **base,
            severity=Severity.INSUFFICIENT_EVIDENCE,
            exceedance_prob=None,
            threshold_value=float(usable[0][1]),
            suppressed_reason="NO_FORECAST: forecast quantiles missing for the window",
            guardrails={name: "not_evaluated" for name in GUARDRAILS},
        )
    p, f, t = max(finite, key=lambda x: x[0])
    candidate = Alert(
        **base,
        severity=Severity.ALERT,
        exceedance_prob=p,
        threshold_value=float(t),
        peak_target_date=f.target_date,
    )
    return apply_guardrails(candidate, reach_state, history, config)


def to_record(outcome: Alert | Suppressed) -> dict[str, Any]:
    """Outcome -> a row shaped like the `alerts` table. A Suppressed outcome is stored
    too (severity NULL is not allowed there, so the caller decides whether to persist
    it); this returns its reason so it is never silently lost."""
    if isinstance(outcome, Suppressed):
        return {
            "reach_id": outcome.reach_id,
            "variable": outcome.variable,
            "issued_date": outcome.issued_at.date(),
            "suppressed": True,
            "suppressed_reason": outcome.reason,
            "exceedance_prob": outcome.exceedance_prob,
            "guardrails": dict(outcome.guardrails),
        }
    return {
        "reach_id": outcome.reach_id,
        "variable": outcome.variable,
        "issued_date": outcome.issued_at.date(),
        "target_window": outcome.target_window,
        "severity": str(outcome.severity),
        "exceedance_prob": outcome.exceedance_prob,
        "withheld_exceedance_prob": outcome.withheld_exceedance_prob,
        "threshold_value": outcome.threshold_value,
        "attribution": [
            {"feature": d.feature, "contribution": d.contribution, "value": d.value, "kind": d.kind}
            for d in outcome.attribution
        ],
        "exposure": dict(outcome.exposure) if outcome.exposure else None,
        "suppressed_reason": outcome.suppressed_reason,
        "guardrails": dict(outcome.guardrails),
        "model_version": outcome.model_version,
        "suppressed": False,
    }
