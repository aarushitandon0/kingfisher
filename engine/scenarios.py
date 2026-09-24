"""Scenario engine - pure functions only (MASTERSPEC 9). No I/O, no DB, no network.

    apply_intervention(features, intervention, coefficients)            -> perturbed features
    response_check(model, features, feature=..., variable=..., ...)     -> ResponseCheck
    run_scenario(reach_ids, interventions, model, baseline_features, ...) -> ScenarioResult

Every output is a PLANNING ESTIMATE. Intervention effect sizes come ONLY from the cited
coefficient table (config/intervention_coefficients.yaml, validated by
engine.coefficients); the model is never asked what an intervention does. It is asked only
what it was trained to answer - the state of the stream given its drivers - on drivers the
cited coefficients have perturbed.

WHICH MODEL (MASTERSPEC 9.6, binding)
-------------------------------------
Variant B only (no reach identity): `run_scenario` refuses any other variant. Before a
lever may perturb a STATIC feature through the model, a response check on held-out data
must show the model moves that feature the way the literature says, by a non-negligible
amount (`response_check`, persisted by models/scenario_response.py with the model version
it ran on). Per lever and target variable the result records the path taken:

  MODEL_PERTURBATION  the cited coefficient perturbs the feature and the model re-infers
  LITERATURE_DIRECT   the model is not trusted for the feature; the lever's cited
                      `direct_effect` scales the forecast quantiles instead
  NOT_ESTIMABLE       the model is not trusted and the table has no cited direct effect -
                      the lever is reported, not estimated. Nothing is made up in its place.

Driver features (precip_max_hourly, first_flush_index) are outside the check's scope in
9.6 and go through the model. Derived features follow their inputs: a multiplicative change
to precip_max_hourly scales first_flush_index (= antecedent_dry_days * precip_max_hourly)
by the same factor, on the issue day and at the target day (`_fut`).

WHAT IS COMPUTED
----------------
For each reach and target variable, over one reference weather year (settings), with one
horizon-`h` row per day:

  p_d            = P(state_d > threshold_d)   engine.probability, the same piecewise-linear
                   CDF through all forecast quantiles the alerts and the reliability diagram
                   use; threshold_d is the reach's own seasonal threshold (engine.thresholds)
  exceedance days = sum_d p_d                  expected days per year above threshold
  interval       = central `ci_level` interval of the Poisson-binomial distribution of the
                   day count (days treated as independent given the forecast - real days are
                   autocorrelated, so this understates the spread; stated on the result)

Baseline: unperturbed features. Scenario: every lever evaluated at its cited `magnitude`
(the central estimate) and at `coefficient_samples` quantiles of a triangular(low,
magnitude, high) distribution over its `uncertainty_range`, all levers at the same quantile
(comonotone: the widest joint spread). The scenario's raw interval is taken from the
equal-weight mixture of the per-sample count distributions - that is how the coefficient
uncertainty reaches the interval - and then WIDENED:

  width = interval_widening_factor * max(raw scenario width, baseline width)

centred so the raw interval (and the central estimate) stay inside it, clipped to [0, days].
The factor must be > 1, so every scenario interval is strictly wider than its baseline
interval; `run_scenario` asserts this for every reach and raises ScenarioInvariantError if
it ever fails. A reach whose baseline interval is degenerate (zero width, or the whole
period) cannot satisfy it and is reported NOT_ESTIMABLE with the reason.

OUTCOMES PER REACH, ALL RETURNED VALUES
--------------------------------------
  OK                     baseline, scenario, delta, and each lever's path and flags
  INSUFFICIENT_EVIDENCE  no threshold for some season of the reference year (driver-only
                         reaches have none), or missing reach-days - never filled
  NOT_ESTIMABLE          no lever could be estimated on the reach, or degenerate baseline
Per-lever flags on a reach: NOT_APPLICABLE (feature NULL, or the extent exceeds what
exists), NO_CHANGE (the lever leaves every value unchanged - e.g. a buffer floor below the
existing width), OUT_OF_SUPPORT (a perturbed value outside the model's training range: trees
cannot extrapolate, so the response is truncated at the edge value).

Spatial scope: a lever changes the attributes of the selected reaches' own catchments only.
It is not propagated to downstream reaches whose catchments contain the treated area.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from engine.alerts import STATIC_FEATURES
from engine.coefficients import CoefficientTable, Intervention, Reference
from engine.probability import exceedance_probability_multi
from engine.thresholds import lookup as threshold_lookup
from engine.thresholds import season_of

LABEL = "planning estimate"
UNITS = "exceedance days per year"
SCENARIO_VARIANT = "B"
FUT_SUFFIX = "_fut"
DEFAULT_VARIABLES: tuple[str, ...] = ("turbidity_proxy", "ndci")
# Derived features that scale with a key feature under a MULTIPLICATIVE change to it
# (pipeline/l2_drivers.py: first_flush_index = antecedent_dry_days * precip_max_hourly).
MULTIPLICATIVE_DEPENDENTS: Mapping[str, tuple[str, ...]] = {
    "precip_max_hourly": ("first_flush_index",),
}
SPATIAL_SCOPE = (
    "Each lever changes the catchment attributes of the selected reaches only. The treated "
    "area also lies inside the catchments of reaches downstream; that is not propagated."
)
_EPS = 1e-12

FloatArray = npt.NDArray[np.float64]


class ScenarioInvariantError(RuntimeError):
    """A scenario interval came out no wider than its baseline interval. A bug, never data."""


class MissingResponseCheck(LookupError):
    """A lever perturbs a static feature and no response check is on record for it."""


class StaleResponseCheck(RuntimeError):
    """The response check on record was run for a different serving model version."""


class LeverPath(StrEnum):
    MODEL_PERTURBATION = "MODEL_PERTURBATION"
    LITERATURE_DIRECT = "LITERATURE_DIRECT"
    NOT_ESTIMABLE = "NOT_ESTIMABLE"


class Verdict(StrEnum):
    TRUSTED = "TRUSTED"
    WRONG_SIGN = "WRONG_SIGN"
    NEGLIGIBLE = "NEGLIGIBLE"


class Status(StrEnum):
    OK = "OK"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_ESTIMABLE = "NOT_ESTIMABLE"


class ReachFlag(StrEnum):
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NO_CHANGE = "NO_CHANGE"
    OUT_OF_SUPPORT = "OUT_OF_SUPPORT"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
class InterventionRequest(_Frozen):
    """One lever on a set of reaches. `extent` is how much of the lever - its meaning is
    set by the coefficient's operation (percentage points of catchment treated, fraction
    of the catchment covered, or unused for a floor). It is NEVER an effect size: those
    come only from the coefficient table."""

    type: str = Field(min_length=1)
    reach_ids: tuple[str, ...] = Field(min_length=1)
    extent: float | None = None

    @field_validator("reach_ids")
    @classmethod
    def _unique(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(v)) != len(v):
            raise ValueError("reach_ids must be unique")
        return v

    @field_validator("extent")
    @classmethod
    def _finite(cls, v: float | None) -> float | None:
        if v is not None and not math.isfinite(v):
            raise ValueError("extent must be finite")
        return v


class ResponseCheck(_Frozen):
    """MASTERSPEC 9.6 rule 2, for one static feature and one target variable."""

    feature: str
    variable: str
    checked_model_version: str
    serving_model_version: str
    fold: str
    rows: int
    reaches: int
    feature_sd: float
    target_sd: float
    mean_p50_change: float  # mean P50 at +1 SD minus mean P50 at -1 SD
    standardised_effect: float  # mean_p50_change / target_sd
    observed_sign: Literal[-1, 0, 1]
    expected_sign: Literal[-1, 1]
    mean_exceedance_prob_change: float | None
    exceedance_rows: int
    share_reaches_up: float
    share_reaches_down: float
    out_of_support_share: float | None
    negligible_effect_sd: float
    verdict: Verdict
    feature_sd_reaches: int | None = None  # reaches the feature SD was taken over
    target_sd_observations: int | None = None  # observations the target SD was taken over


class ScenarioModel(Protocol):
    """What the engine needs from a model. The adapter lives outside engine/ (models/)."""

    @property
    def version(self) -> str: ...
    @property
    def variant(self) -> str: ...
    @property
    def levels(self) -> tuple[float, ...]: ...
    @property
    def features(self) -> Sequence[str]: ...

    def feature_range(self, variable: str, feature: str) -> tuple[float, float] | None:
        """(min, max) of `feature` over the rows the `variable` model was trained on."""
        ...

    def predict_quantiles(self, features: pd.DataFrame, variable: str) -> FloatArray:
        """(n_rows, len(levels)) quantiles, non-decreasing along axis 1."""
        ...


@dataclass(frozen=True)
class ScenarioSettings:
    reference_start: date
    reference_end: date
    horizon: int
    ci_level: float
    coefficient_samples: int
    widening_factor: float
    caveat: str
    seasons: Mapping[str, Sequence[int]]
    threshold_derivation: str = "per-reach seasonal percentile (config/thresholds.yaml)"

    def __post_init__(self) -> None:
        if not (isinstance(self.widening_factor, int | float) and self.widening_factor > 1.0):
            raise ValueError(
                f"interval_widening_factor must be a number > 1 (scenario intervals must be "
                f"strictly wider than baseline), got {self.widening_factor!r}"
            )
        if not 0.0 < self.ci_level < 1.0:
            raise ValueError(f"ci_level must be in (0, 1), got {self.ci_level}")
        if self.coefficient_samples < 1:
            raise ValueError("coefficient_samples must be >= 1")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.period_days not in (365, 366) or (
            self.reference_start.month,
            self.reference_start.day,
            self.reference_end.month,
            self.reference_end.day,
        ) != (1, 1, 12, 31):
            raise ValueError(
                "reference_period must be one whole calendar year, so that exceedance days "
                f"are per year; got {self.reference_start} .. {self.reference_end}"
            )
        if self.reference_start.year != self.reference_end.year:
            raise ValueError("reference_period must be a single calendar year")
        if not self.caveat.strip():
            raise ValueError("the planning-estimate caveat is required")

    @property
    def period_days(self) -> int:
        return (self.reference_end - self.reference_start).days + 1

    @classmethod
    def from_config(
        cls,
        modelling: Mapping[str, Any],
        coefficients: CoefficientTable,
        thresholds: Mapping[str, Any],
    ) -> ScenarioSettings:
        s = modelling["scenario"]
        infl = coefficients.uncertainty_inflation
        factor = infl.get("interval_widening_factor")
        if factor is None:
            raise ValueError("uncertainty_inflation.interval_widening_factor is not set")
        return cls(
            reference_start=_as_date(s["reference_period"]["start"]),
            reference_end=_as_date(s["reference_period"]["end"]),
            horizon=int(s["horizon"]),
            ci_level=float(s["ci_level"]),
            coefficient_samples=int(s["coefficient_samples"]),
            widening_factor=float(factor),
            caveat=" ".join(str(infl["caveat"]).split()),
            seasons={k: list(v) for k, v in thresholds["seasons"].items()},
        )


def _as_date(v: Any) -> date:
    return v if isinstance(v, date) else date.fromisoformat(str(v))


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
class CitationOut(_Frozen):
    intervention_id: str
    role: Literal["effect", "cost", "direct_effect"]  # Role
    authors: str
    year: int
    title: str
    source: str
    doi: str | None
    url: str | None
    locator: str
    supports: tuple[str, ...]


class Interval(_Frozen):
    low: float
    high: float

    @model_validator(mode="after")
    def _ordered(self) -> Interval:
        if not self.low <= self.high:
            raise ValueError(f"interval low {self.low} > high {self.high}")
        return self

    @property
    def width(self) -> float:
        return self.high - self.low


class Estimate(_Frozen):
    exceedance_days: float
    interval: Interval


class LeverOutcome(_Frozen):
    """One lever on one reach for one variable."""

    intervention_id: str
    path: LeverPath
    flags: tuple[ReachFlag, ...] = ()
    detail: tuple[str, ...] = ()
    feature: str
    value_before: float | None = None  # static features: the reach's value
    value_after: float | None = None  # static features, at the cited magnitude


class CostEstimate(_Frozen):
    intervention_id: str
    reach_id: str
    currency: str
    unit: str
    unit_cost_low: float
    unit_cost_high: float
    price_basis: str
    quantity: float | None
    quantity_unit: str | None
    total_low: float | None
    total_high: float | None
    reason: str | None = None  # why there is no quantity
    note: str | None = None


class LeverSummary(_Frozen):
    intervention_id: str
    name: str
    variable: str
    feature: str
    operation: str
    extent: float | None
    reach_ids: tuple[str, ...]
    path: LeverPath
    reason: str
    magnitude: float
    uncertainty_range: tuple[float, float]
    direct_magnitude: float | None = None
    direct_uncertainty_range: tuple[float, float] | None = None
    response_check: ResponseCheck | None = None
    citations: tuple[CitationOut, ...]


class ReachResult(_Frozen):
    reach_id: str
    variable: str
    status: Status
    reason: str | None = None
    days_evaluated: int
    complete: bool  # every lever requested on this reach was estimated on it
    baseline: Estimate | None = None
    scenario: Estimate | None = None
    delta_days: float | None = None
    delta_pct: float | None = None
    # delta at the lowest and highest coefficient samples - the spread the cited
    # uncertainty ranges alone produce, before widening. Not an interval of any kind.
    delta_coefficient_range: Interval | None = None
    levers: tuple[LeverOutcome, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> ReachResult:
        if self.status is Status.OK:
            if self.baseline is None or self.scenario is None or self.delta_days is None:
                raise ValueError("an OK result needs baseline, scenario and delta")
            if not self.scenario.interval.width > self.baseline.interval.width:
                raise ValueError("scenario interval must be strictly wider than baseline")
        else:
            if not self.reason:
                raise ValueError(f"{self.status} must say why")
            if self.scenario is not None or self.delta_days is not None:
                raise ValueError(f"{self.status} must not carry a scenario estimate")
        return self


class ScenarioResult(_Frozen):
    label: Literal["planning estimate"] = "planning estimate"
    caveat: str
    units: Literal["exceedance days per year"] = "exceedance days per year"
    model_version: str
    variant: str
    coefficient_table_version: str
    reference_start: date
    reference_end: date
    horizon: int
    ci_level: float
    widening_factor: float
    coefficient_samples: int
    interval_method: str
    threshold_derivation: str
    spatial_scope: str = SPATIAL_SCOPE
    variables: tuple[str, ...]
    levers: tuple[LeverSummary, ...]
    reaches: tuple[ReachResult, ...]
    costs: tuple[CostEstimate, ...]
    citations: tuple[CitationOut, ...] = Field(min_length=1)


# ---------------------------------------------------------------------------
# coefficients
# ---------------------------------------------------------------------------
def resolve(coefficients: CoefficientTable, intervention_id: str) -> Intervention:
    """The cited coefficient for a lever, or ValueError saying why there is none."""
    try:
        return coefficients.get(intervention_id)
    except KeyError as exc:
        raise ValueError(str(exc.args[0]) if exc.args else str(exc)) from None


def validate_extent(coef: Intervention, extent: float | None) -> None:
    op = coef.effect.operation
    if op == "floor":
        if extent is not None:
            raise ValueError(
                f"{coef.id}: extent is not used by a floor operation (the cited value "
                f"{coef.magnitude} {coef.effect.unit} is the lever); omit it"
            )
        return
    if extent is None:
        raise ValueError(f"{coef.id}: extent is required for {op}")
    if op == "subtract_treated_share" and not 0.0 < extent <= 100.0:
        raise ValueError(
            f"{coef.id}: extent is percentage points of catchment area treated, "
            f"in (0, 100]; got {extent}"
        )
    if op == "scale_down" and not 0.0 < extent <= 1.0:
        raise ValueError(
            f"{coef.id}: extent is the fraction of the catchment covered, in (0, 1]; got {extent}"
        )


def expected_response_sign(coefficients: CoefficientTable, feature: str) -> Literal[-1, 1]:
    """Sign of d(target)/d(feature) the literature implies. Every lever is meant to REDUCE
    the target, so a lever that decreases the feature implies +1, one that increases it -1.
    Two levers implying opposite signs for one feature is a table error."""
    signs = {
        (1 if i.effect.direction == "decrease" else -1)
        for i in coefficients.interventions
        if i.effect.feature == feature
    }
    if not signs:
        raise ValueError(f"no lever in the coefficient table perturbs {feature}")
    if len(signs) > 1:
        raise ValueError(f"levers on {feature} imply opposite response signs")
    return 1 if signs.pop() == 1 else -1


def triangular_ppf(u: float, low: float, mode: float, high: float) -> float:
    """Quantile u of triangular(low, mode, high). Degenerate range -> the single value."""
    if not 0.0 < u < 1.0:
        raise ValueError(f"u must be in (0, 1), got {u}")
    if not low <= mode <= high:
        raise ValueError(f"need low <= mode <= high, got {low}, {mode}, {high}")
    if high - low <= 0.0:
        return float(low)
    fc = (mode - low) / (high - low)
    if u < fc:
        return float(low + math.sqrt(u * (high - low) * (mode - low)))
    return float(high - math.sqrt((1.0 - u) * (high - low) * (high - mode)))


def coefficient_levels(n: int) -> tuple[float, ...]:
    """n equally spaced interior quantile levels, (k + 0.5) / n."""
    return tuple((k + 0.5) / n for k in range(n))


Role = Literal["effect", "cost", "direct_effect"]


def citations_of(
    intervention_id: str, refs: Iterable[Reference], role: Role
) -> tuple[CitationOut, ...]:
    return tuple(
        CitationOut(
            intervention_id=intervention_id,
            role=role,
            authors=r.authors,
            year=r.year,
            title=" ".join(r.title.split()),
            source=r.source,
            doi=r.doi,
            url=r.url,
            locator=" ".join(r.locator.split()),
            supports=tuple(r.supports),
        )
        for r in refs
    )


# ---------------------------------------------------------------------------
# perturbation
# ---------------------------------------------------------------------------
def _columns(features: pd.DataFrame, feature: str) -> list[str]:
    """The feature and its target-day copy, if the frame has one."""
    return [c for c in (feature, f"{feature}{FUT_SUFFIX}") if c in features.columns]


def static_values(features: pd.DataFrame, feature: str) -> pd.Series:
    """reach_id -> the reach's value of a static feature (NaN if NULL). Raises if the value
    varies within a reach - a static attribute that is not static is a data error."""
    g = features.groupby(features["reach_id"].astype(str), sort=True)[feature]
    if (g.nunique(dropna=False) > 1).any():
        bad = list(g.nunique(dropna=False)[lambda s: s > 1].index[:3])
        raise ValueError(f"{feature} varies within reach(es) {bad}; it is not static")
    return g.first()


def lever_applicability(
    features: pd.DataFrame, intervention: InterventionRequest, coef: Intervention
) -> dict[str, str | None]:
    """reach_id -> None if the lever can be applied there, else why not (NOT_APPLICABLE)."""
    col = coef.effect.feature
    if col not in features.columns:
        raise KeyError(f"features have no column {col!r} for {coef.id}")
    present = set(features["reach_id"].astype(str))
    unknown = sorted(set(intervention.reach_ids) - present)
    if unknown:
        raise ValueError(f"{coef.id}: no feature rows for reach(es) {unknown}")
    out: dict[str, str | None] = {}
    if col in STATIC_FEATURES:
        values = static_values(features, col)
        for rid in intervention.reach_ids:
            v = values.get(rid)
            if v is None or pd.isna(v):
                out[rid] = f"{col} is NULL for this reach - nothing to perturb"
            elif (
                coef.effect.operation == "subtract_treated_share"
                and intervention.extent is not None
                and intervention.extent > float(v) + _EPS
            ):
                out[rid] = (
                    f"extent {intervention.extent:g} pp exceeds the reach's current {col} "
                    f"{float(v):.2f} - cannot treat more than exists"
                )
            else:
                out[rid] = None
    else:
        has = features.groupby(features["reach_id"].astype(str))[col].apply(
            lambda s: bool(s.notna().any())
        )
        for rid in intervention.reach_ids:
            out[rid] = None if bool(has.get(rid, False)) else f"{col} is NULL on every day"
    return out


def _perturb(
    features: pd.DataFrame,
    coef: Intervention,
    extent: float | None,
    magnitude: float,
    reaches: Iterable[str],
) -> pd.DataFrame:
    """Apply one lever at `magnitude` to the rows of `reaches`. No checks - callers check."""
    out = features.copy()
    rows = features["reach_id"].astype(str).isin(set(reaches)).to_numpy()
    if not rows.any():
        return out
    col = coef.effect.feature
    op = coef.effect.operation
    if col in MULTIPLICATIVE_DEPENDENTS and op != "scale_down":
        raise NotImplementedError(
            f"{col} has derived features {MULTIPLICATIVE_DEPENDENTS[col]} that are only "
            f"recomputed for multiplicative changes; {op} is not one"
        )
    if op == "subtract_treated_share":
        assert extent is not None
        for c in _columns(features, col):
            out.loc[rows, c] = features.loc[rows, c].to_numpy(dtype="float64") - magnitude * extent
    elif op == "floor":
        for c in _columns(features, col):
            # np.fmax would fill a NaN with the floor; np.maximum keeps NaN (never impute).
            x = features.loc[rows, c].to_numpy(dtype="float64")
            out.loc[rows, c] = np.maximum(x, magnitude)
    elif op == "scale_down":
        assert extent is not None
        factor = 1.0 - magnitude * extent
        for c in [*_columns(features, col)] + [
            d for dep in MULTIPLICATIVE_DEPENDENTS.get(col, ()) for d in _columns(features, dep)
        ]:
            out.loc[rows, c] = features.loc[rows, c].to_numpy(dtype="float64") * factor
    else:  # pragma: no cover - the coefficient schema admits no other operation
        raise ValueError(f"unknown operation {op}")
    return out


def apply_intervention(
    features: pd.DataFrame,
    intervention: InterventionRequest,
    coefficients: CoefficientTable,
    *,
    magnitude: float | None = None,
) -> pd.DataFrame:
    """A copy of `features` with one lever applied to the intervention's reaches.

    The effect size is the coefficient table's `magnitude` for the lever, or - to sample
    its uncertainty - a `magnitude` inside the table's `uncertainty_range` (anything
    outside raises: effect sizes come only from the table). Reaches where the lever is not
    applicable (lever_applicability) are left unchanged; NULLs stay NULL.
    """
    coef = resolve(coefficients, intervention.type)
    validate_extent(coef, intervention.extent)
    lo, hi = coef.uncertainty_range
    m = coef.magnitude if magnitude is None else float(magnitude)
    if not lo - _EPS <= m <= hi + _EPS:
        raise ValueError(
            f"{coef.id}: magnitude {m} is outside the cited uncertainty_range {[lo, hi]}"
        )
    ok = lever_applicability(features, intervention, coef)
    return _perturb(
        features, coef, intervention.extent, m, [r for r, why in ok.items() if why is None]
    )


def direct_coverage(coef: Intervention, extent: float | None, value: float | None) -> float:
    """Share (0-1) of a reach's catchment a LITERATURE_DIRECT lever covers."""
    op = coef.effect.operation
    if op == "subtract_treated_share":
        if value is None or not math.isfinite(value) or value <= 0 or extent is None:
            return 0.0
        return float(min(1.0, extent / value))  # share of the impervious area treated
    if op == "scale_down":
        return float(extent or 0.0)
    if op == "floor":
        return 1.0 if value is not None and math.isfinite(value) and value < coef.magnitude else 0.0
    raise ValueError(f"unknown operation {op}")  # pragma: no cover


# ---------------------------------------------------------------------------
# count distribution
# ---------------------------------------------------------------------------
def poisson_binomial_pmf(p: npt.ArrayLike) -> FloatArray:
    """PMF of the number of successes of independent Bernoulli(p_i) trials.

    p: shape (n,) or (k, n) -> pmf of shape (n + 1,) or (k, n + 1). Exact recursion."""
    arr = np.asarray(p, dtype="float64")
    one = arr.ndim == 1
    arr = np.atleast_2d(arr)
    if np.isnan(arr).any():
        raise ValueError("probabilities contain NaN")
    if (arr < -_EPS).any() or (arr > 1 + _EPS).any():
        raise ValueError("probabilities must be in [0, 1]")
    arr = np.clip(arr, 0.0, 1.0)
    k, n = arr.shape
    pmf = np.zeros((k, n + 1))
    pmf[:, 0] = 1.0
    for i in range(n):
        pi = arr[:, i : i + 1]
        pmf[:, 1 : i + 2] = pmf[:, 1 : i + 2] * (1.0 - pi) + pmf[:, 0 : i + 1] * pi
        pmf[:, 0] *= 1.0 - arr[:, i]
    return pmf[0] if one else pmf


def pmf_interval(pmf: npt.ArrayLike, level: float) -> tuple[int, int]:
    """Central `level` interval [lo, hi] of a count PMF: lo is the smallest k with
    F(k) >= (1 - level) / 2, hi the smallest k with F(k) >= (1 + level) / 2."""
    cdf = np.cumsum(np.asarray(pmf, dtype="float64"))
    lo = int(np.searchsorted(cdf, (1.0 - level) / 2.0 - _EPS, side="left"))
    hi = int(np.searchsorted(cdf, (1.0 + level) / 2.0 - _EPS, side="left"))
    last = len(cdf) - 1
    return min(lo, last), min(hi, last)


def widen_interval(
    central: float,
    raw: Interval,
    baseline: Estimate,
    factor: float,
    upper: float,
) -> Interval:
    """The widened scenario interval (module docstring). Contains `raw` and `central`,
    width factor * max(raw width, baseline width), clipped to [0, upper] by shifting."""
    lo = min(raw.low, central)
    hi = max(raw.high, central)
    base_w = baseline.interval.width
    width = min(factor * max(hi - lo, base_w), upper)
    if hi - lo > 0:
        f = (central - lo) / (hi - lo)
    else:
        f = (baseline.exceedance_days - baseline.interval.low) / base_w if base_w > 0 else 0.5
    f = min(1.0, max(0.0, f))
    new_lo = central - f * width
    new_hi = new_lo + width
    if new_lo < 0.0:
        new_lo, new_hi = 0.0, width
    if new_hi > upper:
        new_lo, new_hi = max(0.0, upper - width), upper
    return Interval(low=new_lo, high=new_hi)


# ---------------------------------------------------------------------------
# response check (MASTERSPEC 9.6 rule 2)
# ---------------------------------------------------------------------------
def judge_response(
    standardised_effect: float, observed_sign: int, expected_sign: int, negligible: float
) -> Verdict:
    if not math.isfinite(standardised_effect) or abs(standardised_effect) < negligible:
        return Verdict.NEGLIGIBLE
    if observed_sign != expected_sign:
        return Verdict.WRONG_SIGN
    return Verdict.TRUSTED


def _sign(x: float) -> Literal[-1, 0, 1]:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def response_check(
    model: ScenarioModel,
    features: pd.DataFrame,
    *,
    feature: str,
    variable: str,
    feature_sd: float,
    target_sd: float,
    expected_sign: Literal[-1, 1],
    negligible_effect_sd: float,
    fold: str,
    serving_model_version: str,
    thresholds: npt.ArrayLike | None = None,
) -> ResponseCheck:
    """Move a static feature -1 SD and +1 SD on held-out rows, everything else fixed,
    re-infer, and judge the sign and size of the mean P50 change.

    `features`: held-out model rows (with reach_id). `thresholds`: optional per-row
    exceedance thresholds (NaN where none) for the mean change in P(exceedance).
    Rows where the feature is NULL are not perturbed and not counted."""
    if feature not in STATIC_FEATURES:
        raise ValueError(f"the response check covers static attributes; {feature} is not one")
    if model.variant != SCENARIO_VARIANT:
        raise ValueError(f"the response check uses variant {SCENARIO_VARIANT}, not {model.variant}")
    for name, v in (("feature_sd", feature_sd), ("target_sd", target_sd)):
        if not (math.isfinite(v) and v > 0):
            raise ValueError(f"{name} must be positive and finite, got {v}")
    if feature not in features.columns:
        raise KeyError(f"features have no column {feature!r}")
    x = features[feature].to_numpy(dtype="float64")
    valid = ~np.isnan(x)
    if not valid.any():
        raise ValueError(f"{feature} is NULL on every held-out row")
    lo_f, hi_f = features.copy(), features.copy()
    lo_f.loc[valid, feature] = x[valid] - feature_sd
    hi_f.loc[valid, feature] = x[valid] + feature_sd
    q_lo = model.predict_quantiles(lo_f, variable)
    q_hi = model.predict_quantiles(hi_f, variable)
    j50 = model.levels.index(0.5)
    d = q_hi[:, j50] - q_lo[:, j50]
    change = float(np.mean(d[valid]))
    effect = change / target_sd
    sign = _sign(change)

    reach_ids = features["reach_id"].astype(str).to_numpy()[valid]
    per_reach = pd.Series(d[valid]).groupby(reach_ids).mean()
    n_reach = len(per_reach)

    p_change: float | None = None
    n_thr = 0
    if thresholds is not None:
        thr = np.asarray(thresholds, dtype="float64")
        if thr.shape != (len(features),):
            raise ValueError("thresholds must align with feature rows")
        m = valid & np.isfinite(thr)
        n_thr = int(m.sum())
        if n_thr:
            e_hi = exceedance_probability_multi(q_hi[m], model.levels, thr[m])
            e_lo = exceedance_probability_multi(q_lo[m], model.levels, thr[m])
            p_change = float(np.mean(e_hi - e_lo))

    rng = model.feature_range(variable, feature)
    oos: float | None = None
    if rng is not None:
        both = np.concatenate([x[valid] - feature_sd, x[valid] + feature_sd])
        oos = float(np.mean((both < rng[0]) | (both > rng[1])))

    return ResponseCheck(
        feature=feature,
        variable=variable,
        checked_model_version=model.version,
        serving_model_version=serving_model_version,
        fold=fold,
        rows=int(valid.sum()),
        reaches=n_reach,
        feature_sd=float(feature_sd),
        target_sd=float(target_sd),
        mean_p50_change=change,
        standardised_effect=effect,
        observed_sign=sign,
        expected_sign=expected_sign,
        mean_exceedance_prob_change=p_change,
        exceedance_rows=n_thr,
        share_reaches_up=float((per_reach > _EPS).mean()) if n_reach else 0.0,
        share_reaches_down=float((per_reach < -_EPS).mean()) if n_reach else 0.0,
        out_of_support_share=oos,
        negligible_effect_sd=float(negligible_effect_sd),
        verdict=judge_response(effect, sign, expected_sign, negligible_effect_sd),
    )


def decide_path(
    coef: Intervention,
    variable: str,
    model: ScenarioModel,
    response_checks: Mapping[tuple[str, str], ResponseCheck],
    coefficients: CoefficientTable,
) -> tuple[LeverPath, ResponseCheck | None, str]:
    """MASTERSPEC 9.6 rules 2-3: which path a lever takes for a target variable, the check
    that decided it, and why - in words shown next to the result."""
    f = coef.effect.feature
    uses = [c for c in (f, f"{f}{FUT_SUFFIX}") if c in model.features]
    if not uses:
        return (
            LeverPath.NOT_ESTIMABLE,
            None,
            f"the scenario model ({model.version}) does not use {f}",
        )
    if f not in STATIC_FEATURES:
        return (
            LeverPath.MODEL_PERTURBATION,
            None,
            f"{f} is a weather driver; the 9.6 response check covers static catchment "
            "attributes, so the cited change goes through the model",
        )
    check = response_checks.get((f, variable))
    if check is None:
        raise MissingResponseCheck(
            f"no response check on record for {f} / {variable} - run `make response-check`"
        )
    if check.serving_model_version != model.version:
        raise StaleResponseCheck(
            f"response check for {f} / {variable} was run for {check.serving_model_version}, "
            f"the scenario model is {model.version} - re-run `make response-check`"
        )
    if check.expected_sign != expected_response_sign(coefficients, f):
        raise StaleResponseCheck(
            f"response check for {f} expected sign {check.expected_sign}, the coefficient "
            "table now implies the opposite - re-run `make response-check`"
        )
    summary = (
        f"response check on {check.fold} fold ({check.checked_model_version}): "
        f"+-1 SD of {f} moves P50 by {check.mean_p50_change:+.4g} "
        f"({check.standardised_effect:+.3f} target SD; negligible below "
        f"{check.negligible_effect_sd}), sign {check.observed_sign:+d} vs literature "
        f"{check.expected_sign:+d} -> {check.verdict}"
    )
    if check.verdict is Verdict.TRUSTED:
        return LeverPath.MODEL_PERTURBATION, check, summary
    direct = coef.direct_effect
    if direct is not None and variable in direct.targets:
        return (
            LeverPath.LITERATURE_DIRECT,
            check,
            f"{summary}; model not trusted, the cited direct effect scales the forecast "
            "quantiles instead",
        )
    return (
        LeverPath.NOT_ESTIMABLE,
        check,
        f"{summary}; model not trusted, and the coefficient table has no cited direct "
        f"effect of {coef.id} on {variable} - not estimated",
    )


# ---------------------------------------------------------------------------
# costs
# ---------------------------------------------------------------------------
def cost_estimate(
    coef: Intervention, extent: float | None, reach_id: str, catchment_area_km2: float | None
) -> CostEstimate:
    """Unit costs from the table, and a total only where the lever's extent converts to the
    cost unit without an assumption: treated percentage points of catchment area -> square
    metres. Currency and price basis as in the source - never converted."""
    c = coef.cost_per_unit
    quantity: float | None = None
    quantity_unit: str | None = None
    reason: str | None = None
    if coef.effect.operation == "subtract_treated_share" and c.unit.startswith("square metre"):
        if catchment_area_km2 is None or not math.isfinite(catchment_area_km2) or extent is None:
            reason = "catchment area unknown for this reach"
        else:
            quantity = extent / 100.0 * catchment_area_km2 * 1e6
            quantity_unit = "square metre"
    else:
        reason = f"no defined conversion from this lever's extent to '{c.unit}'; unit cost only"
    return CostEstimate(
        intervention_id=coef.id,
        reach_id=reach_id,
        currency=c.currency,
        unit=c.unit,
        unit_cost_low=c.low,
        unit_cost_high=c.high,
        price_basis=c.price_basis,
        quantity=quantity,
        quantity_unit=quantity_unit,
        total_low=None if quantity is None else quantity * c.low,
        total_high=None if quantity is None else quantity * c.high,
        reason=reason,
        note=" ".join(c.note.split()) if c.note else None,
    )


# ---------------------------------------------------------------------------
# the scenario run
# ---------------------------------------------------------------------------
@dataclass
class _Lever:
    index: int
    request: InterventionRequest
    coef: Intervention
    applicable: dict[str, str | None]  # reach -> None | reason not applicable


def _usable(lv: _Lever) -> list[str]:
    return [r for r, why in lv.applicable.items() if why is None]


def _lever_columns(features: pd.DataFrame, feature: str) -> list[str]:
    """Every column a lever on `feature` may change: it, its target-day copy, dependents."""
    return _columns(features, feature) + [
        c for dep in MULTIPLICATIVE_DEPENDENTS.get(feature, ()) for c in _columns(features, dep)
    ]


@dataclass
class _EvalContext:
    frame: pd.DataFrame
    model: ScenarioModel
    variable: str
    thr: FloatArray
    q0: FloatArray
    reach_rows: dict[str, npt.NDArray[np.intp]]
    static_vals: dict[str, pd.Series]
    model_levers: list[_Lever]
    direct_levers: list[_Lever]
    changed: dict[tuple[int, str], bool]  # (lever, reach) -> any value changed, any sample
    oos: dict[tuple[int, str], list[str]]  # (lever, reach) -> out-of-support messages


def _track_changes(
    ctx: _EvalContext, lv: _Lever, before: pd.DataFrame, after: pd.DataFrame
) -> None:
    """Record, per reach, whether the lever changed any value and whether a changed value
    left the model's training range (OUT_OF_SUPPORT)."""
    for c in _lever_columns(before, lv.coef.effect.feature):
        b = before[c].to_numpy(dtype="float64")
        a = after[c].to_numpy(dtype="float64")
        diff = ~((b == a) | (np.isnan(b) & np.isnan(a)))
        rng = ctx.model.feature_range(ctx.variable, c) if c in ctx.model.features else None
        for rid in _usable(lv):
            idx = ctx.reach_rows[rid]
            dm = diff[idx]
            key = (lv.index, rid)
            if not dm.any():
                ctx.changed.setdefault(key, False)
                continue
            ctx.changed[key] = True
            if rng is None:
                continue
            vals = a[idx][dm]
            bad = (vals < rng[0]) | (vals > rng[1])
            base = b[idx][dm]
            already = int((bad & ((base < rng[0]) | (base > rng[1]))).sum())
            msgs = ctx.oos.setdefault(key, [])
            if bad.any() and not any(m.startswith(f"{c}:") for m in msgs):
                msgs.append(
                    f"{c}: perturbed values outside the training range "
                    f"[{rng[0]:.4g}, {rng[1]:.4g}] on {int(bad.sum())} day(s)"
                    + (f" ({already} already outside at baseline)" if already else "")
                    + " - the model returns its edge value there, so the response is truncated"
                )


def _evaluate(ctx: _EvalContext, u: float | None) -> FloatArray:
    """Per-row P(exceed) with every estimable lever at quantile u of its coefficient
    distribution (u None = the cited central magnitude)."""
    feats = ctx.frame
    for lv in ctx.model_levers:
        lo, hi = lv.coef.uncertainty_range
        m = lv.coef.magnitude if u is None else triangular_ppf(u, lo, lv.coef.magnitude, hi)
        after = _perturb(feats, lv.coef, lv.request.extent, m, _usable(lv))
        _track_changes(ctx, lv, feats, after)
        feats = after
    q = (
        np.asarray(ctx.model.predict_quantiles(feats, ctx.variable), dtype="float64")
        if ctx.model_levers
        else ctx.q0.copy()
    )
    for lv in ctx.direct_levers:
        de = lv.coef.direct_effect
        if de is None:  # pragma: no cover - decide_path only routes levers that have one
            raise ValueError(f"{lv.coef.id} has no direct effect")
        lo, hi = de.uncertainty_range
        m = de.magnitude if u is None else triangular_ppf(u, lo, de.magnitude, hi)
        cover = np.zeros(len(ctx.frame))
        values = ctx.static_vals.get(lv.coef.effect.feature)
        for rid in _usable(lv):
            v = None if values is None else values.get(rid)
            cv = direct_coverage(
                lv.coef, lv.request.extent, None if v is None or pd.isna(v) else float(v)
            )
            cover[ctx.reach_rows[rid]] = cv
            key = (lv.index, rid)
            ctx.changed[key] = ctx.changed.get(key, False) or (cv > 0 and m > 0)
        q = q * (1.0 - m * cover)[:, None]
    return exceedance_probability_multi(q, ctx.model.levels, ctx.thr)


def _missing_seasons(dates: pd.Series, seasons: Mapping[str, Sequence[int]]) -> list[str]:
    s = season_of(dates, {k: list(v) for k, v in seasons.items()})
    return sorted(set(s.dropna()))


def run_scenario(
    reach_ids: Sequence[str],
    interventions: Sequence[InterventionRequest],
    model: ScenarioModel,
    baseline_features: pd.DataFrame,
    *,
    coefficients: CoefficientTable,
    thresholds: pd.DataFrame,
    response_checks: Mapping[tuple[str, str], ResponseCheck],
    settings: ScenarioSettings,
    variables: Sequence[str] = DEFAULT_VARIABLES,
) -> ScenarioResult:
    """Baseline vs scenario exceedance days per year for every reach, planning estimate.

    baseline_features  one row per (reach_id, target_date) covering the reference year at
                       settings.horizon: columns reach_id, target_date and model.features
    thresholds         engine.thresholds table (reach_id, variable, season, threshold)
    response_checks    {(feature, variable): ResponseCheck} for the static features levers
                       perturb (models/scenario_response.py)
    """
    rids = tuple(dict.fromkeys(str(r) for r in reach_ids))
    if not rids:
        raise ValueError("no reaches")
    if not interventions:
        raise ValueError("no interventions")
    if not variables:
        raise ValueError("no variables")
    if model.variant != SCENARIO_VARIANT:
        raise ValueError(
            f"scenarios use variant {SCENARIO_VARIANT} only (MASTERSPEC 9.6); got variant "
            f"{model.variant} ({model.version})"
        )
    covered = {r for i in interventions for r in i.reach_ids}
    if missing := sorted(set(rids) - covered):
        raise ValueError(f"reach(es) {missing} have no intervention")
    if extra := sorted(covered - set(rids)):
        raise ValueError(f"interventions name reach(es) {extra} that are not in reach_ids")
    pairs = [(i.type, r) for i in interventions for r in i.reach_ids]
    if dup := sorted({p for p in pairs if pairs.count(p) > 1}):
        raise ValueError(f"the same lever is requested twice on a reach: {dup}")

    need = {"reach_id", "target_date", *model.features}
    if lacking := sorted(need - set(baseline_features.columns)):
        raise ValueError(f"baseline_features lack columns {lacking}")
    td = pd.to_datetime(baseline_features["target_date"])
    in_period = (td >= pd.Timestamp(settings.reference_start)) & (
        td <= pd.Timestamp(settings.reference_end)
    )
    frame = baseline_features[
        in_period.to_numpy() & baseline_features["reach_id"].astype(str).isin(rids).to_numpy()
    ].copy()
    frame["reach_id"] = frame["reach_id"].astype(str)
    frame["target_date"] = pd.to_datetime(frame["target_date"]).dt.date
    frame = frame.sort_values(["reach_id", "target_date"]).reset_index(drop=True)
    if frame.duplicated(["reach_id", "target_date"]).any():
        raise ValueError("baseline_features have duplicate (reach_id, target_date) rows")
    if unknown := sorted(set(rids) - set(frame["reach_id"])):
        raise ValueError(f"no feature rows in the reference year for reach(es) {unknown}")

    # ---- levers: coefficients, extents, applicability (combined extents per feature) ----
    levers: list[_Lever] = []
    for k, req in enumerate(interventions):
        coef = resolve(coefficients, req.type)
        validate_extent(coef, req.extent)
        levers.append(_Lever(k, req, coef, lever_applicability(frame, req, coef)))
    for feat in {lv.coef.effect.feature for lv in levers}:
        subtract = [
            lv
            for lv in levers
            if lv.coef.effect.feature == feat
            and lv.coef.effect.operation == "subtract_treated_share"
        ]
        if len(subtract) < 2:
            continue
        values = static_values(frame, feat)
        for rid in rids:
            on = [lv for lv in subtract if rid in lv.request.reach_ids]
            total = sum(lv.request.extent or 0.0 for lv in on)
            v = values.get(rid)
            if len(on) > 1 and v is not None and pd.notna(v) and total > float(v) + _EPS:
                for lv in on:
                    lv.applicable[rid] = (
                        f"combined extent {total:g} pp of levers on {feat} exceeds the reach's "
                        f"current {float(v):.2f}"
                    )

    static_vals = {
        f: static_values(frame, f)
        for f in {lv.coef.effect.feature for lv in levers}
        if f in STATIC_FEATURES
    }
    area = static_values(frame, "catchment_area_km2") if "catchment_area_km2" in frame else None
    samples = coefficient_levels(settings.coefficient_samples)
    reach_of_row = frame["reach_id"].to_numpy()
    reach_rows = {rid: np.flatnonzero(reach_of_row == rid) for rid in rids}

    lever_summaries: list[LeverSummary] = []
    reach_results: list[ReachResult] = []
    direct_used: set[int] = set()

    for variable in variables:
        paths: dict[int, tuple[LeverPath, ResponseCheck | None, str]] = {
            lv.index: decide_path(lv.coef, variable, model, response_checks, coefficients)
            for lv in levers
        }
        for lv in levers:
            path, check, why = paths[lv.index]
            d = lv.coef.direct_effect if path is LeverPath.LITERATURE_DIRECT else None
            if path is LeverPath.LITERATURE_DIRECT:
                direct_used.add(lv.index)
            lever_summaries.append(
                LeverSummary(
                    intervention_id=lv.coef.id,
                    name=lv.coef.name,
                    variable=variable,
                    feature=lv.coef.effect.feature,
                    operation=lv.coef.effect.operation,
                    extent=lv.request.extent,
                    reach_ids=lv.request.reach_ids,
                    path=path,
                    reason=why,
                    magnitude=lv.coef.magnitude,
                    uncertainty_range=lv.coef.uncertainty_range,
                    direct_magnitude=d.magnitude if d else None,
                    direct_uncertainty_range=d.uncertainty_range if d else None,
                    response_check=check,
                    citations=citations_of(lv.coef.id, lv.coef.citation, "effect")
                    + (citations_of(lv.coef.id, d.citation, "direct_effect") if d else ()),
                )
            )

        # thresholds per row and the baseline
        targets = frame[["reach_id", "target_date"]].assign(variable=variable)
        seasons = {k: list(v) for k, v in settings.seasons.items()}
        thr = threshold_lookup(targets, thresholds, seasons)["threshold"].to_numpy(dtype="float64")
        q0 = np.asarray(model.predict_quantiles(frame, variable), dtype="float64")
        p0 = exceedance_probability_multi(q0, model.levels, thr)

        model_levers = [lv for lv in levers if paths[lv.index][0] is LeverPath.MODEL_PERTURBATION]
        direct_levers = [lv for lv in levers if paths[lv.index][0] is LeverPath.LITERATURE_DIRECT]

        changed: dict[tuple[int, str], bool] = {}
        oos: dict[tuple[int, str], list[str]] = {}
        ctx = _EvalContext(
            frame=frame,
            model=model,
            variable=variable,
            thr=thr,
            q0=q0,
            reach_rows=reach_rows,
            static_vals=static_vals,
            model_levers=model_levers,
            direct_levers=direct_levers,
            changed=changed,
            oos=oos,
        )
        any_estimable = bool(model_levers or direct_levers)
        p_central = _evaluate(ctx, None) if any_estimable else None
        p_samples = np.stack([_evaluate(ctx, u) for u in samples]) if any_estimable else None

        for rid in rids:
            idx = reach_rows[rid]
            on = [lv for lv in levers if rid in lv.request.reach_ids]
            outcomes: list[LeverOutcome] = []
            estimated = 0
            for lv in on:
                path = paths[lv.index][0]
                feat = lv.coef.effect.feature
                flags: list[ReachFlag] = []
                detail: list[str] = []
                not_applicable = lv.applicable.get(rid)
                before: float | None = None
                after_c: float | None = None
                if feat in static_vals:
                    sv = static_vals[feat].get(rid)
                    before = None if sv is None or pd.isna(sv) else float(sv)
                if not_applicable is not None:
                    flags.append(ReachFlag.NOT_APPLICABLE)
                    detail.append(not_applicable)
                elif path is not LeverPath.NOT_ESTIMABLE:
                    estimated += 1
                    if not changed.get((lv.index, rid), False):
                        flags.append(ReachFlag.NO_CHANGE)
                        detail.append(
                            f"the lever leaves {feat} unchanged on this reach at every "
                            "coefficient value"
                            + (f" (current {before:.4g})" if before is not None else "")
                        )
                    if msgs := oos.get((lv.index, rid)):
                        flags.append(ReachFlag.OUT_OF_SUPPORT)
                        detail.extend(msgs)
                    if before is not None and path is LeverPath.MODEL_PERTURBATION:
                        one = frame.iloc[idx[:1]]
                        after_c = float(
                            _perturb(one, lv.coef, lv.request.extent, lv.coef.magnitude, [rid])[
                                feat
                            ].iloc[0]
                        )
                outcomes.append(
                    LeverOutcome(
                        intervention_id=lv.coef.id,
                        path=path,
                        flags=tuple(flags),
                        detail=tuple(detail),
                        feature=feat,
                        value_before=before,
                        value_after=after_c,
                    )
                )
            complete = estimated == len(on)

            # evidence: every day of the reference year, with a threshold
            n = len(idx)
            reason: str | None = None
            if n != settings.period_days:
                reason = (
                    f"MISSING_DAYS: {n} of {settings.period_days} reference-year days have features"
                )
            elif not np.isfinite(thr[idx]).all():
                dates = pd.Series(pd.to_datetime(frame["target_date"].to_numpy()[idx]))
                missing = _missing_seasons(dates[~np.isfinite(thr[idx])], settings.seasons)
                reason = (
                    f"NO_THRESHOLD: no seasonal threshold for {', '.join(missing)} "
                    "(too few past observations; driver-only reaches have none)"
                )
            elif not np.isfinite(p0[idx]).all():
                reason = "NO_FORECAST: forecast quantiles missing for some days"
            if reason is not None:
                reach_results.append(
                    ReachResult(
                        reach_id=rid,
                        variable=variable,
                        status=Status.INSUFFICIENT_EVIDENCE,
                        reason=reason,
                        days_evaluated=n,
                        complete=complete,
                        levers=tuple(outcomes),
                    )
                )
                continue

            pmf0 = poisson_binomial_pmf(p0[idx])
            lo0, hi0 = pmf_interval(pmf0, settings.ci_level)
            c0 = float(p0[idx].sum())
            baseline = Estimate(
                exceedance_days=c0, interval=Interval(low=float(lo0), high=float(hi0))
            )
            if estimated == 0 or p_central is None or p_samples is None:
                reach_results.append(
                    ReachResult(
                        reach_id=rid,
                        variable=variable,
                        status=Status.NOT_ESTIMABLE,
                        reason="NO_ESTIMABLE_LEVER: "
                        + "; ".join(
                            f"{o.intervention_id}: "
                            + (o.detail[0] if o.detail else paths[lv.index][2])
                            for o, lv in zip(outcomes, on, strict=True)
                        ),
                        days_evaluated=n,
                        complete=False,
                        baseline=baseline,
                        levers=tuple(outcomes),
                    )
                )
                continue
            if not 0 < baseline.interval.width < n:
                reach_results.append(
                    ReachResult(
                        reach_id=rid,
                        variable=variable,
                        status=Status.NOT_ESTIMABLE,
                        reason=(
                            f"BASELINE_INTERVAL_DEGENERATE: baseline interval "
                            f"[{lo0}, {hi0}] days has no room to widen"
                        ),
                        days_evaluated=n,
                        complete=complete,
                        baseline=baseline,
                        levers=tuple(outcomes),
                    )
                )
                continue

            cs = float(p_central[idx].sum())
            mix = poisson_binomial_pmf(p_samples[:, idx]).mean(axis=0)
            lo_r, hi_r = pmf_interval(mix, settings.ci_level)
            interval = widen_interval(
                cs,
                Interval(low=float(lo_r), high=float(hi_r)),
                baseline,
                settings.widening_factor,
                float(n),
            )
            if not interval.width > baseline.interval.width:
                raise ScenarioInvariantError(
                    f"{rid} {variable}: scenario interval {interval} is not wider than "
                    f"baseline {baseline.interval}"
                )
            sums = np.append(p_samples[:, idx].sum(axis=1), cs) - c0
            reach_results.append(
                ReachResult(
                    reach_id=rid,
                    variable=variable,
                    status=Status.OK,
                    days_evaluated=n,
                    complete=complete,
                    baseline=baseline,
                    scenario=Estimate(exceedance_days=cs, interval=interval),
                    delta_days=cs - c0,
                    delta_pct=(cs - c0) / c0 * 100.0 if c0 > 0 else None,
                    delta_coefficient_range=Interval(low=float(sums.min()), high=float(sums.max())),
                    levers=tuple(outcomes),
                )
            )

    def area_of(rid: str) -> float | None:
        v = None if area is None else area.get(rid)
        return None if v is None or pd.isna(v) else float(v)

    costs = tuple(
        cost_estimate(lv.coef, lv.request.extent, rid, area_of(rid))
        for lv in levers
        for rid in lv.request.reach_ids
        if lv.applicable.get(rid) is None
    )
    cites: dict[tuple[str, str, str, str], CitationOut] = {}
    for lv in levers:
        refs = citations_of(lv.coef.id, lv.coef.citation, "effect") + citations_of(
            lv.coef.id, lv.coef.cost_per_unit.citation, "cost"
        )
        if lv.index in direct_used and lv.coef.direct_effect is not None:
            refs += citations_of(lv.coef.id, lv.coef.direct_effect.citation, "direct_effect")
        for c in refs:
            cites.setdefault((c.intervention_id, c.role, c.title, c.locator), c)

    return ScenarioResult(
        caveat=settings.caveat,
        model_version=model.version,
        variant=model.variant,
        coefficient_table_version=coefficients.version,
        reference_start=settings.reference_start,
        reference_end=settings.reference_end,
        horizon=settings.horizon,
        ci_level=settings.ci_level,
        widening_factor=settings.widening_factor,
        coefficient_samples=settings.coefficient_samples,
        interval_method=(
            f"central {settings.ci_level:.0%} interval of the Poisson-binomial count of "
            "exceedance days (days independent given the forecast - real days are "
            "autocorrelated, so this understates the spread). Scenario: mixture over "
            f"{settings.coefficient_samples} triangular quantiles of each coefficient's "
            f"cited uncertainty_range, widened x{settings.widening_factor:g} of the larger "
            "of its own and the baseline width."
        ),
        threshold_derivation=settings.threshold_derivation,
        variables=tuple(variables),
        levers=tuple(lever_summaries),
        reaches=tuple(reach_results),
        costs=costs,
        citations=tuple(cites.values()),
    )
