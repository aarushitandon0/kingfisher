"""Intervention coefficient table - schema and validation, pure (no I/O).

config/intervention_coefficients.yaml is the ONLY source of scenario effect sizes
(CLAUDE.md #6). This module refuses the table unless every intervention is cited AND
every number in it - magnitude, both uncertainty bounds, both cost bounds - is claimed by
at least one reference's `supports` list. A reference must carry authors, year, title,
source, a DOI or URL, and a locator saying where in the source the number is.

Validation collects every problem in the file and raises once, so a bad edit shows all of
its faults, not the first. Load the file through core.config.load_intervention_coefficients.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Claim = Literal["magnitude", "uncertainty_low", "uncertainty_high", "cost_low", "cost_high"]
EFFECT_CLAIMS: frozenset[str] = frozenset({"magnitude", "uncertainty_low", "uncertainty_high"})
COST_CLAIMS: frozenset[str] = frozenset({"cost_low", "cost_high"})


class UncitedCoefficientError(ValueError):
    """The coefficient table has an uncited or malformed entry. Nothing in it is usable."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class Reference(_Strict):
    authors: str = Field(min_length=1)
    year: int = Field(ge=1900, le=2100)
    title: str = Field(min_length=1)
    source: str = Field(min_length=1)
    doi: str | None = None
    url: str | None = None
    locator: str = Field(min_length=1)
    supports: tuple[Claim, ...]

    @model_validator(mode="after")
    def _needs_doi_or_url(self) -> Reference:
        if not (self.doi or self.url):
            raise ValueError("a reference needs a doi or a url")
        return self


class Effect(_Strict):
    feature: str
    operation: Literal["subtract_treated_share", "floor", "scale_down"]
    direction: Literal["increase", "decrease"]
    unit: str = Field(min_length=1)
    description: str = Field(min_length=1)


class AppliesTo(_Strict):
    reaches: Literal["all", "observable", "driver_only"]
    conditions: tuple[str, ...] = Field(min_length=1)


class Maintenance(_Strict):
    low: float = Field(ge=0)
    high: float = Field(ge=0)
    unit: str = Field(min_length=1)


class Cost(_Strict):
    low: float = Field(ge=0)
    high: float = Field(ge=0)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    unit: str = Field(min_length=1)
    price_basis: str = Field(min_length=1)
    maintenance: Maintenance | None = None
    citation: tuple[Reference, ...] = Field(min_length=1)
    note: str | None = None

    @model_validator(mode="after")
    def _ordered_and_cited(self) -> Cost:
        if self.low > self.high:
            raise ValueError(f"cost low {self.low} > high {self.high}")
        _require_claims(self.citation, COST_CLAIMS, "cost_per_unit")
        return self


class Intervention(_Strict):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    name: str = Field(min_length=1)
    effect: Effect
    magnitude: float
    uncertainty_range: tuple[float, float]
    citation: tuple[Reference, ...] = Field(min_length=1)
    applies_to: AppliesTo
    cost_per_unit: Cost

    @model_validator(mode="after")
    def _consistent(self) -> Intervention:
        low, high = self.uncertainty_range
        if not low <= self.magnitude <= high:
            raise ValueError(
                f"uncertainty_range {[low, high]} must bracket magnitude {self.magnitude}"
            )
        if self.effect.operation in ("subtract_treated_share", "scale_down") and not (
            low >= 0.0 and high <= 1.0
        ):
            raise ValueError(
                f"{self.effect.operation} takes a fraction in [0, 1], got {[low, high]}"
            )
        _require_claims(self.citation, EFFECT_CLAIMS, "citation")
        return self


class NotQuantified(_Strict):
    """A MASTERSPEC lever with no usable coefficient - listed, never loadable."""

    id: str
    name: str
    reason: str = Field(min_length=1)
    citation: tuple[Reference, ...] = Field(min_length=1)


class CoefficientTable(_Strict):
    version: str
    allowed_features: tuple[str, ...] = Field(min_length=1)
    interventions: tuple[Intervention, ...]
    not_quantified: tuple[NotQuantified, ...] = ()
    climate_scenarios: tuple[Any, ...] = ()
    uncertainty_inflation: dict[str, Any]

    @model_validator(mode="after")
    def _table_rules(self) -> CoefficientTable:
        ids = [i.id for i in self.interventions] + [n.id for n in self.not_quantified]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate intervention ids {dupes}")
        for i in self.interventions:
            if i.effect.feature not in self.allowed_features:
                raise ValueError(
                    f"{i.id}: effect.feature {i.effect.feature!r} is not in allowed_features"
                )
        if not str(self.uncertainty_inflation.get("caveat") or "").strip():
            raise ValueError(
                "uncertainty_inflation.caveat is required - the caveat is always shown"
            )
        return self

    def get(self, intervention_id: str) -> Intervention:
        for i in self.interventions:
            if i.id == intervention_id:
                return i
        if any(n.id == intervention_id for n in self.not_quantified):
            raise KeyError(f"{intervention_id} has no cited coefficient (not_quantified)")
        raise KeyError(f"unknown intervention {intervention_id!r}")


def _require_claims(refs: tuple[Reference, ...], needed: frozenset[str], where: str) -> None:
    claimed = {c for r in refs for c in r.supports}
    missing = sorted(needed - claimed)
    if missing:
        raise ValueError(f"{where}: no reference supports {missing}")


def validate_coefficients(raw: Mapping[str, Any]) -> CoefficientTable:
    """The parsed YAML as a validated table. Raises UncitedCoefficientError listing every
    problem, with the id of the entry it belongs to where there is one."""
    if not isinstance(raw, Mapping):
        raise UncitedCoefficientError("intervention coefficients: expected a mapping at top level")
    try:
        return CoefficientTable.model_validate(dict(raw))
    except ValidationError as exc:
        entries = raw.get("interventions") or []
        problems = []
        for err in exc.errors():
            loc = err["loc"]
            label = ".".join(str(p) for p in loc)
            if len(loc) >= 2 and loc[0] == "interventions" and isinstance(loc[1], int):
                idx = loc[1]
                if idx < len(entries) and isinstance(entries[idx], Mapping):
                    entry_id = entries[idx].get("id", f"#{idx}")
                    label = f"{entry_id}: {'.'.join(str(p) for p in loc[2:])}"
            problems.append(f"  - {label}: {err['msg']}")
        raise UncitedCoefficientError(
            "Refusing to load intervention coefficients - every effect size needs a citation "
            "(CLAUDE.md #6):\n" + "\n".join(problems)
        ) from None
