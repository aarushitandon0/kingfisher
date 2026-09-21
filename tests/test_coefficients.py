"""Coefficient-table guardrails (CLAUDE.md #6). The real file must load; then every rule
is broken on purpose and the validator must refuse the whole table."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest
import yaml

from core.config import load_config, load_intervention_coefficients
from core.settings import CONFIG_DIR
from engine.coefficients import UncitedCoefficientError, validate_coefficients

SPEC_LEVERS = {
    "riparian_buffer_restoration",
    "permeable_paving",
    "detention_basin",
    "daylighting",
    "street_sweeping",
    "green_roofs",
}


@pytest.fixture(scope="module")
def raw() -> dict[str, Any]:
    path = CONFIG_DIR / "intervention_coefficients.yaml"
    cfg: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return cfg


def _broken(raw: dict[str, Any], mutate: Callable[[dict[str, Any]], object]) -> dict[str, Any]:
    cfg = copy.deepcopy(raw)
    mutate(cfg)
    return cfg


def test_real_table_loads_and_covers_every_masterspec_lever() -> None:
    table = load_intervention_coefficients()
    ids = {i.id for i in table.interventions} | {n.id for n in table.not_quantified}
    assert ids == SPEC_LEVERS
    assert table.interventions, "no usable intervention at all"


def test_every_loaded_number_is_claimed_by_a_reference() -> None:
    for i in load_intervention_coefficients().interventions:
        effect_claims = {c for r in i.citation for c in r.supports}
        assert {"magnitude", "uncertainty_low", "uncertainty_high"} <= effect_claims, i.id
        cost_claims = {c for r in i.cost_per_unit.citation for c in r.supports}
        assert {"cost_low", "cost_high"} <= cost_claims, i.id
        for ref in (*i.citation, *i.cost_per_unit.citation):
            assert ref.doi or ref.url, (i.id, ref.title)


def test_generic_loader_refuses_the_coefficient_file() -> None:
    with pytest.raises(ValueError, match="load_intervention_coefficients"):
        load_config("intervention_coefficients")


def test_not_quantified_lever_cannot_be_fetched() -> None:
    with pytest.raises(KeyError, match="no cited coefficient"):
        load_intervention_coefficients().get("daylighting")


BREAKAGES: dict[str, Callable[[dict[str, Any]], object]] = {
    "citation_removed": lambda c: c["interventions"][0].pop("citation"),
    "citation_empty": lambda c: c["interventions"][0].update(citation=[]),
    "citation_null": lambda c: c["interventions"][0].update(citation=None),
    "reference_without_doi_or_url": lambda c: [
        r.pop(k, None) for r in c["interventions"][1]["citation"] for k in ("doi", "url")
    ],
    "reference_without_authors": lambda c: c["interventions"][0]["citation"][0].update(authors=""),
    "reference_without_year": lambda c: c["interventions"][0]["citation"][0].pop("year"),
    "reference_without_title": lambda c: c["interventions"][0]["citation"][0].pop("title"),
    "reference_without_locator": lambda c: c["interventions"][0]["citation"][0].pop("locator"),
    "magnitude_unclaimed": lambda c: [
        r.update(supports=[s for s in r["supports"] if s != "magnitude"])
        for r in c["interventions"][0]["citation"]
    ],
    "uncertainty_bound_unclaimed": lambda c: [
        r.update(supports=[s for s in r["supports"] if s != "uncertainty_high"])
        for r in c["interventions"][4]["citation"]
    ],
    "cost_uncited": lambda c: c["interventions"][2]["cost_per_unit"].update(citation=[]),
    "cost_bound_unclaimed": lambda c: c["interventions"][2]["cost_per_unit"]["citation"][0].update(
        supports=["cost_low"]
    ),
    "magnitude_outside_range": lambda c: c["interventions"][1].update(magnitude=1.5),
    "fraction_above_one": lambda c: c["interventions"][1].update(uncertainty_range=[0.3, 1.2]),
    "unknown_feature": lambda c: c["interventions"][0]["effect"].update(feature="reach_id"),
    "unknown_operation": lambda c: c["interventions"][0]["effect"].update(
        operation="ask_the_model"
    ),
    "missing_applies_to": lambda c: c["interventions"][0].pop("applies_to"),
    "missing_cost": lambda c: c["interventions"][0].pop("cost_per_unit"),
    "unknown_field_smuggled_in": lambda c: c["interventions"][0].update(model_estimate=0.4),
    "duplicate_id": lambda c: c["interventions"].append(copy.deepcopy(c["interventions"][0])),
    "caveat_removed": lambda c: c["uncertainty_inflation"].update(caveat=""),
    "cost_currency_not_iso": lambda c: c["interventions"][0]["cost_per_unit"].update(
        currency="euro"
    ),
}


@pytest.mark.parametrize("name", sorted(BREAKAGES))
def test_broken_table_is_refused(raw: dict[str, Any], name: str) -> None:
    with pytest.raises(UncitedCoefficientError):
        validate_coefficients(_broken(raw, BREAKAGES[name]))


def test_error_names_the_offending_entry(raw: dict[str, Any]) -> None:
    cfg = _broken(raw, BREAKAGES["citation_removed"])
    with pytest.raises(UncitedCoefficientError, match=raw["interventions"][0]["id"]):
        validate_coefficients(cfg)
