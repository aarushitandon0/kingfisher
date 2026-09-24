"""Scenario engine (engine/scenarios.py) - MASTERSPEC 9, CLAUDE.md #6 and #7.

A deterministic linear fake stands in for the variant-B model, so every expectation can
be reasoned about by hand. The rules under test:

  - effect sizes come only from the coefficient table, never from the model
  - coefficient uncertainty_range reaches the result
  - every scenario interval is STRICTLY wider than its baseline interval
  - every result carries the citations of every coefficient used
  - the output says "planning estimate", never "prediction"
  - MASTERSPEC 9.6: variant B only, response check decides the path per lever,
    OUT_OF_SUPPORT is flagged, missing evidence is INSUFFICIENT_EVIDENCE (never filled)
"""

from __future__ import annotations

import copy
import itertools
import json
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from core.config import load_config, load_intervention_coefficients
from core.settings import CONFIG_DIR
from engine.coefficients import CoefficientTable, validate_coefficients
from engine.probability import DEFAULT_LEVELS
from engine.scenarios import (
    Estimate,
    Interval,
    InterventionRequest,
    LeverPath,
    MissingResponseCheck,
    ReachFlag,
    ResponseCheck,
    ScenarioResult,
    ScenarioSettings,
    StaleResponseCheck,
    Status,
    Verdict,
    apply_intervention,
    coefficient_levels,
    expected_response_sign,
    judge_response,
    pmf_interval,
    poisson_binomial_pmf,
    response_check,
    run_scenario,
    triangular_ppf,
    widen_interval,
)
from engine.thresholds import TABLE_COLUMNS

Z = np.array([-1.6449, -1.2816, -0.6745, 0.0, 0.6745, 1.2816, 1.6449])
SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}
VERSION = "gbm-test+coimbra.B.production.0000000000"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@dataclass
class LinearModel:
    """P50 = 10 + 3 sin(doy) + sum(slope_f * x_f); quantiles P50 + 2 z_level."""

    slopes: dict[str, float] = field(
        default_factory=lambda: {
            "imperviousness_pct": 0.10,
            "riparian_width_m": -0.02,
            "precip_max_hourly_fut": 0.30,
            "first_flush_index_fut": 0.01,
        }
    )
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    version: str = VERSION
    variant: str = "B"
    levels: tuple[float, ...] = DEFAULT_LEVELS
    features: tuple[str, ...] = (
        "horizon",
        "doy_sin",
        "imperviousness_pct",
        "riparian_width_m",
        "precip_max_hourly",
        "precip_max_hourly_fut",
        "antecedent_dry_days",
        "first_flush_index",
        "first_flush_index_fut",
        "catchment_area_km2",
    )
    calls: int = 0

    def feature_range(self, variable: str, feature: str) -> tuple[float, float] | None:
        return self.ranges.get(feature)

    def predict_quantiles(self, features: pd.DataFrame, variable: str) -> np.ndarray:
        self.calls += 1
        mu = 10.0 + 3.0 * features["doy_sin"].to_numpy(dtype="float64")
        for f, b in self.slopes.items():
            mu = mu + b * np.nan_to_num(features[f].to_numpy(dtype="float64"))
        return mu[:, None] + 2.0 * Z[None, :]


def make_frame(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    statics = {
        "R1": {"imperviousness_pct": 20.0, "riparian_width_m": 5.0, "catchment_area_km2": 2.0},
        "R2": {"imperviousness_pct": 10.0, "riparian_width_m": 50.0, "catchment_area_km2": 4.0},
        "R3": {"imperviousness_pct": 30.0, "riparian_width_m": np.nan, "catchment_area_km2": 1.0},
    }
    parts = []
    for rid, st in statics.items():
        pmh = rng.gamma(0.6, 3.0, len(days))
        pmh_f = rng.gamma(0.6, 3.0, len(days))
        add = rng.integers(0, 20, len(days)).astype(float)
        add_f = rng.integers(0, 20, len(days)).astype(float)
        parts.append(
            pd.DataFrame(
                {
                    "reach_id": rid,
                    "target_date": days.date,
                    "horizon": 1.0,
                    "doy_sin": np.sin(2 * np.pi * days.dayofyear / 365.25),
                    "precip_max_hourly": pmh,
                    "precip_max_hourly_fut": pmh_f,
                    "antecedent_dry_days": add,
                    "first_flush_index": add * pmh,
                    "first_flush_index_fut": add_f * pmh_f,
                    **st,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def make_thresholds(reaches: tuple[str, ...] = ("R1", "R2"), value: float = 13.0) -> pd.DataFrame:
    rows = [
        {
            "reach_id": r,
            "variable": v,
            "season": s,
            "threshold": value,
            "n_obs": 20,
            "clim_exceed_freq": 0.1,
            "percentile": 0.9,
            "fit_end": date(2025, 12, 31),
        }
        for r in reaches
        for v in ("turbidity_proxy", "ndci")
        for s in SEASONS
    ]
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def check(
    feature: str, variable: str, verdict: Verdict, *, version: str = VERSION, sign: int = 1
) -> ResponseCheck:
    effect = {Verdict.TRUSTED: 0.2, Verdict.WRONG_SIGN: -0.2, Verdict.NEGLIGIBLE: 0.001}[verdict]
    return ResponseCheck(
        feature=feature,
        variable=variable,
        checked_model_version="gbm-test+coimbra.B.wf-test.0",
        serving_model_version=version,
        fold="test",
        rows=100,
        reaches=10,
        feature_sd=5.0,
        target_sd=1.0,
        mean_p50_change=effect * sign,
        standardised_effect=effect * sign,
        observed_sign=1 if effect * sign > 0 else -1,
        expected_sign=sign,  # type: ignore[arg-type]
        mean_exceedance_prob_change=None,
        exceedance_rows=0,
        share_reaches_up=0.5,
        share_reaches_down=0.5,
        out_of_support_share=0.0,
        negligible_effect_sd=0.02,
        verdict=verdict,
    )


def checks(verdict: Verdict = Verdict.TRUSTED) -> dict[tuple[str, str], ResponseCheck]:
    table = load_intervention_coefficients()
    return {
        (f, v): check(f, v, verdict, sign=expected_response_sign(table, f))
        for f in ("imperviousness_pct", "riparian_width_m")
        for v in ("turbidity_proxy", "ndci")
    }


@pytest.fixture(scope="module")
def table() -> CoefficientTable:
    return load_intervention_coefficients()


@pytest.fixture(scope="module")
def settings(table: CoefficientTable) -> ScenarioSettings:
    return ScenarioSettings(
        reference_start=date(2025, 1, 1),
        reference_end=date(2025, 12, 31),
        horizon=1,
        ci_level=0.8,
        coefficient_samples=5,
        widening_factor=1.5,
        caveat=str(table.uncertainty_inflation["caveat"]),
        seasons=SEASONS,
    )


def raw_table() -> dict[str, Any]:
    path = CONFIG_DIR / "intervention_coefficients.yaml"
    cfg: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return cfg


def table_with_direct_effect() -> CoefficientTable:
    """The real table plus a TEST-ONLY cited direct effect on permeable paving."""
    raw = copy.deepcopy(raw_table())
    for i in raw["interventions"]:
        if i["id"] == "permeable_paving":
            i["direct_effect"] = {
                "targets": ["turbidity_proxy"],
                "operation": "scale_quantiles",
                "unit": "fractional reduction of the turbidity index at full coverage",
                "description": "test fixture only",
                "magnitude": 0.4,
                "uncertainty_range": [0.2, 0.6],
                "citation": [
                    {
                        "authors": "Test, A.",
                        "year": 2020,
                        "title": "Test fixture",
                        "source": "tests/test_scenarios.py",
                        "doi": "10.0000/test",
                        "locator": "fixture",
                        "supports": [
                            "direct_magnitude",
                            "direct_uncertainty_low",
                            "direct_uncertainty_high",
                        ],
                    }
                ],
            }
    return validate_coefficients(raw)


def run(
    interventions: list[InterventionRequest],
    *,
    table: CoefficientTable,
    settings: ScenarioSettings,
    model: LinearModel | None = None,
    frame: pd.DataFrame | None = None,
    thresholds: pd.DataFrame | None = None,
    response_checks: dict[tuple[str, str], ResponseCheck] | None = None,
    variables: tuple[str, ...] = ("turbidity_proxy",),
) -> ScenarioResult:
    rids = tuple(dict.fromkeys(r for i in interventions for r in i.reach_ids))
    return run_scenario(
        rids,
        interventions,
        model or LinearModel(),
        make_frame() if frame is None else frame,
        coefficients=table,
        thresholds=make_thresholds() if thresholds is None else thresholds,
        response_checks=checks() if response_checks is None else response_checks,
        settings=settings,
        variables=variables,
    )


def reach(result: ScenarioResult, rid: str, variable: str = "turbidity_proxy") -> Any:
    return next(r for r in result.reaches if r.reach_id == rid and r.variable == variable)


PAVING = InterventionRequest(type="permeable_paving", reach_ids=("R1", "R2"), extent=5.0)
DETENTION = InterventionRequest(type="detention_basin", reach_ids=("R1",), extent=1.0)
SWEEPING = InterventionRequest(type="street_sweeping", reach_ids=("R1", "R2"), extent=1.0)
BUFFER = InterventionRequest(type="riparian_buffer_restoration", reach_ids=("R1", "R2"))


# ---------------------------------------------------------------------------
# apply_intervention
# ---------------------------------------------------------------------------
def test_subtract_treated_share_uses_the_table_magnitude(table: CoefficientTable) -> None:
    f = make_frame()
    out = apply_intervention(f, PAVING, table)
    m = table.get("permeable_paving").magnitude
    for rid, x in (("R1", 20.0), ("R2", 10.0)):
        got = out.loc[out.reach_id == rid, "imperviousness_pct"].unique()
        assert got == pytest.approx([x - m * 5.0])
    # a reach not named is untouched; the input is not mutated
    assert (out.loc[out.reach_id == "R3", "imperviousness_pct"] == 30.0).all()
    assert (f.loc[f.reach_id == "R1", "imperviousness_pct"] == 20.0).all()


def test_magnitude_outside_the_cited_range_is_refused(table: CoefficientTable) -> None:
    lo, hi = table.get("permeable_paving").uncertainty_range
    apply_intervention(make_frame(), PAVING, table, magnitude=lo)
    apply_intervention(make_frame(), PAVING, table, magnitude=hi)
    for bad in (lo - 0.01, hi + 0.01):
        with pytest.raises(ValueError, match="outside the cited uncertainty_range"):
            apply_intervention(make_frame(), PAVING, table, magnitude=bad)


def test_floor_keeps_null_null_and_only_raises(table: CoefficientTable) -> None:
    f = make_frame()
    lever = InterventionRequest(type="riparian_buffer_restoration", reach_ids=("R1", "R2", "R3"))
    out = apply_intervention(f, lever, table)
    assert (out.loc[out.reach_id == "R1", "riparian_width_m"] == 10.0).all()  # 5 -> 10
    assert (out.loc[out.reach_id == "R2", "riparian_width_m"] == 50.0).all()  # already wider
    assert out.loc[out.reach_id == "R3", "riparian_width_m"].isna().all()  # NULL stays NULL


def test_scale_down_moves_the_driver_its_target_day_copy_and_the_derived_index(
    table: CoefficientTable,
) -> None:
    f = make_frame()
    lo, hi = table.get("detention_basin").uncertainty_range
    out = apply_intervention(f, DETENTION, table, magnitude=hi)
    r1 = (f.reach_id == "R1").to_numpy()
    factor = 1 - hi * 1.0
    for c in (
        "precip_max_hourly",
        "precip_max_hourly_fut",
        "first_flush_index",
        "first_flush_index_fut",
    ):
        np.testing.assert_allclose(out.loc[r1, c], f.loc[r1, c] * factor)
        np.testing.assert_array_equal(out.loc[~r1, c], f.loc[~r1, c])
    # derived index still equals its definition after the change
    np.testing.assert_allclose(
        out.loc[r1, "first_flush_index"],
        out.loc[r1, "antecedent_dry_days"] * out.loc[r1, "precip_max_hourly"],
    )


def test_extent_is_validated_per_operation(table: CoefficientTable) -> None:
    f = make_frame()
    with pytest.raises(ValueError, match="extent is required"):
        apply_intervention(
            f, InterventionRequest(type="permeable_paving", reach_ids=("R1",)), table
        )
    with pytest.raises(ValueError, match=r"in \(0, 100\]"):
        apply_intervention(
            f, InterventionRequest(type="permeable_paving", reach_ids=("R1",), extent=150), table
        )
    with pytest.raises(ValueError, match=r"in \(0, 1\]"):
        apply_intervention(
            f, InterventionRequest(type="street_sweeping", reach_ids=("R1",), extent=2.0), table
        )
    with pytest.raises(ValueError, match="not used by a floor"):
        apply_intervention(
            f,
            InterventionRequest(type="riparian_buffer_restoration", reach_ids=("R1",), extent=3.0),
            table,
        )


def test_uncited_lever_cannot_be_applied(table: CoefficientTable) -> None:
    with pytest.raises(ValueError, match="no cited coefficient"):
        apply_intervention(
            make_frame(), InterventionRequest(type="daylighting", reach_ids=("R1",)), table
        )
    with pytest.raises(ValueError, match="unknown intervention"):
        apply_intervention(
            make_frame(), InterventionRequest(type="magic", reach_ids=("R1",), extent=1), table
        )


def test_extent_larger_than_what_exists_is_not_applied(table: CoefficientTable) -> None:
    lever = InterventionRequest(type="permeable_paving", reach_ids=("R1", "R2"), extent=15.0)
    out = apply_intervention(make_frame(), lever, table)
    assert (out.loc[out.reach_id == "R2", "imperviousness_pct"] == 10.0).all()  # 15 > 10
    assert (out.loc[out.reach_id == "R1", "imperviousness_pct"] < 20.0).all()


# ---------------------------------------------------------------------------
# effect sizes come from the table, never from the model
# ---------------------------------------------------------------------------
def test_result_effect_sizes_are_the_table_values(
    table: CoefficientTable, settings: ScenarioSettings
) -> None:
    res = run([PAVING, SWEEPING], table=table, settings=settings)
    for lv in res.levers:
        coef = table.get(lv.intervention_id)
        assert lv.magnitude == coef.magnitude
        assert lv.uncertainty_range == coef.uncertainty_range


def test_changing_the_table_changes_the_result_and_the_model_does_not(
    settings: ScenarioSettings,
) -> None:
    """Halving the cited magnitude halves the feature change the model sees. The model is
    only ever asked for quantiles of feature rows - it has no way to supply an effect."""
    raw = copy.deepcopy(raw_table())
    for i in raw["interventions"]:
        if i["id"] == "permeable_paving":
            i["magnitude"] = 0.485
    halved = validate_coefficients(raw)
    full = run([PAVING], table=load_intervention_coefficients(), settings=settings)
    half = run([PAVING], table=halved, settings=settings)
    r_full, r_half = reach(full, "R1"), reach(half, "R1")
    assert r_full.levers[0].value_after == pytest.approx(20 - 0.97 * 5)
    assert r_half.levers[0].value_after == pytest.approx(20 - 0.485 * 5)
    assert r_full.delta_days < r_half.delta_days < 0


def test_model_is_asked_only_for_quantiles(
    table: CoefficientTable, settings: ScenarioSettings
) -> None:
    class NoEffectOracle(LinearModel):
        def __getattr__(self, name: str) -> Any:  # any other question fails the test
            raise AssertionError(f"engine asked the model for {name}")

    m = NoEffectOracle()
    run([PAVING], table=table, settings=settings, model=m)
    # baseline + central + one per coefficient sample
    assert m.calls == 1 + 1 + settings.coefficient_samples


# ---------------------------------------------------------------------------
# intervals: strictly wider, coefficient uncertainty propagated
# ---------------------------------------------------------------------------
LEVER_SETS = [
    [PAVING],
    [SWEEPING],
    [BUFFER],
    [DETENTION, InterventionRequest(type="permeable_paving", reach_ids=("R2",), extent=2.0)],
    [PAVING, SWEEPING],
    [PAVING, BUFFER, InterventionRequest(type="green_roofs", reach_ids=("R1", "R2"), extent=3.0)],
]


@pytest.mark.parametrize("levers", LEVER_SETS, ids=lambda ls: "+".join(i.type for i in ls))
@pytest.mark.parametrize("threshold", [11.0, 13.0, 16.0])
def test_every_scenario_interval_is_strictly_wider_than_baseline(
    levers: list[InterventionRequest],
    threshold: float,
    table: CoefficientTable,
    settings: ScenarioSettings,
) -> None:
    res = run(
        levers,
        table=table,
        settings=settings,
        thresholds=make_thresholds(value=threshold),
        variables=("turbidity_proxy", "ndci"),
    )
    ok = [r for r in res.reaches if r.status is Status.OK]
    assert ok, "fixture produced no OK result to check"
    for r in ok:
        assert r.scenario.interval.width > r.baseline.interval.width, (r.reach_id, r.variable)
        assert r.scenario.interval.low <= r.scenario.exceedance_days <= r.scenario.interval.high
        assert r.scenario.interval.low >= 0 and r.scenario.interval.high <= r.days_evaluated


def test_scenario_interval_widening_holds_even_when_the_lever_does_nothing(
    table: CoefficientTable, settings: ScenarioSettings
) -> None:
    """A lever with no effect on a reach still widens: the baseline's own uncertainty does
    not go away, and the widening factor applies."""
    res = run([BUFFER], table=table, settings=settings)
    r2 = reach(res, "R2")
    assert ReachFlag.NO_CHANGE in r2.levers[0].flags
    assert r2.delta_days == 0.0
    assert r2.scenario.interval.width == pytest.approx(1.5 * r2.baseline.interval.width)


def test_coefficient_uncertainty_reaches_the_result(settings: ScenarioSettings) -> None:
    """Same lever, cited range collapsed to its magnitude vs the real range: the real range
    produces a spread in delta and an interval at least as wide."""
    raw = copy.deepcopy(raw_table())
    for i in raw["interventions"]:
        if i["id"] == "permeable_paving":
            i["uncertainty_range"] = [i["magnitude"], i["magnitude"]]
    point = validate_coefficients(raw)
    real = load_intervention_coefficients()
    lever = [InterventionRequest(type="permeable_paving", reach_ids=("R1",), extent=15.0)]
    a = reach(run(lever, table=point, settings=settings), "R1")
    b = reach(run(lever, table=real, settings=settings), "R1")
    assert a.delta_coefficient_range.width == pytest.approx(0.0, abs=1e-9)
    assert b.delta_coefficient_range.width > 1.0
    assert b.scenario.interval.width >= a.scenario.interval.width
    # the cited central value, not the sample mean, is the central estimate
    assert a.scenario.exceedance_days == pytest.approx(b.scenario.exceedance_days)


def test_zero_central_effect_stays_zero(
    table: CoefficientTable, settings: ScenarioSettings
) -> None:
    """Street sweeping's cited central water-quality effect is 0 (Selbig 2007). The range
    reaches 0.76, so the interval moves - the central estimate must not."""
    r = reach(run([SWEEPING], table=table, settings=settings), "R1")
    assert r.delta_days == 0.0
    assert r.delta_coefficient_range.low < 0.0


def test_widening_factor_must_exceed_one(table: CoefficientTable) -> None:
    kw: dict[str, Any] = {
        "reference_start": date(2025, 1, 1),
        "reference_end": date(2025, 12, 31),
        "horizon": 1,
        "ci_level": 0.8,
        "coefficient_samples": 3,
        "caveat": "x",
        "seasons": SEASONS,
    }
    for bad in (1.0, 0.9, None):
        with pytest.raises(ValueError, match="widening_factor"):
            ScenarioSettings(widening_factor=bad, **kw)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="whole calendar year"):
        ScenarioSettings(widening_factor=1.5, **{**kw, "reference_end": date(2025, 6, 30)})


def test_real_config_builds_settings(table: CoefficientTable) -> None:
    s = ScenarioSettings.from_config(load_config("modelling"), table, load_config("thresholds"))
    assert s.widening_factor > 1
    assert s.period_days in (365, 366)
    assert "predict" not in s.caveat.lower()


# ---------------------------------------------------------------------------
# citations and language
# ---------------------------------------------------------------------------
def test_every_result_carries_every_coefficient_citation(
    table: CoefficientTable, settings: ScenarioSettings
) -> None:
    levers = [PAVING, SWEEPING, BUFFER, DETENTION]
    res = run(levers, table=table, settings=settings)
    got = {(c.intervention_id, c.role, c.title) for c in res.citations}
    for lv in levers:
        coef = table.get(lv.type)
        for ref in coef.citation:
            assert (coef.id, "effect", " ".join(ref.title.split())) in got
        for ref in coef.cost_per_unit.citation:
            assert (coef.id, "cost", " ".join(ref.title.split())) in got
    for summary in res.levers:
        assert summary.citations, summary.intervention_id


def test_a_result_without_citations_cannot_exist(
    table: CoefficientTable, settings: ScenarioSettings
) -> None:
    res = run([PAVING], table=table, settings=settings)
    with pytest.raises(ValueError):
        ScenarioResult.model_validate({**res.model_dump(), "citations": ()})


def test_output_says_planning_estimate_never_prediction(
    table: CoefficientTable, settings: ScenarioSettings
) -> None:
    res = run([PAVING, SWEEPING, BUFFER, DETENTION], table=table, settings=settings)
    assert res.label == "planning estimate"
    assert res.caveat.startswith("Planning estimate")
    text = json.dumps(res.model_dump(mode="json")).lower()
    assert "predict" not in text


# ---------------------------------------------------------------------------
# MASTERSPEC 9.6
# ---------------------------------------------------------------------------
def test_variant_a_is_refused(table: CoefficientTable, settings: ScenarioSettings) -> None:
    with pytest.raises(ValueError, match="variant B only"):
        run([PAVING], table=table, settings=settings, model=LinearModel(variant="A"))


def test_wrong_sign_without_direct_effect_is_not_estimable(
    table: CoefficientTable, settings: ScenarioSettings
) -> None:
    res = run([PAVING], table=table, settings=settings, response_checks=checks(Verdict.WRONG_SIGN))
    (lv,) = res.levers
    assert lv.path is LeverPath.NOT_ESTIMABLE
    assert "WRONG_SIGN" in lv.reason and "no cited direct effect" in lv.reason
    r1 = reach(res, "R1")
    assert r1.status is Status.NOT_ESTIMABLE
    assert r1.scenario is None and r1.delta_days is None
    assert r1.baseline is not None  # the baseline is still shown


def test_negligible_response_is_not_trusted(
    table: CoefficientTable, settings: ScenarioSettings
) -> None:
    res = run([PAVING], table=table, settings=settings, response_checks=checks(Verdict.NEGLIGIBLE))
    assert res.levers[0].path is LeverPath.NOT_ESTIMABLE
    assert "NEGLIGIBLE" in res.levers[0].reason


def test_untrusted_lever_with_cited_direct_effect_goes_literature_direct(
    settings: ScenarioSettings,
) -> None:
    t = table_with_direct_effect()
    model = LinearModel()
    res = run(
        [PAVING],
        table=t,
        settings=settings,
        model=model,
        response_checks=checks(Verdict.WRONG_SIGN),
    )
    (lv,) = res.levers
    assert lv.path is LeverPath.LITERATURE_DIRECT
    assert lv.direct_magnitude == 0.4
    assert any(c.role == "direct_effect" for c in lv.citations)
    assert any(c.role == "direct_effect" for c in res.citations)
    r1 = reach(res, "R1")
    assert r1.status is Status.OK and r1.delta_days < 0
    assert r1.levers[0].path is LeverPath.LITERATURE_DIRECT
    # the model is not re-run on perturbed features: only the baseline inference
    assert model.calls == 1


def test_direct_effect_cannot_scale_a_signed_target() -> None:
    raw = copy.deepcopy(raw_table())
    t = table_with_direct_effect().model_dump(mode="json")
    de = next(i for i in t["interventions"] if i["id"] == "permeable_paving")["direct_effect"]
    de["targets"] = ["ndci"]
    for i in raw["interventions"]:
        if i["id"] == "permeable_paving":
            i["direct_effect"] = de
    with pytest.raises(ValueError, match="cannot scale"):
        validate_coefficients(raw)


def test_trusted_static_lever_goes_through_the_model(
    table: CoefficientTable, settings: ScenarioSettings
) -> None:
    res = run([PAVING], table=table, settings=settings)
    assert res.levers[0].path is LeverPath.MODEL_PERTURBATION
    assert res.levers[0].response_check is not None


def test_driver_lever_goes_through_the_model_without_a_check(
    table: CoefficientTable, settings: ScenarioSettings
) -> None:
    res = run([DETENTION], table=table, settings=settings, response_checks={})
    (lv,) = res.levers
    assert lv.path is LeverPath.MODEL_PERTURBATION and lv.response_check is None
    assert "weather driver" in lv.reason


def test_missing_or_stale_response_check_fails_loudly(
    table: CoefficientTable, settings: ScenarioSettings
) -> None:
    with pytest.raises(MissingResponseCheck, match="make response-check"):
        run([PAVING], table=table, settings=settings, response_checks={})
    stale = {k: v.model_copy(update={"serving_model_version": "old"}) for k, v in checks().items()}
    with pytest.raises(StaleResponseCheck):
        run([PAVING], table=table, settings=settings, response_checks=stale)


def test_out_of_support_flag_names_the_feature_and_range(
    table: CoefficientTable, settings: ScenarioSettings
) -> None:
    model = LinearModel(ranges={"imperviousness_pct": (16.0, 60.0)})
    res = run(
        [InterventionRequest(type="permeable_paving", reach_ids=("R1",), extent=5.0)],
        table=table,
        settings=settings,
        model=model,
    )
    lv = reach(res, "R1").levers[0]
    assert ReachFlag.OUT_OF_SUPPORT in lv.flags  # 20 - 0.97 * 5 = 15.15 < 16
    assert any("imperviousness_pct" in d and "[16, 60]" in d for d in lv.detail)


def test_in_support_perturbation_is_not_flagged(
    table: CoefficientTable, settings: ScenarioSettings
) -> None:
    model = LinearModel(ranges={"imperviousness_pct": (0.0, 60.0)})
    res = run([PAVING], table=table, settings=settings, model=model)
    for rid in ("R1", "R2"):
        assert ReachFlag.OUT_OF_SUPPORT not in reach(res, rid).levers[0].flags


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------
def test_reach_without_threshold_is_insufficient_evidence_not_zero(
    table: CoefficientTable, settings: ScenarioSettings
) -> None:
    lever = InterventionRequest(type="permeable_paving", reach_ids=("R1", "R3"), extent=5.0)
    res = run([lever], table=table, settings=settings)
    r3 = reach(res, "R3")
    assert r3.status is Status.INSUFFICIENT_EVIDENCE
    assert r3.reason.startswith("NO_THRESHOLD")
    assert r3.baseline is None and r3.scenario is None and r3.delta_days is None


def test_missing_season_threshold_names_the_season(
    table: CoefficientTable, settings: ScenarioSettings
) -> None:
    thr = make_thresholds()
    thr = thr[~((thr.reach_id == "R1") & (thr.season == "JJA"))]
    r1 = reach(run([PAVING], table=table, settings=settings, thresholds=thr), "R1")
    assert r1.status is Status.INSUFFICIENT_EVIDENCE and "JJA" in r1.reason


def test_missing_days_are_insufficient_evidence(
    table: CoefficientTable, settings: ScenarioSettings
) -> None:
    f = make_frame()
    f = f[~((f.reach_id == "R1") & (pd.to_datetime(f.target_date).dt.month == 3))]
    r1 = reach(run([PAVING], table=table, settings=settings, frame=f), "R1")
    assert r1.status is Status.INSUFFICIENT_EVIDENCE and r1.reason.startswith("MISSING_DAYS")


def test_null_feature_is_not_applicable(
    table: CoefficientTable, settings: ScenarioSettings
) -> None:
    lever = InterventionRequest(type="riparian_buffer_restoration", reach_ids=("R1", "R3"))
    thr = make_thresholds(("R1", "R3"))
    r3 = reach(run([lever], table=table, settings=settings, thresholds=thr), "R3")
    assert r3.levers[0].flags == (ReachFlag.NOT_APPLICABLE,)
    assert r3.status is Status.NOT_ESTIMABLE and "NULL" in r3.reason


def test_combined_extent_beyond_what_exists_is_not_applicable(
    table: CoefficientTable, settings: ScenarioSettings
) -> None:
    levers = [
        InterventionRequest(type="permeable_paving", reach_ids=("R2",), extent=6.0),
        InterventionRequest(type="green_roofs", reach_ids=("R2",), extent=6.0),
    ]
    r2 = reach(run(levers, table=table, settings=settings), "R2")
    assert all(ReachFlag.NOT_APPLICABLE in lv.flags for lv in r2.levers)
    assert "combined extent 12" in r2.levers[0].detail[0]


def test_request_consistency_is_checked(
    table: CoefficientTable, settings: ScenarioSettings
) -> None:
    with pytest.raises(ValueError, match="twice"):
        run([PAVING, PAVING], table=table, settings=settings)
    with pytest.raises(ValueError, match="no intervention"):
        run_scenario(
            ("R1", "R2"),
            [InterventionRequest(type="permeable_paving", reach_ids=("R1",), extent=1.0)],
            LinearModel(),
            make_frame(),
            coefficients=table,
            thresholds=make_thresholds(),
            response_checks=checks(),
            settings=settings,
        )
    with pytest.raises(ValueError, match="no feature rows"):
        run(
            [InterventionRequest(type="permeable_paving", reach_ids=("R9",), extent=1.0)],
            table=table,
            settings=settings,
        )


# ---------------------------------------------------------------------------
# costs
# ---------------------------------------------------------------------------
def test_costs_keep_source_currency_and_derive_quantity_only_where_defined(
    table: CoefficientTable, settings: ScenarioSettings
) -> None:
    res = run([PAVING, SWEEPING, DETENTION], table=table, settings=settings)
    paving = next(
        c for c in res.costs if c.intervention_id == "permeable_paving" and c.reach_id == "R1"
    )
    assert paving.currency == "EUR" and paving.quantity_unit == "square metre"
    assert paving.quantity == pytest.approx(5.0 / 100 * 2.0 * 1e6)
    assert paving.total_low == pytest.approx(paving.quantity * 40)
    sweep = next(c for c in res.costs if c.intervention_id == "street_sweeping")
    assert sweep.currency == "USD"  # never converted
    assert sweep.quantity is None and sweep.total_low is None and sweep.reason
    det = next(c for c in res.costs if c.intervention_id == "detention_basin")
    assert det.quantity is None and det.reason


# ---------------------------------------------------------------------------
# response check
# ---------------------------------------------------------------------------
def held_out() -> pd.DataFrame:
    return make_frame(seed=3)


@pytest.mark.parametrize(
    ("slope", "verdict"),
    [(0.10, Verdict.TRUSTED), (-0.10, Verdict.WRONG_SIGN), (0.0, Verdict.NEGLIGIBLE)],
)
def test_response_check_verdicts(slope: float, verdict: Verdict) -> None:
    model = LinearModel(slopes={"imperviousness_pct": slope})
    c = response_check(
        model,
        held_out(),
        feature="imperviousness_pct",
        variable="turbidity_proxy",
        feature_sd=8.0,
        target_sd=2.0,
        expected_sign=1,
        negligible_effect_sd=0.02,
        fold="test",
        serving_model_version=VERSION,
        thresholds=np.full(len(held_out()), 13.0),
    )
    assert c.verdict is verdict
    assert c.mean_p50_change == pytest.approx(slope * 16.0)  # +-1 SD = 2 SD swing
    assert c.standardised_effect == pytest.approx(slope * 16.0 / 2.0)
    assert (c.mean_exceedance_prob_change or 0) * np.sign(slope) >= 0
    assert c.checked_model_version == model.version


def test_response_check_refuses_drivers_and_variant_a() -> None:
    kw: dict[str, Any] = {
        "variable": "turbidity_proxy",
        "feature_sd": 1.0,
        "target_sd": 1.0,
        "expected_sign": 1,
        "negligible_effect_sd": 0.02,
        "fold": "test",
        "serving_model_version": VERSION,
    }
    with pytest.raises(ValueError, match="static attributes"):
        response_check(LinearModel(), held_out(), feature="precip_max_hourly", **kw)
    with pytest.raises(ValueError, match="variant B"):
        response_check(LinearModel(variant="A"), held_out(), feature="imperviousness_pct", **kw)


def test_judge_response() -> None:
    assert judge_response(0.5, 1, 1, 0.02) is Verdict.TRUSTED
    assert judge_response(-0.5, -1, 1, 0.02) is Verdict.WRONG_SIGN
    assert judge_response(0.01, 1, 1, 0.02) is Verdict.NEGLIGIBLE
    assert judge_response(-0.01, -1, 1, 0.02) is Verdict.NEGLIGIBLE
    assert judge_response(math.nan, 0, 1, 0.02) is Verdict.NEGLIGIBLE


def test_expected_signs_follow_the_table(table: CoefficientTable) -> None:
    assert expected_response_sign(table, "imperviousness_pct") == 1  # less cover -> less
    assert expected_response_sign(table, "riparian_width_m") == -1  # wider buffer -> less


# ---------------------------------------------------------------------------
# numerics
# ---------------------------------------------------------------------------
def test_poisson_binomial_matches_brute_force() -> None:
    p = np.array([0.1, 0.5, 0.9, 0.3, 0.0, 1.0])
    brute = np.zeros(len(p) + 1)
    for outcome in itertools.product([0, 1], repeat=len(p)):
        prob = np.prod([pi if o else 1 - pi for pi, o in zip(p, outcome, strict=True)])
        brute[sum(outcome)] += prob
    np.testing.assert_allclose(poisson_binomial_pmf(p), brute, atol=1e-12)
    two = poisson_binomial_pmf(np.stack([p, p[::-1]]))
    np.testing.assert_allclose(two[0], two[1], atol=1e-12)
    with pytest.raises(ValueError, match="NaN"):
        poisson_binomial_pmf([0.2, np.nan])


def test_pmf_interval() -> None:
    pmf = np.array([0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05])
    assert pmf_interval(pmf, 0.8) == (1, 5)
    assert pmf_interval(np.array([1.0, 0, 0]), 0.8) == (0, 0)


def test_triangular_ppf() -> None:
    assert triangular_ppf(0.5, 10, 10, 10) == 10
    xs = [triangular_ppf(u, 0.0, 0.0, 0.76) for u in coefficient_levels(9)]
    assert all(0 <= x <= 0.76 for x in xs) and xs == sorted(xs)
    assert triangular_ppf(0.5, 0, 1, 2) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        triangular_ppf(0.0, 0, 1, 2)


@pytest.mark.parametrize(
    ("central", "raw", "base", "upper"),
    [
        (20.0, (15.0, 25.0), (10.0, 30.0), 365.0),  # raw narrower than baseline
        (20.0, (5.0, 40.0), (15.0, 25.0), 365.0),  # raw wider
        (1.0, (0.0, 3.0), (2.0, 12.0), 365.0),  # clipped at zero
        (363.0, (358.0, 365.0), (350.0, 364.0), 365.0),  # clipped at the top
        (5.0, (5.0, 5.0), (3.0, 8.0), 365.0),  # zero-width raw
        (100.0, (0.0, 365.0), (50.0, 300.0), 365.0),  # capped at the period
    ],
)
def test_widen_interval_is_strictly_wider_and_contains_raw(
    central: float, raw: tuple[float, float], base: tuple[float, float], upper: float
) -> None:
    b = Estimate(exceedance_days=sum(base) / 2, interval=Interval(low=base[0], high=base[1]))
    r = Interval(low=raw[0], high=raw[1])
    w = widen_interval(central, r, b, 1.5, upper)
    assert w.width > b.interval.width
    assert 0.0 <= w.low <= min(r.low, central) and max(r.high, central) <= w.high <= upper


def test_degenerate_baseline_cannot_be_widened_and_is_not_estimable(
    table: CoefficientTable, settings: ScenarioSettings
) -> None:
    """A threshold the forecast never reaches: P(exceed) = 0 every day, baseline [0, 0].
    No scenario interval can be 'strictly wider' in a meaningful sense - refuse, say why."""
    r1 = reach(
        run([PAVING], table=table, settings=settings, thresholds=make_thresholds(value=500.0)),
        "R1",
    )
    assert r1.status is Status.NOT_ESTIMABLE
    assert r1.reason.startswith("BASELINE_INTERVAL_DEGENERATE")
    assert r1.baseline.interval.width == 0 and r1.scenario is None
