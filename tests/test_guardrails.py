"""Guardrail tests (CLAUDE.md Testing #1 - the Razorpay pattern).

For every guardrail in engine.alerts.GUARDRAILS there is a case that violates it, and
two tests run on that case:

  test_guardrail_fires[name]            the intact engine stops / downgrades / caps it
  test_breaking_guardrail_is_caught[name]
                                        the guardrail is deliberately disabled (replaced
                                        with a pass-through) and the SAME expectation is
                                        asserted to FAIL - i.e. the suite would catch
                                        anyone removing or neutering that guardrail

test_every_guardrail_has_a_break_case makes it impossible to add a guardrail without one.
Boundary tests pin the configured numbers (0.6, 72 h, 21 days) exactly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import pytest

from core.config import load_config
from engine import alerts
from engine.alerts import (
    GUARDRAILS,
    Alert,
    ForecastDistribution,
    GuardrailConfig,
    HistoryEntry,
    ReachState,
    Residual,
    Severity,
    Suppressed,
    apply_guardrails,
    build_attribution,
    compose_alert,
    compute_exceedance_probability,
    to_record,
)

ISSUED = datetime(2025, 9, 23, 6, 0)
CFG = GuardrailConfig.from_config(load_config("thresholds"))
THRESHOLD = 10.0


def forecast(center: float, target: date, spread: float = 1.0) -> ForecastDistribution:
    """Seven increasing quantiles around `center`."""
    offsets = (-2.0, -1.5, -0.7, 0.0, 0.7, 1.5, 2.0)
    return ForecastDistribution(
        quantiles=tuple(center + o * spread for o in offsets), target_date=target
    )


def healthy_state(**overrides: Any) -> ReachState:
    base: dict[str, Any] = {
        "reach_id": "CMB-0041",
        "observable": True,
        "last_usable_observation": ISSUED.date() - timedelta(days=4),
        "usable_observations_30d": 6,
        "recent_residuals": tuple(
            Residual(ISSUED.date() - timedelta(days=d), 5.0 + 0.1 * (d % 3), 4.0, 5.0, 6.0)
            for d in (5, 10, 15, 20, 25)
        ),
    }
    base.update(overrides)
    return ReachState(**base)


def candidate(prob: float = 0.9) -> Alert:
    return Alert(
        reach_id="CMB-0041",
        variable="turbidity_proxy",
        issued_at=ISSUED,
        severity=Severity.ALERT,
        target_window=(date(2025, 9, 24), date(2025, 9, 26)),
        exceedance_prob=prob,
        threshold_value=THRESHOLD,
        threshold_derivation="per_reach_seasonal_percentile",
    )


# ---------------------------------------------------------------------------
# the healthy path - without this, every "fires" test could pass vacuously
# ---------------------------------------------------------------------------
def test_healthy_candidate_is_issued_as_alert_with_every_guardrail_passing() -> None:
    out = apply_guardrails(candidate(), healthy_state(), [], CFG)
    assert isinstance(out, Alert)
    assert out.severity is Severity.ALERT
    assert out.guardrails == {name: "pass" for name in GUARDRAILS}


# ---------------------------------------------------------------------------
# one violating case per guardrail
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Case:
    candidate: Alert
    state: ReachState
    history: tuple[HistoryEntry, ...]
    expect: Callable[[Alert | Suppressed], bool]


CASES: dict[str, Case] = {
    # minimum confidence 0.6
    "confidence": Case(
        candidate(prob=0.55),
        healthy_state(),
        (),
        lambda o: isinstance(o, Suppressed) and o.guardrail == "confidence",
    ),
    # 72 h cooldown per reach: an ALERT 30 h ago on the same reach (another variable)
    "cooldown": Case(
        candidate(),
        healthy_state(),
        (HistoryEntry("CMB-0041", ISSUED - timedelta(hours=30), Severity.ALERT, "ndci"),),
        lambda o: isinstance(o, Suppressed) and o.guardrail == "cooldown",
    ),
    # staleness: last usable observation 30 days old -> INSUFFICIENT_EVIDENCE
    "staleness": Case(
        candidate(prob=0.97),
        healthy_state(last_usable_observation=ISSUED.date() - timedelta(days=30)),
        (),
        lambda o: (
            isinstance(o, Alert)
            and o.severity is Severity.INSUFFICIENT_EVIDENCE
            and o.exceedance_prob is None
            and o.withheld_exceedance_prob == pytest.approx(0.97)
        ),
    ),
    # driver-only reach: capped at WATCH, never ALERT
    "observability": Case(
        candidate(prob=0.95),
        healthy_state(observable=False),
        (),
        lambda o: isinstance(o, Alert) and o.severity is Severity.WATCH,
    ),
    # drift: every recent observation far outside the forecast band
    "drift": Case(
        candidate(),
        healthy_state(
            recent_residuals=tuple(
                Residual(ISSUED.date() - timedelta(days=d), 15.0, 4.0, 5.0, 6.0)
                for d in (3, 8, 13, 18)
            )
        ),
        (),
        lambda o: isinstance(o, Suppressed) and o.guardrail == "drift",
    ),
}


def test_every_guardrail_has_a_break_case() -> None:
    assert set(CASES) == set(GUARDRAILS), (
        f"guardrails without a deliberate-break case: {sorted(set(GUARDRAILS) - set(CASES))}"
    )


@pytest.mark.parametrize("name", sorted(CASES))
def test_guardrail_fires(name: str) -> None:
    case = CASES[name]
    out = apply_guardrails(case.candidate, case.state, case.history, CFG)
    assert case.expect(out), f"guardrail {name!r} did not fire: {out}"
    assert out.guardrails[name] == "fired"


@pytest.mark.parametrize("name", sorted(CASES))
def test_breaking_guardrail_is_caught(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable exactly one guardrail and check that its case now gets through - proving
    test_guardrail_fires[name] would fail if someone broke it."""
    broken = dict(GUARDRAILS)
    broken[name] = lambda c, s, h, cfg: None  # the deliberate break
    monkeypatch.setattr(alerts, "GUARDRAILS", broken)
    case = CASES[name]
    out = apply_guardrails(case.candidate, case.state, case.history, CFG)
    assert not case.expect(out), f"disabling {name!r} went unnoticed by its test case"
    assert isinstance(out, Alert) and out.severity is Severity.ALERT, (
        f"with {name!r} disabled the case should sail through as an ALERT, got {out}"
    )


# ---------------------------------------------------------------------------
# the configured numbers, exactly
# ---------------------------------------------------------------------------
def test_config_values_are_the_masterspec_values() -> None:
    assert CFG.min_exceedance_prob == 0.60
    assert CFG.cooldown_hours == 72
    assert CFG.staleness_days == 21
    assert CFG.unobservable_max_severity is Severity.WATCH


def test_confidence_boundary_is_inclusive_at_0_6() -> None:
    assert isinstance(apply_guardrails(candidate(0.60), healthy_state(), [], CFG), Alert)
    out = apply_guardrails(candidate(0.5999), healthy_state(), [], CFG)
    assert isinstance(out, Suppressed) and out.guardrail == "confidence"


@pytest.mark.parametrize(
    ("hours", "suppressed"), [(1, True), (71.9, True), (72, False), (100, False)]
)
def test_cooldown_window_is_72_hours(hours: float, suppressed: bool) -> None:
    history = [HistoryEntry("CMB-0041", ISSUED - timedelta(hours=hours), Severity.WATCH)]
    out = apply_guardrails(candidate(), healthy_state(), history, CFG)
    assert isinstance(out, Suppressed) is suppressed


def test_cooldown_is_per_reach_and_ignores_insufficient_evidence() -> None:
    other_reach = [HistoryEntry("CMB-0099", ISSUED - timedelta(hours=2), Severity.ALERT)]
    refusal = [
        HistoryEntry("CMB-0041", ISSUED - timedelta(hours=2), Severity.INSUFFICIENT_EVIDENCE)
    ]
    assert isinstance(apply_guardrails(candidate(), healthy_state(), other_reach, CFG), Alert)
    assert isinstance(apply_guardrails(candidate(), healthy_state(), refusal, CFG), Alert)


@pytest.mark.parametrize(("age", "insufficient"), [(0, False), (21, False), (22, True)])
def test_staleness_boundary_is_21_days(age: int, insufficient: bool) -> None:
    state = healthy_state(last_usable_observation=ISSUED.date() - timedelta(days=age))
    out = apply_guardrails(candidate(), state, [], CFG)
    assert isinstance(out, Alert)
    assert (out.severity is Severity.INSUFFICIENT_EVIDENCE) is insufficient


def test_no_observation_ever_is_insufficient_evidence() -> None:
    out = apply_guardrails(candidate(), healthy_state(last_usable_observation=None), [], CFG)
    assert isinstance(out, Alert) and out.severity is Severity.INSUFFICIENT_EVIDENCE


def test_unassessed_observability_is_also_capped() -> None:
    out = apply_guardrails(candidate(), healthy_state(observable=None), [], CFG)
    assert isinstance(out, Alert) and out.severity is Severity.WATCH
    assert out.suppressed_reason and "never assessed" in out.suppressed_reason


def test_drift_needs_enough_residuals_to_fire() -> None:
    two_bad = tuple(
        Residual(ISSUED.date() - timedelta(days=d), 15.0, 4.0, 5.0, 6.0) for d in (3, 8)
    )
    out = apply_guardrails(candidate(), healthy_state(recent_residuals=two_bad), [], CFG)
    assert isinstance(out, Alert) and out.severity is Severity.ALERT


def test_residuals_outside_the_window_do_not_count_toward_drift() -> None:
    old_bad = tuple(
        Residual(ISSUED.date() - timedelta(days=d), 15.0, 4.0, 5.0, 6.0) for d in (70, 80, 90)
    )
    out = apply_guardrails(candidate(), healthy_state(recent_residuals=old_bad), [], CFG)
    assert isinstance(out, Alert) and out.severity is Severity.ALERT


# ---------------------------------------------------------------------------
# INSUFFICIENT_EVIDENCE is a first-class returned value
# ---------------------------------------------------------------------------
def _compose(**overrides: Any) -> Alert | Suppressed:
    targets = [date(2025, 9, 24), date(2025, 9, 25), date(2025, 9, 26)]
    kwargs: dict[str, Any] = {
        "reach_id": "CMB-0041",
        "variable": "turbidity_proxy",
        "issued_at": ISSUED,
        "forecasts": [forecast(12.0, d) for d in targets],
        "thresholds": [THRESHOLD] * 3,
        "threshold_derivation": "per_reach_seasonal_percentile",
        "reach_state": healthy_state(),
        "history": [],
        "config": CFG,
    }
    kwargs.update(overrides)
    return compose_alert(**kwargs)


def test_compose_healthy_is_alert_with_peak_probability() -> None:
    out = _compose()
    assert isinstance(out, Alert) and out.severity is Severity.ALERT
    assert out.exceedance_prob == pytest.approx(
        compute_exceedance_probability(forecast(12.0, date(2025, 9, 24)), THRESHOLD)
    )
    assert out.target_window == (date(2025, 9, 24), date(2025, 9, 26))


def test_missing_threshold_is_returned_insufficient_evidence_not_raised() -> None:
    out = _compose(thresholds=[None, None, None])
    assert isinstance(out, Alert)
    assert out.severity is Severity.INSUFFICIENT_EVIDENCE
    assert out.exceedance_prob is None
    assert out.suppressed_reason and out.suppressed_reason.startswith("NO_THRESHOLD")


def test_missing_forecast_is_returned_insufficient_evidence() -> None:
    nan = ForecastDistribution(quantiles=(float("nan"),) * 7, target_date=date(2025, 9, 24))
    out = _compose(forecasts=[nan], thresholds=[THRESHOLD])
    assert isinstance(out, Alert) and out.severity is Severity.INSUFFICIENT_EVIDENCE


def test_insufficient_evidence_is_never_coerced_into_a_weak_alert() -> None:
    """A near-certain exceedance on a stale reach is still a refusal, not a WATCH/ALERT
    with a discounted number."""
    out = _compose(
        forecasts=[forecast(40.0, date(2025, 9, 24))],
        thresholds=[THRESHOLD],
        reach_state=healthy_state(last_usable_observation=date(2025, 6, 1)),
    )
    assert isinstance(out, Alert) and out.severity is Severity.INSUFFICIENT_EVIDENCE
    assert out.exceedance_prob is None and out.withheld_exceedance_prob == pytest.approx(1.0)


def test_an_insufficient_evidence_alert_cannot_carry_a_probability() -> None:
    with pytest.raises(ValueError, match="must not carry"):
        Alert(
            reach_id="X",
            variable="ndci",
            issued_at=ISSUED,
            severity=Severity.INSUFFICIENT_EVIDENCE,
            target_window=(date(2025, 9, 24), date(2025, 9, 24)),
            exceedance_prob=0.3,
            threshold_value=None,
            threshold_derivation=None,
            suppressed_reason="STALE_EVIDENCE",
        )


def test_suppressed_outcomes_keep_their_reason_in_the_record() -> None:
    rec = to_record(apply_guardrails(candidate(0.2), healthy_state(), [], CFG))
    assert rec["suppressed"] is True and rec["suppressed_reason"].startswith("BELOW_MIN")


# ---------------------------------------------------------------------------
# probability + attribution
# ---------------------------------------------------------------------------
def test_exceedance_probability_uses_all_seven_quantiles() -> None:
    f = forecast(10.0, date(2025, 9, 24))  # q = 8, 8.5, 9.3, 10, 10.7, 11.5, 12
    assert compute_exceedance_probability(f, 10.0) == pytest.approx(0.5)
    assert compute_exceedance_probability(f, 10.7) == pytest.approx(0.25)
    assert compute_exceedance_probability(f, 11.5) == pytest.approx(0.10)
    assert compute_exceedance_probability(f, 12.0) == pytest.approx(0.05)
    assert compute_exceedance_probability(f, 13.0) == pytest.approx(0.0)


def test_attribution_never_presents_reach_offset_or_horizon_as_a_driver() -> None:
    names = ["reach_id", "horizon", "first_flush_index_fut", "imperviousness_pct", "api_7"]
    out = build_attribution([5.0, 3.0, 0.31, 0.19, 0.0], names, [None, 2, 40.0, 61.0, 1.0])
    assert [d.feature for d in out] == ["first_flush_index_fut", "imperviousness_pct"]
    assert out[1].kind == "catchment_attribute" and out[0].kind == "weather_driver"
