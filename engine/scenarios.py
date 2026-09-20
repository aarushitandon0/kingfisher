"""Scenario engine - pure functions only.

Cited coefficients from config/intervention_coefficients.yaml perturb driver and
static features; the perturbed features are re-inferred. The model is never asked
for an intervention effect. Intervals are widened on every scenario run and the
planning-estimate caveat travels with the result."""

from __future__ import annotations


def apply_scenario() -> None:
    raise NotImplementedError("engine.scenarios - Day 6")
