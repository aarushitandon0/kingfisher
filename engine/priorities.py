"""Prioritisation - pure functions only.

information_value(reach) = forecast_uncertainty x predicted_risk x exposure_weight
Produces the ranked next-field-visit list. One panel, not a subsystem."""

from __future__ import annotations


def rank_reaches() -> None:
    raise NotImplementedError("engine.priorities - Day 6")
