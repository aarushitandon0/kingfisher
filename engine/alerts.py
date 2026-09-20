"""Alert engine - pure functions only. No I/O, no DB, no network, so guardrails are
testable without a server.

An alert is a calibrated probability of threshold exceedance, never a raw
prediction. Guardrails: minimum confidence, cooldown, staleness suppression,
unobservable-reach severity cap, drift suppression. INSUFFICIENT_EVIDENCE is a
first-class output and is never coerced into a low-confidence alert."""

from __future__ import annotations


def evaluate_alert() -> None:
    raise NotImplementedError("engine.alerts - Day 4")
