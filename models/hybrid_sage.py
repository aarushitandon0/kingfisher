"""SAGE-TS hybrid: wavelet burstiness routes smooth regimes to Mamba and bursty
regimes to Transformer, pooled across reaches with a learned embedding.

Hard gate: if it does not beat the baseline on held-out years by end of Day 5,
ship the baseline and report the comparison."""

from __future__ import annotations

from core.logging import get_logger

log = get_logger(__name__)


def train(city: str) -> None:
    raise NotImplementedError("models.hybrid_sage - Day 5")
