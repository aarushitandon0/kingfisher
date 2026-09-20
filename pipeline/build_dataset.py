"""Assembles the modelling table from observations + drivers_daily +
catchment_attributes. Walk-forward splits are constructed here so leakage tests
have one place to assert against."""

from __future__ import annotations

from core.logging import get_logger

log = get_logger(__name__)


def build_dataset(city: str) -> None:
    raise NotImplementedError("pipeline.build_dataset - Day 2")
