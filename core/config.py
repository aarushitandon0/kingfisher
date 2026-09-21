"""YAML config loading. Cities have their own loader (pipeline.l0_network.load_city_config)
because they carry required keys; everything else in config/ comes through here."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import yaml

from core.settings import CONFIG_DIR

if TYPE_CHECKING:
    from engine.coefficients import CoefficientTable

# Files that must never be read unvalidated. The value is the loader to use instead.
_VALIDATED_ONLY = {"intervention_coefficients": "core.config.load_intervention_coefficients()"}


@lru_cache(maxsize=16)
def load_config(name: str) -> dict[str, Any]:
    """config/<name>.yaml as a dict. Raises if the file is missing - no silent defaults."""
    if name in _VALIDATED_ONLY:
        raise ValueError(f"config/{name}.yaml must be loaded through {_VALIDATED_ONLY[name]}")
    return _read_yaml(name)


def load_intervention_coefficients() -> CoefficientTable:
    """The validated scenario coefficient table. Raises UncitedCoefficientError if any
    entry, or any number in one, lacks a citation - an uncited coefficient is never usable."""
    from engine.coefficients import validate_coefficients

    return validate_coefficients(_read_yaml("intervention_coefficients"))


def _read_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing config file {path}")
    cfg: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return cfg


def as_date(value: Any) -> date:
    """YAML parses 2023-12-31 as a date already; strings are accepted too."""
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))
