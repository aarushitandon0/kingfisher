"""FastAPI dependencies. Tests override these (app.dependency_overrides) to run every
route without a database, a trained model or a results directory."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import pandas as pd

from api.repository import PostgresRepository, Repository
from core.settings import REPO_ROOT
from engine.scenarios import InterventionRequest, ScenarioResult


class ScenarioRunner(Protocol):
    """What the scenario routes need from models/scenario_model.ScenarioContext."""

    @property
    def reach_ids(self) -> frozenset[str]: ...

    @property
    def variables(self) -> tuple[str, ...]: ...

    def lever_paths(self) -> dict[str, dict[str, tuple[str, str]]]: ...

    def run(
        self,
        interventions: Sequence[InterventionRequest],
        variables: Sequence[str] | None = None,
    ) -> ScenarioResult: ...


class ScenarioRunnerFactory(Protocol):
    def __call__(self, city: str) -> ScenarioRunner: ...


@lru_cache(maxsize=1)
def _repository() -> PostgresRepository:
    return PostgresRepository()


def get_repository() -> Repository:
    return _repository()


def get_results_dir() -> Path:
    return REPO_ROOT / "results"


class ShapSource(Protocol):
    def __call__(self, city: str, model_version: str, quantile: str) -> pd.DataFrame: ...


def get_shap_source() -> ShapSource:
    from api.services import production_shap

    return production_shap


def _scenario_context(city: str) -> ScenarioRunner:
    from models.scenario_model import scenario_context

    return scenario_context(city)


def get_scenario_runner() -> ScenarioRunnerFactory:
    return _scenario_context
