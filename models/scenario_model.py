"""Scenario inputs from disk: the variant-B LightGBM fit as an engine.scenarios model,
the reference-year feature rows, the thresholds, and the persisted response checks.

engine/scenarios.py is pure; this module does the I/O for it (MASTERSPEC 9.6):

  GBMScenarioModel      a FittedGBM behind engine.scenarios.ScenarioModel. Variant B only.
                        feature_range() is the (min, max) of each feature over the rows the
                        variable's boosters were fitted on - rows with a target - which is
                        the support the trees actually saw.
  reference_features()  one horizon-h row per reach-day of the reference year, from the
                        modelling frame (archive weather at the target day: the scenario
                        asks "under that year's weather", not "tomorrow").
  scenario_thresholds() engine.thresholds table fitted on OK observations up to the
                        model's training cut-off - the same rule the alerts use.
  load_checks()         results/scenario_response_check_<city>.json -> {(feature, variable):
                        ResponseCheck}; written by models/scenario_response.py.
  ScenarioContext       all of the above for one city, cached per process, plus run().
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import load_config, load_intervention_coefficients
from core.settings import REPO_ROOT
from engine.coefficients import CoefficientTable
from engine.scenarios import (
    SCENARIO_VARIANT,
    InterventionRequest,
    ResponseCheck,
    ScenarioResult,
    ScenarioSettings,
    decide_path,
    run_scenario,
)
from engine.thresholds import seasonal_thresholds
from models.baseline_gbm import FittedGBM, load, load_frame, rearrange, to_long

RESULTS_DIR = REPO_ROOT / "results"


def check_path(city: str, results_dir: Path = RESULTS_DIR) -> Path:
    return results_dir / f"scenario_response_check_{city}.json"


class ScenarioInputsMissing(FileNotFoundError):
    """A scenario input (model, frame, response check) is not on disk."""


@dataclass
class GBMScenarioModel:
    fitted: FittedGBM
    ranges: dict[str, dict[str, tuple[float, float]]]  # variable -> feature -> (min, max)

    def __post_init__(self) -> None:
        variant = self.fitted.manifest.get("variant", {}).get("name")
        if variant != SCENARIO_VARIANT:
            raise ValueError(
                f"{self.fitted.version} is variant {variant}; scenarios use variant "
                f"{SCENARIO_VARIANT} only (MASTERSPEC 9.6)"
            )

    @property
    def version(self) -> str:
        return self.fitted.version

    @property
    def variant(self) -> str:
        return SCENARIO_VARIANT

    @property
    def levels(self) -> tuple[float, ...]:
        return tuple(self.fitted.settings.quantiles)

    @property
    def features(self) -> Sequence[str]:
        return tuple(self.fitted.features)

    def feature_range(self, variable: str, feature: str) -> tuple[float, float] | None:
        return self.ranges.get(variable, {}).get(feature)

    def predict_quantiles(self, features: pd.DataFrame, variable: str) -> np.ndarray:
        gs = self.fitted.settings
        horizons = features["horizon"].to_numpy(dtype="float64")
        q = np.full((len(features), len(gs.quantiles)), np.nan)
        done = np.zeros(len(features), dtype=bool)
        X = features[self.fitted.features]
        for bucket, hs in gs.buckets.items():
            m = np.isin(horizons, hs)
            if not m.any():
                continue
            for j, alpha in enumerate(gs.quantiles):
                q[m, j] = self.fitted.boosters[(variable, bucket, alpha)].predict(X.loc[m])
            done |= m
        if not done.all():
            raise ValueError(f"horizons {sorted(set(horizons[~done]))} are in no bucket")
        q, _ = rearrange(q)
        return q


def training_ranges(
    frame: pd.DataFrame, fitted: FittedGBM
) -> dict[str, dict[str, tuple[float, float]]]:
    """Per variable, (min, max) of every model feature over the training rows (issue date
    <= train_end, with a target at some horizon)."""
    fs = fitted.frame_settings
    train = frame[pd.to_datetime(frame["date"]).dt.date <= fitted.train_end]
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for variable in fs.variables:
        long = to_long(train, fs, variable, fs.horizons, fitted.categories, require_target=True)
        long = long[long["target_date"] <= fitted.train_end]
        ranges: dict[str, tuple[float, float]] = {}
        for f in fitted.features:
            if f == "reach_id":
                continue
            v = long[f].to_numpy(dtype="float64")
            v = v[~np.isnan(v)]
            if len(v):
                ranges[f] = (float(v.min()), float(v.max()))
        out[variable] = ranges
    return out


def reference_features(
    frame: pd.DataFrame, fitted: FittedGBM, settings: ScenarioSettings
) -> pd.DataFrame:
    """Horizon-h model rows whose TARGET date falls in the reference year, every reach."""
    fs = fitted.frame_settings
    h = settings.horizon
    first = settings.reference_start - timedelta(days=h)
    last = settings.reference_end - timedelta(days=h)
    d = pd.to_datetime(frame["date"]).dt.date
    rows = frame[(d >= first) & (d <= last)]
    long = to_long(rows, fs, fs.variables[0], [h], fitted.categories, require_target=False)
    long = long.drop(columns=["target"])
    long["reach_id"] = long["reach_id"].astype(str)
    return long.reset_index(drop=True)


def observations(frame: pd.DataFrame, variables: Sequence[str]) -> pd.DataFrame:
    """OK observations (reach_id, date, variable, value): an as-of value of age 0."""
    from models.evaluate import observations_from_frame

    return observations_from_frame(frame, tuple(variables))


def scenario_thresholds(frame: pd.DataFrame, fitted: FittedGBM) -> pd.DataFrame:
    obs = observations(frame, fitted.frame_settings.variables)
    return seasonal_thresholds(obs, fitted.train_end, load_config("thresholds"))


def load_checks(city: str, results_dir: Path = RESULTS_DIR) -> dict[tuple[str, str], ResponseCheck]:
    path = check_path(city, results_dir)
    if not path.exists():
        raise ScenarioInputsMissing(
            f"{path.relative_to(REPO_ROOT)} missing - run `make response-check city={city}`"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("city") != city:
        raise ValueError(f"{path} is for city {raw.get('city')!r}, not {city!r}")
    out: dict[tuple[str, str], ResponseCheck] = {}
    for c in raw["checks"]:
        rc = ResponseCheck.model_validate(c)
        out[(rc.feature, rc.variable)] = rc
    return out


@dataclass
class ScenarioContext:
    city: str
    model: GBMScenarioModel
    features: pd.DataFrame
    thresholds: pd.DataFrame
    checks: dict[tuple[str, str], ResponseCheck]
    settings: ScenarioSettings
    coefficients: CoefficientTable

    @property
    def reach_ids(self) -> frozenset[str]:
        return frozenset(self.features["reach_id"].unique())

    def run(
        self,
        interventions: Sequence[InterventionRequest],
        variables: Sequence[str] | None = None,
    ) -> ScenarioResult:
        rids = tuple(dict.fromkeys(r for i in interventions for r in i.reach_ids))
        return run_scenario(
            rids,
            interventions,
            self.model,
            self.features,
            coefficients=self.coefficients,
            thresholds=self.thresholds,
            response_checks=self.checks,
            settings=self.settings,
            variables=tuple(variables or self.model.fitted.frame_settings.variables),
        )

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(self.model.fitted.frame_settings.variables)

    def lever_paths(self) -> dict[str, dict[str, tuple[str, str]]]:
        """intervention id -> variable -> (path, reason): what a run would do with each
        lever, decided by the same MASTERSPEC 9.6 rule the run applies."""
        return {
            coef.id: {
                v: (str(p), why)
                for v in self.variables
                for p, _, why in [decide_path(coef, v, self.model, self.checks, self.coefficients)]
            }
            for coef in self.coefficients.interventions
        }


def build_context(city: str) -> ScenarioContext:
    modelling = load_config("modelling")
    coefficients = load_intervention_coefficients()
    settings = ScenarioSettings.from_config(modelling, coefficients, load_config("thresholds"))
    fit_name = str(modelling["scenario"]["fit"])
    try:
        fitted = load(city, fit_name)
        frame = load_frame(city)
    except FileNotFoundError as exc:
        raise ScenarioInputsMissing(str(exc)) from exc
    model = GBMScenarioModel(fitted, training_ranges(frame, fitted))
    return ScenarioContext(
        city=city,
        model=model,
        features=reference_features(frame, fitted, settings),
        thresholds=scenario_thresholds(frame, fitted),
        checks=load_checks(city),
        settings=settings,
        coefficients=coefficients,
    )


_BUILD_LOCK = threading.Lock()


@lru_cache(maxsize=4)
def _cached_context(city: str) -> ScenarioContext:
    return build_context(city)


def scenario_context(city: str) -> ScenarioContext:
    """Cached per process: the frame, model and thresholds take seconds to build. The lock
    makes a request that arrives while the startup warm-up is building wait for that
    build instead of starting a second one."""
    with _BUILD_LOCK:
        return _cached_context(city)
