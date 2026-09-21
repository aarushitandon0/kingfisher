"""Synthetic modelling frames for the baseline GBM and evaluation tests.

Built through the real L2 feature code and pipeline.build_dataset.build_frame (via the
leakage-test helpers), with a state that depends on the drivers, so the model has
something real to learn and the baselines have prior years to draw on.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd

from models.baseline_gbm import GBMSettings
from pipeline.build_dataset import build_frame
from tests.test_build_dataset import FS, REACH_ATTRS, REACHES, STATIC, _drivers

START, END = date(2021, 1, 1), date(2025, 6, 30)
OFFSET = {"TST-A": 0.0, "TST-B": 3.0, "TST-C": 1.5}

GS = GBMSettings(
    version="0.0.0-test",
    quantiles=(0.1, 0.5, 0.9),
    buckets={"h01_03": (1, 2, 3), "h04_07": (4, 5, 6, 7), "h08_10": (8, 9, 10)},
    min_train_rows=100,
    params={
        "n_estimators": 60,
        "learning_rate": 0.1,
        "num_leaves": 7,
        "min_data_in_leaf": 10,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "cat_smooth": 10,
        "min_data_per_group": 10,
        "seed": 7,
    },
    shap_quantiles=(0.5, 0.9),
)
EVAL_CFG = {
    "seasonal_naive_window_days": 7,
    "climatology_window_days": 15,
    "min_climatology_obs": 3,
    "reliability_bins": 10,
}
THRESHOLDS = {
    "seasons": {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]},
    "variables": {
        v: {
            "display_name": v,
            "units": "dimensionless index",
            "derivation": "per_reach_seasonal_percentile",
            "percentile": 0.9,
            "min_obs": 8,
        }
        for v in FS.variables
    },
    "guardrails": {"min_exceedance_prob": 0.6},
}


def weather(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(START, END, freq="D", name="date")
    wet = rng.random(len(idx)) < 0.35
    precip = np.where(wet, rng.gamma(1.5, 6.0, len(idx)), 0.0)
    doy = idx.dayofyear.to_numpy()
    season = np.sin(2 * np.pi * (doy - 110) / 365)
    return pd.DataFrame(
        {
            "precip_mm": precip,
            "precip_max_hourly": precip * rng.uniform(0.1, 0.5, len(idx)),
            "precip_duration_h": (precip / 2).round(),
            "temp_mean_c": 15 + 8 * season + rng.normal(0, 1, len(idx)),
            "temp_max_c": 20 + 8 * season,
            "soil_moisture": 0.3,
            "et0": 2.5,
        },
        index=idx,
    )


def observations(w: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    """Every 5th day per reach, ~70% OK. Turbidity responds to same-day rain; NDCI to
    temperature. CLOUD rows carry NULL values, as L1 writes them."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, rid in enumerate(REACHES):
        for day in pd.date_range(pd.Timestamp(START) + pd.Timedelta(days=i), END, freq="5D"):
            ok = rng.random() < 0.7
            p, t = w.at[day, "precip_mm"], w.at[day, "temp_mean_c"]
            rows.append(
                {
                    "reach_id": rid,
                    "obs_date": day.date(),
                    "source": "S2",
                    "quality_flag": "OK" if ok else "CLOUD",
                    "turbidity_proxy": 2 + OFFSET[rid] + 0.4 * p + rng.gamma(2, 0.5)
                    if ok
                    else np.nan,
                    "ndci": -0.3 + 0.015 * t + rng.normal(0, 0.03) if ok else np.nan,
                }
            )
    return pd.DataFrame(rows)


def frame(seed: int = 0) -> pd.DataFrame:
    w = weather(seed)
    obs = observations(w, seed + 1)
    return build_frame(_drivers(w, obs), obs, STATIC, REACH_ATTRS, FS)


def frame_from(w: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    return build_frame(_drivers(w, obs), obs, STATIC, REACH_ATTRS, FS)


def settings(**overrides: object) -> GBMSettings:
    return replace(GS, **overrides)  # type: ignore[arg-type]
