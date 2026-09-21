"""LightGBM baseline: leakage, quality flags, quantile integrity, SHAP, persistence, and
the predict() contract.

The leakage test mirrors test_build_dataset's: perturb everything after the training
cut-off and assert the fitted model does not move. It is checked against a deliberate
leak (an un-embargoed training fold) so it cannot pass vacuously.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

import models.baseline_gbm as gbm
from models.baseline_gbm import (
    InsufficientTrainingData,
    fit,
    forecast_and_explain,
    predict_long,
    rearrange,
    shap_long,
    to_long,
    validate_settings,
)
from pipeline.build_dataset import target_column, walk_forward
from tests import gbm_fixtures as fx
from tests.test_build_dataset import FS, _perturb_after

TRAIN_END = FS.train_end


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return fx.frame()


@pytest.fixture(scope="module")
def cats(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["reach_id"].astype(str).unique())


@pytest.fixture(scope="module")
def val_fold(frame: pd.DataFrame) -> dict:
    return walk_forward(frame, FS)[0]


@pytest.fixture(scope="module")
def model(val_fold: dict, cats: list[str]) -> gbm.FittedGBM:
    return fit(
        val_fold["train"],
        FS,
        fx.GS,
        city="test",
        fit_name="wf-val",
        train_end=TRAIN_END,
        categories=cats,
    )


# ---------------------------------------------------------------------------
# quality flags: missing is never zero, never filled
# ---------------------------------------------------------------------------
def test_long_frame_keeps_only_observed_targets(frame: pd.DataFrame, cats: list[str]) -> None:
    long = to_long(frame, FS, "turbidity_proxy", [1, 2, 3], cats)
    expected = sum(int(frame[target_column("turbidity_proxy", h)].notna().sum()) for h in (1, 2, 3))
    assert len(long) == expected
    assert long["target"].notna().all()


def test_missing_features_stay_nan(frame: pd.DataFrame, cats: list[str]) -> None:
    long = to_long(frame, FS, "turbidity_proxy", [1], cats)
    rows = frame[frame[target_column("turbidity_proxy", 1)].notna()]
    for col in ("turbidity_proxy_asof", "api_30", "upstream_state_lag1_turbidity", "alan_radiance"):
        assert long[col].isna().sum() == rows[col].isna().sum(), col
    assert long["alan_radiance"].isna().all()  # all-NULL static stays all-NULL, not 0


def test_future_rain_sum_is_nan_when_any_day_missing(frame: pd.DataFrame, cats: list[str]) -> None:
    f = frame.copy()
    f["precip_mm_fut_h2"] = np.nan  # day t+2 unknown
    long = to_long(f, FS, "turbidity_proxy", [1, 3], cats, require_target=False)
    assert long.loc[long["horizon"] == 3, "precip_fut_cum_mm"].isna().all()
    assert long.loc[long["horizon"] == 1, "precip_fut_cum_mm"].notna().any()


def test_non_numeric_feature_raises_instead_of_becoming_nan(
    frame: pd.DataFrame, cats: list[str]
) -> None:
    f = frame.head(200).copy()
    f["alan_radiance"] = f["alan_radiance"].astype(object)
    f.loc[f.index[0], "alan_radiance"] = "n/a"
    with pytest.raises(TypeError, match="alan_radiance"):
        to_long(f, FS, "turbidity_proxy", [1], cats, require_target=False)


def test_fit_refuses_thin_training_data(val_fold: dict, cats: list[str]) -> None:
    with pytest.raises(InsufficientTrainingData, match="l1_satellite"):
        fit(
            val_fold["train"],
            FS,
            fx.settings(min_train_rows=10**6),
            city="test",
            fit_name="wf-val",
            train_end=TRAIN_END,
            categories=cats,
        )


# ---------------------------------------------------------------------------
# leakage
# ---------------------------------------------------------------------------
def test_training_targets_never_pass_train_end(
    model: gbm.FittedGBM, val_fold: dict, cats: list[str]
) -> None:
    for var in FS.variables:
        long = to_long(val_fold["train"], FS, var, FS.horizons, cats)
        assert max(long["target_date"]) <= TRAIN_END


def test_unembargoed_fold_is_caught(frame: pd.DataFrame, cats: list[str]) -> None:
    """Break the embargo on purpose: rows up to train_end but with t+h targets in 2024."""
    leaky = frame[pd.to_datetime(frame["date"]).dt.date <= TRAIN_END]
    with pytest.raises(AssertionError, match="train_end"):
        fit(leaky, FS, fx.GS, city="test", fit_name="wf-val", train_end=TRAIN_END, categories=cats)


def test_model_is_invariant_to_everything_after_train_end(
    model: gbm.FittedGBM, val_fold: dict, cats: list[str]
) -> None:
    w = fx.weather()
    obs = fx.observations(w, 1)
    pw, po = _perturb_after(w, obs, TRAIN_END)
    perturbed = walk_forward(fx.frame_from(pw, po), FS)[0]
    refit = fit(
        perturbed["train"],
        FS,
        fx.GS,
        city="test",
        fit_name="wf-val",
        train_end=TRAIN_END,
        categories=cats,
    )
    assert refit.version == model.version
    probe = to_long(val_fold["train"], FS, "turbidity_proxy", [1, 5, 9], cats)
    a = predict_long(model, probe, "turbidity_proxy")
    b = predict_long(refit, probe, "turbidity_proxy")
    np.testing.assert_array_equal(
        a[["p10", "p50", "p90"]].to_numpy(), b[["p10", "p50", "p90"]].to_numpy()
    )


def test_version_changes_when_training_data_changes(
    model: gbm.FittedGBM, val_fold: dict, cats: list[str]
) -> None:
    train = val_fold["train"].copy()
    col = target_column("turbidity_proxy", 1)
    first = train[col].first_valid_index()
    train.loc[first, col] = train.loc[first, col] + 1.0
    other = fit(
        train, FS, fx.GS, city="test", fit_name="wf-val", train_end=TRAIN_END, categories=cats
    )
    assert other.version != model.version
    assert other.version.startswith("gbm-0.0.0-test+test.wf-val.")


# ---------------------------------------------------------------------------
# quantiles and SHAP
# ---------------------------------------------------------------------------
def test_rearrange_sorts_and_counts_crossings() -> None:
    q = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 4.0], [5.0, 5.0, 5.0]])
    out, crossed = rearrange(q)
    assert (np.diff(out, axis=1) >= 0).all()
    assert crossed.tolist() == [False, True, False]


def test_predictions_are_monotone(model: gbm.FittedGBM, val_fold: dict) -> None:
    pred, _ = forecast_and_explain(model, val_fold["eval"], require_target=True)
    assert len(pred) > 0
    assert (pred["p10"] <= pred["p50"]).all() and (pred["p50"] <= pred["p90"]).all()
    assert set(pred["horizon"]) == set(FS.horizons)


def test_shap_is_additive(model: gbm.FittedGBM, val_fold: dict, cats: list[str]) -> None:
    long = to_long(val_fold["eval"], FS, "turbidity_proxy", [2, 6], cats)
    shap = shap_long(model, long, "turbidity_proxy")
    assert set(shap["quantile"]) == {"p50", "p90"}
    contrib = shap.filter(like="shap__").sum(axis=1) + shap["shap_bias"]
    for q, alpha in (("p50", 0.5), ("p90", 0.9)):
        rows = shap["quantile"] == q
        for h in (2, 6):
            m = (long["horizon"] == h).to_numpy()
            raw = model.booster("turbidity_proxy", h, alpha).predict(long.loc[m, model.features])
            got = contrib[rows & (shap["horizon"] == h)].to_numpy()
            np.testing.assert_allclose(got, raw, rtol=1e-6, atol=1e-8)


def test_settings_must_cover_every_horizon() -> None:
    bad = fx.settings(buckets={"h01_03": (1, 2, 3), "h04_07": (4, 5, 6, 7)})
    with pytest.raises(ValueError, match="cover"):
        validate_settings(bad, FS)


# ---------------------------------------------------------------------------
# persistence and predict()
# ---------------------------------------------------------------------------
def test_save_load_roundtrip(
    model: gbm.FittedGBM, val_fold: dict, cats: list[str], tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(gbm.FrameSettings, "from_config", classmethod(lambda cls, cfg=None: FS))
    monkeypatch.setattr(gbm.GBMSettings, "from_config", classmethod(lambda cls, cfg=None: fx.GS))
    gbm.save(model, tmp_path)
    loaded = gbm.load("test", "wf-val", root=tmp_path)
    assert loaded.version == model.version
    assert loaded.train_end == TRAIN_END
    long = to_long(val_fold["eval"], FS, "ndci", [4], cats)
    a = predict_long(model, long, "ndci")[["p10", "p50", "p90"]]
    b = predict_long(loaded, long, "ndci")[["p10", "p50", "p90"]]
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy())


@pytest.fixture
def served(model: gbm.FittedGBM, frame: pd.DataFrame, monkeypatch):
    monkeypatch.setattr(gbm, "city_for_reach", lambda rid: "test")
    monkeypatch.setattr(gbm, "_cached_model", lambda city, fit_name: model)
    monkeypatch.setattr(
        gbm,
        "_cached_reach_frame",
        lambda city, rid: frame[frame["reach_id"].astype(str) == rid],
    )
    return model


def test_predict_contract(served: gbm.FittedGBM) -> None:
    out = gbm.predict("TST-A", date(2024, 3, 10), 4, "turbidity_proxy")
    assert {"p10", "p50", "p90", "model_version", "target_date", "in_sample"} <= set(out)
    assert out["p10"] <= out["p50"] <= out["p90"]
    assert out["target_date"] == date(2024, 3, 14)
    assert out["model_version"] == served.version
    assert out["in_sample"] is False  # 2024 is after the wf-val model's train_end
    assert out["future_drivers_missing"] == 0
    assert gbm.predict("TST-A", date(2023, 3, 10), 1)["in_sample"] is True


def test_predict_flags_missing_future_weather(served: gbm.FittedGBM) -> None:
    out = gbm.predict("TST-C", date(2025, 6, 28), 9, "ndci")  # t+9 is past the archive
    assert out["future_drivers_missing"] > 0
    assert np.isfinite(out["p50"])


def test_predict_rejects_bad_requests(served: gbm.FittedGBM) -> None:
    with pytest.raises(ValueError, match="horizon"):
        gbm.predict("TST-A", date(2024, 3, 10), 11)
    with pytest.raises(LookupError):
        gbm.predict("TST-A", date(2030, 1, 1), 1)
    with pytest.raises(ValueError, match="variable"):
        gbm.predict("TST-A", date(2024, 3, 10), 1, "dissolved_oxygen")


def test_explain_sums_to_raw_prediction(served: gbm.FittedGBM) -> None:
    ex = gbm.explain("TST-B", date(2024, 5, 1), 2, "turbidity_proxy", alpha=0.9)
    total = ex["bias"] + sum(d["contribution"] for d in ex["drivers"])
    assert total == pytest.approx(ex["raw_prediction"])
    mags = [abs(d["contribution"]) for d in ex["drivers"]]
    assert mags == sorted(mags, reverse=True)
