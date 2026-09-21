"""Variants A/B, the land-cover proxy drop, oracle-vs-as-issued substitution, and the
evaluation comparisons that go into metrics.json."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from models.baseline_gbm import (
    GBMSettings,
    VariantSpec,
    all_features,
    feature_names,
    fit,
    forecast_and_explain,
    proxy_dropped_features,
    substitute_future_drivers,
    to_long,
)
from models.evaluate import (
    compare_on_shared_rows,
    crps_weights,
    forecast_records,
    reach_id_shap_share,
)
from pipeline.build_dataset import static_flag_summary, walk_forward
from tests import gbm_fixtures as fx
from tests.test_build_dataset import FS

B = VariantSpec(
    "B", reach_id=False, drop=("turbidity_proxy_asof", "ndci_asof", "obs_asof_age_days")
)


def test_config_defines_both_variants_and_seven_quantiles() -> None:
    gs = GBMSettings.from_config()
    assert gs.quantiles == (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
    assert gs.variants["A"].reach_id is True
    assert gs.variants["B"].reach_id is False
    assert "turbidity_proxy_asof" in gs.variants["B"].drop
    assert gs.drop_when_proxy == ("urban_fraction",)


def test_variant_b_has_no_reach_identity() -> None:
    a = feature_names(FS, VariantSpec("A"))
    b = feature_names(FS, B)
    assert "reach_id" in a and "reach_id" not in b
    assert not {"turbidity_proxy_asof", "ndci_asof", "obs_asof_age_days"} & set(b)
    assert {"imperviousness_pct", "riparian_ndvi_mean", "catchment_area_km2"} <= set(b)
    with pytest.raises(ValueError, match="unknown"):
        feature_names(FS, VariantSpec("X", drop=("not_a_feature",)))


def test_urban_fraction_dropped_only_while_it_is_a_proxy() -> None:
    gs = replace(fx.GS, drop_when_proxy=("urban_fraction",))
    static = pd.DataFrame(
        {
            "flags": [
                {"urban_fraction": "PROXY_BUILT_UP_SHARE"},
                {"urban_fraction": "PROXY_BUILT_UP_SHARE"},
            ]
        }
    )
    dropped = proxy_dropped_features(gs, static_flag_summary(static))
    assert set(dropped) == {"urban_fraction"} and "on 2 reaches" in dropped["urban_fraction"]
    assert "urban_fraction" not in feature_names(FS, VariantSpec("A"), dropped)
    clms = pd.DataFrame({"flags": [{"urban_fraction": "CLMS_OK"}]})
    assert proxy_dropped_features(gs, static_flag_summary(clms)) == {}


def test_variant_b_fits_and_predicts_without_reach_id() -> None:
    frame = fx.frame()
    cats = sorted(frame["reach_id"].astype(str).unique())
    fold = walk_forward(frame, FS)[0]
    model = fit(
        fold["train"],
        FS,
        fx.GS,
        city="t",
        fit_name="B.wf-val",
        train_end=fold["train_end"],
        categories=cats,
        variant=B,
        dropped={"urban_fraction": "PROXY"},
    )
    assert "reach_id" not in model.features and "urban_fraction" not in model.features
    assert model.manifest["variant"]["name"] == "B"
    pred, shap = forecast_and_explain(model, fold["eval"], require_target=True)
    assert set(pred["variant"]) == {"B"} and set(pred["weather"]) == {"ORACLE"}
    assert "shap__reach_id" not in shap


def test_asissued_substitution_replaces_future_drivers_and_drops_uncovered_rows() -> None:
    frame = fx.frame()
    cats = sorted(frame["reach_id"].astype(str).unique())
    long = to_long(frame.head(200), FS, "turbidity_proxy", [1, 2], cats, require_target=False)
    keep = long.iloc[: len(long) // 2]
    asissued = pd.DataFrame(
        {
            "reach_id": keep["reach_id"].astype(str),
            "issued_date": keep["issued_date"],
            "horizon": keep["horizon"].astype(int),
            **{c: 123.0 for c in FS.future_driver_columns},
            "precip_fut_cum_mm": 456.0,
        }
    )
    out = substitute_future_drivers(long, asissued, FS)
    assert len(out) == len(keep)
    assert list(out.columns) == list(long.columns)
    assert (out["precip_mm_fut"] == 123.0).all() and (out["precip_fut_cum_mm"] == 456.0).all()
    # everything that is not a future driver is untouched
    same = [c for c in long.columns if not c.endswith("_fut") and c != "precip_fut_cum_mm"]
    pd.testing.assert_frame_equal(
        out[same].reset_index(drop=True), keep[same].reset_index(drop=True), check_dtype=False
    )


def test_crps_weights_for_seven_levels() -> None:
    w = crps_weights((0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95))
    np.testing.assert_allclose(w, [0.075, 0.1, 0.2, 0.25, 0.2, 0.1, 0.075])
    assert w.sum() == pytest.approx(1.0)


def _scored(offset: float, n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    y = rng.normal(5, 1, n)
    p50 = y + offset
    d = pd.DataFrame(
        {
            "fold": "test",
            "variable": "ndci",
            "reach_id": "R",
            "issued_date": [date(2025, 1, 1 + i % 28) for i in range(n)],
            "horizon": [1 + i // 28 for i in range(n)],
            "target": y,
            "sn": y + 0.5,
        }
    )
    for lv, dz in zip(fx.GS.quantiles, (-1.0, 0.0, 1.0), strict=True):
        d[f"p{round(lv * 100):02d}"] = p50 + dz
    return d


def test_shared_row_comparison_signs() -> None:
    comp = compare_on_shared_rows(_scored(0.1), _scored(2.0), fx.GS, ("A", "B"))
    blk = comp["test"]["ndci"]["all_horizons"]
    assert blk["n"] == 40
    assert blk["skill_A_vs_B_crps"] > 0 and blk["skill_A_vs_B_mae"] > 0
    assert blk["skill_A_vs_seasonal_naive_mae"] > 0 > blk["skill_B_vs_seasonal_naive_mae"]


def test_shared_row_comparison_uses_only_common_rows() -> None:
    comp = compare_on_shared_rows(
        _scored(0.1), _scored(0.1).iloc[:10], fx.GS, ("ORACLE", "ASISSUED")
    )
    assert comp["test"]["ndci"]["all_horizons"]["n"] == 10


def test_reach_id_shap_share() -> None:
    shap = pd.DataFrame(
        {
            "variant": "A",
            "fold": "test",
            "variable": "ndci",
            "quantile": "p50",
            "shap__reach_id": [3.0, -3.0],
            "shap__api_7": [1.0, -1.0],
            "shap_bias": [9.0, 9.0],
        }
    )
    s = reach_id_shap_share(shap)["test"]["ndci"]["p50"]
    assert s["share"] == pytest.approx(0.75) and s["rank"] == 1 and s["of_features"] == 2


def test_forecast_records_carry_all_quantiles_variant_and_weather() -> None:
    pred = _scored(0.0).assign(
        target_date=lambda d: [
            i + pd.Timedelta(days=h)
            for i, h in zip(pd.to_datetime(d["issued_date"]), d["horizon"], strict=True)
        ],
        model_version="v1",
        variant="B",
        weather="ASISSUED",
    )
    rec = forecast_records(pred)
    assert set(rec["variant"]) == {"B"} and set(rec["weather"]) == {"ASISSUED"}
    assert rec["p05"].isna().all()  # the fixture model has no p05 - NULL, not a fill
    assert (rec["target_date"] > rec["issued_date"]).all()
    assert set(rec["fit"]) == {"wf-test"}
    with pytest.raises(ValueError, match="duplicate"):
        forecast_records(pd.concat([pred, pred]))


def test_all_features_is_a_superset_of_every_variant() -> None:
    for v in GBMSettings.from_config().variants.values():
        assert set(feature_names(FS, v)) <= set(all_features(FS))
