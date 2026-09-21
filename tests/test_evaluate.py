"""Evaluation: the metrics are computed correctly, the baselines cannot see the future,
missing baselines are counted rather than filled, and - broken on purpose - a model
that loses to a baseline is reported as losing."""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

import models.evaluate as ev
from engine.probability import exceedance_probability, quantile_cdf
from models.baseline_gbm import fit, forecast_and_explain
from models.evaluate import (
    attach_baselines,
    climatology,
    compute_metrics,
    crps_quantile,
    find_losses,
    format_report,
    observations_from_frame,
    reliability,
    score_block,
    seasonal_naive,
    wilson,
    write_figures,
)
from pipeline.build_dataset import walk_forward
from tests import gbm_fixtures as fx
from tests.test_build_dataset import FS


# ---------------------------------------------------------------------------
# probability
# ---------------------------------------------------------------------------
def test_exceedance_probability_hits_the_quantiles() -> None:
    p = exceedance_probability(1.0, 2.0, 4.0, np.array([1.0, 2.0, 4.0, 3.0]))
    np.testing.assert_allclose(p, [0.9, 0.5, 0.1, 0.3])


def test_exceedance_probability_tails_and_monotone() -> None:
    thr = np.linspace(-5, 10, 301)
    p = exceedance_probability(1.0, 2.0, 4.0, thr)
    assert p[0] == 1.0 and p[-1] == 0.0
    assert (np.diff(p) <= 1e-12).all()
    assert exceedance_probability(1.0, 2.0, 4.0, 4.5) == pytest.approx(0.0)  # p90 + w_hi/4


def test_exceedance_probability_nan_in_nan_out() -> None:
    p = exceedance_probability([1.0, np.nan], [2.0, 2.0], [3.0, 3.0], [np.nan, 2.0])
    assert np.isnan(p).all()


def test_point_mass_quantiles() -> None:
    assert quantile_cdf(2.0, 2.0, 2.0, 1.9) == 0.0
    assert quantile_cdf(2.0, 2.0, 2.0, 2.1) == 1.0


def test_crossed_quantiles_raise() -> None:
    with pytest.raises(ValueError, match="cross"):
        exceedance_probability(3.0, 2.0, 4.0, 1.0)


# ---------------------------------------------------------------------------
# scores
# ---------------------------------------------------------------------------
def test_crps_of_point_forecast_is_absolute_error() -> None:
    y = np.array([1.0, 5.0, -2.0])
    x = np.array([2.0, 3.0, -2.5])
    np.testing.assert_allclose(crps_quantile(y, x, x, x), np.abs(y - x))


def test_crps_rewards_a_sharp_correct_interval() -> None:
    y = np.array([2.0])
    wide = crps_quantile(y, np.array([0.0]), np.array([2.0]), np.array([4.0]))
    narrow = crps_quantile(y, np.array([1.8]), np.array([2.0]), np.array([2.2]))
    assert narrow < wide


def test_wilson_interval_contains_estimate() -> None:
    lo, hi = wilson(0, 3)
    assert lo == 0.0 and 0.4 < hi < 0.8
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi


def test_reliability_bins_count_every_forecast() -> None:
    rng = np.random.default_rng(0)
    prob = rng.random(5000)
    event = rng.random(5000) < prob  # calibrated by construction
    r = reliability(prob, event, np.full(5000, event.mean()), 10, 0.6)
    assert sum(b["n"] for b in r["bins"]) == 5000
    for b in r["bins"]:
        assert abs(b["observed_frequency"] - b["mean_forecast"]) < 0.06
    assert r["brier_skill_vs_climatology"] > 0
    op = r["operating_point_pre_guardrail"]
    assert op["hits"] + op["false_alarms"] == op["flagged"] == int((prob >= 0.6).sum())


# ---------------------------------------------------------------------------
# baselines never see the future; missing is counted, not filled
# ---------------------------------------------------------------------------
def _obs(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reach_id": "R",
            "variable": "v",
            "date": pd.to_datetime([d for d, _, _ in rows]),
            "value": [x for _, _, x in rows],
        }
    )


def _targets(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reach_id": "R",
            "variable": "v",
            "issued_date": [date.fromisoformat(i) for i, _ in pairs],
            "target_date": [date.fromisoformat(t) for _, t in pairs],
        }
    )


def test_seasonal_naive_takes_most_recent_prior_year() -> None:
    obs = _obs(
        [
            ("2021-06-10", "", 1.0),
            ("2022-06-12", "", 2.0),
            ("2023-06-09", "", 3.0),
            ("2023-06-20", "", 99.0),
        ]
    )
    out = seasonal_naive(obs, _targets([("2024-06-05", "2024-06-10")]), window_days=7)
    assert out[0] == 3.0  # 2023-06-09 is 1 day off the anchor; 06-20 is outside the window


def test_seasonal_naive_ignores_the_target_year() -> None:
    obs = _obs([("2023-06-10", "", 3.0), ("2024-06-01", "", 555.0), ("2024-06-10", "", 777.0)])
    out = seasonal_naive(obs, _targets([("2024-06-05", "2024-06-10")]), window_days=7)
    assert out[0] == 3.0


def test_seasonal_naive_missing_is_nan_not_zero() -> None:
    obs = _obs([("2020-01-10", "", 3.0)])
    out = seasonal_naive(obs, _targets([("2024-06-05", "2024-06-10")]), window_days=7)
    assert np.isnan(out[0])


def test_climatology_uses_training_fold_only() -> None:
    obs = _obs(
        [
            ("2021-06-10", "", 1.0),
            ("2022-06-12", "", 2.0),
            ("2023-06-09", "", 3.0),
            ("2024-06-01", "", 1000.0),
        ]
    )
    c = climatology(obs, _targets([("2024-06-05", "2024-06-10")]), date(2023, 12, 31), 15, 3, 0.9)
    assert c.loc[0, "mean"] == pytest.approx(2.0)
    assert c.loc[0, "n"] == 3


def test_climatology_below_min_obs_is_nan_and_counted() -> None:
    obs = _obs([("2022-06-12", "", 2.0), ("2023-06-09", "", 3.0)])
    c = climatology(obs, _targets([("2024-06-05", "2024-06-10")]), date(2023, 12, 31), 15, 3, 0.9)
    assert c.loc[0, "n"] == 2
    assert c.loc[0, ["mean", "q10", "q50", "q90", "threshold"]].isna().all()


def test_climatology_window_wraps_the_year() -> None:
    obs = _obs([("2021-12-28", "", 1.0), ("2022-01-03", "", 2.0), ("2023-12-30", "", 3.0)])
    c = climatology(obs, _targets([("2024-01-01", "2024-01-02")]), date(2023, 12, 31), 15, 3, None)
    assert c.loc[0, "n"] == 3


# ---------------------------------------------------------------------------
# losses are reported - broken on purpose
# ---------------------------------------------------------------------------
def _scored(model_error: float, n: int = 50, observable: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    y = rng.normal(5, 1, n)
    p50 = y + model_error
    return pd.DataFrame(
        {
            "fold": "test",
            "variable": "turbidity_proxy",
            "reach_id": "R",
            "horizon": np.tile([1, 5, 9], n)[:n],
            "target": y,
            "p10": p50 - 1,
            "p50": p50,
            "p90": p50 + 1,
            "reach_observable": observable,
            "sn": y + 0.5,
            "clim_mean": y + 0.7,
            "clim_q10": y - 0.3,
            "clim_q50": y + 0.7,
            "clim_q90": y + 1.7,
            "clim_threshold": np.nan,
            "clim_clim_exceed_freq": np.nan,
        }
    )


def test_a_losing_model_is_reported_as_losing() -> None:
    m = compute_metrics(_scored(model_error=3.0), fx.GS, fx.EVAL_CFG, fx.THRESHOLDS)
    m["not_computed"] = {}
    assert m["losses"], "a model 6x worse than seasonal-naive produced no losses"
    assert {x["baseline"] for x in m["losses"]} == {"seasonal_naive", "climatology"}
    assert {x["scope"] for x in m["losses"]} >= {"all", "h1", "h01_03", "reaches:observable"}
    text = format_report(m)
    assert "LOSES to seasonal_naive:mae" in text
    assert f"THE MODEL LOSES TO A BASELINE IN {len(m['losses'])} COMPARISONS" in text


def test_a_winning_model_is_reported_as_winning() -> None:
    m = compute_metrics(_scored(model_error=0.05), fx.GS, fx.EVAL_CFG, fx.THRESHOLDS)
    m["not_computed"] = {}
    assert not [x for x in m["losses"] if x["metric"] == "mae"]
    assert "beats both" in format_report(m)


def test_skill_sign_convention() -> None:
    blk = score_block(_scored(model_error=3.0))
    assert blk["skill"]["seasonal_naive"]["mae"] < 0
    assert find_losses(blk, {"x": 1})


def test_baseline_gaps_are_disclosed_not_hidden() -> None:
    s = _scored(model_error=0.05)
    s.loc[s.index[:20], "sn"] = np.nan
    blk = score_block(s)
    assert blk["n_model"] == 50 and blk["n_common"] == 30
    assert blk["baseline_coverage"]["seasonal_naive"] == pytest.approx(0.6)


def test_empty_driver_only_group_is_reported_as_unmeasurable() -> None:
    m = compute_metrics(_scored(model_error=0.05), fx.GS, fx.EVAL_CFG, fx.THRESHOLDS)
    grp = m["folds"]["test"]["turbidity_proxy"]["by_observability"]["driver_only"]
    assert grp["n_model"] == 0 and "common" not in grp
    m["not_computed"] = {}
    assert "reaches:driver_only" in format_report(m)


# ---------------------------------------------------------------------------
# end to end on the synthetic frame
# ---------------------------------------------------------------------------
def test_end_to_end_metrics_and_figures(tmp_path) -> None:
    frame = fx.frame()
    cats = sorted(frame["reach_id"].astype(str).unique())
    preds, train_end = [], {}
    for fold in walk_forward(frame, FS):
        model = fit(
            fold["train"],
            FS,
            fx.GS,
            city="test",
            fit_name=f"wf-{fold['name']}",
            train_end=fold["train_end"],
            categories=cats,
        )
        p, _ = forecast_and_explain(model, fold["eval"], require_target=True)
        p["fold"] = fold["name"]
        preds.append(p)
        train_end[fold["name"]] = fold["train_end"]
    obs = observations_from_frame(frame, FS.variables)
    scored = attach_baselines(pd.concat(preds), obs, train_end, fx.EVAL_CFG, fx.THRESHOLDS)
    m = ev._clean(compute_metrics(scored, fx.GS, fx.EVAL_CFG, fx.THRESHOLDS))
    json.dumps(m)  # serialisable, no NaN
    assert list(m["folds"]) == ["val", "test"]
    for fold in ("val", "test"):
        for var in FS.variables:
            v = m["folds"][fold][var]
            assert set(v["by_horizon"]) == {str(h) for h in FS.horizons}
            assert {"observable", "driver_only"} <= set(v["by_observability"])
            assert v["probability"]["n"] > 0
    # turbidity is driven by rain in the fixture: the model should beat both baselines
    tb = m["folds"]["test"]["turbidity_proxy"]["all_horizons"]["skill"]
    assert tb["seasonal_naive"]["mae"] > 0 and tb["climatology"]["mae"] > 0
    m["not_computed"] = ev.NOT_COMPUTED
    written = write_figures(m, tmp_path)
    assert len(written) == 16
    assert all((tmp_path / f).stat().st_size > 10_000 for f in written)


def test_evaluate_refuses_an_empty_fold(tmp_path, monkeypatch) -> None:
    run = {
        "frame_sha256": "abc",
        "fits": {
            "wf-val": {"version": "v", "train_end": "2023-12-31"},
            "wf-test": {"version": "t", "train_end": "2024-12-31"},
        },
    }
    (tmp_path / "gbm_run_x.json").write_text(json.dumps(run))
    pd.DataFrame({"fold": ["val"], "variable": ["ndci"]}).to_parquet(
        tmp_path / "gbm_eval_x.parquet"
    )
    monkeypatch.setattr(ev, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(ev, "frame_path", lambda city: tmp_path / "frame.parquet")
    monkeypatch.setattr(ev, "file_sha256", lambda path: "abc")
    with pytest.raises(ev.NoEvaluationData, match="test"):
        ev.evaluate("x", results_dir=tmp_path / "results")
    assert not (tmp_path / "results" / "metrics.json").exists()


def test_evaluate_refuses_a_stale_model(tmp_path, monkeypatch) -> None:
    (tmp_path / "gbm_run_x.json").write_text(json.dumps({"frame_sha256": "old", "fits": {}}))
    pd.DataFrame().to_parquet(tmp_path / "gbm_eval_x.parquet")
    monkeypatch.setattr(ev, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(ev, "frame_path", lambda city: tmp_path / "frame.parquet")
    monkeypatch.setattr(ev, "file_sha256", lambda path: "new")
    with pytest.raises(RuntimeError, match="retrain"):
        ev.evaluate("x", results_dir=tmp_path / "results")
