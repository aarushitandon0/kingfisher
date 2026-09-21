"""Walk-forward evaluation of the LightGBM baseline (MASTERSPEC 11). Every metric is
reported, favourable or not.

Run (after `python -m models.baseline_gbm --city coimbra`):

    python -m models.evaluate --city coimbra            # also writes the forecasts table
    python -m models.evaluate --city coimbra --no-db

Writes results/metrics.json and results/figures/*.png, prints the comparison table, and
persists every walk-forward prediction (both variants, both weather inputs, all seven
quantiles) to the `forecasts` table. Wherever a model is worse than a baseline, the
table says LOSES and the loss is listed again at the end and in metrics.json["losses"].
Nothing is dropped to make that list shorter.

RUNS - four, scored separately, compared on shared rows
-------------------------------------------------------
  A/ORACLE     variant A (reach_id feature), t+h drivers from the archive (perfect
               prognosis) - the headline table, metrics.json["folds"]
  A/ASISSUED   variant A, t+h drivers rebuilt from the ECMWF IFS 00 UTC run issued on
               the issue date (pipeline.asissued_weather; available from 2024-03-14)
  B/ORACLE     variant B (no reach identity), archive drivers
  B/ASISSUED   variant B, as-issued drivers
metrics.json["runs"] holds all four. Comparisons are always on the rows both sides have:
  variant_comparison   A vs B (per weather input)
  weather_comparison   ORACLE vs ASISSUED (per variant) - what forecast error costs
  reach_id_shap_share  in A, mean |SHAP(reach_id)| / sum of mean |SHAP| over features

FORECASTERS - scored on the same (reach, issue date, horizon) rows
------------------------------------------------------------------
  model           the quantile forecast from models.baseline_gbm (walk-forward fits)
  seasonal_naive  the observation closest to the target's day-of-year in the most recent
                  previous year that has one within +/- seasonal_naive_window_days.
                  A point forecast. Always dated before the issue date.
  climatology     the reach's training-fold observations within +/- climatology_window_days
                  of the target's day-of-year, all training years pooled: mean as the point,
                  empirical quantiles at the model's levels as the distribution. NULL if
                  fewer than min_climatology_obs - counted, never filled.

The headline skill comparison uses rows where all three forecasters exist ("common
rows"); the model is also reported on all of its rows, and each baseline's coverage is
stated, so a baseline that is missing where the model is weak cannot flatter it.

METRICS
-------
  MAE, RMSE        on the point forecast (model P50, climatology mean, seasonal-naive value)
  CRPS             quantile-score quadrature over the K forecast levels:
                     CRPS ~= 2 * sum_k w_k * pinball_{alpha_k}(y, q_k)
                   with w_k the width of the alpha-interval nearest alpha_k (midpoints
                   between adjacent levels, 0 and 1 at the ends; the weights sum to 1).
                   For the seven levels 0.05/0.1/0.25/0.5/0.75/0.9/0.95 the weights are
                   0.075/0.1/0.2/0.25/0.2/0.1/0.075. Applied identically to all three
                   forecasters; for a point forecast and symmetric levels it reduces
                   exactly to MAE. It is an approximation, not the integral CRPS, and is
                   labelled as one.
  coverage_80/90   fraction of observations inside P10-P90 (nominal 0.80) / P05-P95 (0.90)
  skill vs B       1 - metric_model / metric_B   (> 0: model better)

PROBABILITY CALIBRATION
-----------------------
The event is observed > threshold, the threshold being the reach's own seasonal
percentile of its training-fold observations (engine/thresholds.py, config/thresholds.yaml)
- the same table the alert engine uses. The forecast probability comes from
engine.probability.exceedance_probability_multi over all K quantiles - the same function
the alert engine alerts with. Reported: reliability-diagram bins, Brier score, Brier
skill vs the reach-season's climatological exceedance frequency, and hit / false-alarm
counts at the min_exceedance_prob operating point BEFORE guardrails.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.config import load_config
from core.logging import get_logger, stage
from core.settings import REPO_ROOT
from engine.probability import exceedance_probability_multi
from engine.thresholds import lookup as threshold_lookup
from engine.thresholds import seasonal_thresholds
from models.baseline_gbm import GBMSettings, file_sha256, frame_path, load_frame, qname
from pipeline.build_dataset import PROCESSED_DIR, FrameSettings

log = get_logger(__name__)

RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FORECASTERS = ("model", "seasonal_naive", "climatology")
BASELINES = ("seasonal_naive", "climatology")
LOSS_METRICS = ("mae", "rmse", "crps")
MIN_REACH_ROWS = 5  # per-reach comparison needs at least this many common rows
HEADLINE = "A/ORACLE"
KEYS = ["fold", "variable", "reach_id", "issued_date", "horizon"]


class NoEvaluationData(RuntimeError):
    """There is nothing honest to score."""


def display_names(thresholds: dict[str, Any]) -> dict[str, str]:
    return {v: c.get("display_name", v) for v, c in thresholds["variables"].items()}


# ---------------------------------------------------------------------------
# observations (pure)
# ---------------------------------------------------------------------------
def observations_from_frame(frame: pd.DataFrame, variables: tuple[str, ...]) -> pd.DataFrame:
    """Long table of OK observations: reach_id, date, variable, value.

    An OK observation on day t is exactly the frame's as-of value with age 0 on day t.
    """
    today = frame[frame["obs_asof_age_days"] == 0]
    parts = [
        pd.DataFrame(
            {
                "reach_id": today["reach_id"].astype(str).to_numpy(),
                "date": pd.to_datetime(today["date"]).to_numpy(),
                "variable": v,
                "value": today[f"{v}_asof"].to_numpy(dtype="float64"),
            }
        )
        for v in variables
    ]
    obs = pd.concat(parts, ignore_index=True)
    return obs[obs["value"].notna()].reset_index(drop=True)


# ---------------------------------------------------------------------------
# baselines (pure)
# ---------------------------------------------------------------------------
def seasonal_naive(
    obs: pd.DataFrame, targets: pd.DataFrame, window_days: int, max_years_back: int = 15
) -> np.ndarray:
    """Seasonal-naive value per target row.

    targets: reach_id, variable, issued_date, target_date. For k = 1, 2, ...: the
    observation closest to target_date - k years, if within +/- window_days and dated
    on or before the issue date. The first k that has one wins. NaN if none.
    """
    out = np.full(len(targets), np.nan)
    by_key = {k: g.sort_values("date") for k, g in obs.groupby(["reach_id", "variable"])}
    t_all = targets.reset_index(drop=True)
    for (rid, var), idx in t_all.groupby(["reach_id", "variable"]).groups.items():
        g = by_key.get((str(rid), str(var)))
        if g is None or g.empty:
            continue
        od = g["date"].to_numpy(dtype="datetime64[D]")
        ov = g["value"].to_numpy()
        tgt = pd.to_datetime(t_all.loc[idx, "target_date"])
        issued = pd.to_datetime(t_all.loc[idx, "issued_date"]).to_numpy(dtype="datetime64[D]")
        res = np.full(len(idx), np.nan)
        for k in range(1, max_years_back + 1):
            todo = np.isnan(res)
            if not todo.any():
                break
            anchor = (tgt - pd.DateOffset(years=k)).to_numpy(dtype="datetime64[D]")
            pos = np.searchsorted(od, anchor)
            best_val = np.full(len(idx), np.nan)
            best_dist = np.full(len(idx), np.inf)
            for cand in (pos - 1, pos):
                ok = (cand >= 0) & (cand < len(od))
                c = np.clip(cand, 0, len(od) - 1)
                dist = np.abs((od[c] - anchor).astype("int64")).astype("float64")
                ok &= (dist <= window_days) & (od[c] <= issued)
                better = ok & (dist < best_dist)
                best_dist = np.where(better, dist, best_dist)
                best_val = np.where(better, ov[c], best_val)
            res = np.where(todo, best_val, res)
        out[np.asarray(idx)] = res
    return out


def _doy_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a[:, None] - b[None, :])
    return np.asarray(np.minimum(d, 365 - d))


def climatology(
    obs: pd.DataFrame,
    targets: pd.DataFrame,
    train_end: date,
    window_days: int,
    min_obs: int,
    levels: Sequence[float] = (0.1, 0.5, 0.9),
) -> pd.DataFrame:
    """Per target row: mean, the empirical quantiles q<level> at `levels`, and n.
    Uses only observations dated <= train_end."""
    qcols = [f"q{round(a * 100):02d}" for a in levels]
    cols = ["mean", *qcols, "n"]
    out = pd.DataFrame(np.nan, index=range(len(targets)), columns=cols)
    out["n"] = 0
    train = obs[obs["date"] <= pd.Timestamp(train_end)]
    by_key = {k: g for k, g in train.groupby(["reach_id", "variable"])}
    t_all = targets.reset_index(drop=True)
    for (rid, var), idx in t_all.groupby(["reach_id", "variable"]).groups.items():
        g = by_key.get((str(rid), str(var)))
        if g is None or g.empty:
            continue
        vals = g["value"].to_numpy()
        # Day-of-year on a 365-day circle (Feb 29 shares Feb 28's slot).
        odoy = np.minimum(pd.to_datetime(g["date"]).dt.dayofyear.to_numpy(), 365)
        tdoy = np.minimum(
            pd.to_datetime(t_all.loc[idx, "target_date"]).dt.dayofyear.to_numpy(), 365
        )
        near = _doy_distance(tdoy, odoy) <= window_days
        n = near.sum(axis=1)
        enough = n >= min_obs
        rows = np.asarray(idx)
        out.loc[rows, "n"] = n
        if not enough.any():
            continue
        sample = np.where(near[enough], vals[None, :], np.nan)
        r = rows[enough]
        out.loc[r, "mean"] = np.nanmean(sample, axis=1)
        out.loc[r, qcols] = np.nanquantile(sample, list(levels), axis=1).T
    return out


# ---------------------------------------------------------------------------
# metrics (pure)
# ---------------------------------------------------------------------------
def pinball(y: np.ndarray, q: np.ndarray, alpha: float) -> np.ndarray:
    d = y - q
    return np.asarray(np.maximum(alpha * d, (alpha - 1) * d))


def crps_weights(levels: Sequence[float]) -> np.ndarray:
    """Quadrature weights: the width of the alpha-interval nearest each level."""
    lv = np.asarray(levels, dtype="float64")
    edges = np.concatenate([[0.0], (lv[1:] + lv[:-1]) / 2, [1.0]])
    return np.diff(edges)


def crps_quantile(
    y: np.ndarray, quantiles: Sequence[np.ndarray], levels: Sequence[float]
) -> np.ndarray:
    """2 * sum_k w_k * pinball_k - see the module docstring. Equals |y - x| when every
    quantile is x and the levels are symmetric about 0.5."""
    if len(quantiles) != len(levels):
        raise ValueError(f"{len(quantiles)} quantile arrays for {len(levels)} levels")
    w = crps_weights(levels)
    return np.asarray(
        2.0 * sum(wk * pinball(y, q, a) for wk, q, a in zip(w, quantiles, levels, strict=True))
    )


def point_scores(
    y: np.ndarray,
    point: np.ndarray,
    q: Sequence[np.ndarray] | None,
    levels: Sequence[float],
) -> dict[str, Any]:
    err = point - y
    qs = list(q) if q is not None else [point] * len(levels)
    out: dict[str, Any] = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "crps": float(np.mean(crps_quantile(y, qs, levels))),
    }
    if q is not None:
        by = dict(zip(levels, qs, strict=True))
        if 0.1 in by and 0.9 in by:
            out["coverage_80"] = float(np.mean((y >= by[0.1]) & (y <= by[0.9])))
        if 0.05 in by and 0.95 in by:
            out["coverage_90"] = float(np.mean((y >= by[0.05]) & (y <= by[0.95])))
        out["pinball"] = {
            qname(a): float(np.mean(pinball(y, qq, a))) for qq, a in zip(qs, levels, strict=True)
        }
    return out


def _qs(df: pd.DataFrame, prefix: str, levels: Sequence[float]) -> list[np.ndarray]:
    return [df[f"{prefix}{qname(a)[1:]}"].to_numpy() for a in levels]


def score_block(df: pd.DataFrame, levels: Sequence[float]) -> dict[str, Any]:
    """Model on all its rows; all three forecasters on common rows; skill vs baselines."""
    y = df["target"].to_numpy()
    res: dict[str, Any] = {"n_model": int(len(df))}
    if len(df) == 0:
        return res
    res["model_all_rows"] = point_scores(y, df["p50"].to_numpy(), _qs(df, "p", levels), levels)
    res["baseline_coverage"] = {
        "seasonal_naive": float(df["sn"].notna().mean()),
        "climatology": float(df["clim_mean"].notna().mean()),
    }
    common = df[df["sn"].notna() & df["clim_mean"].notna()]
    res["n_common"] = int(len(common))
    if common.empty:
        return res
    yc = common["target"].to_numpy()
    scores = {
        "model": point_scores(yc, common["p50"].to_numpy(), _qs(common, "p", levels), levels),
        "seasonal_naive": point_scores(yc, common["sn"].to_numpy(), None, levels),
        "climatology": point_scores(
            yc, common["clim_mean"].to_numpy(), _qs(common, "clim_q", levels), levels
        ),
    }
    res["common"] = scores
    res["skill"] = {
        b: {
            m: (1 - scores["model"][m] / scores[b][m]) if scores[b][m] > 0 else None
            for m in LOSS_METRICS
        }
        for b in BASELINES
    }
    return res


def reliability(
    prob: np.ndarray, event: np.ndarray, clim_freq: np.ndarray, bins: int, operating_prob: float
) -> dict[str, Any]:
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    table = []
    for b in range(bins):
        m = which == b
        table.append(
            {
                "bin_low": float(edges[b]),
                "bin_high": float(edges[b + 1]),
                "n": int(m.sum()),
                "mean_forecast": float(prob[m].mean()) if m.any() else None,
                "observed_frequency": float(event[m].mean()) if m.any() else None,
            }
        )
    brier = float(np.mean((prob - event) ** 2))
    brier_clim = float(np.mean((clim_freq - event) ** 2))
    flagged = prob >= operating_prob
    hits = int((flagged & event).sum())
    false_alarms = int((flagged & ~event).sum())
    return {
        "n": int(len(prob)),
        "event_rate": float(event.mean()),
        "bins": table,
        "brier": brier,
        "brier_climatology": brier_clim,
        "brier_skill_vs_climatology": (1 - brier / brier_clim) if brier_clim > 0 else None,
        "operating_point_pre_guardrail": {
            "min_exceedance_prob": operating_prob,
            "flagged": int(flagged.sum()),
            "hits": hits,
            "false_alarms": false_alarms,
            "misses": int((~flagged & event).sum()),
            "false_alarm_ratio": false_alarms / int(flagged.sum()) if flagged.any() else None,
            "probability_of_detection": hits / int(event.sum()) if event.any() else None,
        },
    }


def find_losses(block: dict[str, Any], where: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for b, sk in block.get("skill", {}).items():
        for m, s in sk.items():
            if s is not None and s < 0:
                out.append(
                    {
                        **where,
                        "baseline": b,
                        "metric": m,
                        "model": block["common"]["model"][m],
                        "baseline_value": block["common"][b][m],
                        "skill": s,
                        "n": block["n_common"],
                    }
                )
    return out


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def attach_baselines(
    pred: pd.DataFrame,
    obs: pd.DataFrame,
    fold_train_end: dict[str, date],
    ev: dict[str, Any],
    thresholds: dict[str, Any],
    levels: Sequence[float],
) -> pd.DataFrame:
    """Seasonal-naive, climatology (at the model's quantile levels) and the per-reach
    seasonal threshold (engine.thresholds, fitted on the fold's training period)."""
    parts = []
    for fold, g in pred.groupby("fold"):
        g = g.reset_index(drop=True).copy()
        end = fold_train_end[str(fold)]
        g["sn"] = seasonal_naive(obs, g, int(ev["seasonal_naive_window_days"]))
        c = climatology(
            obs,
            g,
            end,
            int(ev["climatology_window_days"]),
            int(ev["min_climatology_obs"]),
            levels,
        ).add_prefix("clim_")
        table = seasonal_thresholds(obs, end, thresholds)
        t = threshold_lookup(g, table, thresholds["seasons"])
        g["threshold"] = t["threshold"].to_numpy()
        g["threshold_clim_freq"] = t["clim_exceed_freq"].to_numpy()
        parts.append(pd.concat([g, c], axis=1))
    return pd.concat(parts, ignore_index=True)


def compute_metrics(
    scored: pd.DataFrame, gs: GBMSettings, ev: dict[str, Any], thresholds: dict[str, Any]
) -> dict[str, Any]:
    levels = gs.quantiles
    op = float(thresholds["guardrails"]["min_exceedance_prob"])
    folds: dict[str, Any] = {}
    losses: list[dict[str, Any]] = []
    order = {"val": 0, "test": 1}
    for fold in sorted(scored["fold"].unique(), key=lambda f: (order.get(f, 9), f)):
        gf = scored[scored["fold"] == fold]
        fres: dict[str, Any] = {}
        for var, g in gf.groupby("variable"):
            vres: dict[str, Any] = {"by_horizon": {}, "by_bucket": {}, "by_observability": {}}
            for h, gh in g.groupby("horizon"):
                blk = score_block(gh, levels)
                vres["by_horizon"][str(h)] = blk
                losses += find_losses(blk, {"fold": fold, "variable": var, "scope": f"h{h}"})
            for b, gb in g.groupby(g["horizon"].map(lambda h: gs.bucket_of(int(h)))):
                blk = score_block(gb, levels)
                vres["by_bucket"][b] = blk
                losses += find_losses(blk, {"fold": fold, "variable": var, "scope": b})
            vres["all_horizons"] = blk = score_block(g, levels)
            losses += find_losses(blk, {"fold": fold, "variable": var, "scope": "all"})

            obs_flag = g["reach_observable"].astype("boolean")
            for label, mask in (
                ("observable", obs_flag == True),  # noqa: E712
                ("driver_only", obs_flag == False),  # noqa: E712
                ("unassessed", obs_flag.isna()),
            ):
                sub = (
                    g[mask.fillna(False).to_numpy()]
                    if label != "unassessed"
                    else g[mask.to_numpy()]
                )
                if label == "unassessed" and sub.empty:
                    continue
                blk = score_block(sub, levels)
                blk["n_reaches"] = int(sub["reach_id"].nunique())
                vres["by_observability"][label] = blk
                losses += find_losses(
                    blk, {"fold": fold, "variable": var, "scope": f"reaches:{label}"}
                )

            reach_rows = []
            for rid, gr in g.groupby("reach_id"):
                c = gr[gr["sn"].notna() & gr["clim_mean"].notna()]
                if len(c) < MIN_REACH_ROWS:
                    continue
                y = c["target"].to_numpy()
                reach_rows.append(
                    {
                        "reach_id": rid,
                        "n": int(len(c)),
                        "mae_model": float(np.mean(np.abs(c["p50"] - y))),
                        "mae_seasonal_naive": float(np.mean(np.abs(c["sn"] - y))),
                        "mae_climatology": float(np.mean(np.abs(c["clim_mean"] - y))),
                    }
                )
            vres["per_reach"] = {
                "min_rows": MIN_REACH_ROWS,
                "reaches_scored": len(reach_rows),
                "model_loses_to_seasonal_naive": sorted(
                    r["reach_id"] for r in reach_rows if r["mae_model"] > r["mae_seasonal_naive"]
                ),
                "model_loses_to_climatology": sorted(
                    r["reach_id"] for r in reach_rows if r["mae_model"] > r["mae_climatology"]
                ),
                "reaches": reach_rows,
            }

            has_thr = g["threshold"].notna().to_numpy()
            qmat = np.column_stack(_qs(g, "p", levels))
            prob = exceedance_probability_multi(qmat, levels, g["threshold"].to_numpy())
            m = has_thr & ~np.isnan(prob)
            event = g["target"].to_numpy()[m] > g["threshold"].to_numpy()[m]
            rel: dict[str, Any] = {
                "threshold_derivation": thresholds["variables"][var]["derivation"],
                "rows_without_threshold": int((~has_thr).sum()),
            }
            if m.any():
                rel.update(
                    reliability(
                        prob[m],
                        event,
                        g["threshold_clim_freq"].to_numpy()[m],
                        int(ev["reliability_bins"]),
                        op,
                    )
                )
            else:
                rel["n"] = 0
            vres["probability"] = rel
            fres[var] = vres
        folds[str(fold)] = fres
    return {"folds": folds, "losses": losses}


def compare_on_shared_rows(
    x: pd.DataFrame, y: pd.DataFrame, gs: GBMSettings, labels: tuple[str, str]
) -> dict[str, Any]:
    """Score two scored prediction sets on the (fold, variable, reach, issue, horizon)
    rows they share. skill = 1 - metric_x / metric_y (> 0: x better)."""
    levels = gs.quantiles
    qc = [qname(a) for a in levels]
    lx, ly = labels
    m = x[[*KEYS, "target", "sn", *qc]].merge(
        y[[*KEYS, *qc]], on=KEYS, suffixes=(f"_{lx}", f"_{ly}"), how="inner"
    )
    out: dict[str, Any] = {}

    def block(d: pd.DataFrame) -> dict[str, Any]:
        t = d["target"].to_numpy()
        res: dict[str, Any] = {"n": int(len(d))}
        for lab in labels:
            qs = [d[f"{c}_{lab}"].to_numpy() for c in qc]
            res[lab] = point_scores(t, d[f"p50_{lab}"].to_numpy(), qs, levels)
        for metric in ("crps", "mae"):
            b = res[ly][metric]
            res[f"skill_{lx}_vs_{ly}_{metric}"] = (1 - res[lx][metric] / b) if b > 0 else None
        sn = d["sn"].notna().to_numpy()
        if sn.any():
            snm = float(np.mean(np.abs(d["sn"].to_numpy()[sn] - t[sn])))
            res["seasonal_naive_mae_on_rows_with_one"] = snm
            for lab in labels:
                mm = float(np.mean(np.abs(d[f"p50_{lab}"].to_numpy()[sn] - t[sn])))
                res[f"skill_{lab}_vs_seasonal_naive_mae"] = (1 - mm / snm) if snm > 0 else None
        return res

    for fold, gf in m.groupby("fold"):
        out[str(fold)] = {}
        for var, g in gf.groupby("variable"):
            v: dict[str, Any] = {"all_horizons": block(g), "by_bucket": {}}
            for b, gb in g.groupby(g["horizon"].map(lambda h: gs.bucket_of(int(h)))):
                v["by_bucket"][b] = block(gb)
            out[str(fold)][str(var)] = v
    return out


def reach_id_shap_share(shap: pd.DataFrame) -> dict[str, Any]:
    """Variant A, evaluation folds: mean |SHAP(reach_id)| / sum over features of mean
    |SHAP|, per fold x variable x quantile, plus reach_id's rank among the features."""
    if shap.empty or "shap__reach_id" not in shap:
        return {}
    variant = shap["variant"] if "variant" in shap else pd.Series("A", index=shap.index)
    a = shap[(variant == "A") & shap["fold"].isin(["val", "test"])]
    cols = [c for c in a.columns if c.startswith("shap__")]
    out: dict[str, Any] = {}
    for (fold, var, q), g in a.groupby(["fold", "variable", "quantile"]):
        mean_abs = g[cols].abs().mean()
        total = float(mean_abs.sum())
        ranked = mean_abs.sort_values(ascending=False)
        out.setdefault(str(fold), {}).setdefault(str(var), {})[str(q)] = {
            "share": float(mean_abs["shap__reach_id"] / total) if total > 0 else None,
            "rank": int(list(ranked.index).index("shap__reach_id")) + 1,
            "of_features": len(cols),
            "top5": [c.removeprefix("shap__") for c in ranked.index[:5]],
            "n": int(len(g)),
        }
    return out


NOT_COMPUTED = {
    "anomaly_precision_recall_f1": "models/anomaly.py is not built, and "
    "data/reference/incidents.csv has no incidents yet.",
    "lead_time_distribution": "needs issued alerts scored against incidents.",
    "false_alarm_rate_after_guardrails": "the guardrails exist (engine/alerts.py) but have "
    "not been replayed over the walk-forward period; the pre-guardrail operating point is "
    "under probability.operating_point_pre_guardrail.",
    "transfer_coimbra_to_pune": "no Pune frame yet (Day 8).",
}


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    if isinstance(o, float | np.floating):
        return None if not math.isfinite(float(o)) else round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, date):
        return o.isoformat()
    return o


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------
def _f(x: Any, w: int = 8, p: int = 3) -> str:
    return (
        f"{'-':>{w}}"
        if x is None or (isinstance(x, float) and not math.isfinite(x))
        else f"{x:>{w}.{p}f}"
    )


def _run_table(L: list[str], folds: dict[str, Any], names: dict[str, str], *, full: bool) -> None:
    for fold, fres in folds.items():
        for var, vres in fres.items():
            L.append(f"\n  FOLD {fold.upper()}  |  {names.get(var, var)}")
            L.append(
                f"  {'scope':<22}{'n':>7}  {'MAE m/sn/cl':>27}  {'CRPS m/sn/cl':>27}  "
                f"{'cov80':>6}  verdict"
            )
            rows = [(f"h{h}", b) for h, b in vres["by_horizon"].items()] if full else []
            rows += list(vres["by_bucket"].items()) + [("ALL", vres["all_horizons"])]
            rows += [(f"reaches:{k}", b) for k, b in vres["by_observability"].items()]
            for scope, b in rows:
                if "common" not in b:
                    L.append(
                        f"  {scope:<22}{b.get('n_common', 0):>7}  no common rows "
                        f"(model rows {b['n_model']}) - not comparable"
                    )
                    continue
                c = b["common"]
                lost = [
                    f"{bl}:{m}"
                    for bl, sk in b["skill"].items()
                    for m, s in sk.items()
                    if s is not None and s < 0
                ]
                verdict = ("LOSES to " + ", ".join(lost)) if lost else "beats both"
                L.append(
                    f"  {scope:<22}{b['n_common']:>7}  "
                    + "  ".join(" ".join(_f(c[f][m]) for f in FORECASTERS) for m in ("mae", "crps"))
                    + f"  {_f(c['model'].get('coverage_80'), 6, 2)}  {verdict}"
                )
            if not full:
                continue
            pr = vres["per_reach"]
            L.append(
                f"  per reach (>= {pr['min_rows']} rows): {pr['reaches_scored']} scored | "
                "model MAE worse than seasonal-naive on "
                f"{len(pr['model_loses_to_seasonal_naive'])}, "
                f"than climatology on {len(pr['model_loses_to_climatology'])}"
            )
            p = vres["probability"]
            if p.get("n"):
                o = p["operating_point_pre_guardrail"]
                L.append(
                    f"  exceedance (> {p['threshold_derivation']} threshold): n {p['n']} | "
                    f"event rate {_f(p['event_rate'], 5, 3)} | Brier {_f(p['brier'], 6, 4)} "
                    f"vs clim {_f(p['brier_climatology'], 6, 4)} "
                    f"(BSS {_f(p['brier_skill_vs_climatology'], 6, 3)}) | "
                    f"at p>={o['min_exceedance_prob']}: {o['flagged']} flagged, {o['hits']} hits, "
                    f"{o['false_alarms']} false alarms"
                )
            else:
                L.append(
                    "  exceedance: no rows with a threshold "
                    f"({p['rows_without_threshold']} without)"
                )


def _comparison_table(
    L: list[str], title: str, comp: dict[str, Any], labels: tuple[str, str], names: dict[str, str]
) -> None:
    lx, ly = labels
    L.append(f"\n  {title}  (shared rows; skill = 1 - {lx}/{ly}, negative = {lx} WORSE)")
    L.append(
        f"  {'fold':<5} {'variable':<26} {'scope':<8} {'n':>7} "
        f"{'CRPS ' + lx:>14} {'CRPS ' + ly:>14} {'skill':>7} {'MAE skill':>9}"
    )
    for fold, fv in comp.items():
        for var, v in fv.items():
            for scope, b in [("ALL", v["all_horizons"]), *v["by_bucket"].items()]:
                L.append(
                    f"  {fold:<5} {names.get(var, var)[:26]:<26} {scope:<8} {b['n']:>7} "
                    f"{_f(b[lx]['crps'], 14, 4)} {_f(b[ly]['crps'], 14, 4)} "
                    f"{_f(b[f'skill_{lx}_vs_{ly}_crps'], 7, 3)} "
                    f"{_f(b[f'skill_{lx}_vs_{ly}_mae'], 9, 3)}"
                )


def format_report(metrics: dict[str, Any]) -> str:
    rule = "=" * 118
    names = {k: v["display_name"] for k, v in metrics.get("variables", {}).items()}
    L = [
        rule,
        "KINGFISHER - BASELINE GBM vs SEASONAL-NAIVE vs CLIMATOLOGY (walk-forward, common rows)",
        rule,
        "  skill = 1 - model/baseline; negative = the model is WORSE. CRPS is the "
        "quantile-quadrature approximation.",
        f"  HEADLINE = {HEADLINE}: variant A, archive (perfect-prognosis) weather.",
    ]
    _run_table(L, metrics["folds"], names, full=True)
    for run, r in metrics.get("runs", {}).items():
        if run == HEADLINE:
            continue
        L.append(f"\n{'-' * 118}\n  RUN {run}")
        _run_table(L, r["folds"], names, full=False)
    for weather, comp in metrics.get("variant_comparison", {}).items():
        _comparison_table(L, f"VARIANT A vs B - {weather} weather", comp, ("A", "B"), names)
    for variant, comp in metrics.get("weather_comparison", {}).items():
        _comparison_table(
            L,
            f"ORACLE vs AS-ISSUED weather - variant {variant}",
            comp,
            ("ORACLE", "ASISSUED"),
            names,
        )
    share = metrics.get("reach_id_shap_share", {})
    if share:
        L.append("\n  REACH_ID SHAP SHARE (variant A): mean|SHAP(reach_id)| / sum mean|SHAP|")
        for fold, fv in share.items():
            for var, qv in fv.items():
                for q, s in qv.items():
                    L.append(
                        f"  {fold:<5} {names.get(var, var)[:26]:<26} {q:<4} share "
                        f"{_f(s['share'], 6, 3)} | rank {s['rank']}/{s['of_features']} | "
                        f"top: {', '.join(s['top5'])}"
                    )
    L.append("\n" + rule)
    losses = metrics["losses"]
    if losses:
        L.append(f"  THE MODEL LOSES TO A BASELINE IN {len(losses)} COMPARISONS (all runs):")
        for x in losses:
            L.append(
                f"    [{x.get('run', HEADLINE)} {x['fold']}] {x['variable']:<16} "
                f"{x['scope']:<22} {x['metric']:<5} model {x['model']:.4f} vs {x['baseline']} "
                f"{x['baseline_value']:.4f} (skill {x['skill']:+.3f}, n={x['n']})"
            )
    else:
        L.append("  Every model run beats both baselines on every scored comparison.")
    L.append("  Not computed: " + "; ".join(metrics["not_computed"]))
    L.append(rule)
    return "\n".join(L)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLORS = {"model": "#2a78d6", "seasonal_naive": "#eb6834", "climatology": "#1baf7a"}
LABELS = {
    "model": "LightGBM (P50)",
    "seasonal_naive": "Seasonal-naive",
    "climatology": "Climatology",
}
MARKERS = {
    "model": "o",
    "seasonal_naive": "s",
    "climatology": "^",
}  # identity is never colour alone


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for an observed frequency k/n - sparse reliability bins are
    shown with their uncertainty, not as bare 0/1 points."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def write_figures(metrics: dict[str, Any], out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    written = []
    names = {k: v["display_name"] for k, v in metrics.get("variables", {}).items()}

    def save(fig: Any, name: str) -> None:
        fig.patch.set_facecolor(SURFACE)
        fig.savefig(out / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(name)

    for fold, fres in metrics["folds"].items():
        for var, v in fres.items():
            hz = [(int(h), b) for h, b in v["by_horizon"].items() if "common" in b]
            if hz:
                hs = [h for h, _ in hz]
                fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
                for ax, metric in zip(axes, ("mae", "crps"), strict=True):
                    _style(ax)
                    for f in FORECASTERS:
                        ys = [b["common"][f][metric] for _, b in hz]
                        ax.plot(
                            hs, ys, color=COLORS[f], lw=2, marker=MARKERS[f], ms=5, label=LABELS[f]
                        )
                    ax.set_xlabel("horizon (days)", color=INK2, fontsize=9)
                    ax.set_title(
                        metric.upper() + (" (quantile approx.)" if metric == "crps" else ""),
                        color=INK,
                        fontsize=10,
                        loc="left",
                    )
                    ax.set_xticks(hs)
                    ax.set_ylim(bottom=0)
                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(
                    handles,
                    labels,
                    frameon=False,
                    fontsize=8,
                    labelcolor=INK2,
                    ncol=3,
                    loc="upper right",
                    bbox_to_anchor=(1.0, 1.02),
                )
                fig.suptitle(
                    f"{names.get(var, var)} - error by horizon, {fold} fold (lower is better)",
                    color=INK,
                    fontsize=11,
                    x=0.01,
                    y=1.02,
                    ha="left",
                )
                save(fig, f"error_by_horizon_{fold}_{var}.png")

                fig, ax = plt.subplots(figsize=(6, 3.6))
                _style(ax)
                ax.axhline(0.8, color=INK2, lw=1, ls="--")
                ax.annotate(
                    "nominal 0.80",
                    (hs[0], 0.8),
                    xytext=(0, 4),
                    textcoords="offset points",
                    color=INK2,
                    fontsize=8,
                )
                for f in ("model", "climatology"):
                    ys = [b["common"][f].get("coverage_80") for _, b in hz]
                    ax.plot(
                        hs,
                        ys,
                        color=COLORS[f],
                        lw=2,
                        marker=MARKERS[f],
                        ms=5,
                        label=LABELS[f].replace(" (P50)", " P10-P90")
                        + ("" if f == "model" else " P10-P90"),
                    )
                ax.set_ylim(0, 1)
                ax.set_xticks(hs)
                ax.set_xlabel("horizon (days)", color=INK2, fontsize=9)
                ax.set_ylabel("share of observations inside interval", color=INK2, fontsize=9)
                ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
                ax.set_title(
                    f"{names.get(var, var)} - 80% interval coverage, {fold} fold",
                    color=INK,
                    fontsize=10,
                    loc="left",
                )
                save(fig, f"coverage_by_horizon_{fold}_{var}.png")

            p = v["probability"]
            if p.get("n"):
                bins = [b for b in p["bins"] if b["n"]]
                fig, (ax, axh) = plt.subplots(
                    2, 1, figsize=(5.2, 6), height_ratios=(3, 1), sharex=True
                )
                _style(ax)
                _style(axh)
                ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="perfect calibration")
                xs = [b["mean_forecast"] for b in bins]
                ys = [b["observed_frequency"] for b in bins]
                ci = [wilson(round(b["observed_frequency"] * b["n"]), b["n"]) for b in bins]
                ax.errorbar(
                    xs,
                    ys,
                    yerr=[
                        [max(0.0, y - lo) for y, (lo, _) in zip(ys, ci, strict=True)],
                        [max(0.0, hi - y) for y, (_, hi) in zip(ys, ci, strict=True)],
                    ],
                    color=COLORS["model"],
                    lw=1.5,
                    marker="o",
                    ms=6,
                    capsize=3,
                    label="LightGBM, 95% Wilson interval",
                )
                for x_, y_, b in zip(xs, ys, bins, strict=True):
                    ax.annotate(
                        f"n={b['n']}",
                        (x_, y_),
                        xytext=(6, -10),
                        textcoords="offset points",
                        color=INK2,
                        fontsize=7,
                    )
                ax.axhline(
                    p["event_rate"],
                    color=COLORS["climatology"],
                    lw=1.2,
                    label=f"observed event rate {p['event_rate']:.2f}",
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1.02)
                ax.set_ylabel("observed frequency", color=INK2, fontsize=9)
                ax.legend(
                    frameon=False,
                    fontsize=8,
                    labelcolor=INK2,
                    loc="lower left",
                    bbox_to_anchor=(0, 1.0),
                    ncol=1,
                )
                ax.set_title(
                    f"{names.get(var, var)} - reliability, {fold} fold\n"
                    f"P(obs > per-reach seasonal threshold), n={p['n']}, "
                    f"Brier {p['brier']:.3f} (clim {p['brier_climatology']:.3f})",
                    color=INK,
                    fontsize=10,
                    loc="left",
                    pad=62,
                )
                centers = [(b["bin_low"] + b["bin_high"]) / 2 for b in p["bins"]]
                axh.bar(centers, [b["n"] for b in p["bins"]], width=0.09, color=COLORS["model"])
                axh.set_yscale("symlog", linthresh=1)
                axh.set_ylabel("forecasts", color=INK2, fontsize=9)
                axh.set_xlabel("forecast exceedance probability", color=INK2, fontsize=9)
                save(fig, f"reliability_{fold}_{var}.png")

            groups = [(k, b) for k, b in v["by_observability"].items() if "common" in b]
            fig, ax = plt.subplots(figsize=(6.4, 3.6))
            _style(ax)
            labels = [k for k, _ in v["by_observability"].items()]
            x = np.arange(len(labels))
            for i, f in enumerate(FORECASTERS):
                ys = [
                    v["by_observability"][k]["common"][f]["mae"]
                    if "common" in v["by_observability"][k]
                    else 0
                    for k in labels
                ]
                ax.bar(
                    x + (i - 1) * 0.26,
                    ys,
                    width=0.24,
                    color=COLORS[f],
                    label=LABELS[f],
                    hatch=("", "//", "..")[i],
                    edgecolor=SURFACE,
                    linewidth=0,
                )
            for j, k in enumerate(labels):
                b = v["by_observability"][k]
                note = (
                    f"n={b.get('n_common', 0)}, {b['n_reaches']} reaches"
                    if "common" in b
                    else f"not measurable\n{b.get('n_model', 0)} scored targets"
                )
                ax.annotate(
                    note,
                    (x[j], 0),
                    xytext=(0, -26),
                    textcoords="offset points",
                    ha="center",
                    color=INK2,
                    fontsize=8,
                )
            ax.set_xticks(x, labels)
            ax.set_ylabel("MAE (common rows)", color=INK2, fontsize=9)
            ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
            ax.set_title(
                f"{names.get(var, var)} - skill by reach observability, {fold} fold",
                color=INK,
                fontsize=10,
                loc="left",
            )
            if groups or labels:
                save(fig, f"skill_by_observability_{fold}_{var}.png")
            else:
                plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# forecasts table
# ---------------------------------------------------------------------------
FORECAST_COLUMNS = [
    "reach_id",
    "issued_date",
    "target_date",
    "variable",
    "p05",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
    "model_version",
    "variant",
    "weather",
    "fit",
    "horizon",
]


def forecast_records(pred: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward predictions -> rows for the `forecasts` table. Quantile columns the
    model did not produce are NULL; the rearranged quantiles are already ordered."""
    out = pd.DataFrame(
        {
            "reach_id": pred["reach_id"].astype(str),
            "issued_date": pd.to_datetime(pred["issued_date"]).dt.date,
            "target_date": pd.to_datetime(pred["target_date"]).dt.date,
            "variable": pred["variable"],
            "model_version": pred["model_version"],
            "variant": pred.get("variant", "A"),
            "weather": pred.get("weather", "ORACLE"),
            "fit": "wf-" + pred["fold"].astype(str),
            "horizon": pred["horizon"].astype(int),
        }
    )
    for q in ("p05", "p10", "p25", "p50", "p75", "p90", "p95"):
        out[q] = pred[q].to_numpy() if q in pred else np.nan
    if out.duplicated(
        ["reach_id", "issued_date", "target_date", "variable", "model_version", "weather"]
    ).any():
        raise ValueError("duplicate forecast keys - one model scored the same row twice")
    return out[FORECAST_COLUMNS]


def persist_forecasts(pred: pd.DataFrame) -> int:
    """Replace every row of these model versions, then bulk-insert."""
    from sqlalchemy import text

    from core.db import bulk_upsert, session_scope

    rows = forecast_records(pred)
    versions = sorted(rows["model_version"].unique())
    with session_scope() as session:
        session.execute(
            text("DELETE FROM forecasts WHERE model_version = ANY(:v)"), {"v": versions}
        )
    written = bulk_upsert(
        rows,
        "forecasts",
        ["reach_id", "issued_date", "target_date", "variable", "model_version", "weather"],
    )
    log.info("evaluate.forecasts_written", rows=written, versions=len(versions))
    return written


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def _fold_of(fit_name: str) -> str | None:
    last = fit_name.split(".")[-1]
    return last.removeprefix("wf-") if last.startswith("wf-") else None


def evaluate(
    city: str, *, results_dir: Path = RESULTS_DIR, write_db: bool = True
) -> dict[str, Any]:
    fs, gs = FrameSettings.from_config(), GBMSettings.from_config()
    cfg = load_config("modelling")
    ev, thresholds = cfg["evaluation"], load_config("thresholds")
    run_path = PROCESSED_DIR / f"gbm_run_{city}.json"
    pred_path = PROCESSED_DIR / f"gbm_eval_{city}.parquet"
    if not run_path.exists() or not pred_path.exists():
        raise FileNotFoundError(
            f"no baseline run for {city} - run `python -m models.baseline_gbm --city {city}`"
        )
    run = json.loads(run_path.read_text())
    if file_sha256(frame_path(city)) != run["frame_sha256"]:
        raise RuntimeError(
            "the modelling frame changed since the model was trained - retrain before evaluating"
        )

    with stage(log, "evaluate", city=city) as counters:
        pred = pd.read_parquet(pred_path)
        if not pred.empty:
            if "variant" not in pred:
                pred["variant"] = "A"
            if "weather" not in pred:
                pred["weather"] = "ORACLE"
        folds_expected = sorted(
            {f for n in run["fits"] if (f := _fold_of(n))}, key=lambda f: (f != "val", f)
        )
        headline = (
            pred[(pred["variant"] == "A") & (pred["weather"] == "ORACLE")]
            if not pred.empty
            else pred
        )
        empty = [f for f in folds_expected if headline.empty or not (headline["fold"] == f).any()]
        if empty:
            raise NoEvaluationData(
                f"walk-forward fold(s) {empty} have no OK Sentinel-2 targets to score. "
                "Refusing to publish metrics from a partial or empty evaluation - run "
                "pipeline.l1_satellite over 2024-2026 and rebuild the frame."
            )
        frame = load_frame(city)
        obs = observations_from_frame(frame, fs.variables)
        train_end: dict[str, date] = {}
        for n, fit in run["fits"].items():
            if (f := _fold_of(n)) is not None:
                train_end[f] = date.fromisoformat(fit["train_end"])

        runs: dict[str, Any] = {}
        scored_by_run: dict[str, pd.DataFrame] = {}
        all_losses: list[dict[str, Any]] = []
        for (variant, weather), sub in pred.groupby(["variant", "weather"]):
            name = f"{variant}/{weather}"
            scored = attach_baselines(sub, obs, train_end, ev, thresholds, gs.quantiles)
            m = compute_metrics(scored, gs, ev, thresholds)
            runs[name] = {"folds": m["folds"], "n_forecasts": int(len(sub))}
            all_losses += [{"run": name, **x} for x in m["losses"]]
            scored_by_run[name] = scored

        variant_comparison = {
            w: compare_on_shared_rows(
                scored_by_run[f"A/{w}"], scored_by_run[f"B/{w}"], gs, ("A", "B")
            )
            for w in ("ORACLE", "ASISSUED")
            if f"A/{w}" in scored_by_run and f"B/{w}" in scored_by_run
        }
        weather_comparison = {
            v: compare_on_shared_rows(
                scored_by_run[f"{v}/ORACLE"],
                scored_by_run[f"{v}/ASISSUED"],
                gs,
                ("ORACLE", "ASISSUED"),
            )
            for v in ("A", "B")
            if f"{v}/ORACLE" in scored_by_run and f"{v}/ASISSUED" in scored_by_run
        }
        shap_path = PROCESSED_DIR / f"gbm_shap_{city}.parquet"
        shap_share = reach_id_shap_share(pd.read_parquet(shap_path)) if shap_path.exists() else {}

        not_computed = dict(NOT_COMPUTED)
        if not weather_comparison:
            not_computed["asissued_weather"] = (
                "pipeline.asissued_weather has not been built - only archive (perfect-"
                "prognosis) weather was scored; live skill will be lower."
            )
        metrics = {
            "city": city,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "model_versions": {k: v["version"] for k, v in run["fits"].items()},
            "frame_sha256": run["frame_sha256"],
            "splits": {f: {"train_end": str(train_end[f])} for f in folds_expected},
            "variables": {
                v: {"display_name": c["display_name"], "units": c["units"]}
                for v, c in thresholds["variables"].items()
            },
            "variants": run.get("variants", {}),
            "dropped_features": run.get("dropped_features", {}),
            "quantile_levels": list(gs.quantiles),
            "config": {
                "evaluation": ev,
                "thresholds": thresholds["variables"],
                "seasons": thresholds["seasons"],
                "min_exceedance_prob": thresholds["guardrails"]["min_exceedance_prob"],
            },
            "quantiles_crossed_rearranged": {
                k: v.get("quantiles_crossed") for k, v in run["fits"].items() if _fold_of(k)
            },
            "headline_run": HEADLINE,
            "folds": runs[HEADLINE]["folds"],
            "losses": all_losses,
            "runs": runs,
            "variant_comparison": variant_comparison,
            "weather_comparison": weather_comparison,
            "reach_id_shap_share": shap_share,
            "not_computed": not_computed,
        }
        metrics = dict(_clean(metrics))
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        figures = write_figures(metrics, results_dir / "figures")
        written = persist_forecasts(pred) if write_db else 0
        counters.record(
            rows_in=len(pred),
            rows_out=sum(len(s) for s in scored_by_run.values()),
            losses=len(all_losses),
            figures=len(figures),
            forecasts_written=written,
        )
    print(format_report(metrics))
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Walk-forward evaluation vs baselines")
    parser.add_argument("--city", default="coimbra")
    parser.add_argument("--no-db", action="store_true", help="do not write the forecasts table")
    parser.add_argument(
        "--head-to-head",
        action="store_true",
        help="P5.4: all forecasters, assimilation + calibration, the production gate "
        "(models/head_to_head.py)",
    )
    args = parser.parse_args(argv)
    if args.head_to_head:
        from models.head_to_head import run

        run(args.city)
        return 0
    evaluate(args.city, write_db=not args.no_db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
