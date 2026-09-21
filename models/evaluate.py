"""Walk-forward evaluation of the LightGBM baseline (MASTERSPEC 11). Every metric is
reported, favourable or not.

Run (after `python -m models.baseline_gbm --city coimbra`):

    python -m models.evaluate --city coimbra

Writes results/metrics.json and results/figures/*.png and prints the comparison table.
Wherever the model is worse than a baseline, the table says LOSES and the loss is listed
again at the end and in metrics.json["losses"]. Nothing is dropped to make that list
shorter.

FORECASTERS - scored on the same (reach, issue date, horizon) rows
------------------------------------------------------------------
  model           P10/P50/P90 from models.baseline_gbm (walk-forward fits)
  seasonal_naive  the observation closest to the target's day-of-year in the most recent
                  previous year that has one within +/- seasonal_naive_window_days.
                  A point forecast. Always dated before the issue date.
  climatology     the reach's training-fold observations within +/- climatology_window_days
                  of the target's day-of-year, all training years pooled: mean as the point,
                  empirical 10/50/90% quantiles as the interval. NULL if fewer than
                  min_climatology_obs - counted, never filled.

The headline skill comparison uses rows where all three forecasters exist ("common
rows"); the model is also reported on all of its rows, and each baseline's coverage is
stated, so a baseline that is missing where the model is weak cannot flatter it.

METRICS
-------
  MAE, RMSE        on the point forecast (model P50, climatology mean, seasonal-naive value)
  CRPS             quantile-score approximation over alpha in {0.1, 0.5, 0.9}:
                   CRPS ~= (2/K) * sum_k pinball_alpha_k(y, q_k). Applied identically to all
                   three forecasters; for a point forecast it reduces exactly to MAE. With
                   three quantiles it is coarse - it ranks forecasters fairly, but it is not
                   the integral CRPS and is not labelled as one.
  coverage_80      fraction of observations inside P10-P90 (nominal 0.80)
  skill vs B       1 - metric_model / metric_B   (> 0: model better)

PROBABILITY CALIBRATION
-----------------------
The event is observed > threshold, the threshold being what config/thresholds.yaml says:
the absolute value if one is set, otherwise the reach's seasonal percentile of its
training-fold observations (same window as climatology). The forecast probability comes
from engine.probability.exceedance_probability - the same function the alert engine
alerts with. Reported: reliability-diagram bins, Brier score, Brier skill vs the
climatological exceedance frequency, and the hit / false-alarm counts at the
min_exceedance_prob operating point BEFORE guardrails (the alert engine's guardrails do
not exist yet).
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.config import load_config
from core.logging import get_logger, stage
from core.settings import REPO_ROOT
from engine.probability import exceedance_probability
from models.baseline_gbm import GBMSettings, file_sha256, frame_path, load_frame
from pipeline.build_dataset import PROCESSED_DIR, FrameSettings

log = get_logger(__name__)

RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FORECASTERS = ("model", "seasonal_naive", "climatology")
BASELINES = ("seasonal_naive", "climatology")
ALPHAS = (0.1, 0.5, 0.9)
LOSS_METRICS = ("mae", "rmse", "crps")
MIN_REACH_ROWS = 5  # per-reach comparison needs at least this many common rows


class NoEvaluationData(RuntimeError):
    """There is nothing honest to score."""


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
    threshold_percentile: float | None,
) -> pd.DataFrame:
    """Per target row: mean, q10, q50, q90, n, and - if threshold_percentile is set -
    the seasonal threshold and the climatological exceedance frequency above it.
    Uses only observations dated <= train_end."""
    cols = ["mean", "q10", "q50", "q90", "n", "threshold", "clim_exceed_freq"]
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
        out.loc[r, ["q10", "q50", "q90"]] = np.nanquantile(sample, ALPHAS, axis=1).T
        if threshold_percentile is not None:
            thr = np.nanquantile(sample, threshold_percentile, axis=1)
            out.loc[r, "threshold"] = thr
            out.loc[r, "clim_exceed_freq"] = (sample > thr[:, None]).sum(axis=1) / n[enough]
    return out


# ---------------------------------------------------------------------------
# metrics (pure)
# ---------------------------------------------------------------------------
def pinball(y: np.ndarray, q: np.ndarray, alpha: float) -> np.ndarray:
    d = y - q
    return np.asarray(np.maximum(alpha * d, (alpha - 1) * d))


def crps_quantile(y: np.ndarray, q10: np.ndarray, q50: np.ndarray, q90: np.ndarray) -> np.ndarray:
    """(2/K) * sum_k pinball - see module docstring. Equals |y - x| when q10=q50=q90=x."""
    qs = (q10, q50, q90)
    return np.asarray(
        2.0 * np.mean([pinball(y, q, a) for q, a in zip(qs, ALPHAS, strict=True)], axis=0)
    )


def point_scores(
    y: np.ndarray, point: np.ndarray, q: tuple[np.ndarray, ...] | None
) -> dict[str, Any]:
    err = point - y
    q = q or (point, point, point)
    out: dict[str, Any] = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "crps": float(np.mean(crps_quantile(y, *q))),
    }
    if q[0] is not point:
        out["coverage_80"] = float(np.mean((y >= q[0]) & (y <= q[2])))
        out["pinball"] = {
            f"p{round(a * 100):02d}": float(np.mean(pinball(y, qq, a)))
            for qq, a in zip(q, ALPHAS, strict=True)
        }
    return out


def score_block(df: pd.DataFrame) -> dict[str, Any]:
    """Model on all its rows; all three forecasters on common rows; skill vs baselines."""
    y = df["target"].to_numpy()
    res: dict[str, Any] = {"n_model": int(len(df))}
    if len(df) == 0:
        return res
    res["model_all_rows"] = point_scores(
        y, df["p50"].to_numpy(), (df["p10"].to_numpy(), df["p50"].to_numpy(), df["p90"].to_numpy())
    )
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
        "model": point_scores(
            yc,
            common["p50"].to_numpy(),
            (common["p10"].to_numpy(), common["p50"].to_numpy(), common["p90"].to_numpy()),
        ),
        "seasonal_naive": point_scores(yc, common["sn"].to_numpy(), None),
        "climatology": point_scores(
            yc,
            common["clim_mean"].to_numpy(),
            (
                common["clim_q10"].to_numpy(),
                common["clim_q50"].to_numpy(),
                common["clim_q90"].to_numpy(),
            ),
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
) -> pd.DataFrame:
    parts = []
    for fold, g in pred.groupby("fold"):
        g = g.reset_index(drop=True).copy()
        g["sn"] = seasonal_naive(obs, g, int(ev["seasonal_naive_window_days"]))
        clim_parts = []
        for var, gv in g.groupby("variable"):
            tcfg = thresholds["variables"][var]
            absolute = tcfg.get("absolute_value")
            pct = None if absolute is not None else float(tcfg["percentile"])
            c = climatology(
                obs,
                gv,
                fold_train_end[str(fold)],
                int(ev["climatology_window_days"]),
                int(ev["min_climatology_obs"]),
                pct,
            )
            c.index = gv.index
            if absolute is not None:
                c["threshold"] = float(absolute)
                c["clim_exceed_freq"] = np.nan
            clim_parts.append(c)
        c = pd.concat(clim_parts).sort_index().add_prefix("clim_")
        parts.append(pd.concat([g, c], axis=1))
    return pd.concat(parts, ignore_index=True)


def bucket_label(h: int, gs: GBMSettings) -> str:
    return gs.bucket_of(h)


def compute_metrics(
    scored: pd.DataFrame, gs: GBMSettings, ev: dict[str, Any], thresholds: dict[str, Any]
) -> dict[str, Any]:
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
                blk = score_block(gh)
                vres["by_horizon"][str(h)] = blk
                losses += find_losses(blk, {"fold": fold, "variable": var, "scope": f"h{h}"})
            for b, gb in g.groupby(g["horizon"].map(lambda h: bucket_label(int(h), gs))):
                blk = score_block(gb)
                vres["by_bucket"][b] = blk
                losses += find_losses(blk, {"fold": fold, "variable": var, "scope": b})
            vres["all_horizons"] = blk = score_block(g)
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
                blk = score_block(sub)
                blk["n_reaches"] = int(sub["reach_id"].nunique())
                vres["by_observability"][label] = blk
                losses += find_losses(
                    blk, {"fold": fold, "variable": var, "scope": f"reaches:{label}"}
                )

            # per reach
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

            # probability calibration
            has_thr = g["clim_threshold"].notna().to_numpy()
            prob = exceedance_probability(g["p10"], g["p50"], g["p90"], g["clim_threshold"])
            m = has_thr & ~np.isnan(prob)
            event = g["target"].to_numpy()[m] > g["clim_threshold"].to_numpy()[m]
            rel: dict[str, Any] = {
                "threshold_derivation": thresholds["variables"][var]["derivation"],
                "rows_without_threshold": int((~has_thr).sum()),
            }
            if m.any():
                rel.update(
                    reliability(
                        prob[m],
                        event,
                        g["clim_clim_exceed_freq"].to_numpy()[m],
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


NOT_COMPUTED = {
    "anomaly_precision_recall_f1": "models/anomaly.py is Day 5, and data/reference/incidents.csv "
    "has no incidents yet.",
    "lead_time_distribution": "needs issued alerts from engine/alerts.py (Day 4) scored "
    "against incidents.",
    "false_alarm_rate_after_guardrails": "engine/alerts.py guardrails not built yet; the "
    "pre-guardrail operating point is under probability.operating_point_pre_guardrail.",
    "transfer_coimbra_to_pune": "no Pune frame yet (Day 8).",
    "live_forecast_skill": "all skill here uses archive weather for t+h drivers (perfect "
    "prognosis). Skill with real Open-Meteo forecasts will be lower.",
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


def format_report(metrics: dict[str, Any]) -> str:
    rule = "=" * 118
    L = [
        rule,
        "KINGFISHER - BASELINE GBM vs SEASONAL-NAIVE vs CLIMATOLOGY (walk-forward, common rows)",
        rule,
        "  skill = 1 - model/baseline; negative = the model is WORSE. "
        "CRPS is the 3-quantile approximation.",
        "  Future drivers are archive weather (perfect prognosis) - live skill will be lower.",
    ]
    for fold, fres in metrics["folds"].items():
        for var, vres in fres.items():
            L.append(f"\n  FOLD {fold.upper()}  |  {var}")
            L.append(
                f"  {'scope':<22}{'n':>7}  {'MAE m/sn/cl':>27}  {'CRPS m/sn/cl':>27}  "
                f"{'cov80':>6}  verdict"
            )
            rows = [(f"h{h}", b) for h, b in vres["by_horizon"].items()]
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
    L.append("\n" + rule)
    losses = metrics["losses"]
    if losses:
        L.append(f"  THE MODEL LOSES TO A BASELINE IN {len(losses)} COMPARISONS:")
        for x in losses:
            L.append(
                f"    [{x['fold']}] {x['variable']:<16} {x['scope']:<22} {x['metric']:<5} "
                f"model {x['model']:.4f} vs {x['baseline']} {x['baseline_value']:.4f} "
                f"(skill {x['skill']:+.3f}, n={x['n']})"
            )
    else:
        L.append("  The model beats both baselines on every scored comparison.")
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
                        metric.upper() + (" (3-quantile approx.)" if metric == "crps" else ""),
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
                    f"{var} - error by horizon, {fold} fold (lower is better)",
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
                    f"{var} - 80% interval coverage, {fold} fold",
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
                    f"{var} - reliability, {fold} fold\nP(obs > seasonal threshold), n={p['n']}, "
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
                f"{var} - skill by reach observability, {fold} fold",
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
# I/O
# ---------------------------------------------------------------------------
def evaluate(city: str, *, results_dir: Path = RESULTS_DIR) -> dict[str, Any]:
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
        folds_expected = [n.removeprefix("wf-") for n in run["fits"] if n.startswith("wf-")]
        empty = [f for f in folds_expected if pred.empty or not (pred["fold"] == f).any()]
        if empty:
            raise NoEvaluationData(
                f"walk-forward fold(s) {empty} have no OK Sentinel-2 targets to score. "
                "Refusing to publish metrics from a partial or empty evaluation - run "
                "pipeline.l1_satellite over 2024-2026 and rebuild the frame."
            )
        frame = load_frame(city)
        obs = observations_from_frame(frame, fs.variables)
        train_end = {
            f: date.fromisoformat(run["fits"][f"wf-{f}"]["train_end"]) for f in folds_expected
        }
        scored = attach_baselines(pred, obs, train_end, ev, thresholds)
        metrics = compute_metrics(scored, gs, ev, thresholds)
        metrics = {
            "city": city,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "model_versions": {k: v["version"] for k, v in run["fits"].items()},
            "frame_sha256": run["frame_sha256"],
            "splits": {f: {"train_end": str(train_end[f])} for f in folds_expected},
            "config": {
                "evaluation": ev,
                "thresholds": thresholds["variables"],
                "min_exceedance_prob": thresholds["guardrails"]["min_exceedance_prob"],
            },
            "quantiles_crossed_rearranged": {
                k: v.get("quantiles_crossed") for k, v in run["fits"].items() if k.startswith("wf-")
            },
            "not_computed": NOT_COMPUTED,
            **metrics,
        }
        metrics = dict(_clean(metrics))
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        figures = write_figures(metrics, results_dir / "figures")
        counters.record(
            rows_in=len(pred),
            rows_out=len(scored),
            losses=len(metrics["losses"]),
            figures=len(figures),
        )
    print(format_report(metrics))
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Walk-forward evaluation vs baselines")
    parser.add_argument("--city", default="coimbra")
    args = parser.parse_args(argv)
    evaluate(args.city)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
