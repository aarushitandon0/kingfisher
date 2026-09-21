"""P5.4 - head-to-head, the production gate, attribution and scenario sensitivity.

Run (after models.baseline_gbm and `models.ealstm hindcast`):

    python -m models.head_to_head --city coimbra
    python -m models.evaluate --city coimbra --head-to-head      (same thing)

FORECASTERS - one harness, same splits, same rows, all calibrated identically
----------------------------------------------------------------------------
  seasonal_naive   point forecast (no interval, so nothing to calibrate; CRPS = MAE)
  climatology      reach-season empirical quantiles             + CQR
  lightgbm         variant A walk-forward fits (config gate.lightgbm_variant) + CQR
  lightgbm+assim   the same, residual-assimilated (models/assimilation.py)    + CQR
  ealstm           5-seed CMAL ensemble, hindcast simulation    + CQR
  ealstm+assim     the same, residual-assimilated                             + CQR
tau, the conformal adjustments and the isotonic maps are fitted on 2024 ONLY
(models/calibration.py refuses anything else) and applied to 2024 and 2025-26.

Every forecaster here uses ARCHIVE weather for t+1..t+h. metrics.json carries
"weather_label": "hindcast with observed weather - upper bound on live skill".

REPORTED - CRPS (primary; quantile quadrature, models/evaluate.py), MAE, RMSE, 80%
coverage, reliability bins + Brier, by horizon bucket, by observability class, on the
spatial-holdout reaches (EA-LSTM never trained on them; LightGBM DID - said in the
output), per fold, on shared rows.

THE GATE - decided on 2024 VALIDATION, never on test (config/modelling.yaml gate)
EA-LSTM becomes production iff, for EVERY target, ealstm+assim's CRPS beats
lightgbm+assim's in >= 2 of the 3 horizon buckets on shared rows AND its 80% coverage is
no further from 0.80 than LightGBM's. Written to results/gate.json BEFORE the test fold
is scored; test is then scored once and every model is reported, whatever the outcome.

ATTRIBUTION - only if EA-LSTM is production: integrated gradients (models.ealstm.explain),
with a storm sanity check (precipitation features must dominate on the wettest day).

SCENARIO SENSITIVITY - both models: +10 points imperviousness_pct on every reach,
re-infer, report the sign and size of the turbidity response. A model that moves the
wrong way on most reaches is printed as such - the scenario engine relies on this.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pandas as pd

from core.config import load_config
from core.logging import get_logger, stage
from core.settings import REPO_ROOT
from engine.probability import DEFAULT_LEVELS, exceedance_probability_multi
from models import assimilation as assim
from models import calibration as cal
from models.evaluate import (
    _clean,
    attach_baselines,
    observations_from_frame,
    point_scores,
    reliability,
)

log = get_logger(__name__)

RESULTS_DIR = REPO_ROOT / "results"
LEVELS = tuple(DEFAULT_LEVELS)
Q = [f"p{round(a * 100):02d}" for a in LEVELS]
KEYS = ["fold", "variable", "reach_id", "issued_date", "horizon"]
HINDCAST_LABEL = "hindcast with observed weather - upper bound on live skill"
PRECIP_FEATURES = frozenset(
    {
        "precip_mm",
        "precip_max_hourly",
        "precip_duration_h",
        "api_7",
        "api_14",
        "api_30",
        "first_flush_index",
        "antecedent_dry_days",
    }
)
MODELS = ("seasonal_naive", "climatology", "lightgbm", "lightgbm+assim", "ealstm", "ealstm+assim")


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------
def common_keys(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Rows every forecaster has (the KEYS intersection)."""
    keys = None
    for df in frames.values():
        k = df[KEYS].assign(reach_id=df["reach_id"].astype(str)).drop_duplicates()
        keys = k if keys is None else keys.merge(k, on=KEYS, how="inner")
    return keys if keys is not None else pd.DataFrame(columns=KEYS)


def restrict(df: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    d = df.assign(reach_id=df["reach_id"].astype(str))
    return d.merge(keys, on=KEYS, how="inner")


def exceedance(rows: pd.DataFrame) -> np.ndarray:
    q = rows[Q].to_numpy(dtype="float64")
    return exceedance_probability_multi(
        np.sort(q, axis=1), LEVELS, rows["threshold"].to_numpy(dtype="float64")
    )


def block_scores(
    rows: pd.DataFrame, *, point_only: bool, prob: np.ndarray | None, op: float, bins: int
) -> dict[str, Any]:
    if rows.empty:
        return {"n": 0}
    y = rows["target"].to_numpy(dtype="float64")
    if point_only:
        s = point_scores(y, rows["p50"].to_numpy(dtype="float64"), None, LEVELS)
    else:
        s = point_scores(
            y, rows["p50"].to_numpy(dtype="float64"), [rows[c].to_numpy() for c in Q], LEVELS
        )
    s["n"] = int(len(rows))
    s["n_reaches"] = int(rows["reach_id"].nunique())
    if prob is not None:
        thr = rows["threshold"].to_numpy(dtype="float64")
        m = ~np.isnan(prob) & ~np.isnan(thr)
        if m.any():
            r = reliability(
                prob[m], y[m] > thr[m], rows["threshold_clim_freq"].to_numpy()[m], bins, op
            )
            s["probability"] = {
                k: r[k]
                for k in (
                    "n",
                    "event_rate",
                    "brier",
                    "brier_climatology",
                    "brier_skill_vs_climatology",
                    "bins",
                )
            }
    return s


def score_model(
    rows: pd.DataFrame,
    buckets: Mapping[str, Sequence[int]],
    *,
    point_only: bool,
    prob: np.ndarray | None,
    holdout: set[str],
    op: float,
    bins: int,
) -> dict[str, Any]:
    """fold -> variable -> {all, by_bucket, by_observability, spatial_holdout}."""
    out: dict[str, Any] = {}
    rows = rows.reset_index(drop=True)
    p = prob if prob is not None else None
    bk = rows["horizon"].map(lambda h: cal.bucket_of(int(h), buckets))
    cls = cal.obs_class(rows)
    for (fold, var), g in rows.groupby(["fold", "variable"]):
        idx = g.index.to_numpy()

        def sc(mask: np.ndarray, idx: np.ndarray = idx) -> dict[str, Any]:
            ii = idx[mask]
            return block_scores(
                rows.loc[ii],
                point_only=point_only,
                prob=None if p is None else p[ii],
                op=op,
                bins=bins,
            )

        v: dict[str, Any] = {
            "all": sc(np.ones(len(idx), bool)),
            "by_bucket": {},
            "by_observability": {},
        }
        for b in buckets:
            v["by_bucket"][b] = sc((bk.loc[idx] == b).to_numpy())
        for c in cal.OBS_CLASSES:
            m = (cls.loc[idx] == c).to_numpy()
            if m.any():
                v["by_observability"][c] = sc(m)
        v["spatial_holdout"] = sc(g["reach_id"].astype(str).isin(holdout).to_numpy())
        out.setdefault(str(fold), {})[str(var)] = v
    return out


def decide_gate(
    val_scores: Mapping[str, Any],
    gate_cfg: Mapping[str, Any],
    buckets: Sequence[str],
    challenger: str = "ealstm+assim",
    incumbent: str = "lightgbm+assim",
) -> dict[str, Any]:
    """Pure. val_scores[model][variable] = score_model(...)['val'][variable]."""
    need = int(gate_cfg["min_buckets_won"])
    tol = float(gate_cfg["coverage_tolerance"])
    per_var: dict[str, Any] = {}
    for var in sorted(val_scores[incumbent]):
        c, i = val_scores[challenger].get(var), val_scores[incumbent][var]
        if c is None:
            per_var[var] = {"passed": False, "reason": "challenger has no validation rows"}
            continue
        wins = {
            b: (c["by_bucket"][b].get("crps", np.inf) < i["by_bucket"][b].get("crps", -np.inf))
            for b in buckets
        }
        cov_c = abs(c["all"].get("coverage_80", np.nan) - 0.8)
        cov_i = abs(i["all"].get("coverage_80", np.nan) - 0.8)
        cov_ok = bool(cov_c <= cov_i + tol)
        n_won = int(sum(wins.values()))
        per_var[var] = {
            "crps": {
                b: {
                    "challenger": c["by_bucket"][b].get("crps"),
                    "incumbent": i["by_bucket"][b].get("crps"),
                    "challenger_wins": wins[b],
                }
                for b in buckets
            },
            "buckets_won": n_won,
            "coverage_80": {
                "challenger": c["all"].get("coverage_80"),
                "incumbent": i["all"].get("coverage_80"),
            },
            "coverage_no_worse": cov_ok,
            "passed": bool(n_won >= need and cov_ok),
        }
    passed = bool(per_var) and all(v["passed"] for v in per_var.values())
    return {
        "rule": f"{challenger} beats {incumbent} on validation CRPS in >= {need} of {len(buckets)} "
        f"buckets for every target, with |coverage_80 - 0.80| no worse (+{tol})",
        "decided_on": "val",
        "per_variable": per_var,
        "production": "ealstm" if passed else "lightgbm",
    }


# ---------------------------------------------------------------------------
# calibration of one forecaster (fits on val, applies to both folds)
# ---------------------------------------------------------------------------
def calibrate(
    rows: pd.DataFrame,
    *,
    transforms: Mapping[str, str],
    buckets: Mapping[str, Sequence[int]],
    ccfg: Mapping[str, Any],
    val_end: date,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    val = rows[rows["fold"] == "val"]
    fit = cal.fit_conformal(
        val,
        transforms=transforms,
        buckets=buckets,
        val_end=val_end,
        target_coverage=float(ccfg["target_coverage"]),
        min_rows=int(ccfg["min_conformal_rows"]),
    )
    out = cal.apply_conformal(rows, fit, transforms=transforms, buckets=buckets)
    prob = exceedance(out)
    vm = (out["fold"] == "val").to_numpy()
    maps, iso_rec = cal.fit_isotonic_by_class(
        out[vm], prob[vm], min_events=int(ccfg["min_exceedance_events"]), val_end=val_end
    )
    prob_cal, flag = cal.apply_isotonic(out, prob, maps)
    record = {
        "conformal": fit.as_dict(),
        "isotonic": iso_rec,
        "rows_resorted": int(out["quantiles_resorted"].sum()),
        "rows_probability_calibrated": int(flag.sum()),
    }
    return out, prob_cal, record


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_lightgbm(city: str, variant: str) -> pd.DataFrame:
    from pipeline.build_dataset import PROCESSED_DIR

    ev = pd.read_parquet(PROCESSED_DIR / f"gbm_eval_{city}.parquet")
    ev = ev[(ev["variant"] == variant) & (ev["weather"] == "ORACLE")].copy()
    ev["reach_id"] = ev["reach_id"].astype(str)
    return ev


def load_ealstm_rows(city: str, fs: Any, last: date) -> pd.DataFrame | None:
    from models.ealstm import forecast_rows
    from pipeline.build_dataset import PROCESSED_DIR

    path = PROCESSED_DIR / f"ealstm_sim_{city}.parquet"
    if not path.exists():
        return None
    sim = pd.read_parquet(path)
    bounds = fs.fold_bounds(last)
    folds = {f: bounds[f] for f in ("val", "test")}
    rows = forecast_rows(sim, fs.horizons, folds)
    return rows[rows["p50"].notna()].copy()


# ---------------------------------------------------------------------------
# scenario sensitivity
# ---------------------------------------------------------------------------
def sensitivity_lightgbm(city: str, fs: Any, delta: float) -> dict[str, Any]:
    from models.baseline_gbm import load, load_frame, predict_long, to_long

    frame = load_frame(city)
    d = pd.to_datetime(frame["date"])
    dates = pd.date_range(fs.val_start, fs.val_end, freq="MS")
    rows = frame[d.isin(dates)]
    out: dict[str, Any] = {}
    for fit_name in ("B.production", "A.production"):
        model = load(city, fit_name)
        if "imperviousness_pct" not in model.features:
            out[fit_name] = {"status": "imperviousness_pct is not a feature of this model"}
            continue
        per: dict[str, Any] = {}
        for var in ("turbidity_proxy", "ndci"):
            base = to_long(rows, fs, var, [1], model.categories, require_target=False)
            pert_rows = rows.assign(
                imperviousness_pct=np.minimum(rows["imperviousness_pct"] + delta, 100.0)
            )
            pert = to_long(pert_rows, fs, var, [1], model.categories, require_target=False)
            p0 = predict_long(model, base, var)["p50"].to_numpy()
            p1 = predict_long(model, pert, var)["p50"].to_numpy()
            per[var] = summarise_response(base["reach_id"].astype(str).to_numpy(), p0, p1)
        out[fit_name] = per
    return out


def sensitivity_ealstm(fs: Any, delta: float) -> dict[str, Any]:
    from models import cmal
    from models.ealstm import load_ensemble

    out: dict[str, Any] = {}
    dates = list(pd.date_range(fs.val_start, fs.val_end, freq="MS").date)
    for var in ("turbidity_proxy", "ndci"):
        ens = load_ensemble(var)
        reaches = (ens.data.nh_dir / "all_reaches.txt").read_text().split()
        rid_all, p0_all, p1_all = [], [], []
        for rid in reaches:
            s = ens.data.statics(rid)
            imp = float(s[list(ens.data.static_attributes).index("imperviousness_pct")])
            m0 = ens.simulate(rid, dates)
            m1 = ens.simulate(
                rid, dates, static_overrides={"imperviousness_pct": min(imp + delta, 100.0)}
            )
            assert isinstance(m0, cmal.Mixture) and isinstance(m1, cmal.Mixture)
            q0 = ens.target.inverse(cmal.quantiles(m0, (0.5,))[:, 0])
            q1 = ens.target.inverse(cmal.quantiles(m1, (0.5,))[:, 0])
            rid_all += [rid] * len(dates)
            p0_all.append(q0)
            p1_all.append(q1)
        out[var] = summarise_response(
            np.array(rid_all), np.concatenate(p0_all), np.concatenate(p1_all)
        )
    return out


def summarise_response(reach: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> dict[str, Any]:
    """Per-reach mean change in P50; sign shares and sizes. Pure."""
    df = pd.DataFrame(
        {
            "reach_id": reach,
            "d": p1 - p0,
            "rel": (p1 - p0) / np.where(np.abs(p0) > 0, np.abs(p0), np.nan),
        }
    )
    per = df.groupby("reach_id")[["d", "rel"]].mean().dropna(subset=["d"])
    n = len(per)
    up = int((per["d"] > 1e-9).sum())
    down = int((per["d"] < -1e-9).sum())
    return {
        "reaches": n,
        "share_up": up / n if n else None,
        "share_down": down / n if n else None,
        "share_no_change": (n - up - down) / n if n else None,
        "median_change": float(per["d"].median()) if n else None,
        "median_relative_change": float(per["rel"].median()) if n else None,
        "mean_change": float(per["d"].mean()) if n else None,
    }


def sensitivity_verdict(model: str, var: str, r: Mapping[str, Any]) -> str | None:
    """Turbidity is expected to RISE with imperviousness (more runoff, more wash-off)."""
    if var != "turbidity_proxy" or not r.get("reaches"):
        return None
    if (r["share_down"] or 0) > 0.5:
        return (
            f"WRONG SIGN: {model} lowers turbidity on {r['share_down']:.0%} of reaches "
            "when imperviousness rises"
        )
    if (r["share_no_change"] or 0) > 0.5:
        return (
            f"INSENSITIVE: {model} does not respond to imperviousness on "
            f"{r['share_no_change']:.0%} of reaches"
        )
    return None


# ---------------------------------------------------------------------------
# attribution sanity check
# ---------------------------------------------------------------------------
def storm_sanity(fs: Any, manifest: Mapping[str, Any]) -> dict[str, Any]:
    from models.ealstm import explain, load_ensemble

    out: dict[str, Any] = {}
    train = manifest["basin_lists"]["train_reaches"]
    rid = max(train, key=lambda r: manifest["per_reach"][r]["ok"])
    for var in ("turbidity_proxy", "ndci"):
        ens = load_ensemble(var)
        s = ens.data.series(rid)
        win = s[(s.index >= pd.Timestamp(fs.val_start)) & (s.index <= pd.Timestamp(fs.val_end))]
        storm = win["precip_mm"].idxmax().date()
        ex = explain(rid, storm, var, ensemble=ens)
        dyn = [d for d in ex["drivers"] if d.kind == "weather_driver"]
        tot = sum(abs(d.contribution) for d in dyn)
        precip = sum(abs(d.contribution) for d in dyn if d.feature in PRECIP_FEATURES)
        share = precip / tot if tot > 0 else None
        top = dyn[0].feature if dyn else None
        out[var] = {
            "reach_id": rid,
            "storm_date": str(storm),
            "precip_mm": float(win["precip_mm"].max()),
            "top_weather_driver": top,
            "precipitation_share_of_weather_attribution": share,
            "passed": bool(share is not None and share > 0.5 and top in PRECIP_FEATURES),
            "top5": [(d.feature, round(d.contribution, 5)) for d in ex["drivers"][:5]],
            "completeness_delta": ex["completeness_delta_transformed"],
            "attribution_complete": ex["attribution_complete"],
        }
    return out


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def run(city: str) -> dict[str, Any]:
    from models.baseline_gbm import load_frame
    from models.ealstm import EALSTMSettings, load_manifest
    from pipeline.build_dataset import FrameSettings

    cfg = load_config("modelling")
    thresholds = load_config("thresholds")
    fs = FrameSettings.from_config()
    buckets = {k: tuple(v) for k, v in cfg["baseline_gbm"]["horizon_buckets"].items()}
    transforms = assim.transforms_from_config(cfg)
    aset = assim.AssimilationSettings.from_config(cfg)
    ccfg, gcfg, ev = cfg["calibration"], cfg["gate"], cfg["evaluation"]
    op = float(thresholds["guardrails"]["min_exceedance_prob"])
    bins = int(ev["reliability_bins"])
    manifest = load_manifest(EALSTMSettings.from_config())
    holdout = set(manifest["basin_lists"]["spatial_holdout"])

    with stage(log, "head_to_head", city=city) as counters:
        frame = load_frame(city)
        last = pd.to_datetime(frame["date"]).max().date()
        obs = observations_from_frame(frame, fs.variables)
        fold_train_end = {"val": fs.train_end, "test": fs.val_end}

        raw: dict[str, pd.DataFrame] = {
            "lightgbm": load_lightgbm(city, str(gcfg["lightgbm_variant"]))
        }
        ea = load_ealstm_rows(city, fs, last)
        notes: list[str] = []
        if ea is None:
            notes.append(
                "EA-LSTM not trained / no hindcast: gate NOT decided, LightGBM stays production"
            )
        else:
            raw["ealstm"] = ea
        counters.record(rows_in=int(sum(len(v) for v in raw.values())))

        # ---- assimilation (identical for both) ----------------------------------
        taus: dict[str, Any] = {}
        for m in list(raw):
            rows = raw[m]
            res = assim.residuals(rows, transforms)
            val = rows[rows["fold"] == "val"]
            fit = assim.fit_tau(
                val,
                res[res["fold"] == "val"],
                aset.tau_grid,
                max_obs_age_days=aset.max_obs_age_days,
                transforms=transforms,
                val_end=fs.val_end,
            )
            taus[m] = fit
            tau = {v: f["tau_days"] for v, f in fit.items()}
            raw[f"{m}+assim"] = assim.assimilate(
                rows, res, tau, max_obs_age_days=aset.max_obs_age_days, transforms=transforms
            )

        # ---- shared rows + baselines ---------------------------------------------
        keys = common_keys(raw)
        shared = {
            m: restrict(df, keys).sort_values(KEYS).reset_index(drop=True) for m, df in raw.items()
        }
        base = attach_baselines(shared["lightgbm"], obs, fold_train_end, ev, thresholds, LEVELS)
        base = base.sort_values(KEYS).reset_index(drop=True)
        extra = [
            "sn",
            "clim_mean",
            *[f"clim_q{c[1:]}" for c in Q],
            "threshold",
            "threshold_clim_freq",
        ]
        for m in shared:
            shared[m] = (
                shared[m]
                .drop(columns=[c for c in extra if c in shared[m]])
                .merge(
                    base[[*KEYS, *extra]].assign(reach_id=base["reach_id"].astype(str)),
                    on=KEYS,
                    how="left",
                )
            )
        sn = shared["lightgbm"].assign(p50=shared["lightgbm"]["sn"])
        clim = shared["lightgbm"].assign(**{c: shared["lightgbm"][f"clim_q{c[1:]}"] for c in Q})
        shared["seasonal_naive"] = sn[sn["p50"].notna()]
        shared["climatology"] = clim[clim["p50"].notna()]

        # ---- calibration (identical for all probabilistic forecasters) ---------
        calibrated: dict[str, pd.DataFrame] = {}
        probs: dict[str, np.ndarray] = {}
        cal_records: dict[str, Any] = {}
        for m in [x for x in MODELS if x in shared and x != "seasonal_naive"]:
            out, p, rec = calibrate(
                shared[m], transforms=transforms, buckets=buckets, ccfg=ccfg, val_end=fs.val_end
            )
            calibrated[m], probs[m], cal_records[m] = out, p, rec

        # ---- VALIDATION scores -> the gate (before test is looked at) ------------
        def scores_for(fold: str) -> dict[str, Any]:
            res: dict[str, Any] = {}
            for m in MODELS:
                if m == "seasonal_naive" and m in shared:
                    d = shared[m][shared[m]["fold"] == fold]
                    res[m] = score_model(
                        d, buckets, point_only=True, prob=None, holdout=holdout, op=op, bins=bins
                    ).get(fold, {})
                elif m in calibrated:
                    msk = (calibrated[m]["fold"] == fold).to_numpy()
                    res[m] = score_model(
                        calibrated[m][msk],
                        buckets,
                        point_only=False,
                        prob=probs[m][msk],
                        holdout=holdout,
                        op=op,
                        bins=bins,
                    ).get(fold, {})
            return res

        val_scores = scores_for("val")
        if "ealstm+assim" in val_scores:
            gate = decide_gate(val_scores, gcfg, list(buckets))
            gate["unassimilated_comparison"] = decide_gate(
                val_scores, gcfg, list(buckets), "ealstm", "lightgbm"
            )
        else:
            gate = {
                "production": "lightgbm",
                "decided_on": "val",
                "rule": "not decided",
                "reason": notes[0],
            }
        gate_path = RESULTS_DIR / "gate.json"
        previous = json.loads(gate_path.read_text()) if gate_path.exists() else {}
        gate["decided"] = "ealstm+assim" in val_scores
        gate["decided_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        # only a DECIDED gate's test scoring counts; a LightGBM-only provisional run
        # re-scores test numbers evaluate.py has already published
        gate["test_evaluations"] = (
            int(previous.get("test_evaluations", 0)) if previous.get("decided") else 0
        )
        RESULTS_DIR.mkdir(exist_ok=True)
        gate_path.write_text(json.dumps(_clean(gate), indent=2))
        log.info("head_to_head.gate", production=gate["production"])

        # ---- TEST, once, every model regardless -------------------------------
        test_scores = scores_for("test")
        if gate["decided"]:
            gate["test_evaluations"] += 1
        gate_path.write_text(json.dumps(_clean(gate), indent=2))

        # ---- attribution + sensitivity -----------------------------------------
        attribution: dict[str, Any]
        if gate["production"] == "ealstm":
            try:
                attribution = storm_sanity(fs, manifest)
            except ImportError as e:
                attribution = {"status": f"not computed: {e} (pip install captum)"}
        else:
            attribution = {
                "status": "not computed: EA-LSTM is not production (SHAP from LightGBM is served)"
            }
        delta = 10.0
        sens: dict[str, Any] = {
            "delta_imperviousness_pct": delta,
            "lightgbm": sensitivity_lightgbm(city, fs, delta),
        }
        if "ealstm" in raw:
            sens["ealstm"] = sensitivity_ealstm(fs, delta)
        verdicts = []
        for fit_name, per in sens["lightgbm"].items():
            for var, r in per.items():
                if isinstance(r, dict) and (
                    v := sensitivity_verdict(f"LightGBM {fit_name}", var, r)
                ):
                    verdicts.append(v)
        for var, r in sens.get("ealstm", {}).items():
            if v := sensitivity_verdict("EA-LSTM", var, r):
                verdicts.append(v)
        sens["verdicts"] = verdicts

        report = {
            "city": city,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "weather_label": HINDCAST_LABEL,
            "notes": notes
            + [
                "LightGBM's test fit is retrained on data through 2024 (walk-forward); the "
                "EA-LSTM is trained on 2016-2023 only (2024 = early stopping), so on test "
                "LightGBM has one more year of training data.",
                "Spatial holdout: the EA-LSTM never saw these reaches; LightGBM variant A "
                "was trained on them (reach_id feature) - not a like-for-like comparison.",
                "seasonal_naive is a point forecast: no interval, no calibration, CRPS = MAE.",
            ],
            "shared_rows": int(len(keys)),
            "assimilation_tau": taus,
            "calibration": cal_records,
            "gate": gate,
            "scores": {"val": val_scores, "test": test_scores},
            "attribution_sanity": attribution,
            "scenario_sensitivity": sens,
        }
        report = dict(_clean(report))
        (RESULTS_DIR / "head_to_head.json").write_text(json.dumps(report, indent=2))
        text = format_report(report)
        (RESULTS_DIR / "head_to_head.txt").write_text(text, encoding="utf-8")
        mpath = RESULTS_DIR / "metrics.json"
        if mpath.exists():
            m = json.loads(mpath.read_text())
            m["head_to_head"] = report
            m["production_model"] = gate["production"]
            mpath.write_text(json.dumps(m, indent=2))
        counters.record(rows_out=int(len(keys)), production=gate["production"])
    print(text)
    return report


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------
def _f(x: Any, w: int = 8, p: int = 4) -> str:
    return f"{'-':>{w}}" if x is None else f"{x:>{w}.{p}f}"


def format_report(r: Mapping[str, Any]) -> str:
    rule = "=" * 110
    L = [rule, "KINGFISHER P5.4 - HEAD-TO-HEAD (calibrated; " + r["weather_label"] + ")", rule]
    for fold in ("val", "test"):
        sc = r["scores"][fold]
        variables = sorted({v for m in sc.values() for v in m})
        for var in variables:
            L.append(f"\n  {fold.upper()}  |  {var}")
            L.append(
                f"  {'model':<16}{'scope':<18}{'n':>7}{'CRPS':>9}{'MAE':>9}{'RMSE':>9}"
                f"{'cov80':>8}{'Brier':>8}"
            )
            for m in MODELS:
                v = sc.get(m, {}).get(var)
                if not v:
                    continue
                scopes = [
                    ("all", v["all"]),
                    *[(b, x) for b, x in v["by_bucket"].items()],
                    ("spatial_holdout", v["spatial_holdout"]),
                ]
                for scope, s in scopes:
                    if not s.get("n"):
                        continue
                    L.append(
                        f"  {m:<16}{scope:<18}{s['n']:>7}{_f(s.get('crps'), 9)}"
                        f"{_f(s.get('mae'), 9)}"
                        f"{_f(s.get('rmse'), 9)}{_f(s.get('coverage_80'), 8, 3)}"
                        f"{_f(s.get('probability', {}).get('brier'), 8, 4)}"
                    )
    g = r["gate"]
    L += [
        "",
        rule,
        f"  GATE (decided on validation): production = {g['production'].upper()}",
        f"  rule: {g['rule']}",
    ]
    for var, pv in g.get("per_variable", {}).items():
        L.append(
            f"   {var}: buckets won {pv.get('buckets_won')} | "
            f"coverage no worse: {pv.get('coverage_no_worse')} | "
            f"{'PASS' if pv.get('passed') else 'FAIL'}"
        )
    L.append(f"  test fold evaluated {g.get('test_evaluations')} time(s)")
    sens = r["scenario_sensitivity"]
    L += [
        "",
        f"  SCENARIO SENSITIVITY: +{sens['delta_imperviousness_pct']:.0f} points "
        "imperviousness_pct, every reach",
    ]
    for model, per in [("LightGBM " + k, v) for k, v in sens["lightgbm"].items()] + [
        ("EA-LSTM", sens.get("ealstm", {}))
    ]:
        for var, s in per.items():
            if isinstance(s, dict) and s.get("reaches"):
                L.append(
                    f"   {model:<22}{var:<16} up {s['share_up']:.0%}  down {s['share_down']:.0%}  "
                    f"flat {s['share_no_change']:.0%}  median change {s['median_change']:+.4f} "
                    f"({(s['median_relative_change'] or 0):+.1%})"
                )
    for v in sens["verdicts"]:
        L.append(f"   !!! {v}")
    a = r["attribution_sanity"]
    L += [
        "",
        "  ATTRIBUTION SANITY: "
        + (
            a.get("status")
            or json.dumps(
                {k: {"passed": x["passed"], "top": x["top_weather_driver"]} for k, x in a.items()}
            )
        ),
    ]
    L.append(rule)
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P5.4 head-to-head + gate")
    parser.add_argument("--city", default="coimbra")
    args = parser.parse_args(argv)
    run(args.city)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
