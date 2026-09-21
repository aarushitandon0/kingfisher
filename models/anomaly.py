"""Weather-explained anomaly detector (P5.3). An anomaly is an observation the weather
cannot explain - not merely a high one.

Run (after models.head_to_head has decided the production model):

    python -m models.anomaly --city coimbra

For every OK observation in the out-of-sample period (validation + test folds):

  PIT    = F(observed) under the production model's UN-assimilated forecast for that
           date - the driver -> state simulation, "what the weather predicts" -
           EA-LSTM: the exact CMAL mixture CDF of the hindcast simulation;
           LightGBM: variant B (no reach identity, no observation history - its
           driver -> state analogue), shortest horizon, engine.probability's quantile CDF.
  score  = -log(1 - PIT)
  label  UNEXPLAINED        PIT > pit_threshold AND observed > P90 + margin * (P90 - P10)
                            (transformed space; config/thresholds.yaml anomaly). The only
                            label that counts as an anomaly.
         WEATHER_EXPLAINED  a high reading (observed > the reach-season threshold) that
                            the weather accounts for
         NORMAL             everything else
  localised  true when the reach is UNEXPLAINED and every upstream reach observed on the
             SAME acquisition date is unremarkable (not UNEXPLAINED, PIT <= threshold);
             false if an upstream reach is also UNEXPLAINED; NULL when no upstream reach
             was observed that day. A LABEL ONLY. Kingfisher never attributes a source
             or a polluter (MASTERSPEC 3, non-goal).

SCORING - against data/reference/incidents.csv: a detection matches an incident within
+/- match_window_days on the incident's reach or up to match_downstream_reaches reaches
downstream of it. Precision, recall, F1 with reach-cluster bootstrap 95% CIs. With fewer
than min_incidents incidents, the reference is the PROXY - observations above the
reach-season 95th percentile of the fitting period - and every number says so. The
proxy measures how often an extreme reading is one the weather cannot explain; it is not
skill against real pollution events.

Pure core (classify, localise, match, bootstrap) + one I/O runner.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pandas as pd

from core.config import load_config
from core.logging import get_logger, stage
from core.settings import REPO_ROOT

log = get_logger(__name__)

UNEXPLAINED = "UNEXPLAINED"
WEATHER_EXPLAINED = "WEATHER_EXPLAINED"
NORMAL = "NORMAL"
PROXY_LABEL = "PROXY: observations above the reach-season 95th percentile (not incidents)"


@dataclass(frozen=True)
class AnomalySettings:
    pit_threshold: float
    p90_margin_fraction: float
    match_window_days: int
    match_downstream_reaches: int
    min_incidents: int
    proxy_percentile: float
    bootstrap_resamples: int
    bootstrap_seed: int

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None = None) -> AnomalySettings:
        a = (cfg or load_config("thresholds"))["anomaly"]
        return cls(
            pit_threshold=float(a["pit_threshold"]),
            p90_margin_fraction=float(a["p90_margin_fraction"]),
            match_window_days=int(a["match_window_days"]),
            match_downstream_reaches=int(a["match_downstream_reaches"]),
            min_incidents=int(a["min_incidents"]),
            proxy_percentile=float(a["proxy_percentile"]),
            bootstrap_resamples=int(a["bootstrap_resamples"]),
            bootstrap_seed=int(a["bootstrap_seed"]),
        )


# ---------------------------------------------------------------------------
# PIT + classification (pure)
# ---------------------------------------------------------------------------
def pit_from_quantiles(q: np.ndarray, levels: Sequence[float], y: np.ndarray) -> np.ndarray:
    """PIT under engine.probability's piecewise-linear CDF - the same CDF the alerts use."""
    from engine.probability import quantile_cdf_multi

    return quantile_cdf_multi(np.sort(q, axis=1), levels, y)


def score(pit: np.ndarray) -> np.ndarray:
    """-log(1 - PIT); PIT == 1 (beyond the whole distribution) scores +inf, never capped."""
    with np.errstate(divide="ignore"):
        return np.asarray(-np.log1p(-np.asarray(pit, dtype="float64")))


def classify(obs: pd.DataFrame, s: AnomalySettings, transforms: Mapping[str, str]) -> pd.DataFrame:
    """obs: variable, observed, pit, p10, p90 (original units) and threshold (NaN if the
    reach-season has none). Adds anomaly_score, exceeds_p90_margin, high, label."""
    from models.assimilation import _fwd

    out = obs.copy()
    margin_ok = np.zeros(len(out), dtype=bool)
    for var in out["variable"].unique():
        m = (out["variable"] == var).to_numpy()
        tr = transforms[str(var)]
        y = _fwd(out.loc[m, "observed"].to_numpy(dtype="float64"), tr)
        lo = _fwd(out.loc[m, "p10"].to_numpy(dtype="float64"), tr)
        hi = _fwd(out.loc[m, "p90"].to_numpy(dtype="float64"), tr)
        margin_ok[m] = y > hi + s.p90_margin_fraction * (hi - lo)
    pit = out["pit"].to_numpy(dtype="float64")
    anomalous = (pit > s.pit_threshold) & margin_ok
    thr = (
        out["threshold"].to_numpy(dtype="float64")
        if "threshold" in out
        else np.full(len(out), np.nan)
    )
    high = out["observed"].to_numpy(dtype="float64") > thr  # NaN threshold -> not "high"
    label = np.where(anomalous, UNEXPLAINED, np.where(high, WEATHER_EXPLAINED, NORMAL))
    label = np.where(np.isnan(pit), None, label)  # no forecast -> no judgement, never NORMAL
    out["anomaly_score"] = score(pit)
    out["exceeds_p90_margin"] = margin_ok
    out["high"] = high
    out["label"] = label
    return out


def localise(
    labelled: pd.DataFrame, upstream: Mapping[str, Sequence[str]], pit_threshold: float
) -> pd.Series:
    """True / False / None per row (see module docstring). labelled: reach_id, date,
    variable, label, pit."""
    key = labelled.set_index(["variable", "reach_id", "date"])
    lab = key["label"].to_dict()
    pit = key["pit"].to_dict()
    out = []
    for r in labelled.itertuples(index=False):
        if r.label != UNEXPLAINED:
            out.append(None)
            continue
        ups = [u for u in upstream.get(str(r.reach_id), ()) if (r.variable, u, r.date) in lab]
        if not ups:
            out.append(None)
            continue
        quiet = all(
            lab[(r.variable, u, r.date)] != UNEXPLAINED
            and not (pit[(r.variable, u, r.date)] > pit_threshold)
            for u in ups
        )
        out.append(quiet)
    return pd.Series(out, index=labelled.index, dtype=object)


# ---------------------------------------------------------------------------
# topology helpers (pure)
# ---------------------------------------------------------------------------
def adjacency(edges: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """(upstream_of, downstream_of) from reach_topology rows (upstream_id, downstream_id)."""
    up: dict[str, list[str]] = defaultdict(list)
    down: dict[str, list[str]] = defaultdict(list)
    for u, d in edges[["upstream_id", "downstream_id"]].itertuples(index=False):
        up[str(d)].append(str(u))
        down[str(u)].append(str(d))
    return dict(up), dict(down)


def within_downstream(reach: str, down: Mapping[str, Sequence[str]], k: int) -> set[str]:
    """The reach and every reach reachable within k downstream hops."""
    seen = {reach}
    q: deque[tuple[str, int]] = deque([(reach, 0)])
    while q:
        r, d = q.popleft()
        if d == k:
            continue
        for n in down.get(r, ()):
            if n not in seen:
                seen.add(n)
                q.append((n, d + 1))
    return seen


# ---------------------------------------------------------------------------
# scoring (pure)
# ---------------------------------------------------------------------------
def match(
    detections: pd.DataFrame,
    events: pd.DataFrame,
    down: Mapping[str, Sequence[str]],
    *,
    window_days: int,
    k_downstream: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(detection matched?, event matched?). Both frames: reach_id, date."""
    det_ok = np.zeros(len(detections), dtype=bool)
    ev_ok = np.zeros(len(events), dtype=bool)
    if detections.empty or events.empty:
        return det_ok, ev_ok
    by_reach: dict[str, list[tuple[int, pd.Timestamp]]] = defaultdict(list)
    for i, (r, d) in enumerate(detections[["reach_id", "date"]].itertuples(index=False)):
        by_reach[str(r)].append((i, pd.Timestamp(d)))
    w = pd.Timedelta(days=window_days)
    for j, (r, d) in enumerate(events[["reach_id", "date"]].itertuples(index=False)):
        t = pd.Timestamp(d)
        for reach in within_downstream(str(r), down, k_downstream):
            for i, td in by_reach.get(reach, ()):
                if abs(td - t) <= w:
                    det_ok[i] = True
                    ev_ok[j] = True
    return det_ok, ev_ok


def prf(det_ok: np.ndarray, ev_ok: np.ndarray) -> dict[str, float | None]:
    p = float(det_ok.mean()) if len(det_ok) else None
    r = float(ev_ok.mean()) if len(ev_ok) else None
    f = 2 * p * r / (p + r) if p is not None and r is not None and (p + r) > 0 else None
    return {"precision": p, "recall": r, "f1": f}


def bootstrap_prf(
    det_reach: np.ndarray,
    det_ok: np.ndarray,
    ev_reach: np.ndarray,
    ev_ok: np.ndarray,
    *,
    n: int,
    seed: int,
) -> dict[str, list[float | None]]:
    """95% percentile CIs, resampling REACHES with replacement (observations within a
    reach are not independent). Match flags are computed once on the full data."""
    reaches = np.array(sorted(set(det_reach) | set(ev_reach)))
    if len(reaches) == 0:
        return {"precision": [None, None], "recall": [None, None], "f1": [None, None]}
    rng = np.random.default_rng(seed)
    d_idx = {r: np.flatnonzero(det_reach == r) for r in reaches}
    e_idx = {r: np.flatnonzero(ev_reach == r) for r in reaches}
    stats: dict[str, list[float]] = {"precision": [], "recall": [], "f1": []}
    for _ in range(n):
        pick = rng.choice(reaches, len(reaches), replace=True)
        di = np.concatenate([d_idx[r] for r in pick]) if len(pick) else np.array([], int)
        ei = np.concatenate([e_idx[r] for r in pick]) if len(pick) else np.array([], int)
        m = prf(det_ok[di], ev_ok[ei])
        for k in stats:
            if m[k] is not None:
                stats[k].append(float(m[k]))  # type: ignore[arg-type]
    return {
        k: ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [None, None])
        for k, v in stats.items()
    }


def proxy_events(
    obs: pd.DataFrame, fit_end: date, percentile: float, thresholds: dict[str, Any]
) -> pd.DataFrame:
    """Observations above the reach-season `percentile` of the fitting period (<= fit_end),
    judged only after fit_end."""
    from engine.thresholds import lookup, seasonal_thresholds

    cfg = json.loads(json.dumps(thresholds))
    for v in cfg["variables"].values():
        v["percentile"] = percentile
    long = obs.rename(columns={"observed": "value"})[["reach_id", "date", "variable", "value"]]
    table = seasonal_thresholds(long, fit_end, cfg)
    after = obs[pd.to_datetime(obs["date"]) > pd.Timestamp(fit_end)].copy()
    t = lookup(after.assign(target_date=after["date"]), table, cfg["seasons"])
    after["proxy_threshold"] = t["threshold"].to_numpy()
    return after[after["observed"] > after["proxy_threshold"]][["reach_id", "date", "variable"]]


# ---------------------------------------------------------------------------
# I/O runner
# ---------------------------------------------------------------------------
RESULTS_DIR = REPO_ROOT / "results"


def _production_model() -> str:
    gate = RESULTS_DIR / "gate.json"
    if not gate.exists():
        raise FileNotFoundError(
            f"{gate} missing - run `python -m models.head_to_head` first: the anomaly "
            "detector uses whichever model the gate made production"
        )
    return str(json.loads(gate.read_text())["production"])


def _observations_ealstm(city: str) -> pd.DataFrame:
    from pipeline.build_dataset import PROCESSED_DIR

    sim = pd.read_parquet(PROCESSED_DIR / f"ealstm_sim_{city}.parquet")
    keep = [
        "variable",
        "reach_id",
        "date",
        "observed",
        "pit",
        "p10",
        "p50",
        "p90",
        "reach_observable",
        "model_version",
    ]
    return sim[keep].copy()


def _observations_lightgbm(city: str) -> pd.DataFrame:
    from engine.probability import DEFAULT_LEVELS
    from pipeline.build_dataset import PROCESSED_DIR

    ev = pd.read_parquet(PROCESSED_DIR / f"gbm_eval_{city}.parquet")
    b = ev[(ev["variant"] == "B") & (ev["weather"] == "ORACLE") & ev["target"].notna()]
    b = b.sort_values("horizon").drop_duplicates(
        ["variable", "reach_id", "target_date"], keep="first"
    )
    names = [f"p{round(a * 100):02d}" for a in DEFAULT_LEVELS]
    pit = pit_from_quantiles(b[names].to_numpy(), DEFAULT_LEVELS, b["target"].to_numpy())
    return pd.DataFrame(
        {
            "variable": b["variable"].to_numpy(),
            "reach_id": b["reach_id"].astype(str).to_numpy(),
            "date": pd.to_datetime(b["target_date"]).dt.date.to_numpy(),
            "observed": b["target"].to_numpy(),
            "pit": pit,
            "p10": b["p10"].to_numpy(),
            "p50": b["p50"].to_numpy(),
            "p90": b["p90"].to_numpy(),
            "reach_observable": b["reach_observable"].to_numpy(),
            "model_version": b["model_version"].to_numpy(),
        }
    )


def load_topology(city: str) -> pd.DataFrame:
    from sqlalchemy import text

    from core.db import session_scope

    with session_scope() as session:
        edges = pd.read_sql(
            text(
                "SELECT t.upstream_id, t.downstream_id FROM reach_topology t "
                "JOIN reaches r ON r.reach_id = t.upstream_id WHERE r.city = :c"
            ),
            session.connection(),
            params={"c": city},
        )
    if edges.empty:
        raise RuntimeError(f"no reach_topology edges for {city} - run pipeline.l0_network")
    return edges


def load_incidents(city: str) -> pd.DataFrame:
    path = REPO_ROOT / "data" / "reference" / "incidents.csv"
    inc = pd.read_csv(path, dtype={"reach_id": str})
    return inc[inc["city"] == city].assign(date=lambda d: pd.to_datetime(d["date"]).dt.date)


def run(city: str) -> dict[str, Any]:
    from engine.thresholds import lookup, seasonal_thresholds
    from models.assimilation import transforms_from_config
    from pipeline.build_dataset import PROCESSED_DIR, FrameSettings

    s = AnomalySettings.from_config()
    thresholds = load_config("thresholds")
    cfg = load_config("modelling")
    transforms = transforms_from_config(cfg)
    fs = FrameSettings.from_config()
    production = _production_model()
    with stage(log, "anomaly", city=city, model=production) as counters:
        obs = _observations_ealstm(city) if production == "ealstm" else _observations_lightgbm(city)
        obs["date"] = pd.to_datetime(obs["date"]).dt.date
        counters.record(rows_in=len(obs))
        # seasonal "high" threshold: fitted on everything before the observation's fold
        fold_end = {"val": fs.train_end, "test": fs.val_end}
        parts = []
        from models.baseline_gbm import load_frame
        from models.evaluate import observations_from_frame

        all_obs = observations_from_frame(load_frame(city), fs.variables)
        d = pd.to_datetime(obs["date"])
        in_fold = {
            "val": (d >= pd.Timestamp(fs.val_start)) & (d <= pd.Timestamp(fs.val_end)),
            "test": d >= pd.Timestamp(fs.test_start),
        }
        for fold, m in in_fold.items():
            g = obs[m.to_numpy()].copy()
            table = seasonal_thresholds(all_obs, fold_end[fold], thresholds)
            g["threshold"] = lookup(g.assign(target_date=g["date"]), table, thresholds["seasons"])[
                "threshold"
            ].to_numpy()
            g["fold"] = fold
            parts.append(g)
        obs = pd.concat(parts, ignore_index=True)
        no_pit = int(obs["pit"].isna().sum())
        if no_pit:
            counters.drop(no_pit, "NO_FORECAST_FOR_OBS_DATE")
        lab = classify(obs, s, transforms)
        up, down = adjacency(load_topology(city))
        lab["localised"] = localise(lab, up, s.pit_threshold)

        incidents = load_incidents(city)
        use_proxy = len(incidents) < s.min_incidents
        ref_label = PROXY_LABEL if use_proxy else "data/reference/incidents.csv"
        scoring: dict[str, Any] = {}
        for var, g in lab.groupby("variable"):
            det = g[g["label"] == UNEXPLAINED][["reach_id", "date"]].reset_index(drop=True)
            if use_proxy:
                allo = all_obs.rename(columns={"value": "observed"})
                allo = allo[allo["variable"] == var].assign(
                    date=lambda d: pd.to_datetime(d["date"]).dt.date
                )
                # proxy thresholds fitted on the training period; events judged after it
                ev = proxy_events(allo, fs.train_end, s.proxy_percentile, thresholds)
            else:
                ev = incidents[["reach_id", "date"]]
            d_ok, e_ok = match(
                det,
                ev,
                down,
                window_days=s.match_window_days,
                k_downstream=s.match_downstream_reaches,
            )
            scoring[str(var)] = {
                "reference": ref_label,
                "detections": int(len(det)),
                "events": int(len(ev)),
                **prf(d_ok, e_ok),
                "ci95": bootstrap_prf(
                    det["reach_id"].to_numpy(),
                    d_ok,
                    ev["reach_id"].to_numpy(),
                    e_ok,
                    n=s.bootstrap_resamples,
                    seed=s.bootstrap_seed,
                ),
                "labels": g["label"].value_counts(dropna=False).to_dict(),
                "localised": g.loc[g["label"] == UNEXPLAINED, "localised"]
                .value_counts(dropna=False)
                .to_dict(),
            }
        lab.to_parquet(PROCESSED_DIR / f"anomalies_{city}.parquet", index=False)
        report = {
            "city": city,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "production_model": production,
            "pit_source": (
                "EA-LSTM hindcast simulation (un-assimilated), exact CMAL CDF"
                if production == "ealstm"
                else "LightGBM variant B, shortest horizon, quantile CDF (engine.probability)"
            ),
            "weather_label": "hindcast with observed weather - upper bound on live skill",
            "settings": s.__dict__,
            "reference": ref_label,
            "incidents_available": int(len(incidents)),
            "no_attribution": (
                "labels only; no source or polluter is ever attributed (MASTERSPEC 3)"
            ),
            "by_variable": scoring,
        }
        counters.record(
            rows_out=len(lab),
            unexplained=int((lab["label"] == UNEXPLAINED).sum()),
            weather_explained=int((lab["label"] == WEATHER_EXPLAINED).sum()),
        )
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "anomaly_metrics.json").write_text(json.dumps(_clean(report), indent=2))
    metrics_path = RESULTS_DIR / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        m["anomaly"] = _clean(report)
        m.get("not_computed", {}).pop("anomaly_precision_recall_f1", None)
        metrics_path.write_text(json.dumps(m, indent=2))
    return report


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    if isinstance(o, float | np.floating):
        return None if not math.isfinite(float(o)) else round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    return o


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Weather-explained anomaly detector (P5.3)")
    parser.add_argument("--city", default="coimbra")
    args = parser.parse_args(argv)
    print(json.dumps(_clean(run(args.city)), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
