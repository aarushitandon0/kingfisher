"""Is there a level shift in the Sentinel-2 indices at the Sentinel-2C platform change?

Run (after L1):

    python scripts/check_s2_platform_shift.py --city coimbra

Sentinel-2C replaced Sentinel-2A in the operational constellation in January 2025 (S2A
continued on an extended campaign). A sensor change can shift a reflectance-derived
index even after harmonisation. The Statistical API does not say which platform took an
acquisition, so the platform per acquisition day comes from the CDSE STAC catalogue
(free: no processing units), cached to disk like every other response.

Two comparisons, reported separately because they answer different questions:

  CONTEMPORANEOUS  (the test) within the S2C era, S2C acquisitions vs S2B acquisitions
                   of the same reaches in the same months. Both sensors see the same
                   water in the same season, so a difference is the sensor, not the
                   weather or the year.
  BEFORE / AFTER   (context, confounded) S2A-era vs S2C-era residuals. Also changes with
                   the year's hydrology, so it is never read as a sensor effect alone.
  REFERENCE        S2A vs S2B before S2C - the inter-sensor difference the series always
                   had, for scale.

Values are deseasonalised per reach and calendar month against the median of that
reach-month before the S2C era. Days imaged by two platforms are excluded (MIXED): the
Statistical API's daily aggregate cannot be attributed to one sensor. Writes
results/s2_platform_shift.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.cache import DiskCache  # noqa: E402
from core.settings import REPO_ROOT  # noqa: E402
from pipeline.l0_network import load_city_config  # noqa: E402

STAC_URL = "https://stac.dataspace.copernicus.eu/v1/search"
S2C_ERA_START = date(2025, 1, 21)  # S2C operational in place of S2A
VARIABLES = ("turbidity_proxy", "ndci", "mndwi")
BOOTSTRAP = 4000


def stac_month(bbox: dict[str, float], year: int, month: int) -> list[dict[str, Any]]:
    lo = date(year, month, 1)
    hi = date(year + (month == 12), month % 12 + 1, 1)
    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": [bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]],
        "datetime": f"{lo.isoformat()}T00:00:00Z/{hi.isoformat()}T00:00:00Z",
        "limit": 200,
        "fields": {
            "include": ["id", "properties.datetime", "properties.platform"],
            "exclude": ["assets", "links", "geometry"],
        },
    }

    def fetch() -> dict[str, Any]:
        r = requests.post(STAC_URL, json=body, timeout=120)
        r.raise_for_status()
        payload: dict[str, Any] = r.json()
        if payload.get("numberReturned", 0) >= 200:
            raise RuntimeError(f"STAC {year}-{month:02d}: page full - paginate")
        return payload

    entry = DiskCache("stac").get_or_fetch(
        body, fetch, slug=f"s2l2a_{year}-{month:02d}", url=STAC_URL, source="CDSE STAC"
    )
    feats: list[dict[str, Any]] = entry.payload.get("features", [])
    return feats


def platform_by_day(bbox: dict[str, float], start: date, end: date) -> pd.Series:
    rows = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        for f in stac_month(bbox, y, m):
            rows.append(
                {
                    "date": pd.Timestamp(f["properties"]["datetime"][:10]),
                    "platform": f["properties"]["platform"].upper().replace("SENTINEL-", "S"),
                }
            )
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    if not rows:
        raise RuntimeError("STAC returned no Sentinel-2 L2A items - refusing to continue")
    df = pd.DataFrame(rows).drop_duplicates()
    per_day = df.groupby("date")["platform"].agg(lambda s: sorted(set(s)))
    return per_day.map(lambda p: p[0] if len(p) == 1 else "MIXED")


def load_ok_observations(city: str) -> pd.DataFrame:
    from sqlalchemy import text

    from core.db import session_scope

    with session_scope() as session:
        return pd.read_sql(
            text(
                "SELECT o.reach_id, o.obs_date, o.turbidity_proxy, o.ndci, o.mndwi "
                "FROM observations o JOIN reaches r USING (reach_id) "
                "WHERE r.city = :c AND o.source = 'S2' AND o.quality_flag = 'OK'"
            ),
            session.connection(),
            params={"c": city},
        )


def deseasonalise(obs: pd.DataFrame, var: str) -> pd.Series:
    """value - median of the same reach and calendar month before the S2C era."""
    d = obs.assign(month=pd.to_datetime(obs["obs_date"]).dt.month)
    pre = d[pd.to_datetime(d["obs_date"]) < pd.Timestamp(S2C_ERA_START)]
    med = pre.groupby(["reach_id", "month"])[var].median().rename("_med")
    joined = d.join(med, on=["reach_id", "month"])
    return joined[var] - joined["_med"]


def compare(a: np.ndarray, b: np.ndarray, seed: int = 0) -> dict[str, Any]:
    """Mean difference a - b with a bootstrap 95% CI, plus Mann-Whitney U if scipy is
    available. Returns counts even when a side is empty."""
    out: dict[str, Any] = {"n_a": int(len(a)), "n_b": int(len(b))}
    if len(a) < 5 or len(b) < 5:
        out["verdict"] = "TOO_FEW_OBSERVATIONS"
        return out
    rng = np.random.default_rng(seed)
    diffs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(BOOTSTRAP)]
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    pooled_sd = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
    out.update(
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        mean_diff=float(a.mean() - b.mean()),
        ci95=[float(lo), float(hi)],
        median_diff=float(np.median(a) - np.median(b)),
        effect_size_d=float((a.mean() - b.mean()) / pooled_sd) if pooled_sd > 0 else None,
    )
    try:
        from scipy.stats import mannwhitneyu

        out["mann_whitney_p"] = float(mannwhitneyu(a, b).pvalue)
    except ImportError:
        out["mann_whitney_p"] = None
    out["verdict"] = "SHIFT" if (lo > 0 or hi < 0) else "NO_DETECTABLE_SHIFT"
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--city", default="coimbra")
    args = parser.parse_args(argv)
    cfg = load_city_config(args.city)
    obs = load_ok_observations(args.city)
    if obs.empty:
        raise RuntimeError("no OK observations - run L1 first")
    obs["obs_date"] = pd.to_datetime(obs["obs_date"])
    platforms = platform_by_day(
        cfg["bbox"], obs["obs_date"].min().date(), obs["obs_date"].max().date()
    )
    obs["platform"] = obs["obs_date"].map(platforms).fillna("NOT_IN_STAC")
    era = np.where(obs["obs_date"] >= pd.Timestamp(S2C_ERA_START), "S2C_ERA", "PRE_S2C")
    obs["era"] = era

    report: dict[str, Any] = {
        "city": args.city,
        "s2c_era_start": S2C_ERA_START.isoformat(),
        "platform_counts": {
            e: obs.loc[obs["era"] == e, "platform"].value_counts().to_dict()
            for e in ("PRE_S2C", "S2C_ERA")
        },
        "method": "per reach-month median (pre-S2C) removed; MIXED platform days excluded",
        "variables": {},
    }
    for var in VARIABLES:
        r = obs.assign(resid=deseasonalise(obs, var)).dropna(subset=["resid"])
        s2c = r[r["era"] == "S2C_ERA"]
        pre = r[r["era"] == "PRE_S2C"]
        report["variables"][var] = {
            "contemporaneous_S2C_minus_S2B": compare(
                s2c.loc[s2c["platform"] == "S2C", "resid"].to_numpy(),
                s2c.loc[s2c["platform"] == "S2B", "resid"].to_numpy(),
            ),
            "reference_pre_S2A_minus_S2B": compare(
                pre.loc[pre["platform"] == "S2A", "resid"].to_numpy(),
                pre.loc[pre["platform"] == "S2B", "resid"].to_numpy(),
            ),
            "before_after_confounded_S2C_era_minus_pre": compare(
                s2c["resid"].to_numpy(), pre["resid"].to_numpy()
            ),
        }
    out = REPO_ROOT / "results" / "s2_platform_shift.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
