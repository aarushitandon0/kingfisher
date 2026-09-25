"""Production calibration: the CQR fit that the head-to-head scored is the one applied to
the forecasts the alert engine reads. Variant B (the scenario model) is never touched,
LIVE rows use the as-issued fit, and a city without one falls back to ORACLE loudly
(in the record), never silently to nothing.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from models import production_calibration as PC

TR = {"ndci": "identity", "turbidity_proxy": "log1p"}
BUCKETS = {"h01_03": (1, 2, 3), "h04_07": (4, 5, 6, 7), "h08_10": (8, 9, 10)}
VAL_END = date(2024, 12, 31)
CCFG = {"target_coverage": 0.80, "min_conformal_rows": 30}
Q = ("p05", "p10", "p25", "p50", "p75", "p90", "p95")
# A deliberately over-confident model: P10-P90 = +-0.3 around a truth with sd 1.
OFFS = (-0.4, -0.3, -0.15, 0.0, 0.15, 0.3, 0.4)


def _rows(fold: str, weather: str, variant: str = "A", n_days: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0 if fold == "val" else 1)
    start = date(2024, 3, 1) if fold == "val" else date(2025, 3, 1)
    recs = []
    for i in range(n_days):
        t = start + timedelta(days=i)
        for h in range(1, 11):
            r = {
                "fold": fold,
                "variable": "ndci",
                "reach_id": "CMB-0001",
                "issued_date": t,
                "horizon": h,
                "target_date": t + timedelta(days=h),
                "target": float(rng.normal()),
                "reach_observable": True,
                "weather": weather,
                "variant": variant,
            }
            for c, o in zip(Q, OFFS, strict=True):
                r[c] = o
            recs.append(r)
    return pd.DataFrame(recs)


def _fits(ev: pd.DataFrame) -> dict:
    return PC.fit_all(ev, transforms=TR, buckets=BUCKETS, ccfg=CCFG, val_end=VAL_END)


def test_calibration_restores_held_out_coverage() -> None:
    ev = pd.concat([_rows("val", "ORACLE"), _rows("test", "ORACLE")], ignore_index=True)
    fits = _fits(ev)
    test = ev[ev["fold"] == "test"]
    raw = PC.coverage_80(test, TR)
    after = PC.coverage_80(PC.apply(test, fits, transforms=TR, buckets=BUCKETS), TR)
    assert raw is not None and after is not None
    assert raw < 0.4  # the fixture really is over-confident
    assert after > 0.7  # and the held-out rows are covered after calibration


def test_variant_b_is_never_touched() -> None:
    ev = _rows("val", "ORACLE")
    fits = _fits(ev)
    b = _rows("test", "ORACLE", variant="B")
    out = PC.apply(b, fits, transforms=TR, buckets=BUCKETS)
    pd.testing.assert_frame_equal(out[list(Q)], b[list(Q)])


def test_live_uses_asissued_fit_when_present() -> None:
    ev = pd.concat([_rows("val", "ORACLE"), _rows("val", "ASISSUED")], ignore_index=True)
    fits = _fits(ev)
    assert PC.fit_used_for("LIVE", fits) == "ASISSUED"
    assert PC.fit_used_for("ORACLE", fits) == "ORACLE"


def test_live_falls_back_to_oracle_without_asissued() -> None:
    fits = _fits(_rows("val", "ORACLE"))
    assert "ASISSUED" not in fits
    assert PC.fit_used_for("LIVE", fits) == "ORACLE"
    live = _rows("test", "LIVE")
    out = PC.apply(live, fits, transforms=TR, buckets=BUCKETS)
    assert (out["p90"] - out["p10"] > live["p90"] - live["p10"]).all()


def test_no_oracle_validation_rows_fails_loudly() -> None:
    with pytest.raises(RuntimeError, match="no ORACLE"):
        _fits(_rows("val", "ASISSUED"))


def test_fit_refuses_test_fold_rows() -> None:
    ev = _rows("val", "ORACLE")
    ev.loc[ev.index[:5], "issued_date"] = date(2025, 6, 1)
    with pytest.raises(AssertionError):
        _fits(ev)


def test_unassessed_rows_are_not_adjusted() -> None:
    fits = _fits(_rows("val", "ORACLE"))
    rows = _rows("test", "ORACLE").assign(reach_observable=None)
    out = PC.apply(rows, fits, transforms=TR, buckets=BUCKETS)
    pd.testing.assert_frame_equal(out[list(Q)], rows[list(Q)])
