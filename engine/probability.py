"""Exceedance probability from a quantile forecast - pure, no I/O.

The forecast carries K quantiles (the baseline emits seven: 0.05, 0.10, 0.25, 0.50, 0.75,
0.90, 0.95), not a full distribution, so P(state > threshold) needs a stated CDF. This is
it, and it is the ONLY place it is defined: the alert engine issues alerts with it and
models/evaluate.py scores the reliability diagram with it, so the probability that is
calibrated is the probability that is alerted on.

CDF: piecewise linear through the points (q_k, alpha_k). Below the lowest quantile the
first segment's slope continues down to F = 0; above the highest, the last segment's
slope continues up to F = 1 - i.e. with the default levels the lower tail reaches 0 at
q05 - (q10 - q05) and the upper tail reaches 1 at q95 + (q95 - q90). A zero-width segment
is a point mass. Missing input -> NaN out, never 0.

`quantile_cdf` / `exceedance_probability` keep the original three-quantile signature
(P10/P50/P90) as a special case of the same construction.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

DEFAULT_LEVELS: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def _check_levels(levels: Sequence[float]) -> FloatArray:
    lv = np.asarray(levels, dtype="float64")
    if lv.ndim != 1 or len(lv) < 2:
        raise ValueError("need at least two quantile levels")
    if np.any(np.diff(lv) <= 0) or lv[0] <= 0 or lv[-1] >= 1:
        raise ValueError(f"levels must be strictly increasing inside (0, 1): {levels}")
    return lv


def quantile_cdf_multi(
    quantiles: npt.ArrayLike, levels: Sequence[float], x: npt.ArrayLike
) -> FloatArray:
    """F(x) under the piecewise-linear CDF through (quantiles[..., k], levels[k]).

    quantiles: shape (..., K), K = len(levels); x broadcasts against quantiles[..., 0].
    """
    lv = _check_levels(levels)
    q = np.asarray(quantiles, dtype="float64")
    if q.shape[-1] != len(lv):
        raise ValueError(f"last axis of quantiles is {q.shape[-1]}, levels has {len(lv)}")
    v = np.asarray(x, dtype="float64")
    shape = np.broadcast_shapes(q.shape[:-1], v.shape)
    q = np.broadcast_to(q, (*shape, len(lv)))
    v = np.broadcast_to(v, shape)

    known = ~(np.isnan(q).any(axis=-1) | np.isnan(v))
    if np.any(known & (np.diff(q, axis=-1) < 0).any(axis=-1)):
        raise ValueError("quantiles cross (non-decreasing order violated) - rearrange first")

    K = len(lv)
    cdf = np.full(shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        # lower tail
        w0 = q[..., 1] - q[..., 0]
        lo_tail = np.where(
            w0 > 0, np.maximum(0.0, lv[0] - (lv[1] - lv[0]) * (q[..., 0] - v) / w0), 0.0
        )
        cdf = np.where(v < q[..., 0], lo_tail, cdf)
        # interior segments (a zero-width segment matches nothing: a point mass)
        for k in range(K - 1):
            a, b = q[..., k], q[..., k + 1]
            seg = (v >= a) & (v < b)
            val = lv[k] + (lv[k + 1] - lv[k]) * (v - a) / (b - a)
            cdf = np.where(seg, val, cdf)
        # upper tail
        wk = q[..., K - 1] - q[..., K - 2]
        hi_tail = np.where(
            wk > 0,
            np.minimum(1.0, lv[-1] + (lv[-1] - lv[-2]) * (v - q[..., K - 1]) / wk),
            np.where(v > q[..., K - 1], 1.0, lv[-1]),
        )
        cdf = np.where(v >= q[..., K - 1], hi_tail, cdf)
    return np.where(known, cdf, np.nan)


def exceedance_probability_multi(
    quantiles: npt.ArrayLike, levels: Sequence[float], threshold: npt.ArrayLike
) -> FloatArray:
    """P(state > threshold) = 1 - F(threshold). NaN wherever an input is NaN."""
    return 1.0 - quantile_cdf_multi(quantiles, levels, threshold)


def quantile_cdf(
    p10: npt.ArrayLike, p50: npt.ArrayLike, p90: npt.ArrayLike, x: npt.ArrayLike
) -> FloatArray:
    """F(x) from P10/P50/P90 only (the three-quantile special case). Arrays broadcast."""
    q10, q50, q90, v = np.broadcast_arrays(
        *(np.asarray(a, dtype="float64") for a in (p10, p50, p90, x))
    )
    return quantile_cdf_multi(np.stack([q10, q50, q90], axis=-1), (0.1, 0.5, 0.9), v)


def exceedance_probability(
    p10: npt.ArrayLike, p50: npt.ArrayLike, p90: npt.ArrayLike, threshold: npt.ArrayLike
) -> FloatArray:
    """P(state > threshold) = 1 - F(threshold), from P10/P50/P90 only."""
    return 1.0 - quantile_cdf(p10, p50, p90, threshold)
