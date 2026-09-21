"""Exceedance probability from a P10/P50/P90 forecast - pure, no I/O.

The forecast carries three quantiles, not a full distribution, so P(state > threshold)
needs a stated CDF. This is it, and it is the ONLY place it is defined: the alert engine
issues alerts with it and models/evaluate.py scores the reliability diagram with it, so
the probability that is calibrated is the probability that is alerted on.

CDF: piecewise linear through (p10, 0.1), (p50, 0.5), (p90, 0.9). The tails continue at
the slope of the adjacent segment and are clipped to [0, 1], i.e. the lower tail reaches
0 at p10 - (p50 - p10)/4 and the upper tail reaches 1 at p90 + (p90 - p50)/4. A
zero-width segment is a point mass. Missing input -> NaN out, never 0.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def quantile_cdf(
    p10: npt.ArrayLike, p50: npt.ArrayLike, p90: npt.ArrayLike, x: npt.ArrayLike
) -> FloatArray:
    """F(x) under the piecewise-linear quantile CDF. Arrays broadcast."""
    q10, q50, q90, v = np.broadcast_arrays(
        *(np.asarray(a, dtype="float64") for a in (p10, p50, p90, x))
    )
    known = ~(np.isnan(q10) | np.isnan(q50) | np.isnan(q90) | np.isnan(v))
    if np.any(known & ((q10 > q50) | (q50 > q90))):
        raise ValueError("quantiles cross (p10 <= p50 <= p90 violated) - rearrange first")

    w_lo = q50 - q10
    w_hi = q90 - q50
    with np.errstate(divide="ignore", invalid="ignore"):
        lower_tail = np.where(w_lo > 0, np.maximum(0.0, 0.1 - 0.4 * (q10 - v) / w_lo), 0.0)
        lower_mid = 0.1 + 0.4 * (v - q10) / w_lo
        upper_mid = 0.5 + 0.4 * (v - q50) / w_hi
        upper_tail = np.where(
            w_hi > 0,
            np.minimum(1.0, 0.9 + 0.4 * (v - q90) / w_hi),
            np.where(v > q90, 1.0, 0.9),
        )
    cdf = np.select(
        [v < q10, v < q50, v < q90],
        [lower_tail, lower_mid, upper_mid],
        default=upper_tail,
    )
    return np.where(known, cdf, np.nan)


def exceedance_probability(
    p10: npt.ArrayLike, p50: npt.ArrayLike, p90: npt.ArrayLike, threshold: npt.ArrayLike
) -> FloatArray:
    """P(state > threshold) = 1 - F(threshold). NaN wherever an input is NaN."""
    return 1.0 - quantile_cdf(p10, p50, p90, threshold)
