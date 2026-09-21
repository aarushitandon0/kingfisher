"""Countable Mixture of Asymmetric Laplacians (CMAL) - the EA-LSTM's predictive
distribution, as pure numpy. No torch, no I/O.

Klotz, Kratzert, Gauch, Sampson, Brandstetter, Klambauer, Hochreiter & Nearing (2022),
"Uncertainty estimation with deep learning for rainfall-runoff modeling", HESS 26:1673.

ONE COMPONENT - asymmetric Laplace, location mu, scale b > 0, asymmetry tau in (0, 1)
-------------------------------------------------------------------------------------
The density NeuralHydrology's MaskedCMALLoss maximises is

    f(y) = tau (1 - tau) / b * exp(-rho_tau((y - mu) / b)),  rho_tau(u) = max(tau u, (tau - 1) u)

which integrates to the closed-form CDF

    F(y) = tau * exp((1 - tau) (y - mu) / b)              y <  mu
    F(y) = 1 - (1 - tau) * exp(-tau (y - mu) / b)          y >= mu

so F(mu) = tau: mu is the component's tau-quantile, not its mean.

THE SERVED DISTRIBUTION
-----------------------
One network: a pi-weighted mixture of K components. The ensemble: an EQUAL-WEIGHT
mixture of the S seeds' mixtures (config/modelling.yaml ealstm.seeds), i.e. one
mixture of S*K components with weights pi_k / S. Quantiles come from inverting that
mixture CDF by bisection (the CDF is continuous and strictly increasing, so the inverse
is unique); no sampling, so the numbers are reproducible.

Parameters arrive with a leading "row" axis and a trailing component axis: shape
(n, C) for C = S*K components. A row with any NaN parameter gives NaN everywhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

BISECTION_STEPS = 60  # interval halves 60 times: far below float32 resolution of the inputs


@dataclass(frozen=True)
class Mixture:
    """A batch of CMAL mixtures, one per row. Arrays are (n, C)."""

    mu: FloatArray
    b: FloatArray
    tau: FloatArray
    pi: FloatArray

    def __post_init__(self) -> None:
        shapes = {a.shape for a in (self.mu, self.b, self.tau, self.pi)}
        if len(shapes) != 1 or self.mu.ndim != 2:
            raise ValueError(f"CMAL parameters must share one (n, C) shape, got {shapes}")

    @property
    def n(self) -> int:
        return int(self.mu.shape[0])

    def valid(self) -> npt.NDArray[np.bool_]:
        """Rows whose parameters are all finite and in range."""
        finite = np.all(
            np.isfinite(self.mu)
            & np.isfinite(self.b)
            & np.isfinite(self.tau)
            & np.isfinite(self.pi),
            axis=1,
        )
        in_range = np.all((self.b > 0) & (self.tau > 0) & (self.tau < 1) & (self.pi >= 0), axis=1)
        return np.asarray(finite & in_range)

    def shifted(self, delta: npt.ArrayLike) -> Mixture:
        """Location shift of every component by delta (n,) - exact for a location family."""
        d = np.asarray(delta, dtype="float64").reshape(-1, 1)
        return Mixture(self.mu + d, self.b, self.tau, self.pi)

    def perturbed_scale(self, factor: npt.ArrayLike) -> Mixture:
        f = np.asarray(factor, dtype="float64").reshape(-1, 1)
        return Mixture(self.mu, self.b * f, self.tau, self.pi)

    def rows(self, index: npt.ArrayLike) -> Mixture:
        i = np.asarray(index)
        return Mixture(self.mu[i], self.b[i], self.tau[i], self.pi[i])


def normalise_weights(pi: npt.ArrayLike) -> FloatArray:
    """NeuralHydrology's head adds eps to every weight (sum = 1 + (K-1) eps); renormalise."""
    p = np.asarray(pi, dtype="float64")
    return np.asarray(p / p.sum(axis=-1, keepdims=True))


def ensemble(members: Sequence[Mixture]) -> Mixture:
    """Equal-weight mixture of S member mixtures: concatenate components, pi / S."""
    if not members:
        raise ValueError("an ensemble needs at least one member")
    n = {m.n for m in members}
    if len(n) != 1:
        raise ValueError(f"members disagree on the number of rows: {n}")
    s = len(members)
    return Mixture(
        mu=np.concatenate([m.mu for m in members], axis=1),
        b=np.concatenate([m.b for m in members], axis=1),
        tau=np.concatenate([m.tau for m in members], axis=1),
        pi=np.concatenate([normalise_weights(m.pi) / s for m in members], axis=1),
    )


def ald_cdf(y: FloatArray, mu: FloatArray, b: FloatArray, tau: FloatArray) -> FloatArray:
    """Asymmetric-Laplace CDF, elementwise (broadcasting)."""
    z = (y - mu) / b
    with np.errstate(over="ignore"):
        lower = tau * np.exp(np.minimum((1.0 - tau) * z, 0.0))
        upper = 1.0 - (1.0 - tau) * np.exp(np.minimum(-tau * z, 0.0))
    return np.asarray(np.where(z < 0, lower, upper))


def cdf(m: Mixture, y: npt.ArrayLike) -> FloatArray:
    """F(y) per row. y: (n,) or scalar. NaN for an invalid row or a NaN y."""
    v = np.broadcast_to(np.asarray(y, dtype="float64"), (m.n,))
    w = normalise_weights(m.pi)
    comp = ald_cdf(v[:, None], m.mu, m.b, m.tau)
    out = np.sum(w * comp, axis=1)
    return np.asarray(np.where(m.valid() & np.isfinite(v), out, np.nan))


def cdf_grid(m: Mixture, y: npt.ArrayLike) -> FloatArray:
    """F at every value of y for every row: (n, len(y))."""
    grid = np.asarray(y, dtype="float64")
    w = normalise_weights(m.pi)
    comp = ald_cdf(grid[None, :, None], m.mu[:, None, :], m.b[:, None, :], m.tau[:, None, :])
    out = np.sum(w[:, None, :] * comp, axis=2)
    return np.asarray(np.where(m.valid()[:, None], out, np.nan))


def _bracket(m: Mixture) -> tuple[FloatArray, FloatArray]:
    """Per-row [lo, hi] with F(lo) < 1e-9 and F(hi) > 1 - 1e-9 for every component:
    an ALD's tails decay like exp(-min(tau, 1-tau) |z|), so 40 / min(tau, 1-tau) scales
    past the component covers it."""
    reach = 40.0 * m.b / np.minimum(m.tau, 1.0 - m.tau)
    return np.min(m.mu - reach, axis=1), np.max(m.mu + reach, axis=1)


def quantiles(m: Mixture, levels: Sequence[float]) -> FloatArray:
    """(n, len(levels)) quantiles by bisection on the mixture CDF. Monotone in the level
    by construction (one bracket, one CDF)."""
    lv = np.asarray(levels, dtype="float64")
    if np.any((lv <= 0) | (lv >= 1)):
        raise ValueError(f"levels must be inside (0, 1): {levels}")
    ok = m.valid()
    out = np.full((m.n, len(lv)), np.nan)
    if not ok.any():
        return out
    mm = m.rows(np.flatnonzero(ok))
    lo0, hi0 = _bracket(mm)
    lo = np.repeat(lo0[:, None], len(lv), axis=1)
    hi = np.repeat(hi0[:, None], len(lv), axis=1)
    w = normalise_weights(mm.pi)
    for _ in range(BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        f = np.sum(
            w[:, None, :]
            * ald_cdf(mid[:, :, None], mm.mu[:, None, :], mm.b[:, None, :], mm.tau[:, None, :]),
            axis=2,
        )
        below = f < lv[None, :]
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    out[ok] = 0.5 * (lo + hi)
    return out


def mean(m: Mixture) -> FloatArray:
    """Mixture mean. An ALD(mu, b, tau) has mean mu + b (1 - 2 tau) / (tau (1 - tau))."""
    w = normalise_weights(m.pi)
    comp = m.mu + m.b * (1.0 - 2.0 * m.tau) / (m.tau * (1.0 - m.tau))
    return np.asarray(np.where(m.valid(), np.sum(w * comp, axis=1), np.nan))


def rescale(m: Mixture, center: float, scale: float) -> Mixture:
    """Undo NeuralHydrology's target normalisation y_n = (y - center) / scale. Location
    and scale transform; tau and pi are invariant."""
    if not scale > 0:
        raise ValueError(f"target scale must be > 0, got {scale}")
    return Mixture(m.mu * scale + center, m.b * scale, m.tau, m.pi)


def sample(m: Mixture, n_samples: int, rng: np.random.Generator) -> FloatArray:
    """(n, n_samples) draws - for tests only; served numbers never come from sampling."""
    w = normalise_weights(m.pi)
    u_comp = rng.random((m.n, n_samples))
    idx = (u_comp[:, :, None] > np.cumsum(w, axis=1)[:, None, :]).sum(axis=2)
    idx = np.minimum(idx, m.mu.shape[1] - 1)
    take = np.take_along_axis
    mu, b, t = (take(a, idx, axis=1) for a in (m.mu, m.b, m.tau))
    u = rng.random((m.n, n_samples))
    # inverse ALD CDF
    left = u < t
    with np.errstate(divide="ignore"):
        y_left = mu + b / (1.0 - t) * np.log(u / t)
        y_right = mu - b / t * np.log((1.0 - u) / (1.0 - t))
    return np.asarray(np.where(left, y_left, y_right))
