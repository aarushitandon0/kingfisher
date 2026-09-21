"""CMAL mixture maths (models/cmal.py) and NaN-target masking in NeuralHydrology's own
CMAL loss - proved on a toy batch, not assumed (P5.1 item 4).

The masking tests call neuralhydrology.training.loss.MaskedCMALLoss - the exact class the
training runs use (config loss: CMALLoss) - and check three things a leak would break:
the loss equals the loss over the non-NaN samples alone, it does not move when a masked
sample's predictions move, and the gradient reaching a masked sample is exactly zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from models import cmal

RNG = np.random.default_rng(20260921)


def _mix(n: int = 4, k: int = 3, seed: int = 0) -> cmal.Mixture:
    r = np.random.default_rng(seed)
    return cmal.Mixture(
        mu=r.normal(0, 1, (n, k)),
        b=r.uniform(0.2, 1.5, (n, k)),
        tau=r.uniform(0.1, 0.9, (n, k)),
        pi=r.dirichlet(np.ones(k), n),
    )


def _nh_density(y: np.ndarray, mu: float, b: float, t: float) -> np.ndarray:
    """The density MaskedCMALLoss uses, transcribed."""
    e = y - mu
    return t * (1 - t) / b * np.exp(-np.maximum(t * e, (t - 1) * e) / b)


# ---------------------------------------------------------------------------
# the mixture
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("mu", "b", "t"), [(0.0, 1.0, 0.5), (2.0, 0.3, 0.1), (-1.0, 2.0, 0.85)])
def test_ald_cdf_integrates_nh_density(mu: float, b: float, t: float) -> None:
    grid = np.linspace(mu - 60 * b, mu + 60 * b, 400_001)
    dens = _nh_density(grid, mu, b, t)
    numeric = np.cumsum(dens) * (grid[1] - grid[0])
    for y in (mu - 2 * b, mu - 0.1 * b, mu, mu + 0.5 * b, mu + 3 * b):
        i = np.searchsorted(grid, y)
        closed = cmal.ald_cdf(np.array(y), np.array(mu), np.array(b), np.array(t))
        assert closed == pytest.approx(numeric[i], abs=2e-3)
    # mu is the tau-quantile
    assert cmal.ald_cdf(np.array(mu), np.array(mu), np.array(b), np.array(t)) == pytest.approx(t)


def test_quantiles_invert_cdf_and_are_monotone() -> None:
    m = _mix(6)
    levels = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
    q = cmal.quantiles(m, levels)
    assert (np.diff(q, axis=1) > 0).all()
    for j, a in enumerate(levels):
        assert cmal.cdf(m, q[:, j]) == pytest.approx(np.full(6, a), abs=1e-9)


def test_quantiles_match_sampling() -> None:
    m = _mix(3, seed=7)
    draws = cmal.sample(m, 200_000, RNG)
    q = cmal.quantiles(m, (0.1, 0.5, 0.9))
    emp = np.quantile(draws, (0.1, 0.5, 0.9), axis=1).T
    assert q == pytest.approx(emp, abs=0.03)
    assert cmal.mean(m) == pytest.approx(draws.mean(axis=1), abs=0.03)


def test_ensemble_is_equal_weight_mixture_of_members() -> None:
    members = [_mix(5, seed=s) for s in range(5)]
    ens = cmal.ensemble(members)
    assert ens.mu.shape == (5, 15)
    assert ens.pi.sum(axis=1) == pytest.approx(np.ones(5))
    y = RNG.normal(0, 1, 5)
    assert cmal.cdf(ens, y) == pytest.approx(np.mean([cmal.cdf(m, y) for m in members], axis=0))


def test_nh_eps_weights_are_renormalised() -> None:
    m = _mix(2)
    eps_pi = (1 - 1e-5) * m.pi + 1e-5  # what NH's head emits
    m2 = cmal.Mixture(m.mu, m.b, m.tau, eps_pi)
    far = np.array([1e6, 1e6])  # beyond every component's tail
    assert (cmal.cdf(m2, far) <= 1.0).all()
    assert cmal.cdf(m2, far) == pytest.approx([1.0, 1.0], abs=1e-12)
    # without renormalising, the eps weights would sum above 1
    assert (eps_pi.sum(axis=1) > 1.0).all()


def test_rescale_undoes_normalisation_exactly() -> None:
    m = _mix(4)
    center, scale = 3.2, 0.7
    r = cmal.rescale(m, center, scale)
    y = RNG.normal(0, 1, 4)
    assert cmal.cdf(r, y * scale + center) == pytest.approx(cmal.cdf(m, y))


def test_shift_moves_every_quantile_by_delta() -> None:
    m = _mix(4)
    d = np.array([0.5, -1.0, 0.0, 2.0])
    q0 = cmal.quantiles(m, (0.1, 0.5, 0.9))
    q1 = cmal.quantiles(m.shifted(d), (0.1, 0.5, 0.9))
    assert q1 - q0 == pytest.approx(np.repeat(d[:, None], 3, axis=1), abs=1e-9)


def test_invalid_row_is_nan_never_zero() -> None:
    m = _mix(3)
    mu = m.mu.copy()
    mu[1, 0] = np.nan
    bad = cmal.Mixture(mu, m.b, m.tau, m.pi)
    assert np.isnan(cmal.cdf(bad, np.zeros(3))[1])
    assert np.isnan(cmal.quantiles(bad, (0.5,))[1]).all()
    assert np.isfinite(cmal.quantiles(bad, (0.5,))[[0, 2]]).all()


# ---------------------------------------------------------------------------
# NaN targets are masked out of NeuralHydrology's CMAL loss
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch")
nh_loss = pytest.importorskip("neuralhydrology.training.loss")
nh_config = pytest.importorskip("neuralhydrology.utils.config")


def _loss_obj(n_dist: int = 3) -> object:
    cfg = nh_config.Config(
        {"predict_last_n": 1, "target_variables": ["y"], "n_distributions": n_dist},
        dev_mode=True,
    )
    return nh_loss.MaskedCMALLoss(cfg)


def _toy_batch(bs: int = 8, seq: int = 5, k: int = 3) -> tuple[dict, dict, list[int]]:
    g = torch.Generator().manual_seed(0)
    y = torch.randn(bs, seq, 1, generator=g)
    masked = [1, 4, 6]
    y[masked, -1, 0] = float("nan")  # the predicted (last) step is unobserved
    y[0, 0, 0] = float("nan")  # a NaN outside predict_last_n: irrelevant to the loss
    pred = {
        "mu": torch.randn(bs, seq, k, generator=g).requires_grad_(),
        "b": (torch.rand(bs, seq, k, generator=g) + 0.2).requires_grad_(),
        "tau": (torch.rand(bs, seq, k, generator=g) * 0.8 + 0.1).requires_grad_(),
        "pi": torch.softmax(torch.randn(bs, seq, k, generator=g), -1).detach().requires_grad_(),
    }
    return pred, {"y": y}, masked


def test_nh_cmal_loss_ignores_nan_targets() -> None:
    loss_fn = _loss_obj()
    pred, data, masked = _toy_batch()
    total, _ = loss_fn(pred, data)
    assert torch.isfinite(total)

    keep = [i for i in range(8) if i not in masked]
    sub_pred = {k: v.detach()[keep] for k, v in pred.items()}
    sub_total, _ = loss_fn(sub_pred, {"y": data["y"][keep]})
    assert float(total) == pytest.approx(float(sub_total), rel=1e-6)


def test_nh_cmal_loss_invariant_to_masked_predictions() -> None:
    loss_fn = _loss_obj()
    pred, data, masked = _toy_batch()
    base = float(loss_fn(pred, data)[0])
    wild = {k: v.detach().clone() for k, v in pred.items()}
    wild["mu"][masked] = 1e6  # absurd predictions where there is no target
    wild["b"][masked] = 1e-3
    assert float(loss_fn(wild, data)[0]) == pytest.approx(base, rel=1e-6)


def test_nh_cmal_loss_sends_zero_gradient_to_masked_samples() -> None:
    loss_fn = _loss_obj()
    pred, data, masked = _toy_batch()
    total, _ = loss_fn(pred, data)
    total.backward()
    keep = [i for i in range(8) if i not in masked]
    for name in ("mu", "b", "tau", "pi"):
        g = pred[name].grad
        assert torch.all(g[masked] == 0), f"{name}: gradient leaks into NaN-target samples"
        assert torch.any(g[keep] != 0), f"{name}: no gradient at all - test is vacuous"


def test_nh_cmal_loss_would_be_nan_without_the_mask() -> None:
    """Guard against a vacuous test: the NaNs really are in the predicted step, so an
    unmasked likelihood over the batch is NaN."""
    pred, data, _ = _toy_batch()
    y = data["y"][:, -1:, :]
    e = y - pred["mu"][:, -1:, :]
    t, b = pred["tau"][:, -1:, :], pred["b"][:, -1:, :]
    ll = torch.log(t) + torch.log(1 - t) - torch.log(b) - torch.max(t * e, (t - 1) * e) / b
    assert torch.isnan(ll).any()


def test_select_epoch_criterion_equals_nh_loss() -> None:
    """models.ealstm.cmal_nll (epoch selection) is the same number as NH's loss."""
    from models.ealstm import cmal_nll

    loss_fn = _loss_obj()
    pred, data, masked = _toy_batch()
    keep = [i for i in range(8) if i not in masked]
    nh = float(loss_fn({k: v.detach()[keep] for k, v in pred.items()}, {"y": data["y"][keep]})[0])
    last = {k: v.detach()[keep][:, -1, :].double().numpy() for k, v in pred.items()}
    mix = cmal.Mixture(last["mu"], last["b"], last["tau"], last["pi"])
    ours = cmal_nll(mix, data["y"][keep][:, -1, 0].double().numpy())
    assert ours.mean() == pytest.approx(nh, rel=1e-4)
