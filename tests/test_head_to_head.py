"""P5.4 - the gate (decided on validation only), scenario-sensitivity verdicts, and the
differentiable mixture quantile the integrated-gradients attribution is built on."""

from __future__ import annotations

import numpy as np
import pytest

from models import cmal
from models.head_to_head import decide_gate, sensitivity_verdict, summarise_response

BUCKETS = ["h01_03", "h04_07", "h08_10"]
GATE = {"min_buckets_won": 2, "coverage_tolerance": 0.0}


def _scores(crps: dict[str, list[float]], cov: dict[str, float]) -> dict:
    return {
        var: {
            "all": {"crps": float(np.mean(c)), "coverage_80": cov[var]},
            "by_bucket": {b: {"crps": x} for b, x in zip(BUCKETS, c, strict=True)},
        }
        for var, c in crps.items()
    }


def _val(ea: dict, gbm: dict, ea_cov: dict, gbm_cov: dict) -> dict:
    return {"ealstm+assim": _scores(ea, ea_cov), "lightgbm+assim": _scores(gbm, gbm_cov)}


GBM = {"ndci": [1.0, 1.0, 1.0], "turbidity_proxy": [5.0, 5.0, 5.0]}
COV = {"ndci": 0.78, "turbidity_proxy": 0.78}


def test_ealstm_wins_two_of_three_for_both_targets() -> None:
    ea = {"ndci": [0.9, 0.95, 1.1], "turbidity_proxy": [4.0, 6.0, 4.9]}
    g = decide_gate(_val(ea, GBM, COV, COV), GATE, BUCKETS)
    assert g["production"] == "ealstm"
    assert all(v["passed"] for v in g["per_variable"].values())
    assert g["decided_on"] == "val"


def test_one_target_losing_keeps_lightgbm() -> None:
    ea = {"ndci": [0.9, 0.9, 0.9], "turbidity_proxy": [4.0, 6.0, 6.0]}  # turbidity wins 1/3
    g = decide_gate(_val(ea, GBM, COV, COV), GATE, BUCKETS)
    assert g["production"] == "lightgbm"
    assert g["per_variable"]["turbidity_proxy"]["buckets_won"] == 1


def test_worse_coverage_blocks_even_with_better_crps() -> None:
    ea = {"ndci": [0.5, 0.5, 0.5], "turbidity_proxy": [2.0, 2.0, 2.0]}
    g = decide_gate(_val(ea, GBM, {"ndci": 0.60, "turbidity_proxy": 0.79}, COV), GATE, BUCKETS)
    assert g["production"] == "lightgbm"
    assert not g["per_variable"]["ndci"]["coverage_no_worse"]


def test_tie_is_not_a_win() -> None:
    ea = {"ndci": [1.0, 1.0, 0.9], "turbidity_proxy": [5.0, 5.0, 4.0]}
    assert decide_gate(_val(ea, GBM, COV, COV), GATE, BUCKETS)["production"] == "lightgbm"


def test_missing_target_for_challenger_fails() -> None:
    ea = {"ndci": [0.1, 0.1, 0.1]}
    val = {"ealstm+assim": _scores(ea, COV), "lightgbm+assim": _scores(GBM, COV)}
    assert decide_gate(val, GATE, BUCKETS)["production"] == "lightgbm"


# ---------------------------------------------------------------------------
# sensitivity
# ---------------------------------------------------------------------------
def test_summarise_response_signs() -> None:
    reach = np.array(["A", "A", "B", "B", "C", "C"])
    p0 = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    p1 = np.array([1.5, 1.5, 1.0, 1.0, 3.0, 3.0])
    r = summarise_response(reach, p0, p1)
    assert r["reaches"] == 3
    assert r["share_up"] == pytest.approx(1 / 3)
    assert r["share_down"] == pytest.approx(1 / 3)
    assert r["share_no_change"] == pytest.approx(1 / 3)


def test_wrong_sign_is_printed_plainly() -> None:
    r = {"reaches": 10, "share_up": 0.2, "share_down": 0.7, "share_no_change": 0.1}
    v = sensitivity_verdict("LightGBM B", "turbidity_proxy", r)
    assert v is not None and v.startswith("WRONG SIGN") and "70%" in v
    ok = {"reaches": 10, "share_up": 0.8, "share_down": 0.1, "share_no_change": 0.1}
    assert sensitivity_verdict("EA-LSTM", "turbidity_proxy", ok) is None
    flat = {"reaches": 10, "share_up": 0.1, "share_down": 0.1, "share_no_change": 0.8}
    assert sensitivity_verdict("EA-LSTM", "turbidity_proxy", flat).startswith("INSENSITIVE")  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# differentiable mixture quantile (integrated gradients)
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch")


def test_torch_quantile_value_and_implicit_gradient() -> None:
    from models.ealstm import _torch_mixture_quantile

    rng = np.random.default_rng(4)
    mu = rng.normal(0, 1, (3, 6))
    b = rng.uniform(0.2, 1.0, (3, 6))
    tau = rng.uniform(0.2, 0.8, (3, 6))
    pi = rng.dirichlet(np.ones(6), 3)
    ref = cmal.quantiles(cmal.Mixture(mu, b, tau, pi), (0.5,))[:, 0]
    t = lambda a: torch.tensor(a, dtype=torch.float64, requires_grad=True)  # noqa: E731
    tm, tb, tt, tp = t(mu), t(b), t(tau), t(pi)
    q = _torch_mixture_quantile(tm, tb, tt, tp, 0.5)
    assert q.detach().numpy() == pytest.approx(ref, abs=1e-9)
    q.sum().backward()
    # finite differences on every mu
    eps = 1e-6
    fd = np.zeros_like(mu)
    for i in range(3):
        for k in range(6):
            up, dn = mu.copy(), mu.copy()
            up[i, k] += eps
            dn[i, k] -= eps
            fd[i, k] = (
                cmal.quantiles(cmal.Mixture(up, b, tau, pi), (0.5,))[i, 0]
                - cmal.quantiles(cmal.Mixture(dn, b, tau, pi), (0.5,))[i, 0]
            ) / (2 * eps)
    assert tm.grad.numpy() == pytest.approx(fd, abs=1e-5)
    # a location shift of every component moves the median one-for-one
    assert tm.grad.numpy().sum(axis=1) == pytest.approx(np.ones(3), abs=1e-8)
