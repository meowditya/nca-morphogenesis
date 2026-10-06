"""Finite-difference check of the hand-written backward pass in tools/train_numpy.py.

Run: python tests/test_gradients.py        # or: python -m pytest tests
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from train_numpy import backward, init_params, rollout  # noqa: E402


def check(seed=0, B=2, H=7, W=8, C=6, hidden=5, steps=4, eps=1e-6, island=False):
    rng = np.random.default_rng(seed)
    params = init_params(C, hidden, rng, dtype=np.float64)
    params["W2"] = rng.normal(0, 0.3, params["W2"].shape)   # non-zero so every path carries gradient
    x0 = rng.uniform(0, 1, (B, H, W, C))
    x0[:, :3, :3] = 0.0                                       # a dead corner, so life masks matter
    if island:                                                # live patch inside an empty grid, so the
        big = np.zeros((B, H + 9, W + 7, C))                  # trainer's cropping is exercised
        big[:, 5:5 + H, 2:2 + W] = x0
        x0 = big
    weights = rng.normal(size=x0.shape)                       # L = sum(weights * x_final)

    def loss(p, x):
        xf, _ = rollout(x, p, steps, np.random.default_rng(123), 0.5, 0.1)
        return float((weights * xf).sum())

    xf, cache = rollout(x0, params, steps, np.random.default_rng(123), 0.5, 0.1)
    grads, gx0 = backward(weights.copy(), cache, params)

    worst = 0.0
    for name in ("W1", "b1", "W2"):
        for flat in rng.choice(params[name].size, min(12, params[name].size), replace=False):
            i = np.unravel_index(flat, params[name].shape)
            p_plus = {k: v.copy() for k, v in params.items()}
            p_minus = {k: v.copy() for k, v in params.items()}
            p_plus[name][i] += eps
            p_minus[name][i] -= eps
            num = (loss(p_plus, x0) - loss(p_minus, x0)) / (2 * eps)
            rel = abs(num - grads[name][i]) / max(1e-8, abs(num) + abs(grads[name][i]))
            worst = max(worst, rel)
    for flat in rng.choice(x0.size, 20, replace=False):
        i = np.unravel_index(flat, x0.shape)
        if x0[i] == 0.0:
            continue
        xp, xm = x0.copy(), x0.copy()
        xp[i] += eps
        xm[i] -= eps
        num = (loss(params, xp) - loss(params, xm)) / (2 * eps)
        rel = abs(num - gx0[i]) / max(1e-8, abs(num) + abs(gx0[i]))
        worst = max(worst, rel)
    return worst


def crop_matches_full_grid(seed=0, steps=30):
    """With every cell firing, the trainer's cropped rollout must equal plain full-grid steps."""
    from nca.numpy_nca import make_seed, step
    rng = np.random.default_rng(seed)
    params = init_params(16, 32, rng, dtype=np.float64)
    params["W2"] = rng.normal(0, 0.05, params["W2"].shape)
    params["W2"][:, 3] = np.abs(params["W2"][:, 3])            # alpha grows, so the organism spreads
    x0 = make_seed(2, 30, 26, 16, np.float64)
    xc, _ = rollout(x0, params, steps, rng, 1.0, 0.1)
    xf = x0
    for _ in range(steps):
        xf = step(xf, params, np.ones(xf.shape[:3] + (1,), bool), 0.1)
    return np.abs(xc - xf).max(), int((np.abs(xf).sum(-1) > 0).sum())


def test_step_rule_matches_masked_full_grid():
    """nca.numpy_nca.step must equal the PyTorch rule: compute dx on the full grid,
    apply it only to cells alive before the step, then clear cells with no mature
    neighbour before and after. (Computing dx for dead cells too, as the Distill code
    does, gives a different result, which is what this guards against.)"""
    from nca.checkpoint import load_npz
    from nca.numpy_nca import alive_mask, make_seed, step, update
    p, meta = load_npz(os.path.join(ROOT, "checkpoints", "spiderweb_starter.npz"))
    thr, rng = meta["alive_threshold"], np.random.default_rng(0)
    a = b = make_seed(1, 48, 48, meta["channels"])
    worst = 0.0
    for _ in range(60):
        fire = rng.random((1, 48, 48, 1)) < 0.5
        pre = alive_mask(b, thr)
        dx, _, _ = update(b, p)                       # full grid, like a conv net
        b = b + dx * (fire & pre)
        b = b * (pre & alive_mask(b, thr))
        a = step(a, p, fire, thr)
        worst = max(worst, float(np.abs(a - b).max()))
    print(f"numpy step vs full-grid masked rule: max |diff| {worst:.1e}")
    assert worst < 1e-4


def test_crop_matches_full_grid():
    diff, live = crop_matches_full_grid()
    print(f"cropped vs full-grid rollout: max |diff| {diff:.1e} over {live} live cells")
    assert diff < 1e-9 and live > 50


def test_backward_matches_finite_differences():
    results = [check(seed=s) for s in range(3)] + [check(seed=s, island=True, steps=6) for s in range(3)]
    for s, r in enumerate(results):
        print(f"case {s} ({'island' if s >= 3 else 'full'}): worst relative error {r:.2e}")
    assert max(results) < 1e-5, "backward pass disagrees with finite differences"


if __name__ == "__main__":
    test_step_rule_matches_masked_full_grid()
    test_crop_matches_full_grid()
    test_backward_matches_finite_differences()
    print("gradient check passed")
