"""Stress tests: how a trained model behaves beyond the conditions it was trained on.

    python tools/stress_test.py --checkpoint checkpoints/spiderweb.npz

tools/evaluate.py measures damage of the same kinds and sizes as training. This
script goes further, on the 72x72 training grid unless noted:

  * growth and stability over 16 independent random seeds, up to 5,000 steps;
  * erasure: training-sized discs, then bigger discs, cutting the web in half,
    keeping only one quarter, a slot through the centre, and removing the hub;
  * noise: training strength, then stronger, over the whole organism, and on
    the hidden channels only;
  * growth from off-centre seeds in a 160x160 dish.

Each damage case is applied to 16 grown webs and run for 1,000 steps. A web counts
as recovered when its error ends below twice the undamaged median (plus 1e-4).
IoU is the overlap between solid pixels (alpha > 0.5) and the target's.
Prints a table and writes all numbers to --out (JSON).
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nca.checkpoint import load_npz  # noqa: E402
from nca.numpy_nca import alive_mask, make_seed, step  # noqa: E402
from nca.target import load_target  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=os.path.join(ROOT, "checkpoints", "spiderweb.npz"))
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="JSON file for all numbers (default: <checkpoint>_stress.json)")
    args = ap.parse_args()

    params, meta = load_npz(args.checkpoint)
    target = load_target(os.path.join(ROOT, "data", meta["target"]), meta["target_size"], meta["pad"])
    H, W, _ = target.shape
    C, thr, n = meta["channels"], meta["alive_threshold"], args.trials
    rng = np.random.default_rng(args.seed)
    ys, xs = np.mgrid[0:H, 0:W]
    cy, cx = H / 2, W / 2

    def run(x, steps):
        for _ in range(steps):
            x = step(x, params, rng.random(x.shape[:3] + (1,)) < meta["fire_rate"], thr)
        return x

    def mse(x):
        return ((x[..., :4] - target) ** 2).mean(axis=(1, 2, 3))

    def iou(x):
        a, t = x[..., 3] > 0.5, target[..., 3] > 0.5
        return (a & t).sum(axis=(1, 2)) / np.maximum((a | t).sum(axis=(1, 2)), 1)

    def disc(x0, y0, r):
        return (xs + 0.5 - x0) ** 2 + (ys + 0.5 - y0) ** 2 <= r * r

    results = {}

    # ---- growth and stability ------------------------------------------------------
    x = run(make_seed(n, H, W, C), 200)
    growth = {"t200": mse(x), "iou200": iou(x)}
    x = run(x, 800)
    growth["t1000"] = mse(x)
    x = run(x, 4000)
    growth["t5000"], growth["iou5000"] = mse(x), iou(x)
    grown = x
    base = float(np.median(growth["t5000"]))
    results["growth"] = {k: dict(median=float(np.median(v)), worst=float(v.max() if "iou" not in k else v.min()))
                         for k, v in growth.items()}
    print(f"growth over {n} seeds: MSE median/worst  t200 {np.median(growth['t200']):.5f}/{growth['t200'].max():.5f}"
          f"  t1000 {np.median(growth['t1000']):.5f}/{growth['t1000'].max():.5f}"
          f"  t5000 {np.median(growth['t5000']):.5f}/{growth['t5000'].max():.5f}"
          f"   IoU min t200 {growth['iou200'].min():.3f}  t5000 {growth['iou5000'].min():.3f}")

    # ---- damage cases -----------------------------------------------------------------
    reaches_hub = []                                     # filled by erase_discs, per trial

    def erase_discs(rmin, rmax):
        def f(x):
            reaches_hub.clear()
            for k in range(len(x)):
                x0, y0, r = rng.uniform(W / 4, 3 * W / 4), rng.uniform(H / 4, 3 * H / 4), rng.uniform(rmin, rmax)
                x[k][disc(x0, y0, r)] = 0
                reaches_hub.append(np.hypot(x0 - cx, y0 - cy) < r + 2)   # within 2 px of the centre
            return x
        return f

    def erase_mask(mask):
        def f(x):
            x[:, mask] = 0
            return x
        return f

    def noise(std, radius, hidden_only=False):
        def f(x):
            living = alive_mask(x, thr)[..., 0]
            ch = slice(4, None) if hidden_only else slice(None)
            width = C - 4 if hidden_only else C
            for k in range(len(x)):
                cells = living[k] & disc(cx, cy, radius)
                x[k][cells, ch] += rng.normal(0.0, std, (int(cells.sum()), width)).astype(x.dtype)
            return x
        return f

    half = W / 2
    cases = [
        ("erase: disc r 4-14 (training range)", erase_discs(0.1 * half, 0.4 * half)),
        ("erase: disc r 16-22 (bigger)", erase_discs(0.45 * half, 0.6 * half)),
        ("erase: left half", erase_mask(xs < cx)),
        ("erase: all but one quarter", erase_mask((xs >= cx) | (ys >= cy))),
        ("erase: 6 px slot through the centre", erase_mask(np.abs(ys + 0.5 - cy) <= 3)),
        ("erase: the hub (centre disc r 8)", erase_mask(disc(cx, cy, 8))),
        ("noise 0.6, r 8 (training range)", noise(0.6, 8)),
        ("noise 1.0, r 8 (stronger)", noise(1.0, 8)),
        ("noise 2.0, r 8 (much stronger)", noise(2.0, 8)),
        ("noise 0.6 over the whole web", noise(0.6, half + 4)),
        ("noise 1.0 on hidden channels only", noise(1.0, half + 4, hidden_only=True)),
    ]
    print(f"\n{'damage':38s} {'damaged':>9s} {'+1000 median':>13s} {'worst':>9s} {'recovered':>10s}")
    for name, damage in cases:
        reps = 3 if "training range" in name and name.startswith("erase") else 1   # more trials here
        x = damage(np.concatenate([grown] * reps))
        hit = mse(x)
        x = run(x, 1000)
        end = mse(x)
        good = end < 2 * base + 1e-4
        ok, total = int(good.sum()), len(end)
        results[name] = dict(damaged_median=float(np.median(hit)), end_median=float(np.median(end)),
                             end_worst=float(end.max()), recovered=ok, trials=total, iou_end_min=float(iou(x).min()))
        print(f"{name:38s} {np.median(hit):9.5f} {np.median(end):13.5f} {end.max():9.5f} {ok:>6d}/{total}")
        if reps > 1:
            hub = np.array(reaches_hub)
            results[name]["hub"] = dict(reached=int(hub.sum()), recovered=int(good[hub].sum()),
                                        missed=int((~hub).sum()), missed_recovered=int(good[~hub].sum()))
            print(f"{'':38s} wounds reaching the hub: {good[hub].sum()}/{hub.sum()} recovered;"
                  f" missing it: {good[~hub].sum()}/{(~hub).sum()}")

    # ---- position independence: off-centre seeds in a bigger dish -----------------------
    G = 160
    spots = [(40, 50), (118, 40), (75, 120), (121, 121)]
    x = np.zeros((len(spots), G, G, C), np.float32)
    for k, (sx, sy) in enumerate(spots):
        x[k, sy, sx, 3:] = 1.0
    x = run(x, 300)
    hy, hx = H // 2, W // 2
    crops = np.stack([x[k, sy - hy:sy + hy, sx - hx:sx + hx] for k, (sx, sy) in enumerate(spots)])
    stray = [float(np.abs(x[k, ..., 3]).sum() - np.abs(crops[k, ..., 3]).sum()) for k in range(len(spots))]
    results["offcentre_160"] = dict(mse=mse(crops).tolist(), iou=iou(crops).tolist(), alpha_outside=stray)
    print("\n160x160 dish, 4 off-centre seeds, 300 steps: MSE " + ", ".join(f"{v:.5f}" for v in mse(crops))
          + "  IoU " + ", ".join(f"{v:.3f}" for v in iou(crops))
          + "  alpha outside the web's square " + ", ".join(f"{v:.3f}" for v in stray))

    results["undamaged_median_mse"] = base
    out = args.out or os.path.splitext(args.checkpoint)[0] + "_stress.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
