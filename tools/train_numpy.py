"""Reference trainer in pure NumPy, with a hand-written backward pass.

It produced checkpoints/spiderweb_starter.npz and needs only NumPy, which also
makes it a readable reference for the maths. It follows the same recipe
as nca/train.py (pool, reseeding, disc damage, loss, per-tensor gradient
normalisation, Adam and its learning-rate drop), and its gradients are checked
against finite differences in tests/test_gradients.py. To save time it only
evaluates cells near living ones, which gives identical results (also
checked in that test). For real work use the PyTorch trainer, which is far
faster and runs on a GPU:

    python -m nca.train --help

The starter checkpoint was made with the original Distill damage settings
(from the project root):

    python tools/train_numpy.py --iters 4000 --damage 3 --noise-damage 0 \
        --out checkpoints/spiderweb_starter.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nca.checkpoint import load_npz, save_npz  # noqa: E402
from nca.numpy_nca import (  # noqa: E402
    alive_mask, make_seed, perceive, sobel_x, sobel_y, update)
from nca.target import load_target  # noqa: E402


def init_params(channels, hidden, rng, dtype=np.float32):
    """Same distribution as torch.nn.Conv2d's default init; last layer zeroed."""
    bound = 1.0 / np.sqrt(3 * channels)
    return {
        "W1": rng.uniform(-bound, bound, (3 * channels, hidden)).astype(dtype),
        "b1": rng.uniform(-bound, bound, (hidden,)).astype(dtype),
        "W2": np.zeros((hidden, channels), dtype),
    }


def live_box(x, margin=2):
    """Bounding box (y0, y1, x0, x1) of all non-zero cells in the batch, grown by `margin`.

    One step can only change cells within 1 of a live cell, and perception reads
    1 further, so with margin 2 a step computed on this crop (zero-padded) is
    identical to the full-grid step, and every cell outside it stays 0.
    """
    occ = np.any(x != 0, axis=(0, 3))
    ys, xs = np.flatnonzero(occ.any(1)), np.flatnonzero(occ.any(0))
    if ys.size == 0:
        return None
    H, W = occ.shape
    return (max(ys[0] - margin, 0), min(ys[-1] + margin + 1, H),
            max(xs[0] - margin, 0), min(xs[-1] + margin + 1, W))


def rollout(x, params, n_steps, rng, fire_rate, threshold):
    """Run n_steps forward. Keeps, per step, what backward needs (on the live crop only)."""
    cache = []
    B = x.shape[0]
    for _ in range(n_steps):
        box = live_box(x)
        if box is None:                      # everything is dead and stays dead
            cache.append(None)
            continue
        y0, y1, x0, x1 = box
        xc = np.ascontiguousarray(x[:, y0:y1, x0:x1])
        fire = rng.random((B, y1 - y0, x1 - x0, 1)) < fire_rate
        pre = alive_mask(xc, threshold)
        rows = np.flatnonzero(pre)
        dx, _, h = update(xc, params, rows)
        x_new = xc + dx * fire
        life = pre & alive_mask(x_new, threshold)
        cache.append((box, xc, rows, h, fire, life))
        x = np.zeros_like(x)
        x[:, y0:y1, x0:x1] = x_new * life
    return x, cache


def backward(g, cache, params):
    """Backprop dL/dx_final through the cached rollout. Returns (param grads, dL/dx0).

    Per step:  x' = x + fire * W2^T relu(W1^T perceive(x) + b1);  x_next = life * x'.
    Masks are treated as constants, exactly as autograd does for comparisons.
    Only pre-alive rows can carry gradient (life is a subset of pre), and cells
    outside the step's crop are constant zeros, so their gradient is dropped.
    """
    grads = {k: np.zeros_like(v) for k, v in params.items()}
    for entry in reversed(cache):
        if entry is None:
            g = np.zeros_like(g)
            continue
        (y0, y1, x0, x1), x, rows, h, fire, life = entry
        B, H, W, C = x.shape
        gp = g[:, y0:y1, x0:x1] * life                  # dL/dx'
        G = (gp * fire).reshape(-1, C)[rows]            # dL/d(dx) on live rows
        grads["W2"] += h.T @ G
        gh = G @ params["W2"].T
        gh *= h > 0                                     # through relu
        y = perceive(x).reshape(-1, 3 * C)[rows]
        grads["W1"] += y.T @ gh
        grads["b1"] += gh.sum(axis=0)
        gy = np.zeros((B * H * W, 3 * C), x.dtype)
        gy[rows] = gh @ params["W1"].T
        gy = gy.reshape(B, H, W, C, 3)
        # the transpose of correlating with a Sobel kernel is correlating with
        # the kernel flipped 180 degrees, which for Sobel is its negative
        g = np.zeros_like(g)
        g[:, y0:y1, x0:x1] = gp + gy[..., 0] - sobel_x(gy[..., 1]) - sobel_y(gy[..., 2])
    return grads, g


def circle_masks(n, h, w, rng):
    """n random discs (centre in the middle half, radius 0.1-0.4 of the half-width), as in Distill."""
    xs = np.linspace(-1.0, 1.0, w)[None, None, :]
    ys = np.linspace(-1.0, 1.0, h)[None, :, None]
    cx, cy = rng.uniform(-0.5, 0.5, (2, n, 1, 1))
    r = rng.uniform(0.1, 0.4, (n, 1, 1))
    return (((xs - cx) / r) ** 2 + ((ys - cy) / r) ** 2) < 1.0


def noise_damage(x, rng, threshold, std_range):
    """In place: Gaussian noise on every channel of the living cells inside a random disc.

    Mirrors the sandbox's noise brush, so the model learns to recover from it.
    The standard deviation is drawn per sample from std_range.
    """
    n, H, W, C = x.shape
    discs = circle_masks(n, H, W, rng)
    living = alive_mask(x, threshold)[..., 0]
    for k in range(n):
        m = discs[k] & living[k]
        std = rng.uniform(*std_range)
        x[k][m] += rng.normal(0.0, std, (int(m.sum()), C)).astype(x.dtype)


def per_sample_loss(x, target):
    return ((x[..., :4] - target) ** 2).mean(axis=(1, 2, 3))


class Adam:
    def __init__(self, params, b1=0.9, b2=0.999, eps=1e-8):
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t, self.b1, self.b2, self.eps = 0, b1, b2, eps

    def step(self, params, grads, lr):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= (lr * mh / (np.sqrt(vh) + self.eps)).astype(params[k].dtype)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="data/spiderweb_128.png")
    ap.add_argument("--size", type=int, default=40)
    ap.add_argument("--pad", type=int, default=16)
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--fire-rate", type=float, default=0.5)
    ap.add_argument("--alive-threshold", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--pool", type=int, default=1024)
    ap.add_argument("--damage", type=int, default=2, help="samples per batch that get a random disc erased")
    ap.add_argument("--noise-damage", type=int, default=2,
                    help="further samples per batch that get Gaussian noise inside a random disc")
    ap.add_argument("--noise-std", default="0.1,0.6", help="range the noise standard deviation is drawn from")
    ap.add_argument("--min-steps", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=96)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lr-decay-at", type=int, default=2000, help="lr drops 10x at this iteration")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/spiderweb_numpy.npz")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--init", default=None, help="start from the weights in this .npz (fine-tuning)")
    args = ap.parse_args()
    if args.batch < 1 + args.damage + args.noise_damage:
        ap.error("--batch must be at least 1 + --damage + --noise-damage")
    noise_std = tuple(float(v) for v in args.noise_std.split(","))

    rng = np.random.default_rng(args.seed)
    target = load_target(args.target, args.size, args.pad).astype(np.float32)
    H, W, _ = target.shape
    init_meta = None
    if args.init:
        params, init_meta = load_npz(args.init)
        args.channels, args.hidden = init_meta["channels"], init_meta["hidden"]
    else:
        params = init_params(args.channels, args.hidden, rng)
    C = args.channels
    opt = Adam(params)
    seed_state = make_seed(1, H, W, C)[0]
    pool = make_seed(args.pool, H, W, C)
    meta = dict(channels=C, hidden=args.hidden, fire_rate=args.fire_rate,
                alive_threshold=args.alive_threshold, target=os.path.basename(args.target),
                target_size=args.size, pad=args.pad, grid=[H, W], trained_by="tools/train_numpy.py",
                damage=args.damage, noise_damage=args.noise_damage, noise_std=list(noise_std),
                pool=args.pool, batch=args.batch, steps=[args.min_steps, args.max_steps])
    done_before = 0
    if init_meta:  # keep the full history: iterations counts the earlier run too
        done_before = int(init_meta.get("iterations", 0))
        meta["fine_tuned_from"] = dict(file=os.path.basename(args.init), iterations=done_before,
                                       trained_by=init_meta.get("trained_by"),
                                       damage=init_meta.get("damage"),
                                       noise_damage=init_meta.get("noise_damage", 0))
    log_path = os.path.splitext(args.out)[0] + "_log.json"
    log = []
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    t0 = time.time()
    for it in range(1, args.iters + 1):
        idx = rng.choice(args.pool, args.batch, replace=False)
        x0 = pool[idx]
        order = np.argsort(per_sample_loss(x0, target))[::-1]   # worst first
        idx, x0 = idx[order], x0[order]
        x0[0] = seed_state                                      # replace the worst with a fresh seed
        if args.damage:
            x0[-args.damage:] *= ~circle_masks(args.damage, H, W, rng)[..., None]  # erase from the best
        if args.noise_damage:                                   # and add noise to the next best
            end = args.batch - args.damage
            noise_damage(x0[end - args.noise_damage:end], rng, args.alive_threshold, noise_std)

        n_steps = int(rng.integers(args.min_steps, args.max_steps + 1))
        x, cache = rollout(x0, params, n_steps, rng, args.fire_rate, args.alive_threshold)
        diff = x[..., :4] - target
        loss = float((diff ** 2).mean())
        if not np.isfinite(loss):
            raise SystemExit(f"loss became {loss} at iteration {it}")
        g = np.zeros_like(x)
        g[..., :4] = 2.0 * diff / diff.size
        grads, _ = backward(g, cache, params)
        del cache
        for k in grads:
            grads[k] /= np.linalg.norm(grads[k]) + 1e-8          # per-tensor gradient normalisation
        lr = args.lr if it <= args.lr_decay_at else args.lr * 0.1
        opt.step(params, grads, lr)
        pool[idx] = x

        log.append([it, loss, n_steps])
        if it % args.log_every == 0 or it == 1:
            el = time.time() - t0
            print(f"it {it:5d}  loss {loss:.5f}  log10 {np.log10(loss):+.3f}  steps {n_steps}  "
                  f"{el / it:.2f}s/it  elapsed {el / 60:.1f}m", flush=True)
        if it % args.save_every == 0 or it == args.iters:
            save_npz(args.out, params["W1"], params["b1"], params["W2"], dict(meta, iterations=done_before + it))
            with open(log_path, "w") as f:
                json.dump(log, f)


if __name__ == "__main__":
    main()
