"""Measure growth, persistence and regeneration of a trained checkpoint.

    python tools/evaluate.py --checkpoint checkpoints/spiderweb.npz

Runs with NumPy only, on the grid size the model was trained on:
  1. growth      - from one seed cell; MSE against the target every 10 steps
  2. persistence - keep running to --long steps; does the pattern hold?
  3. regeneration- erase a random disc (training-style damage) from the grown
                   pattern, `--trials` times, and track how fast the MSE recovers
  4. noise       - add Gaussian noise (each --noise strength) to the living
                   cells within 8 cells of the centre; MSE right after and 300 steps later

Writes <out>/<name>_eval.png (growth and regrowth filmstrips) and
<out>/<name>_eval.json (all numbers), and prints a summary.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nca.checkpoint import load_npz  # noqa: E402
from nca.numpy_nca import alive_mask, make_seed, step  # noqa: E402
from nca.target import load_target, to_rgb  # noqa: E402


def circle_mask(h, w, rng):
    xs = np.linspace(-1.0, 1.0, w)[None, :]
    ys = np.linspace(-1.0, 1.0, h)[:, None]
    cx, cy = rng.uniform(-0.5, 0.5, 2)
    r = rng.uniform(0.1, 0.4)
    return ((xs - cx) / r) ** 2 + ((ys - cy) / r) ** 2 < 1.0


def run(x, params, meta, n, rng, every=10, target=None, frames_at=()):
    """Advance n steps; return final state, [(t, mse)], {t: rgb frame}."""
    losses, frames = [], {}
    H, W = x.shape[1:3]
    for t in range(1, n + 1):
        fire = rng.random((1, H, W, 1)) < meta["fire_rate"]
        x = step(x, params, fire, meta["alive_threshold"])
        if target is not None and (t % every == 0 or t == n or t in frames_at):
            losses.append((t, float(((x[0, ..., :4] - target) ** 2).mean())))
        if t in frames_at:
            frames[t] = to_rgb(x[0, ..., :4])
    return x, losses, frames


def strip(images, labels, scale=3):
    tiles = []
    for img, label in zip(images, labels):
        im = Image.fromarray((img * 255).astype(np.uint8)).resize(
            (img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST)
        canvas = Image.new("RGB", (im.width, im.height + 16), (255, 255, 255))
        canvas.paste(im, (0, 16))
        ImageDraw.Draw(canvas).text((4, 2), label, fill=(60, 56, 54))
        tiles.append(canvas)
    out = Image.new("RGB", (sum(t.width for t in tiles) + 4 * (len(tiles) - 1), tiles[0].height), (255, 255, 255))
    x = 0
    for t in tiles:
        out.paste(t, (x, 0))
        x += t.width + 4
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/spiderweb.npz")
    ap.add_argument("--target", default=None, help="defaults to data/<target in checkpoint metadata>")
    ap.add_argument("--grow", type=int, default=200)
    ap.add_argument("--long", type=int, default=1000)
    ap.add_argument("--regrow", type=int, default=200)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--noise", default="0.1,0.3,0.6", help="comma-separated noise strengths ('' to skip)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()
    if args.long <= args.grow or args.grow < 10 or args.regrow < 5 or args.trials < 1:
        ap.error("need --long > --grow >= 10, --regrow >= 5 and --trials >= 1")

    params, meta = load_npz(args.checkpoint)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_path = args.target or os.path.join(root, "data", meta["target"])
    target = load_target(target_path, meta["target_size"], meta["pad"])
    H, W, _ = target.shape
    rng = np.random.default_rng(args.seed)
    empty_mse = float((target ** 2).mean())

    # 1 + 2: growth and persistence
    grow_frames = tuple(sorted({t for t in (0, 10, 20, 30, 40, 60, 80, 120) if t < args.grow} | {args.grow}))
    x0 = make_seed(1, H, W, meta["channels"])
    x, growth, frames = run(x0, params, meta, args.grow, rng, 10, target, grow_frames)
    frames[0] = to_rgb(x0[0, ..., :4])
    grown = x.copy()
    x_long, long_losses, long_frames = run(x, params, meta, args.long - args.grow, rng, 50, target,
                                           (args.long - args.grow,))
    long_losses = [(t + args.grow, v) for t, v in long_losses]

    # 3: regeneration trials from the grown pattern
    trials, trial_frames = [], []
    marks = tuple(sorted({t for t in (0, 10, 25, 50, 100) if t < args.regrow} | {args.regrow}))
    for _ in range(args.trials):
        damaged = grown * ~circle_mask(H, W, rng)[None, ..., None]
        before = float(((grown[0, ..., :4] - target) ** 2).mean())
        hit = float(((damaged[0, ..., :4] - target) ** 2).mean())
        _, losses, fr = run(damaged, params, meta, args.regrow, rng, 5, target, marks[1:])
        curve = dict(losses)
        trials.append(dict(before=before, damaged=hit, curve=losses,
                           after={t: curve[t] for t in marks[1:] if t in curve}))
        fr[0] = to_rgb(damaged[0, ..., :4])
        trial_frames.append(fr)
    # filmstrip: the trial with the median amount of damage (representative, not cherry-picked)
    order = np.argsort([tr["damaged"] for tr in trials])
    regrow_frames = trial_frames[order[len(order) // 2]]

    # summary
    g = dict(growth)
    print(f"checkpoint {args.checkpoint}  ({meta.get('trained_by')}, {meta.get('iterations')} iterations)")
    print(f"grid {H}x{W}; MSE of an empty grid vs target: {empty_mse:.5f}")
    print("growth MSE:  " + "  ".join(f"t{t}={g[t]:.5f}" for t in sorted({20, 40, 60, 80, 100, 150, args.grow}) if t in g))
    lp = dict(long_losses)
    print("persistence: MSE " + "  ".join(f"t{t}={lp[t]:.5f}" for t in sorted({500, 1000, 2000, args.long}) if t in lp))
    med = lambda key: float(np.median([tr[key] for tr in trials]))  # noqa: E731
    med_after = {t: float(np.median([tr["after"][t] for tr in trials])) for t in trials[0]["after"]}
    print(f"regeneration ({args.trials} random discs): median MSE before {med('before'):.5f} -> "
          f"damaged {med('damaged'):.5f} -> " + "  ".join(f"+{t}: {v:.5f}" for t, v in med_after.items()))
    # share of the damage-induced error that has been repaired: 1 = back to the pre-damage MSE.
    # Discs that barely touched the pattern are left out (the ratio would be noise / ~0).
    hit = [tr for tr in trials if tr["damaged"] - tr["before"] > 1e-4]
    repaired = {t: float(np.median([(tr["damaged"] - tr["after"][t]) / (tr["damaged"] - tr["before"]) for tr in hit]))
                for t in trials[0]["after"]} if hit else {}
    if hit:
        print(f"damage repaired (median of {len(hit)} trials that hit the pattern): "
              + "  ".join(f"+{t}: {100 * v:.0f}%" for t, v in repaired.items()))
    else:
        print("damage repaired: no trial's disc hit the pattern noticeably")
    last = marks[-1]
    recovered = sum(tr["after"][last] < 2 * tr["before"] + 1e-4 for tr in trials)
    worst = max(trials, key=lambda tr: tr["after"][last])
    print(f"recovered after +{last} (MSE below 2x its pre-damage value + 1e-4): {recovered}/{len(trials)}; "
          f"worst trial: {worst['before']:.5f} -> damaged {worst['damaged']:.5f} -> {worst['after'][last]:.5f}")

    # 4: noise (runs last so it does not change the random stream of the sections above)
    noise = []
    for strength in [float(v) for v in args.noise.split(",") if v.strip()]:
        ys, xs = np.mgrid[0:H, 0:W]
        near = ((ys + 0.5 - H / 2) ** 2 + (xs + 0.5 - W / 2) ** 2 <= 8 ** 2)
        living = near & alive_mask(grown, meta["alive_threshold"])[0, ..., 0]
        noisy = grown.copy()
        noisy[0][living] += rng.normal(0.0, strength, (int(living.sum()), meta["channels"])).astype(np.float32)
        after, _, _ = run(noisy, params, meta, 300, rng)
        mse = lambda z: float(((z[0, ..., :4] - target) ** 2).mean())  # noqa: E731
        noise.append(dict(strength=strength, before=mse(grown), noisy=mse(noisy), after300=mse(after)))
        print(f"noise {strength}: MSE {mse(grown):.5f} -> {mse(noisy):.5f} right after -> {mse(after):.5f} after 300 steps")

    name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    os.makedirs(args.out, exist_ok=True)
    top = strip([frames[t] for t in grow_frames] + [long_frames[args.long - args.grow]],
                [f"t={t}" for t in grow_frames] + [f"t={args.long}"])
    bottom = strip([to_rgb(target)] + [regrow_frames[t] for t in marks],
                   ["target", "damaged"] + [f"+{t}" for t in marks[1:]])
    sheet = Image.new("RGB", (max(top.width, bottom.width), top.height + bottom.height + 10), (255, 255, 255))
    sheet.paste(top, (0, 0))
    sheet.paste(bottom, (0, top.height + 10))
    sheet.save(os.path.join(args.out, f"{name}_eval.png"))
    with open(os.path.join(args.out, f"{name}_eval.json"), "w") as f:
        json.dump(dict(checkpoint=args.checkpoint, meta=meta, empty_mse=empty_mse, growth=growth,
                       persistence=long_losses, regeneration=trials, repaired_median=repaired,
                       recovered=int(recovered), noise=noise), f, indent=1)
    print(f"wrote {args.out}/{name}_eval.png and .json")


if __name__ == "__main__":
    main()
