"""Train a growing + regenerating NCA (Distill "Growing Neural Cellular Automata", experiment 3).

Training recipe
  * every sample starts from (or continues from) the persistent sample pool,
  * the worst sample in each batch is replaced by a fresh seed, so the model
    never forgets how to grow from one pixel,
  * the best `--damage` samples get a random disc erased, and the next
    `--noise-damage` samples get Gaussian noise inside a random disc, so the
    model learns to regrow missing parts and to clean up scrambled ones,
  * a further `--seed-damage` samples get a disc erased that is guaranteed to
    cover the grid centre (the original seed cell), forcing the model to
    regenerate even when its "origin" is destroyed,
  * the rollout is split in half; at the midpoint, `--redamage` of the best
    samples get a random disc erased, so the model trains on sequential
    damage-regrow-damage cycles (hidden-state resilience),
  * after the full rollout the RGBA channels are compared with the target
    (MSE) plus a small `--overflow-weight` penalty on hidden channels that
    exceed 1.0 in absolute value (keeps regrown hidden states stable),
  * a consistency loss (`--consistency-weight`) penalises hidden-channel drift:
    for damaged samples the regrown hidden state is compared to its pre-damage
    snapshot, forcing the model to reproduce the same internal state,
  * per-step Gaussian noise (`--hidden-noise`) is injected into hidden channels
    during training, making the model tolerant of small state perturbations,
  * gradients are normalised per tensor before Adam (stabilises long rollouts),
  * the final states are written back into the pool.

Usage (from the project root):

    python -m nca.train --target data/spiderweb_128.png --out checkpoints/my_web.npz
    python -m nca.train --resume checkpoints/my_web.pt            # continue a run

Outputs: <out>.npz (weights for the simulator), <out>.pt (full training
state for --resume), <out>_log.json (loss per iteration) and preview PNGs.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.checkpoint import checkpoint

from .checkpoint import load_npz
from .model import NCA, make_seed
from .target import load_target, to_rgb


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def circle_masks(n, h, w, device):
    """n random discs (centre in the middle half, radius 0.1-0.4 of the half-width), as in Distill."""
    xs = torch.linspace(-1.0, 1.0, w, device=device)[None, None, :]
    ys = torch.linspace(-1.0, 1.0, h, device=device)[None, :, None]
    cx, cy = torch.rand(2, n, 1, 1, device=device) - 0.5
    r = torch.rand(n, 1, 1, device=device) * 0.3 + 0.1
    return (((xs - cx) / r) ** 2 + ((ys - cy) / r) ** 2) < 1.0


def centre_kill_masks(n, h, w, device, min_r=0.15, max_r=0.4):
    """n random discs guaranteed to cover the grid centre, for seed-cell damage.

    The disc is jittered slightly off-centre but the jitter is small enough
    that the disc always covers the (0, 0) normalised coordinate (grid centre).
    """
    xs = torch.linspace(-1.0, 1.0, w, device=device)[None, None, :]
    ys = torch.linspace(-1.0, 1.0, h, device=device)[None, :, None]
    r = torch.rand(n, 1, 1, device=device) * (max_r - min_r) + min_r
    max_jitter = r * 0.4  # |jitter| < r so the disc still covers (0, 0)
    cx = (torch.rand(n, 1, 1, device=device) - 0.5) * 2 * max_jitter
    cy = (torch.rand(n, 1, 1, device=device) - 0.5) * 2 * max_jitter
    return (((xs - cx) / r) ** 2 + ((ys - cy) / r) ** 2) < 1.0


def noise_damage(x, model, std_range):
    """In place: Gaussian noise on every channel of the living cells inside a random disc."""
    n, _, H, W = x.shape
    disc = circle_masks(n, H, W, x.device)[:, None]                       # (n, 1, H, W)
    hit = (disc & model.alive(x)).to(x.dtype)
    lo, hi = std_range
    std = torch.rand(n, 1, 1, 1, device=x.device) * (hi - lo) + lo
    x += torch.randn_like(x) * std * hit


def per_sample_loss(x, target):
    return ((x[:, :4] - target) ** 2).mean(dim=(1, 2, 3))


def save_preview(path, x):
    """Save the batch's RGBA (composited over white) side by side."""
    rgba = x[:, :4].detach().permute(0, 2, 3, 1).cpu().numpy()
    row = np.concatenate([to_rgb(s) for s in rgba], axis=1)
    Image.fromarray((row * 255).astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="data/spiderweb_128.png", help="RGBA image to grow (PNG with transparency)")
    ap.add_argument("--size", type=int, default=40, help="target is fitted into size x size pixels")
    ap.add_argument("--pad", type=int, default=16, help="empty border around the target")
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--fire-rate", type=float, default=0.5)
    ap.add_argument("--alive-threshold", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--pool", type=int, default=1024)
    ap.add_argument("--damage", type=int, default=2, help="samples per batch that get a random disc erased")
    ap.add_argument("--noise-damage", type=int, default=2,
                    help="further samples per batch that get Gaussian noise inside a random disc")
    ap.add_argument("--seed-damage", type=int, default=2,
                    help="samples per batch that get a disc erased covering the centre (seed cell)")
    ap.add_argument("--noise-std", default="0.1,0.6", help="range the noise standard deviation is drawn from")
    ap.add_argument("--min-steps", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=96)
    ap.add_argument("--redamage", type=int, default=2,
                    help="samples re-damaged with a random disc mid-rollout for multi-round resilience")
    ap.add_argument("--overflow-weight", type=float, default=1e-3,
                    help="penalty on hidden channels exceeding 1.0 in abs value (stabilises regrown states)")
    ap.add_argument("--consistency-weight", type=float, default=0.05,
                    help="penalty on hidden-channel drift between pre-damage and regrown state")
    ap.add_argument("--hidden-noise", type=float, default=0.02,
                    help="std of Gaussian noise added to hidden channels each step during training")
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lr-decay-at", type=int, default=2000, help="lr drops 10x at this iteration")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="recompute activations in the backward pass (much less memory, ~30%% slower)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/run.npz",
                    help="weights file to write (also <out>.pt, <out>_log.json, <out>_previews/)")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--resume", default=None, help="a .pt file written by a previous run")
    ap.add_argument("--init", default=None, help="start from the weights in this .npz (fine-tuning)")
    args = ap.parse_args()

    if args.resume:  # restore the original run's settings, but keep this run's --iters/--out/--device
        resume_path = args.resume
        if not os.path.isfile(resume_path):
            stem = os.path.splitext(os.path.basename(resume_path))[0]
            candidate = os.path.join(os.path.dirname(resume_path), stem, os.path.basename(resume_path))
            if os.path.isfile(candidate):
                resume_path = candidate
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        keep = {k: getattr(args, k) for k in ("iters", "out", "device", "resume", "log_every", "save_every")}
        args = argparse.Namespace(**{**vars(args), **state["args"], **keep})

    if args.batch < 1 + args.damage + args.noise_damage + args.seed_damage:
        raise SystemExit("--batch must be at least 1 + --damage + --noise-damage + --seed-damage")
    noise_std = tuple(float(v) for v in str(args.noise_std).split(","))
    init_meta = None
    if args.init and not args.resume:
        _, init_meta = load_npz(args.init)
        args.channels, args.hidden = init_meta["channels"], init_meta["hidden"]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    target_np = load_target(args.target, args.size, args.pad)
    H, W, _ = target_np.shape
    target = torch.from_numpy(target_np).permute(2, 0, 1)[None].to(device)
    C = args.channels

    if init_meta:
        model, _ = NCA.from_npz(args.init)
        model.fire_rate, model.alive_threshold = args.fire_rate, args.alive_threshold
        model = model.to(device)
    else:
        model = NCA(C, args.hidden, args.fire_rate, args.alive_threshold).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[args.lr_decay_at], gamma=0.1)
    seed = make_seed(1, H, W, C, device)
    pool = make_seed(args.pool, H, W, C, device)
    log, start = [], 1

    out_clean = args.out[:-4] if args.out.endswith(".npz") else args.out
    parent_dir, weight_name = os.path.split(out_clean)
    if not parent_dir:
        parent_dir = "checkpoints"
    if os.path.basename(parent_dir) == weight_name:
        weight_dir = parent_dir
    else:
        weight_dir = os.path.join(parent_dir, weight_name)
    os.makedirs(weight_dir, exist_ok=True)

    npz_path = os.path.join(weight_dir, f"{weight_name}.npz")
    pt_path = os.path.join(weight_dir, f"{weight_name}.pt")
    log_path = os.path.join(weight_dir, f"{weight_name}_log.json")
    preview_dir = os.path.join(weight_dir, f"{weight_name}_previews")
    os.makedirs(preview_dir, exist_ok=True)

    if args.resume:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        pool = state["pool"].to(device)
        log, start = state["log"], state["iter"] + 1
        print(f"resumed {args.resume} at iteration {start}")

    meta = dict(target=os.path.basename(args.target), target_size=args.size, pad=args.pad, grid=[H, W],
                trained_by="nca/train.py", damage=args.damage, noise_damage=args.noise_damage,
                seed_damage=args.seed_damage,
                noise_std=list(noise_std), pool=args.pool, batch=args.batch,
                steps=[args.min_steps, args.max_steps])
    done_before = 0
    if init_meta:  # keep the full history: iterations counts the earlier run too
        done_before = int(init_meta.get("iterations", 0))
        meta["fine_tuned_from"] = dict(file=os.path.basename(args.init), iterations=done_before,
                                       trained_by=init_meta.get("trained_by"),
                                       damage=init_meta.get("damage"),
                                       noise_damage=init_meta.get("noise_damage", 0),
                                       seed_damage=init_meta.get("seed_damage", 0))
    if args.resume:  # a resumed fine-tune keeps its provenance and iteration offset
        meta, done_before = state.get("meta", meta), state.get("done_before", done_before)

    def step_fn(x):
        x = model(x)
        if args.hidden_noise:                                            # per-step hidden noise
            noise = torch.zeros_like(x)
            noise[:, 4:].normal_(0, args.hidden_noise)
            x = x + noise
        return x

    def save(it):
        model.export_npz(npz_path, **meta, iterations=done_before + it)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        pool=pool.cpu(), log=log, iter=it, args=vars(args), meta=meta,
                        done_before=done_before), pt_path)
        with open(log_path, "w") as f:
            json.dump(log, f)

    print(f"device {device}, grid {H}x{W}, {sum(p.numel() for p in model.parameters())} parameters")
    t0 = time.time()
    for it in range(start, args.iters + 1):
        idx = torch.randperm(args.pool)[: args.batch].to(device)
        x0 = pool[idx]
        with torch.no_grad():
            order = per_sample_loss(x0, target).argsort(descending=True)  # worst first
        idx, x0 = idx[order], x0[order]
        x0[:1] = seed                                                    # worst -> fresh seed
        n_dmg = args.damage + args.noise_damage + args.seed_damage
        dmg_start = args.batch - n_dmg
        with torch.no_grad():                                            # snapshot for consistency loss
            pre_damage_hidden = x0[dmg_start:, 4:].clone()
        if args.damage:
            intact = (~circle_masks(args.damage, H, W, device)).to(x0.dtype)[:, None]
            x0[-args.damage:] *= intact                                  # best -> erased
        if args.noise_damage:                                            # next best -> noised
            end = args.batch - args.damage
            noise_damage(x0[end - args.noise_damage:end], model, noise_std)
        if args.seed_damage:                                             # next -> centre-killed
            end_sd = args.batch - args.damage - args.noise_damage
            start_sd = end_sd - args.seed_damage
            intact_c = (~centre_kill_masks(args.seed_damage, H, W, device)).to(x0.dtype)[:, None]
            x0[start_sd:end_sd] *= intact_c

        x = x0
        steps = random.randint(args.min_steps, args.max_steps)
        mid = steps // 2
        for _ in range(mid):
            x = checkpoint(step_fn, x, use_reentrant=False) if args.grad_checkpoint else step_fn(x)
        if args.redamage:                                                # mid-rollout re-damage
            n_rd = min(args.redamage, args.batch - 1)
            rd_intact = (~circle_masks(n_rd, H, W, device)).to(x.dtype)[:, None]
            rd_mask = torch.ones_like(x[:, :1])
            rd_mask[-n_rd:] = rd_intact
            x = x * rd_mask
        for _ in range(steps - mid):
            x = checkpoint(step_fn, x, use_reentrant=False) if args.grad_checkpoint else step_fn(x)

        loss = per_sample_loss(x, target).mean()
        if args.overflow_weight:                                         # hidden-channel regulariser
            overflow = (x[:, 4:].abs() - 1.0).clamp(min=0).mean()
            loss = loss + args.overflow_weight * overflow
        if args.consistency_weight and n_dmg > 0:                        # hidden-state consistency
            hidden_diff = ((x[dmg_start:, 4:] - pre_damage_hidden) ** 2).mean()
            loss = loss + args.consistency_weight * hidden_diff
        loss_v = loss.item()
        if not np.isfinite(loss_v):
            raise SystemExit(f"loss became {loss_v} at iteration {it}; try a lower --lr")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p.grad /= p.grad.norm() + 1e-8                           # per-tensor normalisation
        opt.step()
        sched.step()
        pool[idx] = x.detach()

        log.append([it, loss_v])
        if it % args.log_every == 0 or it == start:
            el = time.time() - t0
            print(f"it {it:5d}  loss {loss_v:.5f}  log10 {np.log10(loss_v):+.3f}  "
                  f"lr {sched.get_last_lr()[0]:.0e}  {el / (it - start + 1):.2f}s/it", flush=True)
            save_preview(os.path.join(preview_dir, f"it{it:05d}.png"), x)
        if it % args.save_every == 0 or it == args.iters:
            save(it)
    print(f"done: {npz_path}")


if __name__ == "__main__":
    main()
