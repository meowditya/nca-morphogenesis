"""Training-loop tests for the NumPy trainer (no PyTorch needed).

    python tests/test_training.py        # or: python -m pytest tests
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from nca.checkpoint import load_npz  # noqa: E402
from nca.numpy_nca import alive_mask, make_seed, step  # noqa: E402
from train_numpy import circle_masks, noise_damage  # noqa: E402

CKPT = os.path.join(ROOT, "checkpoints", "spiderweb_starter.npz")


def grown_state(steps=150):
    params, meta = load_npz(CKPT)
    rng = np.random.default_rng(0)
    x = make_seed(1, 72, 72, meta["channels"])
    for _ in range(steps):
        x = step(x, params, rng.random((1, 72, 72, 1)) < 0.5)
    return x[0]


def test_noise_damage_is_confined():
    batch = np.stack([grown_state()] * 5)
    before = batch.copy()
    noise_damage(batch[1:3], np.random.default_rng(1), 0.1, (0.1, 0.6))   # the slice form the trainer uses
    changed = np.any(batch != before, axis=-1)
    discs = circle_masks(2, 72, 72, np.random.default_rng(1))             # same draws as above
    assert changed[[0, 3, 4]].sum() == 0, "only the selected samples may change"
    assert changed[1:3].sum() > 0
    assert not (changed & ~alive_mask(before, 0.1)[..., 0]).any(), "noise only on living cells"
    assert not (changed[1:3] & ~discs).any(), "noise only inside the disc"


def test_fine_tune_run():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "ft.npz")
        cmd = [sys.executable, os.path.join(ROOT, "tools", "train_numpy.py"), "--init", CKPT,
               "--iters", "2", "--pool", "16", "--batch", "6", "--damage", "1", "--noise-damage", "2",
               "--min-steps", "8", "--max-steps", "10", "--save-every", "1", "--log-every", "1",
               "--lr", "2e-4", "--out", out]
        subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True)
        _, meta = load_npz(out)
        _, start = load_npz(CKPT)
        assert meta["iterations"] == start["iterations"] + 2
        assert meta["fine_tuned_from"]["iterations"] == start["iterations"]
        assert meta["noise_damage"] == 2 and meta["damage"] == 1
        bad = subprocess.run(cmd[:-2] + ["--batch", "3", "--out", out], cwd=ROOT, capture_output=True)
        assert bad.returncode != 0, "a batch too small for the damage settings must be rejected"


def test_persistent_trainer_records_settings():
    """nca/train_persistant.py must save every damage and loss setting in the checkpoint."""
    try:
        import torch  # noqa: F401
    except ImportError:
        print("skip test_persistent_trainer_records_settings: PyTorch not installed")
        return
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "pt")
        cmd = [sys.executable, "-m", "nca.train_persistant", "--init", CKPT, "--device", "cpu",
               "--iters", "2", "--pool", "16", "--batch", "8", "--min-steps", "8", "--max-steps", "10",
               "--save-every", "1", "--log-every", "1", "--lr", "2e-4",
               "--hidden-noise", "0", "--consistency-weight", "0", "--redamage", "1",
               "--half-damage", "1", "--out", out]
        subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True)
        _, meta = load_npz(os.path.join(out, "pt.npz"))
        assert meta["trained_by"] == "nca/train_persistant.py"
        assert meta["hidden_noise"] == 0 and meta["consistency_weight"] == 0
        assert meta["redamage"] == 1 and meta["overflow_weight"] == 1e-3
        assert meta["seed_damage"] == 2 and meta["half_damage"] == 1 and meta["batch"] == 8
        bad = subprocess.run(cmd[:-2] + ["--half-damage", "2", "--out", out], cwd=ROOT, capture_output=True)
        assert bad.returncode != 0, "a batch too small for the damage settings must be rejected"


def test_halfplane_masks():
    """Half-plane wounds cover about half the grid and cut close to the centre, in every direction."""
    try:
        import torch
    except ImportError:
        print("skip test_halfplane_masks: PyTorch not installed")
        return
    from nca.train_persistant import halfplane_masks
    torch.manual_seed(0)
    n, h, w = 512, 72, 72
    m = halfplane_masks(n, h, w, "cpu")
    assert m.shape == (n, h, w) and m.dtype == torch.bool
    frac = m.float().mean((1, 2))
    assert frac.min() > 0.3 and frac.max() < 0.7, (frac.min(), frac.max())
    c = m[:, h // 2 - 6:h // 2 + 6, w // 2 - 6:w // 2 + 6]        # 12x12 patch around the centre
    assert bool((c.any((1, 2)) & ~c.all((1, 2))).all()), "every cut must pass within a few px of the centre"
    # every direction is used: the erased side's centroid points all around the circle
    ys, xs = torch.meshgrid(torch.arange(h) - h / 2 + 0.5, torch.arange(w) - w / 2 + 0.5, indexing="ij")
    ang = torch.atan2((m * ys).sum((1, 2)), (m * xs).sum((1, 2)))
    hist = torch.histc(ang, bins=8, min=-torch.pi, max=torch.pi)
    assert hist.min() > n / 8 * 0.5, hist


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
