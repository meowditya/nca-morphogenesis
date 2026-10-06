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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
