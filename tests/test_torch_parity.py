"""PyTorch <-> NumPy parity. Run this once on a machine with PyTorch installed:

    python tests/test_torch_parity.py

It checks that
  0. the trainer's noise damage only touches living cells inside its disc,
  1. weights survive the .npz export/import round trip,
  2. nca.model.NCA and nca.numpy_nca produce the same states, step for step,
  3. autograd gradients from the PyTorch model match the hand-written NumPy
     backward pass (itself checked against finite differences in
     tests/test_gradients.py) on the same rollout.

Every cell fires (fire_rate = 1) so both sides see identical randomness.
"""
import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from nca import numpy_nca  # noqa: E402
from nca.checkpoint import load_npz  # noqa: E402
from nca.model import NCA, make_seed  # noqa: E402
from train_numpy import backward, rollout  # noqa: E402

CKPT = os.path.join(ROOT, "checkpoints", "spiderweb_starter.npz")


def test_noise_damage_is_confined():
    from nca.train import noise_damage
    model, meta = NCA.from_npz(CKPT)
    torch.manual_seed(0)
    x = make_seed(4, 72, 72, meta["channels"])
    with torch.no_grad():
        for _ in range(150):
            x = model(x)
        before = x.clone()
        noise_damage(x[1:3], model, (0.1, 0.6))           # the slice form the trainer uses
    changed = (x != before).any(dim=1)                     # (4, H, W)
    living = model.alive(before)[:, 0]
    assert not changed[[0, 3]].any(), "only the selected samples may change"
    assert changed[1:3].any() and not (changed & ~living).any(), "noise only on living cells"


def test_roundtrip():
    torch.manual_seed(0)
    m = NCA(16, 32)
    torch.nn.init.normal_(m.fc2.weight, std=0.1)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "w.npz")
        m.export_npz(path, note="roundtrip")
        m2, meta = NCA.from_npz(path)
    for a, b in zip(m.parameters(), m2.parameters()):
        assert torch.equal(a, b)
    assert meta["note"] == "roundtrip"


def test_forward_parity(steps=40):
    model, meta = NCA.from_npz(CKPT)
    params, _ = load_npz(CKPT)
    H = W = 48
    xt = make_seed(1, H, W, meta["channels"])
    xn = numpy_nca.make_seed(1, H, W, meta["channels"])
    ones = np.ones((1, H, W, 1), bool)
    worst = scale = 0.0
    with torch.no_grad():
        for _ in range(steps):
            xt = model(xt, fire_rate=1.0)
            xn = numpy_nca.step(xn, params, ones, meta["alive_threshold"])
            worst = max(worst, float(np.abs(xt[0].permute(1, 2, 0).numpy() - xn[0]).max()))
            scale = max(scale, float(np.abs(xn).max()))
    print(f"forward: max |torch - numpy| over {steps} steps = {worst:.2e} (largest |state| {scale:.1f})")
    # float32 rounding differs between conv2d and matmul; a real mismatch (wrong kernel
    # orientation, channel order or mask) shows up as differences of order |state|
    assert worst < 1e-4 * max(1.0, scale)


def test_gradient_parity(steps=12):
    torch.manual_seed(0)
    model = NCA(16, 32, fire_rate=1.0)
    torch.nn.init.normal_(model.fc2.weight, std=0.05)
    with torch.no_grad():
        model.fc2.weight[3].abs_()  # make alpha grow so the rollout is non-trivial
    params = {
        "W1": model.fc1.weight.detach()[:, :, 0, 0].T.double().numpy().copy(),
        "b1": model.fc1.bias.detach().double().numpy().copy(),
        "W2": model.fc2.weight.detach()[:, :, 0, 0].T.double().numpy().copy(),
    }
    model = model.double()
    H = W = 24
    target = torch.rand(1, 4, H, W, dtype=torch.float64)

    x = make_seed(1, H, W, 16).double()
    for _ in range(steps):
        x = model(x)
    loss = ((x[:, :4] - target) ** 2).mean()
    loss.backward()

    xn = numpy_nca.make_seed(1, H, W, 16, np.float64)
    xf, cache = rollout(xn, params, steps, np.random.default_rng(0), 1.0, 0.1)
    tn = target[0].permute(1, 2, 0).numpy()
    g = np.zeros_like(xf)
    g[..., :4] = 2 * (xf[..., :4] - tn) / (H * W * 4)
    grads, _ = backward(g, cache, params)

    pairs = {
        "W1": (model.fc1.weight.grad[:, :, 0, 0].T.numpy(), grads["W1"]),
        "b1": (model.fc1.bias.grad.numpy(), grads["b1"]),
        "W2": (model.fc2.weight.grad[:, :, 0, 0].T.numpy(), grads["W2"]),
    }
    for name, (a, b) in pairs.items():
        rel = np.abs(a - b).max() / (np.abs(a).max() + 1e-12)
        print(f"grad {name}: max relative difference {rel:.2e}")
        assert rel < 1e-8, name


if __name__ == "__main__":
    test_noise_damage_is_confined()
    test_roundtrip()
    test_forward_parity()
    test_gradient_parity()
    print("torch parity checks passed")
