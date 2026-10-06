"""Stateful NCA world: the grid tensor, obstacles, brushes and the step loop.

The state lives as a NumPy array (H, W, C). Steps run on one of two
backends with identical semantics:

  * "torch" - nca.model.NCA under torch.no_grad() (CPU, CUDA or MPS),
  * "numpy" - nca.numpy_nca, which only evaluates live cells.

"auto" picks torch when it is importable. Obstacles ("walls") are cells
whose state is forced to zero after every step: they are permanently dead,
so neighbours perceive nothing there and growth must route around them.
"""
import time

import numpy as np

from nca import numpy_nca
from nca.checkpoint import load_npz

VIEWS = ("rgba", "hidden", "alive")


class NumpyBackend:
    name = "numpy"

    def __init__(self, params, meta, rng):
        self.params = params
        self.fire_rate = meta["fire_rate"]
        self.threshold = meta["alive_threshold"]
        self.rng = rng

    def run(self, x, n, keep):
        x = x[None]
        H, W = x.shape[1:3]
        for _ in range(n):
            fire = self.rng.random((1, H, W, 1)) < self.fire_rate
            x = numpy_nca.step(x, self.params, fire, self.threshold)
            x *= keep
        return x[0]


class TorchBackend:
    def __init__(self, checkpoint, device):
        import torch
        from nca.model import NCA

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model, _ = NCA.from_npz(checkpoint)
        self.model.to(self.device).eval()
        self.name = f"torch:{self.device.type}"

    def run(self, x, n, keep):
        torch = self.torch
        with torch.no_grad():
            t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].to(self.device)
            k = torch.from_numpy(keep.astype(np.float32)).permute(2, 0, 1)[None].to(self.device)
            for _ in range(n):
                t = self.model(t) * k
            return t[0].permute(1, 2, 0).contiguous().cpu().numpy()


def make_backend(kind, checkpoint, params, meta, rng, device="auto"):
    if kind in ("auto", "torch"):
        try:
            return TorchBackend(checkpoint, device)
        except ImportError:
            if kind == "torch":
                raise
    return NumpyBackend(params, meta, rng)


class Simulation:
    def __init__(self, checkpoint, height=96, width=96, backend="auto", device="auto", seed=None):
        self.params, self.meta = load_npz(checkpoint)
        self.checkpoint = checkpoint
        self.C = self.meta["channels"]
        self.threshold = self.meta["alive_threshold"]
        self.H, self.W = height, width
        self.rng = np.random.default_rng(seed)
        self.backend = make_backend(backend, checkpoint, self.params, self.meta, self.rng, device)
        self.walls = np.zeros((height, width), bool)
        self.ys, self.xs = np.mgrid[0:height, 0:width]
        self.reset()

    # ---- lifecycle ------------------------------------------------------------------
    def reset(self):
        """Empty dish with a single seed in the centre (walls are kept)."""
        self.state = np.zeros((self.H, self.W, self.C), np.float32)
        self.plant(self.W // 2, self.H // 2)
        self.steps = 0
        self.step_time = 0.0

    def clear(self):
        self.state[:] = 0.0

    def step(self, n=1):
        if n <= 0:
            return
        t0 = time.perf_counter()
        keep = ~self.walls[..., None]
        self.state = self.backend.run(self.state, n, keep)
        self.steps += n
        self.step_time = (time.perf_counter() - t0) / n

    # ---- brushes (coordinates in cells, may be fractional) ------------------------------
    def disc_mask(self, points, radius):
        """Union of discs of `radius` around each (x, y) point."""
        mask = np.zeros((self.H, self.W), bool)
        r = max(float(radius), 0.5)
        for px, py in points:
            y0, y1 = max(int(py - r) - 1, 0), min(int(py + r) + 2, self.H)
            x0, x1 = max(int(px - r) - 1, 0), min(int(px + r) + 2, self.W)
            if y0 >= y1 or x0 >= x1:
                continue
            dy = self.ys[y0:y1, x0:x1] + 0.5 - py
            dx = self.xs[y0:y1, x0:x1] + 0.5 - px
            mask[y0:y1, x0:x1] |= dx * dx + dy * dy <= r * r
        return mask

    def erase(self, points, radius):
        self.state[self.disc_mask(points, radius)] = 0.0

    def noise(self, points, radius, strength=0.5):
        """Add Gaussian noise to every channel of the *living* cells under the brush."""
        mask = self.disc_mask(points, radius) & self.alive()
        n = int(mask.sum())
        if n:
            self.state[mask] += self.rng.normal(0.0, strength, (n, self.C)).astype(np.float32)

    def paint_walls(self, points, radius, value=True):
        mask = self.disc_mask(points, radius)
        self.walls[mask] = value
        if value:
            self.state[mask] = 0.0

    def plant(self, x, y):
        """A seed: one cell with alpha and all hidden channels set to 1."""
        ix, iy = int(np.clip(x, 0, self.W - 1)), int(np.clip(y, 0, self.H - 1))
        if not self.walls[iy, ix]:
            self.state[iy, ix, 3:] = 1.0

    def random_damage(self):
        """Erase a random disc over the organism, like the training-time damage."""
        alive = np.argwhere(self.state[..., 3] > self.threshold)
        if len(alive) == 0:
            return
        cy, cx = alive[self.rng.integers(len(alive))]
        span = max(np.ptp(alive[:, 0]), np.ptp(alive[:, 1]), 8)
        self.erase([(cx + 0.5, cy + 0.5)], self.rng.uniform(0.15, 0.35) * span)

    # ---- observation ----------------------------------------------------------------
    def alive(self):
        return numpy_nca.alive_mask(self.state[None], self.threshold)[0, ..., 0]

    def render(self, view="rgba"):
        """(H, W, 4) uint8. rgba: premultiplied colour + alpha; hidden: channels 4-6; alive: mask."""
        s = self.state
        out = np.empty((self.H, self.W, 4), np.float32)
        if view == "hidden" and self.C >= 7:
            out[..., :3] = 0.5 + 0.5 * np.tanh(s[..., 4:7])
            out[..., 3] = self.alive()
            out[..., :3] *= out[..., 3:4]
        elif view == "alive":
            a = self.alive().astype(np.float32)
            out[..., :3] = a[..., None] * np.clip(s[..., 3:4], 0.25, 1.0)
            out[..., 3] = a
        else:
            out[..., :3] = s[..., :3]
            out[..., 3] = s[..., 3]
        return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    def mature_cells(self):
        """Cells with alpha above the threshold ("mature" in the Distill article)."""
        return int((self.state[..., 3] > self.threshold).sum())
