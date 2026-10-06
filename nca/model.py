"""The growing Neural Cellular Automaton, in PyTorch.

Each cell holds a 16-channel state: RGBA (channels 0-3) plus 12 hidden
channels. Every step, each cell

1. perceives its 3x3 neighbourhood through fixed filters (identity, Sobel-x,
   Sobel-y) applied per channel -> a 48-vector,
2. maps that through a tiny per-cell MLP (two 1x1 convolutions) to a
   residual update dx,
3. applies dx only if a random "fire" mask says so (asynchronous updates),
4. dies (state zeroed) unless some cell in its 3x3 neighbourhood has
   alpha > 0.1 both before and after the update.

Weights are stored in the framework-neutral format of nca/checkpoint.py so the
simulator can also run them with NumPy.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .checkpoint import load_npz, save_npz


class NCA(nn.Module):
    def __init__(self, channels=16, hidden=128, fire_rate=0.5, alive_threshold=0.1):
        super().__init__()
        self.channels = channels
        self.hidden = hidden
        self.fire_rate = fire_rate
        self.alive_threshold = alive_threshold

        ident = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        kernels = torch.stack([ident, sobel_x, sobel_x.T])          # (3, 3, 3)
        # depthwise filter bank, output channel 3*c + k sees input channel c through kernel k
        self.register_buffer("kernels", kernels.repeat(channels, 1, 1)[:, None], persistent=False)

        self.fc1 = nn.Conv2d(3 * channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.fc2.weight)  # start as the identity map: "do nothing"

    def perceive(self, x):
        return F.conv2d(x, self.kernels, padding=1, groups=self.channels)

    def alive(self, x):
        return F.max_pool2d(x[:, 3:4], kernel_size=3, stride=1, padding=1) > self.alive_threshold

    def update(self, x):
        return self.fc2(F.relu(self.fc1(self.perceive(x))))

    def forward(self, x, fire_rate=None):
        """One step. x: (B, C, H, W)."""
        fire_rate = self.fire_rate if fire_rate is None else fire_rate
        pre = self.alive(x)
        fire = torch.rand_like(x[:, :1]) < fire_rate
        x = x + self.update(x) * fire.to(x.dtype)
        return x * (pre & self.alive(x)).to(x.dtype)

    # ---- framework-neutral weights -------------------------------------------------
    def export_npz(self, path, **meta):
        save_npz(
            path,
            W1=self.fc1.weight.detach().cpu().numpy()[:, :, 0, 0].T,
            b1=self.fc1.bias.detach().cpu().numpy(),
            W2=self.fc2.weight.detach().cpu().numpy()[:, :, 0, 0].T,
            meta=dict(meta, channels=self.channels, hidden=self.hidden,
                      fire_rate=self.fire_rate, alive_threshold=self.alive_threshold),
        )

    @classmethod
    def from_npz(cls, path):
        params, meta = load_npz(path)
        model = cls(meta["channels"], meta["hidden"], meta["fire_rate"], meta["alive_threshold"])
        with torch.no_grad():
            model.fc1.weight.copy_(torch.from_numpy(np.ascontiguousarray(params["W1"].T))[:, :, None, None])
            model.fc1.bias.copy_(torch.from_numpy(params["b1"]))
            model.fc2.weight.copy_(torch.from_numpy(np.ascontiguousarray(params["W2"].T))[:, :, None, None])
        return model, meta


def make_seed(batch, height, width, channels, device=None):
    """Empty grid with one live cell in the centre: alpha and hidden channels = 1."""
    x = torch.zeros(batch, channels, height, width, device=device)
    x[:, 3:, height // 2, width // 2] = 1.0
    return x
