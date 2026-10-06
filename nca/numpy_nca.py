"""NumPy implementation of the growing-NCA update rule.

This mirrors nca/model.py operation for operation, so a checkpoint trained in
PyTorch runs here unchanged. It exists for two reasons: the simulation server
can run without PyTorch installed, and tools/train_numpy.py builds a
hand-written backward pass on top of these functions.

Layout is channels-last: states are (B, H, W, C).
"""
import numpy as np

IDENT = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.float32)
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32) / 8.0
SOBEL_Y = SOBEL_X.T.copy()


def correlate3x3(x, k):
    """Zero-padded 3x3 cross-correlation over (H, W) of a (B, H, W, C) array.

    Same convention as torch.nn.functional.conv2d(padding=1):
    out[i, j] = sum_{a,b} k[a, b] * x[i + a - 1, j + b - 1].
    """
    B, H, W, C = x.shape
    p = np.zeros((B, H + 2, W + 2, C), x.dtype)
    p[:, 1:-1, 1:-1] = x
    out = np.zeros_like(x)
    for a in range(3):
        for b in range(3):
            w = k[a, b]
            if w != 0:
                out += x.dtype.type(w) * p[:, a:a + H, b:b + W]
    return out


def sobel_x(x):
    """correlate3x3(x, SOBEL_X), computed separably: smooth along H, difference along W."""
    p = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    d = p[:, :, 2:] - p[:, :, :-2]
    return (d[:, :-2] + 2 * d[:, 1:-1] + d[:, 2:]) * x.dtype.type(0.125)


def sobel_y(x):
    """correlate3x3(x, SOBEL_Y), computed separably: smooth along W, difference along H."""
    p = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    s = p[:, :, :-2] + 2 * p[:, :, 1:-1] + p[:, :, 2:]
    return (s[:, 2:] - s[:, :-2]) * x.dtype.type(0.125)


def perceive(x):
    """(B, H, W, C) -> (B, H, W, 3C), channel 3c+k = [identity, sobel_x, sobel_y][k] of c."""
    B, H, W, C = x.shape
    y = np.empty((B, H, W, C, 3), x.dtype)
    y[..., 0] = x
    y[..., 1] = sobel_x(x)
    y[..., 2] = sobel_y(x)
    return y.reshape(B, H, W, 3 * C)


def maxpool3x3(a):
    """3x3 max filter, stride 1, padded with -inf (torch max_pool2d semantics). a: (B, H, W)."""
    B, H, W = a.shape
    p = np.full((B, H + 2, W + 2), -np.inf, a.dtype)
    p[:, 1:-1, 1:-1] = a
    out = p[:, 0:H, 0:W].copy()
    for i in range(3):
        for j in range(3):
            np.maximum(out, p[:, i:i + H, j:j + W], out=out)
    return out


def alive_mask(x, threshold):
    """A cell is alive if any cell in its 3x3 neighbourhood has alpha > threshold."""
    return (maxpool3x3(x[..., 3]) > threshold)[..., None]


def update(x, params, rows=None):
    """The learned residual dx = W2^T relu(W1^T perceive(x) + b1).

    rows: optional flat indices (into B*H*W) of the cells to evaluate; every
    other cell gets dx = 0. step() passes the pre-alive cells, which is exact
    because everything outside them is zeroed by the life mask anyway.
    Returns (dx, rows, h) where h holds the hidden activations of those rows.
    """
    B, H, W, C = x.shape
    y = perceive(x).reshape(-1, 3 * C)
    if rows is not None:
        y = y[rows]
    h = y @ params["W1"]
    h += params["b1"]
    np.maximum(h, 0, out=h)
    out = h @ params["W2"]
    if rows is None:
        return out.reshape(B, H, W, C), rows, h
    dx = np.zeros((B * H * W, C), x.dtype)
    dx[rows] = out
    return dx.reshape(B, H, W, C), rows, h


def step(x, params, fire_mask, alive_threshold=0.1):
    """One NCA step. fire_mask: (B, H, W, 1) bool, which cells update this step."""
    pre = alive_mask(x, alive_threshold)
    dx, _, _ = update(x, params, np.flatnonzero(pre))
    x = x + dx * fire_mask
    post = alive_mask(x, alive_threshold)
    return x * (pre & post)


def make_seed(batch, height, width, channels, dtype=np.float32):
    """Empty grid with one live cell in the centre: alpha and hidden channels = 1."""
    x = np.zeros((batch, height, width, channels), dtype)
    x[:, height // 2, width // 2, 3:] = 1.0
    return x
