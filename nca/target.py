"""Load a target image as premultiplied RGBA, padded with empty space."""
import numpy as np
from PIL import Image


def load_target(path, size=40, pad=16):
    """Return a (H, W, 4) float32 array in [0, 1] with premultiplied RGB.

    The image is fitted into a size x size square (aspect ratio kept), then
    `pad` transparent pixels are added on every side, as in the Distill
    article (40 px emoji + 16 px padding = 72 x 72 grid).
    """
    im = Image.open(path).convert("RGBA")
    scale = size / max(im.size)  # shrink or enlarge so the longer side is exactly `size`
    im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    a = np.asarray(canvas, np.float32) / 255.0
    a[..., :3] *= a[..., 3:4]  # premultiply
    return np.pad(a, ((pad, pad), (pad, pad), (0, 0)))


def to_rgb(rgba, background=1.0):
    """Composite premultiplied RGBA over a solid background (for previews)."""
    a = np.clip(rgba[..., 3:4], 0.0, 1.0)
    return np.clip(rgba[..., :3] + (1.0 - a) * background, 0.0, 1.0)
