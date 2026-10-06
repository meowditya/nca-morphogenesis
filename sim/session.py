"""Transport-agnostic protocol between one browser tab and one Simulation.

The server owns the timing loop; this class owns the rules:

  client -> server  JSON text messages, see `handle()`
  server -> client  JSON text ("hello", "stats", "error") and binary frames

Binary frame layout (little endian):

  offset  size        field
  0       4           magic b"NCA1"
  4       4  uint32   simulation step
  8       2  uint16   height H
  10      2  uint16   width W
  12      1  uint8    view (0 rgba, 1 hidden, 2 alive)
  13      1  uint8    flags (bit 0: paused)
  14      2  uint16   reserved
  16      H*W*4       RGBA bytes, row-major, premultiplied colour
  ...     ceil(H*W/8) wall bitmask, row-major, most significant bit first

Brush and reset commands are queued by `handle()` (called from the network
task) and applied by `tick()` (called from the stepping thread) between
steps, so the grid is never touched by two threads at once.
"""
import collections
import math
import struct
import time

import numpy as np

from .engine import VIEWS

HEADER = struct.Struct("<4sIHHBBH")
TOOLS = ("erase", "noise", "wall", "unwall", "seed")
MAX_POINTS = 512


def _number(value, lo, hi):
    """A finite float clipped to [lo, hi]; ValueError for anything else (NaN, inf, strings, bools)."""
    if isinstance(value, bool):
        raise ValueError("bool is not a number here")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError("not finite")
    return min(max(v, lo), hi)


class Session:
    def __init__(self, sim, steps_per_frame=2):
        self.sim = sim
        self.steps_per_frame = steps_per_frame
        self.paused = False
        self.view = "rgba"
        self.commands = collections.deque()
        self.frames = 0
        self.stats_t = time.perf_counter()
        self.stats_frames = 0
        self.stats_steps = 0

    # ---- network side ---------------------------------------------------------------
    def hello(self):
        return {
            "type": "hello",
            "height": self.sim.H,
            "width": self.sim.W,
            "backend": self.sim.backend.name,
            "checkpoint": self.sim.checkpoint.replace("\\", "/").split("/")[-1],
            "meta": self.sim.meta,
            "steps_per_frame": self.steps_per_frame,
            "tools": list(TOOLS),
            "views": list(VIEWS),
        }

    def handle(self, msg):
        """Apply one client message. Returns an error dict for malformed input, else None.

        Never raises on bad input: every number is checked to be finite and clipped,
        so a buggy or hostile client cannot crash its session.
        """
        kind = msg.get("type") if isinstance(msg, dict) else None
        try:
            if kind == "stroke":
                tool, pts = msg.get("tool"), msg.get("points")
                if tool not in TOOLS or not isinstance(pts, list) or not pts:
                    raise ValueError
                lim = 4 * max(self.sim.H, self.sim.W)
                points = [(_number(p[0], -lim, lim), _number(p[1], -lim, lim)) for p in pts[:MAX_POINTS]]
                radius = _number(msg.get("radius", 4), 0.5, 64)
                strength = _number(msg.get("strength", 0.5), 0.0, 5.0)
                self.commands.append(("stroke", tool, points, radius, strength))
            elif kind == "pause":
                self.paused = bool(msg.get("value", not self.paused))
            elif kind == "step":
                self.commands.append(("step", int(_number(msg.get("n", 1), 1, 1000))))
            elif kind == "speed":
                self.steps_per_frame = int(_number(msg.get("steps_per_frame", 2), 0, 16))
            elif kind == "view":
                if msg.get("mode") not in VIEWS:
                    raise ValueError
                self.view = msg["mode"]
            elif kind in ("reset", "clear", "clear_walls", "damage"):
                self.commands.append((kind,))
            else:
                return {"type": "error", "message": f"unknown message type {kind!r}"}
        except (TypeError, ValueError, IndexError, KeyError, OverflowError):
            return {"type": "error", "message": f"malformed {kind!r} message"}
        return None

    # ---- stepping thread ------------------------------------------------------------
    def _apply(self, cmd):
        """Run one queued command; returns how many extra steps it asks for (only "step" does)."""
        sim = self.sim
        if cmd[0] == "stroke":
            _, tool, points, radius, strength = cmd
            if tool == "erase":
                sim.erase(points, radius)
            elif tool == "noise":
                sim.noise(points, radius, strength)
            elif tool == "wall":
                sim.paint_walls(points, radius, True)
            elif tool == "unwall":
                sim.paint_walls(points, radius, False)
            elif tool == "seed":
                for x, y in points:
                    sim.plant(x, y)
            return 0
        elif cmd[0] == "reset":
            sim.reset()
        elif cmd[0] == "clear":
            sim.clear()
        elif cmd[0] == "clear_walls":
            sim.walls[:] = False
        elif cmd[0] == "damage":
            sim.random_damage()
        elif cmd[0] == "step":
            return cmd[1]
        return 0

    def tick(self):
        """Apply queued commands, advance the world, return the encoded frame."""
        extra = 0
        while self.commands:
            extra += self._apply(self.commands.popleft())
        n = extra if self.paused else self.steps_per_frame + extra
        self.sim.step(n)
        self.frames += 1
        self.stats_frames += 1
        self.stats_steps += n
        return self.encode()

    def encode(self):
        sim = self.sim
        header = HEADER.pack(b"NCA1", sim.steps & 0xFFFFFFFF, sim.H, sim.W,
                             VIEWS.index(self.view), int(self.paused), 0)
        return header + sim.render(self.view).tobytes() + np.packbits(sim.walls).tobytes()

    def stats(self):
        now = time.perf_counter()
        dt = max(now - self.stats_t, 1e-6)
        out = {
            "type": "stats",
            "step": self.sim.steps,
            "fps": round(self.stats_frames / dt, 1),
            "steps_per_sec": round(self.stats_steps / dt, 1),
            "ms_per_step": round(self.sim.step_time * 1000, 2),
            "mature_cells": self.sim.mature_cells(),
            "paused": self.paused,
            "steps_per_frame": self.steps_per_frame,
        }
        self.stats_t, self.stats_frames, self.stats_steps = now, 0, 0
        return out


def decode(frame):
    """Inverse of Session.encode (used by tests and tools)."""
    magic, step, H, W, view, flags, _ = HEADER.unpack_from(frame, 0)
    assert magic == b"NCA1"
    rgba = np.frombuffer(frame, np.uint8, H * W * 4, HEADER.size).reshape(H, W, 4)
    bits = np.frombuffer(frame, np.uint8, offset=HEADER.size + H * W * 4)
    walls = np.unpackbits(bits)[: H * W].reshape(H, W).astype(bool)
    return dict(step=step, H=H, W=W, view=VIEWS[view], paused=bool(flags & 1), rgba=rgba, walls=walls)
