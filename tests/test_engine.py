"""Simulation engine, session protocol and server endpoint tests (NumPy backend, no PyTorch needed).

    python tests/test_engine.py        # or: python -m pytest tests
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sim.engine import Simulation  # noqa: E402
from sim.session import Session, decode  # noqa: E402

CKPT = os.path.join(ROOT, "checkpoints", "spiderweb_starter.npz")


def make_sim(size=64):
    return Simulation(CKPT, size, size, backend="numpy", seed=0)


def test_seed_and_growth():
    sim = make_sim()
    assert sim.mature_cells() == 1
    sim.step(60)
    assert sim.mature_cells() > 50, "the starter checkpoint should grow from a single seed"


def test_brushes():
    sim = make_sim()
    sim.step(60)
    c = sim.W / 2
    sim.erase([(c, c)], 6)
    assert np.all(sim.state[int(c) - 3:int(c) + 3, int(c) - 3:int(c) + 3] == 0)

    before = sim.state.copy()
    sim.noise([(c, c + 12)], 4, strength=0.5)
    changed = np.any(sim.state != before, axis=-1)
    assert changed.any() and not (changed & ~sim.disc_mask([(c, c + 12)], 4)).any()

    sim.paint_walls([(5, 5)], 3)
    walls = sim.walls.copy()
    assert walls.sum() > 20
    sim.plant(5, 5)                       # planting on a wall is refused
    assert sim.state[5, 5, 3] == 0
    sim.step(5)
    assert np.all(sim.state[walls] == 0), "walls must stay dead"
    sim.paint_walls([(5, 5)], 3, value=False)
    assert not sim.walls.any()


def test_session_protocol():
    sess = Session(make_sim(), steps_per_frame=3)
    hello = sess.hello()
    assert hello["height"] == 64 and hello["backend"] == "numpy"
    f = decode(sess.tick())
    assert f["step"] == 3 and f["rgba"].shape == (64, 64, 4) and not f["paused"]

    assert sess.handle({"type": "pause", "value": True}) is None
    assert sess.handle({"type": "step", "n": 2}) is None
    assert sess.handle({"type": "view", "mode": "alive"}) is None
    assert sess.handle({"type": "stroke", "tool": "wall", "points": [[10, 10]], "radius": 2}) is None
    f = decode(sess.tick())
    assert f["step"] == 5 and f["paused"] and f["view"] == "alive" and f["walls"].sum() > 0
    assert decode(sess.tick())["step"] == 5   # paused: no progress without "step"

    bad_messages = [
        {"type": "nope"}, None, [], {"type": "view", "mode": "xray"},
        {"type": "stroke", "tool": "laser", "points": [[1, 1]]},
        {"type": "stroke", "tool": "erase", "points": "x"},
        {"type": "stroke", "tool": "erase", "points": [[1]]},
        {"type": "stroke", "tool": "erase", "points": [[float("nan"), 3]]},
        {"type": "stroke", "tool": "erase", "points": [[3, 3]], "radius": float("inf")},
        {"type": "step", "n": "x"}, {"type": "step", "n": float("inf")},
        {"type": "speed", "steps_per_frame": "abc"}, {"type": "speed", "steps_per_frame": None},
    ]
    for bad in bad_messages:
        assert sess.handle(bad)["type"] == "error", bad
    sess.handle({"type": "stroke", "tool": "seed", "points": [[1e300, -1e300]]})  # clipped, harmless
    sess.handle({"type": "pause", "value": False})
    assert decode(sess.tick())["step"] > 5, "session must keep running after bad input"


def test_server_websocket():
    from starlette.testclient import TestClient
    import server

    app = server.create_app(CKPT, size=48, backend="numpy", fps=60)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        with client.websocket_connect("/ws") as ws:
            assert json.loads(ws.receive_text())["type"] == "hello"
            ws.send_text("{not json")
            got_error = got_frame = False
            for _ in range(50):
                m = ws.receive()
                if m.get("bytes") is not None:
                    got_frame = decode(m["bytes"])["H"] == 48
                elif m.get("text"):
                    got_error |= json.loads(m["text"]).get("message") == "invalid JSON"
                if got_frame and got_error:
                    break
            assert got_frame and got_error


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
