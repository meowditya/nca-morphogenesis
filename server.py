"""Real-time NCA sandbox server: serves the web client and streams the grid over WebSockets.

    python server.py --checkpoint checkpoints/spiderweb.npz
    # then open http://localhost:8000

Every browser tab gets its own Simulation. Per connection, a loop runs at a
fixed frame rate: apply queued brush commands -> advance `steps_per_frame`
NCA steps (in a worker thread, so the event loop stays responsive) -> send
one binary frame. Stats go out as JSON twice a second.
"""
import argparse
import asyncio
import json
import os
import traceback

import uvicorn
from starlette.applications import Starlette
from starlette.routing import Mount, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from nca.checkpoint import load_npz
from sim.engine import Simulation
from sim.session import Session

ROOT = os.path.dirname(os.path.abspath(__file__))


def create_app(checkpoint, size=96, backend="auto", device="auto", fps=30, steps_per_frame=2):
    # fail at startup, not on every connection attempt, if something is misconfigured
    load_npz(checkpoint)
    if backend == "torch":
        import torch  # noqa: F401  (raises ImportError here if PyTorch is missing)

    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        sim = await asyncio.to_thread(Simulation, checkpoint, size, size, backend, device)
        session = Session(sim, steps_per_frame)
        await websocket.send_text(json.dumps(session.hello()))

        async def reader():
            while True:
                text = await websocket.receive_text()
                try:
                    reply = session.handle(json.loads(text))
                except json.JSONDecodeError:
                    reply = {"type": "error", "message": "invalid JSON"}
                except Exception as exc:  # a bug in a handler should not end the session
                    traceback.print_exc()
                    reply = {"type": "error", "message": f"server error: {exc}"}
                if reply:
                    await websocket.send_text(json.dumps(reply))

        reader_task = asyncio.create_task(reader())
        loop = asyncio.get_running_loop()
        period = 1.0 / fps
        next_frame = next_stats = loop.time()
        try:
            while not reader_task.done():
                frame = await asyncio.to_thread(session.tick)
                await websocket.send_bytes(frame)
                now = loop.time()
                if now >= next_stats:
                    await websocket.send_text(json.dumps(session.stats()))
                    next_stats = now + 0.5
                next_frame += period
                if next_frame < now:          # fell behind (slow machine): don't try to catch up
                    next_frame = now
                await asyncio.sleep(next_frame - now)
        except (WebSocketDisconnect, RuntimeError, OSError):
            pass  # client went away mid-send
        finally:
            reader_task.cancel()
            if reader_task.done() and not reader_task.cancelled():
                exc = reader_task.exception()
                if exc and not isinstance(exc, (WebSocketDisconnect, RuntimeError, OSError)):
                    raise exc

    return Starlette(routes=[
        WebSocketRoute("/ws", ws_endpoint),
        Mount("/", app=StaticFiles(directory=os.path.join(ROOT, "web"), html=True)),
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=os.path.join(ROOT, "checkpoints", "spiderweb.npz"))
    ap.add_argument("--size", type=int, default=96, help="grid is size x size cells")
    ap.add_argument("--backend", default="auto", choices=["auto", "torch", "numpy"])
    ap.add_argument("--device", default="auto", help="torch device: auto | cpu | cuda | mps")
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--steps-per-frame", type=int, default=2)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    app = create_app(args.checkpoint, args.size, args.backend, args.device, args.fps, args.steps_per_frame)
    print(f"NCA sandbox on http://{args.host}:{args.port}  (checkpoint {args.checkpoint})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
