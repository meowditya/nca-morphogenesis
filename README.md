# NCA Morphogenesis

A neural cellular automaton that grows a pattern from a single cell, maintains it,
and regrows it after damage. It comes with a live web sandbox where you
can wound the organism, inject noise and build walls while it runs. It is based on
[Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)
(Mordvintsev et al., Distill 2020), and the default target is the 🕸 spider-web emoji
from Google's Noto font.

![growth from one cell (top) and regrowth after a random wound (bottom)](checkpoints/spiderweb_starter_eval.png)

*Top: growth from a single seed cell. Bottom: the target, then a randomly damaged
pattern regrowing. Produced by `tools/evaluate.py` with the bundled starter checkpoint.*

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python server.py                                       # then open http://localhost:8000
```

The sandbox loads `checkpoints/spiderweb_starter.npz` by default. If PyTorch is
installed it runs the model with PyTorch (CUDA when available, otherwise CPU; pass
`--device mps` on Apple silicon), otherwise with NumPy. Both implement the same update
rule, and `tests/test_torch_parity.py` checks that they agree. The sandbox alone does
not need PyTorch: `pip install numpy starlette "uvicorn[standard]"` is enough.

## Train your own model

```bash
python -m nca.train --out checkpoints/spiderweb.npz             # uses a GPU if one is visible
python server.py --checkpoint checkpoints/spiderweb.npz
python tools/evaluate.py --checkpoint checkpoints/spiderweb.npz  # growth / persistence / regrowth / noise numbers
```

The defaults follow the Distill "regenerating" experiment: 40 px target with 16 px
padding (a 72×72 grid), 16 channels, 128 hidden units, fire rate 0.5, a pool of
1024 samples, batch 8, 64-96 steps per rollout, Adam at 2e-3 dropping to 2e-4 after
2000 of 8000 iterations, and per-tensor gradient normalisation.

One deliberate difference: Distill erases a disc from 3 samples per batch. Here 2
samples get a disc erased and 2 more get Gaussian noise (standard deviation 0.1-0.6)
on the living cells inside a disc. The brief asks for regeneration after injected
noise, and a model trained only on erasure does not recover from noise (see below).
`--damage 3 --noise-damage 0` gives the original Distill recipe. Useful flags:

| flag | what it does |
|---|---|
| `--target path.png` | any RGBA image with a transparent background |
| `--damage N` / `--noise-damage N` | samples per batch erased / noised (both 0 = growing-only model) |
| `--noise-std lo,hi` | range of the noise strength used in training |
| `--init model.npz` | start from existing weights (fine-tuning); iteration counts carry over |
| `--grad-checkpoint` | recomputes activations in the backward pass; much less GPU memory |
| `--resume run.pt` | continue a run; `.pt` holds weights, optimiser, schedule and the pool |
| `--device cpu\|cuda\|mps` | defaults to the first available |

With `--out checkpoints/spiderweb.npz` the trainer writes `spiderweb.npz` (weights for
the simulator), `spiderweb.pt` (full training state for `--resume`), `spiderweb_log.json`
(loss per iteration) and a `spiderweb_previews/` folder with a PNG of the batch every
`--log-every` iterations, all in `checkpoints/`. On Colab: upload the zip, switch to a
GPU runtime and run
`!unzip -q nca-morphogenesis.zip && cd nca-morphogenesis && pip install -r requirements.txt && python -m nca.train`.

## Run the checks

```bash
python tests/test_torch_parity.py   # run this first: PyTorch model == NumPy port (forward and gradients)
python tests/test_gradients.py      # hand-written NumPy backward pass vs finite differences
python tests/test_engine.py         # engine, brushes, protocol, WebSocket endpoint
python tests/test_training.py       # noise damage and fine-tuning in the NumPy trainer
```

With pytest installed, `python -m pytest tests -s` runs them all.

## How the pieces map to the brief

| brief | where |
|---|---|
| **NCA architecture**: cellular update rule as a light 2-D conv net in PyTorch | `nca/model.py`: fixed depthwise perception (identity, Sobel x/y), two 1×1 convs (48 → 128 → 16, last layer zero-initialised), stochastic per-cell updates, alive masking on alpha |
| **Training**: grow from one seed with a persistent sample pool and random damage | `nca/train.py`: pool of 1024, worst sample reseeded, the best get a random disc erased (2) or noised (2), BPTT over 64-96 steps |
| **Simulation engine**: frame-by-frame recursive grid updates, state persistence, temporal flow | `sim/engine.py` (grid state, walls, brushes, PyTorch or NumPy backend) and the fixed-rate loop in `server.py` (pause, single-step, steps-per-frame) |
| **Interactive sandbox and transport**: WebSocket canvas, inject noise, paint obstacles, erase | `server.py` (Starlette WebSocket), `sim/session.py` (protocol), `web/` (canvas client) |

## Sandbox

| tool | key | effect |
|---|---|---|
| Erase | `E` | zeroes every channel under the brush (a wound) |
| Noise | `N` | adds Gaussian noise to all 16 channels of living cells under the brush (strength = std. dev.) |
| Wall | `W` | paints obstacles: cells forced to zero after every step, so growth has to route around them |
| Unwall | `U` | removes obstacles |
| Seed | `S` | plants a new seed cell where you click; grow a second organism |

Also: `Space` pause, `.` single step, `R` reset, `D` erase a random disc over the organism, `[` `]` brush size, `1` `2` `3` switch between the RGBA view, hidden channels
4-6 as colour, and the alive mask. "Steps per frame" sets simulation speed; the server
streams up to 30 frames per second (`--fps`).

Each browser tab gets its own simulation. Server flags: `--size` (grid, default 96),
`--fps`, `--steps-per-frame`, `--backend auto|torch|numpy`, `--device`, `--host`, `--port`.

### Wire protocol

Client to server, JSON text: `stroke {tool, points: [[x, y], ...], radius, strength}`,
`pause {value}`, `step {n}`, `speed {steps_per_frame}`, `view {mode}`, `reset`, `clear`,
`clear_walls`, `damage`. Coordinates are in cells and may be fractional; the client
interpolates drags so strokes have no gaps.

Server to client: a JSON `hello` (grid size, backend, checkpoint metadata), JSON `stats`
twice a second, and one binary frame per tick: a 16-byte header (`NCA1`, step, H, W, view,
flags), H×W×4 RGBA bytes with premultiplied colour, then a bit-packed wall mask. The
full layout is in the docstring of `sim/session.py`.

## Layout

```
nca/model.py          PyTorch NCA (perceive -> update -> stochastic fire -> alive mask)
nca/train.py          PyTorch trainer: pool, damage, BPTT, export
nca/numpy_nca.py      the same update rule in NumPy (simulator fallback, reference trainer)
nca/checkpoint.py     framework-neutral .npz weight format shared by everything
nca/target.py         target loading (premultiplied RGBA + padding)
sim/engine.py         grid state, obstacles, brushes, backends
sim/session.py        per-connection protocol and frame encoding
server.py             Starlette app: static client + /ws stream loop
web/                  canvas client (index.html, app.js, style.css)
tools/train_numpy.py  NumPy trainer with a hand-written backward pass (made the starter checkpoint)
tools/evaluate.py     growth / persistence / regeneration / noise metrics and filmstrips
tests/                gradient check, engine/server tests, PyTorch parity test
docs/                 sandbox screenshot
data/                 spiderweb_128.png target, rendered from the Noto Color Emoji font (SIL OFL 1.1)
checkpoints/          spiderweb_starter.npz and its evaluation
```

## About the starter checkpoint

`checkpoints/spiderweb_starter.npz` was trained with `tools/train_numpy.py`, because
PyTorch could not be installed in the environment where this project was built. It uses
the same architecture and recipe as `nca/train.py`, but ran for 4,000 iterations (half the
Distill schedule) on 2 CPU cores, taking 2 h 48 min. Numbers from one run of
`python tools/evaluate.py` (mean squared error against the target; they vary a little
with the random seed):

| state | MSE |
|---|---|
| empty grid | 0.0296 |
| after 60 / 80 / 200 growth steps | 0.0092 / 0.0032 / 0.0027 |
| left running: step 500 / 1,000 / 2,000 | 0.0028 / 0.0039 / 0.0055 (2,000 needs `--long 2000`) |
| 8 random wounds: just after, +50, +200 steps (median) | 0.0055, 0.0035, 0.0030 |

Growth takes about 80 steps. 200 steps after a wound, a median of 86% of the extra
error is gone; the outline and spokes come back solidly. Two weaknesses: the inner rings
are still faint, and the web is not perfectly stable. Left alone it slowly thickens, as
the rising error after step 500 shows. The loss was still falling when training stopped,
so a full 8,000-iteration PyTorch run should improve both.

![the sandbox: grown web, an erase stroke, and the regrown web](docs/sandbox_wound.png)

The model was only ever trained on erasure. In the same evaluation, noise injected into
the centre at strength 0.1 does little harm (0.0027 → 0.0031 after 300 steps), but at
0.3 and 0.6 it leaves scars that only partly heal (0.0050 and 0.0112).

**Noise-damage experiment.** I fine-tuned the starter for 1,200 more iterations with the
new default damage (2 erased + 2 noised per batch, learning rate 2e-4):

```
python tools/train_numpy.py --init checkpoints/spiderweb_starter.npz --iters 1200 \
    --lr 2e-4 --lr-decay-at 1000000 --seed 1 --out checkpoints/spiderweb_ft.npz
```

| `tools/evaluate.py` measure | starter | after fine-tune |
|---|---|---|
| grown, step 200 | 0.0027 | 0.0024 |
| noise 0.3 / 0.6, then 300 steps | 0.0050 / 0.0112 | 0.0045 / 0.0080 |
| left running to step 1,000 | 0.0039 | 0.0069 |
| erase wounds repaired after 200 steps (median) | 86% | 65% |

Noise healing improved, but the web drifted more on long runs, so the fine-tuned weights
were not adopted. The likely cause is the short run: a fine-tune starts a fresh sample
pool, so in 1,200 iterations no sample lives long enough to teach long-term stability.
A full-length run with noise damage from the start, which is what `python -m nca.train`
now does, avoids that, but it has not been run here; check it with `tools/evaluate.py`.

`nca/model.py` and `nca/train.py` were written but not executed where this was built.
`tests/test_torch_parity.py` checks them against the NumPy path, whose gradients are
verified by finite differences, so run that test once before a long training run.

## Where to take it next

- Train other targets (any transparent PNG), or a growing-only model (`--damage 0`) and
  compare how each reacts to the same wound in the sandbox.
- Add walls to training (zero random rectangles every step) so obstacle avoidance is
  learned rather than accidental.
- Move inference into the browser (WebGL or WebGPU) to remove the server from the loop;
  the model is only 8,320 parameters.
