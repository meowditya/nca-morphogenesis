"""Framework-neutral checkpoint format shared by both trainers and the simulator.

A checkpoint is a single .npz file holding three weight arrays plus a JSON
metadata string:

    W1   (3*C, hidden)  float32   perception -> hidden  (1x1 conv as a matrix)
    b1   (hidden,)      float32
    W2   (hidden, C)    float32   hidden -> state update (no bias)
    meta JSON           channels, hidden, fire_rate, alive_threshold, target info

Perception channel j = 3*c + k, where c is the state channel and
k = 0 identity, 1 Sobel-x, 2 Sobel-y. nca/model.py (PyTorch) and
nca/numpy_nca.py both use this ordering, so weights move between them freely.
"""
import json

import numpy as np

FORMAT = "nca-npz-v1"


def save_npz(path, W1, b1, W2, meta):
    meta = dict(meta)
    meta["format"] = FORMAT
    meta.setdefault("channels", int(W2.shape[1]))
    meta.setdefault("hidden", int(W2.shape[0]))
    np.savez(
        path,
        W1=np.asarray(W1, np.float32),
        b1=np.asarray(b1, np.float32),
        W2=np.asarray(W2, np.float32),
        meta=np.array(json.dumps(meta)),
    )


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        params = {k: z[k].astype(np.float32) for k in ("W1", "b1", "W2")}
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path}: not an {FORMAT} checkpoint")
    C, hidden = meta["channels"], meta["hidden"]
    if params["W1"].shape != (3 * C, hidden) or params["W2"].shape != (hidden, C):
        raise ValueError(f"{path}: weight shapes do not match metadata")
    return params, meta
