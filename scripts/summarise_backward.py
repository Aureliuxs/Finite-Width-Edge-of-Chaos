"""Reduce the large outputs of backward_walk.py to the compact summaries used by make_figS_backward.py
(data/backward/summary_*.npz): per-network stationary mean steps and block sums of the forward and backward walks,
the total sums X_L and Y_L, and the backward walk accumulated over the 96 layers below the output."""
import glob
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "backward")
BURN, BLOCK, KMAX = 100, 100, 96


def summarise(path):
    d = pickle.load(open(path, "rb"))
    L = d["L"]
    out = dict(N=d["N"], L=L, M=d["M"], sw2=d["sw2"], sb2=d["sb2"])
    for key, s in (("f", d["steps_f"][:, BURN:]), ("b", d["steps_b"][:, :L - BURN])):
        nb = s.shape[1] // BLOCK
        out["mean_" + key] = s.mean(1)
        out["blocks_" + key] = s[:, :nb * BLOCK].reshape(len(s), nb, BLOCK).sum(2)
    out["X"] = d["steps_f"].sum(1)
    out["Y"] = d["steps_b"].sum(1)
    s = d["steps_b"][:, L - KMAX:][:, ::-1]                    # backward steps starting at the output
    out["Yk"] = np.concatenate([np.zeros((len(s), 1)), np.cumsum(s, axis=1)], axis=1).astype(np.float32)
    name = os.path.basename(path).replace("walk_", "summary_").replace(".pkl", ".npz")
    np.savez_compressed(os.path.join(DATA, name), **out)
    return name


if __name__ == "__main__":
    for f in sorted(glob.glob(os.path.join(DATA, "walk_N*_L400_*.pkl"))):
        print(summarise(f))
