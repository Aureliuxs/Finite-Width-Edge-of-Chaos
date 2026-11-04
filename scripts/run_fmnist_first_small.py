"""
Small test: what in the first layer makes the difference between Crinit and the tuned zero-bias rule?

Crinit brings the first hidden layer to the steady signal q* by multiplying the input by s = sqrt(q*/sigma_w^2)
(the first-layer weights keep the variance sigma_w^2/d_in); the standard rules leave the input alone, so their first
hidden layer has the signal sigma_w^2.  The multiplier changes two things at once: the signal of the first hidden
layer (q* instead of sigma_w^2) and the gradient of the first-layer weights (s times smaller).  The variants below
separate the two.  Writing W_1 for the first-layer weights and eta for the learning rate:

  std        q_1 = sigma_w^2, first-layer update as the standard rules            (q1 large, gradient x1)
  in         q_1 = q* by input scaling (the paper's runs so far)                   (q1 small, gradient x s)
  weight     q_1 = q* by scaling W_1 instead (what crinit.torch_init does)         (q1 small, gradient x1, W_1 s times
                                                                                    smaller, so a 1/s larger relative step)
  in+lr      input scaling with the first-layer learning rate multiplied by 1/s    (q1 small, update as std)
  std-lr     standard first layer with the first-layer learning rate multiplied by s  (q1 large, update s times smaller)
  q1 scan    input scaling to q_1 in {0.5, 1, 2, 3} (gradient x sqrt(q_1/sigma_w^2))

tanh, Gaussian weights, Crinit at gamma = 1 and (for L = 32) the tuned zero-bias sigma_w^2 = 1.41; three learning rates
(m = 1/2, 1, 2); 12 networks each; validation accuracy after two epochs.  Results: data/fmnist/first_small_L{L}.pkl.

Usage: python run_fmnist_first_small.py [L] [threads]
"""
import math
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_fmnist_tuned as r  # noqa: E402
from eocfw.tuned import Batch, load_fashion  # noqa: E402


def variants(L, C):
    V = {}
    for base in (["tanh_crinit", "tanh_zb:1.4142"] if L == 32 else ["tanh_crinit"]):
        c = C[base]
        q = c["q1"] if base == "tanh_crinit" else r._crinit().fixed_point("tanh", c["sw2"], 0.0)
        s = math.sqrt(max(q - c["sb2"], 1e-12) / c["sw2"])
        tag = "crinit" if base == "tanh_crinit" else "zb1.41"
        V[f"{tag}:std"] = dict(c, q1=c["sw2"] + c["sb2"])
        V[f"{tag}:in"] = dict(c, q1=q)
        V[f"{tag}:weight"] = dict(c, q1=q, first_layer="weight")
        V[f"{tag}:in+lr"] = dict(c, q1=q, lr1_mult=1 / s)
        if L == 32:
            V[f"{tag}:std-lr"] = dict(c, q1=c["sw2"] + c["sb2"], lr1_mult=s)
        if base == "tanh_crinit" and L == 32:
            for q1 in (0.5, 1.0, 2.0, 3.0):
                V[f"{tag}:q1={q1:g}"] = dict(c, q1=q1)
            for f in (0.5, 2.0):                        # is the relative step 1/s of the weight-scaled first layer near the best?
                V[f"{tag}:weight*lr{f:g}"] = dict(c, q1=q, first_layer="weight", lr1_mult=f)
    return V


if __name__ == "__main__":
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
    torch.set_flush_denormal(True)
    data = load_fashion(r.RAW)
    V = variants(L, r.configs(L))
    mults = (0.5, 1.0, 2.0)
    n = 12
    E = 2 if L == 32 else 4
    path = os.path.join(r.D, f"first_small_L{L}.pkl")
    out = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
    todo = [k for k in V if k not in out]
    per = 126
    for i in range(0, len(todo), max(1, per // (len(mults) * n))):
        chunk = todo[i:i + max(1, per // (len(mults) * n))]
        t0 = time.time()
        mem = []
        for k in chunk:
            mem += r.members_for(k, V[k], L, mults, [3000 + j for j in range(n)])
        b = Batch(data, L, r.N, mem)
        for _ in range(E):
            b.run_epoch(val=True, test=False)
        S = b.summary()
        for k in chunk:
            res = {}
            for m in mults:
                idx = [j for j, mm in enumerate(S["members"]) if mm["key"] == k and mm["lr_mult"] == m]
                v = np.where(np.isnan(S["val"][:, idx]), 0.1, S["val"][:, idx])
                res[m] = dict(val=v, diverged=S["diverged"][idx])
            out[k] = dict(cfg=V[k], epochs=E, res=res)
            best = max(mults, key=lambda m: res[m]["val"][-1].mean())
            v = res[best]["val"][-1]
            r.log(f"L={L} {k:16s} best m={best:<4g} val {v.mean():.4f} +- {v.std(ddof=1) / math.sqrt(v.size):.4f}  "
                  + "  ".join(f"m={m:g}: {res[m]['val'][-1].mean():.4f}" for m in mults)
                  + (f"  ({int(sum(res[m]['diverged'].sum() for m in mults))} div)" if any(res[m]["diverged"].any() for m in mults) else ""))
        pickle.dump(out, open(path, "wb"))
        r.log(f"chunk done ({time.time() - t0:.0f}s)")
