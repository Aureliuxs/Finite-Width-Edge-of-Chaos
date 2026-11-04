"""
Information about input similarity transmitted through L layers of a finite tanh network, along the mean-field
edge of chaos (q* from 1e-4 to 2), with ReLU (He) and linear references.  Output: data/info/info_L{L}_N{N}.npy
(columns q, I, I_se, sig, res, Qratio, comp, logspread, sw2, sb2) and ref_L{L}_N{N}.npy (rows relu, linear: I, I_se, comp,
logspread).  Takes a few hours on two cores; the data folder ships with the results.
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eocfw.info import LIN, RELU, run_point  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "info")
os.makedirs(D, exist_ok=True)
qs = np.logspace(-4, 0.3, 18)
V1 = os.path.join(D, "..", "info_v1")       # the earlier (fewer-network) results; a point is redone unless it already differs


def done(L, N):
    f, g = os.path.join(D, f"info_L{L}_N{N}.npy"), os.path.join(V1, f"info_L{L}_N{N}.npy")
    return os.path.exists(f) and (not os.path.exists(g) or not np.array_equal(np.load(f), np.load(g)))


for L in [10, 40, 160]:
    for N in ([64, 128, 256, 512] if L < 160 else [128, 256, 512]):
        if done(L, N):
            continue
        t = time.time()
        rows = []
        nn = {10: 40, 40: 40, 160: 16}[L]          # networks per point (the error bars of Fig. 4 are over these)
        for q in qs:
            r = run_point(N, L, q, n_nets=nn, seed=int(1000 * q) + N + L)
            rows.append([q, r["I"], r["I_se"], r["sig"], r["res"], r["Qratio"], r["comp"], r["logspread"], r["sw2"],
                         r["sb2"]])
        ref = []
        for act, sw2 in [(RELU, 2.0), (LIN, 1.0)]:
            r = run_point(N, L, 1.0, n_nets=nn, seed=7 + N + L, act=act, sw2=sw2, sb2=0.0)
            ref.append([r["I"], r["I_se"], r["comp"], r["logspread"]])
        np.save(os.path.join(D, f"info_L{L}_N{N}.npy"), np.array(rows))
        np.save(os.path.join(D, f"ref_L{L}_N{N}.npy"), np.array(ref))
        print(f"L={L} N={N} done ({time.time() - t:.0f}s)", flush=True)

# a second smooth activation (erf) along its own edge of chaos, L = 40: the plateau and its fall are the same
from scipy.special import erf as _erf  # noqa: E402
from eocfw.meanfield import eoc_point  # noqa: E402

ERF = lambda x: _erf(x / math.sqrt(2))
for N in [64, 256]:
    if os.path.exists(os.path.join(D, f"info_erf_L40_N{N}.npy")):
        continue
    t = time.time()
    rows = []
    for q in qs:
        sw2, sb2 = eoc_point("erf", q)
        r = run_point(N, 40, q, n_nets=40, seed=int(1000 * q) + N + 3, act=ERF, sw2=sw2, sb2=sb2)
        rows.append([q, r["I"], r["I_se"], r["sig"], r["res"], r["Qratio"], r["comp"], r["logspread"], r["sw2"], r["sb2"]])
    np.save(os.path.join(D, f"info_erf_L40_N{N}.npy"), np.array(rows))
    print(f"erf L=40 N={N} done ({time.time() - t:.0f}s)", flush=True)
