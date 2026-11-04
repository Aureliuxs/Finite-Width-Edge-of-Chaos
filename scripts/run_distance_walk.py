"""
Drift and long-run variance of ln D (distance between two nearby inputs) on the mean-field edge of chaos:
theory a_1, b_1 against the exact finite-N chain, for tanh and erf (Gaussian weights) and for tanh with
orthogonal weights.  Output: data/distance_walk.json
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eocfw.finite_width import a1_b1  # noqa: E402
from eocfw.meanfield import eoc_point  # noqa: E402
from eocfw.simulate import distance_walk  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "distance_walk.json")
QS = [0.02, 0.05, 0.15, 0.4, 1.0]
res = {"gaussian": {}, "orthogonal": {}}
for act in ["tanh", "erf"]:
    res["gaussian"][act] = []
    for q in QS:
        sw2, sb2 = eoc_point(act, q)
        th = a1_b1(act, q)
        row = dict(q=q, sw2=sw2, sb2=sb2, theory={k: float(v) for k, v in th.items()}, sims=[])
        for N in [200, 800]:
            t = time.time()
            r = distance_walk(act, N, sw2, sb2, B=1500 if N <= 200 else 500, L=2000, burn=600, block=200, seed=N)
            row["sims"].append(dict(N=N, N_mu=float(N * r["mu"]), N_var_long=float(N * r["var_long"]),
                                    N_var_step=float(N * r["var_step"]), N_mu_mean=float(N * r["mu_mean"])))
            print(f"{act} q*={q} N={N}: N mu={N * r['mu']:.3f} (a1={th['a1']:.3f})  N var={N * r['var_long']:.3f} "
                  f"(b1={th['b1']:.3f})  N mu_mean={N * r['mu_mean']:.3f} (a1+b1/2={th['a1'] + th['b1'] / 2:.3f})"
                  f"  [{time.time() - t:.0f}s]", flush=True)
        res["gaussian"][act].append(row)
        json.dump(res, open(OUT, "w"), indent=1)

res["orthogonal"]["tanh"] = []
for q in QS:
    sw2, sb2 = eoc_point("tanh", q)
    th = a1_b1("tanh", q, orthogonal=True)
    row = dict(q=q, sw2=sw2, sb2=sb2, theory={k: float(v) for k, v in th.items()}, sims=[])
    for N in [200]:
        t = time.time()
        r = distance_walk("tanh", N, sw2, sb2, B=600, L=2000, burn=400, block=200, seed=N + 1, orthogonal=True)
        row["sims"].append(dict(N=N, N_mu=float(N * r["mu"]), N_var_long=float(N * r["var_long"]),
                                N_var_step=float(N * r["var_step"]), N_mu_mean=float(N * r["mu_mean"])))
        print(f"orthogonal tanh q*={q} N={N}: N mu={N * r['mu']:.3f} (a1_orth={th['a1']:.3f})  "
              f"N var={N * r['var_long']:.3f} (b1_orth={th['b1']:.3f})  [{time.time() - t:.0f}s]", flush=True)
    res["orthogonal"]["tanh"].append(row)
    json.dump(res, open(OUT, "w"), indent=1)
print("saved", OUT)
