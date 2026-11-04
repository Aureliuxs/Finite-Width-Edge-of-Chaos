"""Many pairs of inputs through one network (SM, Sec. S3.7).  Explicit Gaussian weight matrices, tanh at the
mean-field edge of chaos with sigma_b^2 = 0.05 (sigma_w^2 = 1.761, q* = 0.570), N = 64, L = 200.  For each of
`nets` networks, P = 200 pairs of nearby inputs (a base input at the steady signal and a random tangent direction)
are propagated and ln D_L is recorded per pair.  Output: data/pairs_one_network.json with the mean of ln D_L, the
variance across networks of the per-network mean over pairs, and the within-network variance across pairs.
Takes about ten minutes on one core for 160 networks."""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from eocfw.finite_width import a1_b1  # noqa: E402
from eocfw.meanfield import fixed_point  # noqa: E402

N, L, SW2, SB2, P = 64, 200, 1.7609546396067868, 0.05, 200


def one_network(rng):
    q = fixed_point("tanh", SW2, SB2)
    z = rng.normal(0, np.sqrt(q), (P, N))                   # P base inputs at the steady signal
    v = rng.normal(0, 1, (P, N))
    v /= np.linalg.norm(v, axis=1, keepdims=True)           # unit tangent vectors (the small differences)
    logD = np.zeros(P)
    for _ in range(L):
        W = rng.normal(0, np.sqrt(SW2 / N), (N, N))
        b = rng.normal(0, np.sqrt(SB2), N)
        a = np.tanh(z)
        v = ((1 - a ** 2) * v) @ W.T                        # J v with J = W diag(phi')
        z = a @ W.T + b
        nv = np.linalg.norm(v, axis=1)
        logD += 2 * np.log(nv)
        v /= nv[:, None]
    return logD


if __name__ == "__main__":
    nets = int(sys.argv[1]) if len(sys.argv) > 1 else 160
    rng = np.random.default_rng(1)
    t0 = time.time()
    res = np.array([one_network(rng) for _ in range(nets)])  # nets x P
    q = fixed_point("tanh", SW2, SB2)
    th = a1_b1("tanh", q, SW2)
    per_net = res.mean(1)
    out = dict(N=N, L=L, sw2=SW2, sb2=SB2, q=q, nets=nets, pairs=P, a1=th["a1"], b1=th["b1"],
               a1_L_over_N=th["a1"] * L / N, b1_L_over_N=th["b1"] * L / N,
               mean_lnD=float(res.mean()), se_mean=float(per_net.std(ddof=1) / np.sqrt(nets)),
               var_between=float(per_net.var(ddof=1)), var_within=float(res.var(1, ddof=1).mean()),
               var_total=float(res.var(ddof=1)), frac_pairs_shrunk=float((res < 0).mean()))
    json.dump(out, open(os.path.join(HERE, "..", "data", "pairs_one_network.json"), "w"), indent=1)
    for k, v in out.items():
        print(k, v)
    print(f"{time.time() - t0:.0f}s")
