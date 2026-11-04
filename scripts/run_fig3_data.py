"""
Data of Fig. 3 at the widths N = 16 and 64, over 300 layers.

(a) two inputs at finite separation through a tanh network at the mean-field edge of chaos of Tamai et al.
    (sigma_b^2 = 0.09): mean and typical 1 - C_l over many networks, with the mean-field map;  -> data/two_input_chain_v2.npz
(b) typical signal of one input: standard sigma_w^2 = 1 against the typical-preserving sigma_c^2(N);  -> data/signal_walk_v2.npz
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eocfw.finite_width import sigma_c2_smooth  # noqa: E402
from eocfw.gauss import gavg, get  # noqa: E402
from eocfw.meanfield import dist_map, eoc_at_bias  # noqa: E402
from eocfw.simulate import signal_walk, two_input_chain, typical_recursion  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
L = 300
NS = [16, 64]

# (a)
sb2 = 0.09
sw2, q = eoc_at_bias("tanh", sb2)
out = dict(sw2=sw2, q=q)
for N in NS:
    t = time.time()
    B = {16: 40000, 64: 20000}[N]
    rho = two_input_chain("tanh", N, sw2, sb2, c0=0.5, L=L, B=B, q1=q, seed=N)
    lr = np.log(np.clip(rho, 1e-300, None))
    out[f"N{N}_mean"] = rho.mean(1)
    out[f"N{N}_mean_se"] = rho.std(1) / np.sqrt(B)
    out[f"N{N}_typical"] = np.exp(lr.mean(1))
    out[f"N{N}_typical_se"] = lr.std(1) / np.sqrt(B)
    print(f"(a) N={N}: mean/typical at l={L}: {out[f'N{N}_mean'][-1]:.3e} / {out[f'N{N}_typical'][-1]:.3e} ({time.time() - t:.0f}s)", flush=True)
eps = [0.5]
for _ in range(L):
    eps.append(dist_map("tanh", eps[-1], q, sw2))
out["meanfield"] = np.array(eps)
np.savez(os.path.join(D, "two_input_chain_v2.npz"), **out)

# (b)
out = {}
for N in NS:
    for tag, s in [("std", 1.0), ("fin", sigma_c2_smooth(N))]:
        t = time.time()
        typ, se, mean = signal_walk("tanh", N, s, L, B={16: 8000, 64: 3000}[N], seed=N)
        out[f"tanh_{N}_{tag}"] = np.stack([typ, se, mean, typical_recursion("tanh", s, N, L)])
        print(f"(b) tanh N={N} {tag} sw2={s:.5f}: typical q_L {typ[-1]:.3e}, mean {mean[-1]:.3e} ({time.time() - t:.0f}s)", flush=True)
phi = get("tanh")[0]
mf = [1.0]
for _ in range(L - 1):
    mf.append(gavg(lambda u: phi(u) ** 2, mf[-1]))
out["tanh_meanfield"] = np.array(mf)
np.savez(os.path.join(D, "signal_walk_v2.npz"), **out)
print("saved")
