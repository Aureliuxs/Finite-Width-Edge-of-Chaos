"""
Typical signal exp(E ln q_l) of one input through 128 layers: standard initialisation against the
typical-preserving sigma_c^2(N), for tanh and ReLU with Gaussian weights, and for tanh with orthogonal weights.
Output: data/signal_walk.npz
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eocfw.finite_width import sigma_c2_leaky, sigma_c2_smooth  # noqa: E402
from eocfw.gauss import gavg, get  # noqa: E402
from eocfw.simulate import signal_walk, typical_recursion  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
L = 128
out = {}
for act in ["tanh", "relu"]:
    for N in [32, 64]:
        for tag, sw2 in [("std", 1.0 if act == "tanh" else 2.0),
                         ("fin", sigma_c2_smooth(N) if act == "tanh" else sigma_c2_leaky(N, 0.0))]:
            typ, se, mean = signal_walk(act, N, sw2, L, B=2000, seed=N)
            out[f"{act}_{N}_{tag}"] = np.stack([typ, se, mean, typical_recursion(act, sw2, N, L)])
            print(f"{act} N={N} {tag} sw2={sw2:.5f}: typical q_L {typ[-1]:.4e}, mean {mean[-1]:.4e}", flush=True)
# orthogonal weights: tanh, sigma_w^2 = 1 and sigma_c^2(N); linear reference
for N in [32, 64]:
    for tag, sw2 in [("std", 1.0), ("fin", sigma_c2_smooth(N))]:
        typ, se, mean = signal_walk("tanh", N, sw2, L, B=1000, seed=N + 7, orthogonal=True)
        out[f"orth_tanh_{N}_{tag}"] = np.stack([typ, se, mean, typical_recursion("tanh", sw2, N, L)])
        print(f"orthogonal tanh N={N} {tag}: typical q_L {typ[-1]:.4e}", flush=True)
# mean-field curve for tanh (sigma_w^2 = 1)
phi = get("tanh")[0]
mf = [1.0]
for _ in range(L - 1):
    mf.append(gavg(lambda u: phi(u) ** 2, mf[-1]))
out["tanh_meanfield"] = np.array(mf)
np.savez(os.path.join(D, "signal_walk.npz"), **out)
print("saved")
