"""
Two inputs at finite separation through a finite-width tanh network at the mean-field edge of chaos of
Tamai et al. (sigma_w = 1.39558, sigma_b = 0.3, kappa = 0.2335): decay of rho_l = 1 - C_l for N = 100, 400,
mean and typical (exp of the mean log), compared with the mean-field map and with the finite-width prediction
of a drift a_1/N of ln rho at small rho.  Output: data/two_input_chain.npz
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eocfw.finite_width import a1_b1  # noqa: E402
from eocfw.meanfield import dist_map, eoc_at_bias, kappa  # noqa: E402
from eocfw.simulate import two_input_chain  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
sb2 = 0.09
sw2, q = eoc_at_bias("tanh", sb2)
th = a1_b1("tanh", q, sw2)
print(f"sigma_w={math.sqrt(sw2):.5f} q*={q:.4f} kappa={kappa('tanh', q, sw2):.4f} a1={th['a1']:.3f} b1={th['b1']:.3f}")
out = dict(sw2=sw2, q=q, a1=th["a1"], b1=th["b1"], kappa=kappa("tanh", q, sw2))
for N, L, B in [(100, 1500, 4000), (400, 3000, 2000)]:
    rho = two_input_chain("tanh", N, sw2, sb2, c0=0.5, L=L, B=B, q1=q, seed=N)
    out[f"N{N}_mean"] = rho.mean(1)
    out[f"N{N}_median"] = np.median(rho, 1)
    out[f"N{N}_typical"] = np.exp(np.log(np.clip(rho, 1e-300, None)).mean(1))
    out[f"N{N}_std_log"] = np.log(np.clip(rho, 1e-300, None)).std(1)
    print(f"N={N}: rho mean/typical at l={L}: {out[f'N{N}_mean'][-1]:.3e} / {out[f'N{N}_typical'][-1]:.3e}", flush=True)
# mean field
eps = [0.5]
for _ in range(3000):
    eps.append(dist_map("tanh", eps[-1], q, sw2))
out["meanfield"] = np.array(eps)
np.savez(os.path.join(D, "two_input_chain.npz"), **out)
print("saved")
