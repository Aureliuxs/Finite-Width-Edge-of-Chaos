"""First-order against all-orders Crinit for tanh (Gaussian weights) at depths 32 and 96 and widths 16-256.
The all-orders value is the zero of the drift in the exact finite-width chain (crinit order='exact').
Output: data/crinit_orders.json (Table S2 of the SM)."""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from eocfw.mnist import _crinit  # noqa: E402  (finds the crinit package next to this folder)

crinit = _crinit()
out = []
for L in (32, 96):
    for N in (16, 32, 64, 128, 256):
        t = time.time()
        r1 = crinit.crinit("tanh", depth=L, width=N)
        re = crinit.crinit("tanh", depth=L, width=N, order="exact")
        se = float(re.notes[-1].split("+-")[1].split("(")[0])
        out.append(dict(L=L, N=N, q1=r1.q_star, sw2_1=r1.sigma_w2, sb2_1=r1.sigma_b2, mf=r1.sigma_w2_meanfield,
                        qe=re.q_star, sw2_e=re.sigma_w2, sb2_e=re.sigma_b2, se=se))
        print(f"L={L} N={N}: first {r1.sigma_w2:.4f} exact {re.sigma_w2:.4f} +- {se:.4f} ({time.time() - t:.0f}s)", flush=True)
json.dump(out, open(os.path.join(HERE, "..", "data", "crinit_orders.json"), "w"), indent=1)
print("saved")
