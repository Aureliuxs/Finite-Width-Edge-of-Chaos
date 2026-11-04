"""
Finite-width edge of chaos: predicted shift Delta sigma_w^2 = -a_1/(N d ln chi_1/d sigma_w^2) against the measured
zero of the drift of ln D (exact chain), for tanh at sigma_b^2 = 0.05 and 0.09 (the point of Tamai et al.),
Gaussian weights; and for orthogonal weights at sigma_b^2 = 0.05.  Also the avalanche survival at the finite-N
edge of chaos.  Output: data/eoc_shift.json, data/survival_N400.npz
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eocfw.finite_width import eoc_shift  # noqa: E402
from eocfw.simulate import distance_walk, survival  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
out = []
for sb2, Ns, orth in [(0.05, [100, 200, 400, 800], False), (0.09, [100, 200, 400], False), (0.05, [100, 200, 400], True)]:
    for N in Ns:
        t = time.time()
        sw2, q, dpred = eoc_shift("tanh", sb2, N, orthogonal=orth)
        grid = [sw2, sw2 + 3 * dpred]
        mus = []
        for s in grid:
            r = distance_walk("tanh", N, s, sb2, B=(1200 if N <= 200 else 500) if not orth else 500, L=1500,
                              burn=500, block=150, seed=N, orthogonal=orth)
            mus.append(r["mu"])
        meas = grid[0] - mus[0] * (grid[1] - grid[0]) / (mus[1] - mus[0])
        out.append(dict(sb2=sb2, N=N, orthogonal=orth, sw2_mf=sw2, q=q, shift_pred=dpred, shift_meas=meas - sw2,
                        mu0=mus[0], mu1=mus[1]))
        print(f"sb2={sb2} orth={orth} N={N}: MF eoc {sw2:.5f} (q*={q:.4f}); shift predicted {dpred:.5f}, measured "
              f"{meas - sw2:.5f}   ({time.time() - t:.0f}s)", flush=True)
        json.dump(out, open(os.path.join(D, "eoc_shift.json"), "w"), indent=1)

N = 400
sw2, q, dpred = eoc_shift("tanh", 0.05, N)
T, A, B = survival("tanh", N, sw2 + dpred, 0.05, B=30000, L=500)
np.savez(os.path.join(D, "survival_N400.npz"), T=T, A=A, B=B)
print("survival", [(k, round((T >= k).sum() / B, 4)) for k in [1, 2, 4, 8, 16, 32, 64, 128, 256]])
