"""
Data of Fig. 2 at the widths N = 16, 64, 256 (and 1024 for the shift).

(a, b) drift and long-run variance of ln D on the mean-field edge of chaos (tanh, Gaussian weights) against
       a_1 and b_1, at five steady signals;                       -> data/distance_walk_v2.json
(c)    the measured zero of the drift (finite-width edge of chaos) against the predicted shift, tanh at
       sigma_b^2 = 0.05 and 0.20 (N = 16 ... 1024), 0.09 (three widths), and orthogonal weights at 0.05
       (N = 16 ... 1024);                                                  -> data/eoc_shift_v2.json
(d)    survival of distance avalanches at the measured finite-width edge of chaos.  -> data/survival_v2.npz

Usage: python run_fig2_data.py [walk|shift|survival]   (default: all three in this order)
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eocfw.finite_width import a1_b1, eoc_shift  # noqa: E402
from eocfw.meanfield import eoc_point  # noqa: E402
from eocfw.simulate import distance_walk, survival  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
QS = [0.02, 0.05, 0.15, 0.4, 1.0]
NS = [16, 64, 256]


def walk():
    path = os.path.join(D, "distance_walk_v2.json")
    res = json.load(open(path)) if os.path.exists(path) else {"gaussian": {"tanh": []}}
    done = {(r["q"], s["N"]) for r in res["gaussian"]["tanh"] for s in r["sims"]}
    rows = {r["q"]: r for r in res["gaussian"]["tanh"]}
    for q in QS:
        sw2, sb2 = eoc_point("tanh", q)
        th = a1_b1("tanh", q)
        row = rows.setdefault(q, dict(q=q, sw2=sw2, sb2=sb2, theory={k: float(v) for k, v in th.items()}, sims=[]))
        for N in NS:
            if (q, N) in done:
                continue
            t = time.time()
            B = {16: 6000, 64: 3000, 256: 800}[N]
            r = distance_walk("tanh", N, sw2, sb2, B=B, L=2000, burn=600, block=200, seed=N + 3)
            row["sims"].append(dict(N=N, N_mu=float(N * r["mu"]), N_var_long=float(N * r["var_long"]),
                                    N_var_step=float(N * r["var_step"]), N_mu_mean=float(N * r["mu_mean"])))
            print(f"tanh q*={q} N={N}: N mu={N * r['mu']:.3f} (a1={th['a1']:.3f})  N var={N * r['var_long']:.3f} "
                  f"(b1={th['b1']:.3f})  N(1-F')={N * (1 - th['Fp']):.2f}  [{time.time() - t:.0f}s]", flush=True)
            res["gaussian"]["tanh"] = [rows[k] for k in QS if k in rows]
            json.dump(res, open(path, "w"), indent=1)


def measured_zero(N, sb2, orth, sw2, dpred, B, seed):
    """Zero of the drift mu(sigma_w^2) from a scan of several values around the predicted point."""
    span = max(3 * abs(dpred), 0.02)
    grid = sw2 + span * np.array([0.0, 0.5, 1.0, 1.5, 2.0]) if N >= 64 else sw2 + span * np.array([0.0, 0.33, 0.67, 1.0, 1.33, 1.67])
    mus = [distance_walk("tanh", N, s, sb2, B=B, L=1500, burn=500, block=150, seed=seed, orthogonal=orth)["mu"] for s in grid]
    mus = np.array(mus)
    # linear interpolation between the bracketing points (mu decreases through zero from below as sw2 grows)
    i = np.where(np.diff(np.sign(mus)) != 0)[0]
    if i.size == 0:
        c = np.polyfit(grid, mus, 1)
        return -c[1] / c[0], grid.tolist(), mus.tolist()
    i = i[0]
    z = grid[i] - mus[i] * (grid[i + 1] - grid[i]) / (mus[i + 1] - mus[i])
    return float(z), grid.tolist(), mus.tolist()


def shift():
    path = os.path.join(D, "eoc_shift_v2.json")
    out = json.load(open(path)) if os.path.exists(path) else []
    done = {(r["sb2"], r["N"], r["orthogonal"]) for r in out}
    for sb2, Ns, orth in [(0.05, [16, 32, 64, 128, 256, 512, 1024], False), (0.09, [16, 64, 256], False),
                          (0.05, [16, 32, 64, 128, 256, 512, 1024], True),
                          (0.20, [16, 32, 64, 128, 256, 512, 1024], False)]:
        for N in Ns:
            if (sb2, N, orth) in done:
                continue
            t = time.time()
            sw2, q, dpred = eoc_shift("tanh", sb2, N, orthogonal=orth)
            B = {16: 4000, 32: 3000, 64: 2000, 128: 1500, 256: 800, 512: 400, 1024: 200}[N]
            if orth:
                B = min(B, 600)
            meas, grid, mus = measured_zero(N, sb2, orth, sw2, dpred, B, seed=N + 11)
            out.append(dict(sb2=sb2, N=N, orthogonal=orth, sw2_mf=sw2, q=q, shift_pred=dpred, shift_meas=meas - sw2,
                            grid=grid, mus=mus))
            print(f"sb2={sb2} orth={orth} N={N}: MF eoc {sw2:.5f} (q*={q:.4f}); shift predicted {dpred:.5f}, "
                  f"measured {meas - sw2:.5f}   ({time.time() - t:.0f}s)", flush=True)
            json.dump(out, open(path, "w"), indent=1)


def surv():
    sh = json.load(open(os.path.join(D, "eoc_shift_v2.json")))
    out = {}
    for N in NS:
        r = [x for x in sh if x["N"] == N and abs(x["sb2"] - 0.05) < 1e-9 and not x["orthogonal"]][0]
        sw2 = r["sw2_mf"] + r["shift_meas"]
        t = time.time()
        B = {16: 60000, 64: 40000, 256: 30000}[N]
        T, A, Btot = survival("tanh", N, sw2, 0.05, B=B, L=4000, seed=N + 5)
        out[f"T_N{N}"], out[f"A_N{N}"], out[f"B_N{N}"] = T, A, Btot
        print(f"survival N={N} at sw2={sw2:.5f}: " +
              str([(k, round((T >= k).sum() / Btot, 4)) for k in [1, 4, 16, 64, 256, 1024, 4000]]) +
              f"  ({time.time() - t:.0f}s)", flush=True)
        np.savez(os.path.join(D, "survival_v2.npz"), **out)


if __name__ == "__main__":
    jobs = sys.argv[1:] or ["walk", "shift", "survival"]
    for j in jobs:
        {"walk": walk, "shift": shift, "survival": surv}[j]()
    print("done", jobs)
