"""Figure 2: drift and variance of ln D, the finite-width edge of chaos, and the survival of distance avalanches,
for N = 16, 64, 256.  Needs data/distance_walk_v2.json, data/eoc_shift_v2.json, data/survival_v2.npz
(scripts/run_fig2_data.py)."""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eocfw import style  # noqa: E402
from eocfw.finite_width import a1_b1  # noqa: E402
from eocfw.style import BLUE, DEEP_BLUE, DEEP_RED, GREY, INK, LIGHT_BLUE, ORANGE, RED, dots, panel, ylab  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "data")
FIG = os.path.join(HERE, "..", "figures")
style.apply(8.5)
NS = [16, 64, 256]
COLS = {16: RED, 64: ORANGE, 256: BLUE, 1024: DEEP_BLUE}


def main():
    dw = json.load(open(os.path.join(D, "distance_walk_v2.json")))
    sh = json.load(open(os.path.join(D, "eoc_shift_v2.json")))
    sv = np.load(os.path.join(D, "survival_v2.npz"))
    fig, ax = plt.subplots(1, 4, figsize=(7.06, 1.88))
    fig.subplots_adjust(left=0.06, right=0.995, top=0.86, bottom=0.18, wspace=0.40)
    qs = np.logspace(-2, 0.35, 70)
    th = [a1_b1("tanh", x) for x in qs]
    rows = dw["gaussian"]["tanh"]
    # breakdown of the expansion for N = 16: N (1 - F') < 3
    q_break = brentq(lambda q: 16 * (1 - a1_b1("tanh", q)["Fp"]) - 3.0, 1e-3, 2.0)
    # (a) drift
    for a in ax[:2]:
        a.axvspan(1e-2, q_break, color="#EFEFEF", zorder=0, lw=0)
    ax[0].plot(qs, [t["a1"] for t in th], "-", color=INK, lw=1.2, label="$a_1$, theory")
    for N in NS:
        x = [r["q"] for r in rows]
        y = [[s["N_mu"] for s in r["sims"] if s["N"] == N][0] for r in rows]
        dots(ax[0], x, y, color=COLS[N], label=f"$N={N}$", size=16)
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"$q^*$")
    ylab(ax[0], r"$N\mu_d$")
    ax[0].set_ylim(-3.0, 0.3)
    ax[0].set_xlim(1e-2, 2.3)
    ax[0].axhline(0, color=GREY, lw=0.5)
    ax[0].legend(loc="lower left", fontsize=6.3)
    # (b) variance
    ax[1].plot(qs, [t["b1"] for t in th], "-", color=INK, lw=1.2, label="$b_1$")
    for N in NS:
        x = [r["q"] for r in rows]
        y = [[s["N_var_long"] for s in r["sims"] if s["N"] == N][0] for r in rows]
        dots(ax[1], x, y, color=COLS[N], label=f"$N={N}$", size=16)
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"$q^*$")
    ylab(ax[1], r"$N\sigma_d^2$")
    ax[1].set_ylim(0, 6.5)
    ax[1].set_xlim(1e-2, 2.3)
    ax[1].legend(loc="lower right", fontsize=6.3)
    # (c) shift of the edge of chaos
    NN = np.logspace(1.1, 3.1, 50)
    for sb2, orth, col, lab in [(0.05, False, RED, r"$\sigma_b^2=0.05$"), (0.20, False, ORANGE, r"$\sigma_b^2=0.20$"),
                                (0.05, True, BLUE, "orthogonal, $0.05$")]:
        r = [x for x in sh if abs(x["sb2"] - sb2) < 1e-9 and x["orthogonal"] == orth]
        if not r:
            continue
        c = r[0]["shift_pred"] * r[0]["N"]
        ax[2].plot(NN, c / NN, "-", color=col, lw=1.0)
        # statistical error of the measured zero: error of the drift (b_1/N over L B steps) over the slope of mu(sw2)
        err = []
        for x in r:
            B = {16: 4000, 32: 3000, 64: 2000, 128: 1500, 256: 800, 512: 400, 1024: 200}[x["N"]]
            if orth:
                B = min(B, 600)
            b1 = a1_b1("tanh", x["q"], x["sw2_mf"], orthogonal=orth)["b1"]
            slope = abs(np.polyfit(x["grid"], x["mus"], 1)[0])
            err.append(np.sqrt(b1 / x["N"] / (1500 * B)) / slope)
        ax[2].errorbar([x["N"] for x in r], [x["shift_meas"] for x in r], yerr=err, fmt="none", ecolor=col, elinewidth=0.6,
                       capsize=1.5, zorder=2)
        dots(ax[2], [x["N"] for x in r], [x["shift_meas"] for x in r], color=col, label=lab, size=16)
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xlabel(r"$N$")
    ylab(ax[2], r"$\Delta\sigma_w^2$")
    ax[2].set_xticks([10, 100, 1000])
    ax[2].legend(loc="lower left", fontsize=5.8, labelspacing=0.15, handletextpad=0.3, borderaxespad=0.15, borderpad=0.2)
    # (d) survival: lines through every n, markers at n = 1, 2, 4, ... (evenly spaced on the log axis)
    n = np.arange(1, 4000)
    sel = 2 ** np.arange(0, 12) - 1
    for N in NS:
        T, B = sv[f"T_N{N}"], int(sv[f"B_N{N}"])
        surv = np.array([(T >= k).sum() for k in n]) / B
        ok = surv > 0
        ax[3].plot(n[ok], surv[ok], "-", color=COLS[N], lw=0.9)
        dots(ax[3], n[sel], surv[sel], color=COLS[N], label=f"$N={N}$", size=13)
    ax[3].plot(n, 0.5 * n ** -0.5, "--", color=GREY, lw=0.8, label=r"$n^{-1/2}$")
    ax[3].set_xscale("log")
    ax[3].set_yscale("log")
    ax[3].set_xlabel(r"$n$")
    ylab(ax[3], r"$P(T\geq n)$")
    ax[3].set_ylim(5e-3, 1.2)
    ax[3].legend(loc="lower left", fontsize=6.3)
    for a, l in zip(ax, "abcd"):
        panel(a, l, x=-0.3, y=1.02)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig2.{ext}"))
    print("saved fig2; breakdown q* for N=16:", round(q_break, 4))


if __name__ == "__main__":
    main()
