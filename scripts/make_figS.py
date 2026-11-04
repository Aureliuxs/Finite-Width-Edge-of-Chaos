"""Figures of the Supplemental Material: figS_erf, figS_spread, figS_mnist, figS_finiteN.
(figS_classes is produced by legacy/make_theory_figures.py, see README.)"""
import json
import math
import os
import pickle
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import digamma

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eocfw import style  # noqa: E402
from eocfw.finite_width import a1_b1  # noqa: E402
from eocfw.info import logspread_theory  # noqa: E402
from eocfw.meanfield import eoc_point, sweet_spot  # noqa: E402
from eocfw.style import BLUE, CMAP_BLUE, DEEP_BLUE, DEEP_RED, GREY, INK, LIGHT_BLUE, ORANGE, RED, dots, panel, ylab  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "data")
FIG = os.path.join(HERE, "..", "figures")
style.apply(8.5)


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{stem}.{ext}"))
    plt.close(fig)
    print("saved", stem)


def fig_erf():
    dw = json.load(open(os.path.join(D, "distance_walk.json")))
    fig, ax = plt.subplots(1, 2, figsize=(5.2, 2.0))
    fig.subplots_adjust(left=0.1, right=0.99, top=0.86, bottom=0.2, wspace=0.4)
    qs = np.logspace(-2, 0.35, 60)
    th = [a1_b1("erf", x) for x in qs]
    ax[0].plot(qs, [t["a1"] for t in th], "-", color=INK, lw=1.2, label="$a_1$")
    ax[0].plot(qs, [t["a_jensen"] for t in th], ":", color=INK, lw=0.9, label="averaging term")
    ax[0].plot(qs, [t["a_r"] + t["a_mean"] + t["a_curv"] for t in th], "--", color=GREY, lw=0.9, label="surroundings")
    ax[1].plot(qs, [t["b1"] for t in th], "-", color=INK, lw=1.2, label="$b_1$")
    ax[1].plot(qs, [t["b_white"] for t in th], ":", color=INK, lw=0.9, label="$3P_4/P^2-1$")
    rows = dw["gaussian"]["erf"]
    for N, col in [(200, BLUE), (800, DEEP_RED)]:
        x = [r["q"] for r in rows]
        dots(ax[0], x, [[s["N_mu"] for s in r["sims"] if s["N"] == N][0] for r in rows], color=col, label=f"$N={N}$", size=16)
        dots(ax[1], x, [[s["N_var_long"] for s in r["sims"] if s["N"] == N][0] for r in rows], color=col, label=f"$N={N}$", size=16)
    for a in ax:
        a.set_xscale("log")
        a.set_xlabel(r"$q^*$")
    ax[0].legend(loc="lower left", fontsize=6.3)
    ax[1].legend(loc="upper left", fontsize=6.0, ncol=2, columnspacing=0.8)
    ylab(ax[0], r"$N\mu_d$")
    ylab(ax[1], r"$N\sigma_d^2$")
    ax[0].set_ylim(-3, 0.6)
    ax[0].axhline(0, color=GREY, lw=0.5)
    ax[1].set_ylim(0, 7.5)
    panel(ax[0], "a", x=-0.25, y=1.02)
    panel(ax[1], "b", x=-0.25, y=1.02)
    save(fig, "figS_erf")


def fig_spread():
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.0))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.2, wspace=0.42)
    cols = {64: LIGHT_BLUE, 128: BLUE, 256: DEEP_BLUE, 512: INK}
    L = 40
    qfine = np.logspace(-4, 0.3, 40)
    for N, col in cols.items():
        a = np.load(os.path.join(D, "info", f"info_L{L}_N{N}.npy"))
        dots(ax[0], a[:, 0], a[:, 7], color=col, size=10, label=f"$N={N}$")
        ax[0].plot(qfine, [logspread_theory(x, L, N) for x in qfine], "-", color=col, lw=0.9)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"$q^*$")
    ylab(ax[0], r"std $\ln\hat q_L$")
    ax[0].legend(loc="lower left", fontsize=6.3)
    for Lx, mk in [(10, "o"), (40, "s"), (160, "D")]:
        for N in [64, 128, 256, 512]:
            f = os.path.join(D, "info", f"info_L{Lx}_N{N}.npy")
            if not os.path.exists(f):
                continue
            a = np.load(f)
            r = np.load(os.path.join(D, "info", f"ref_L{Lx}_N{N}.npy"))
            pl = np.mean(a[a[:, 0] < 0.02, 1])
            ax[1].scatter([N], [pl], marker=mk, s=16, color=BLUE, edgecolors="k", linewidths=0.4, zorder=3,
                          label=f"$L={Lx}$" if N == 64 else None)
            ax[1].scatter([N], [r[0, 0]], marker=mk, s=16, color=RED, edgecolors="k", linewidths=0.4, zorder=3)
    ax[1].scatter([], [], marker="o", s=16, color=RED, edgecolors="k", linewidths=0.4, label="ReLU")
    ax[1].set_xscale("log")
    ax[1].set_xticks([64, 128, 256, 512])
    ax[1].set_xticklabels(["64", "128", "256", "512"])
    ax[1].minorticks_off()
    ax[1].set_xlabel(r"$N$")
    ylab(ax[1], r"$I$ (bits)")
    ax[1].legend(loc="upper left", fontsize=6.3)
    # (c) a second smooth activation: erf along its own edge of chaos, L = 40
    for N, col in [(64, LIGHT_BLUE), (256, DEEP_BLUE)]:
        f = os.path.join(D, "info", f"info_erf_L40_N{N}.npy")
        if not os.path.exists(f):
            continue
        a = np.load(f)
        t = np.load(os.path.join(D, "info", f"info_L40_N{N}.npy"))
        ax[2].errorbar(a[:, 0], a[:, 1], yerr=a[:, 2], fmt="none", ecolor=col, elinewidth=0.5, capsize=1, zorder=2)
        dots(ax[2], a[:, 0], a[:, 1], color=col, size=10, label=f"erf, $N={N}$")
        ax[2].plot(t[:, 0], t[:, 1], "-", color=col, lw=0.9, label=rf"$\tanh$, $N={N}$")
    ax[2].set_xscale("log")
    ax[2].set_xlabel(r"$q^*$")
    ax[2].set_ylim(0.2, 1.75)
    ylab(ax[2], r"$I$ (bits)")
    ax[2].legend(loc="lower left", fontsize=6.0, ncol=2, columnspacing=0.8, handlelength=1.5)
    for a, l in zip(ax, "abc"):
        panel(a, l, x=-0.25, y=1.02)
    save(fig, "figS_spread")


def fig_mnist_sm():
    R = pickle.load(open(os.path.join(D, "mnist", "main.pkl"), "rb"))
    # the loss curves of panel (a) come from the three-seed runs of main.pkl (the large runs record test accuracy only)
    fig = plt.figure(figsize=(7.0, 2.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.62, 0.9], left=0.07, right=0.98, top=0.86, bottom=0.18, wspace=0.35)
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2])]
    axl = fig.add_subplot(gs[0, 1])
    axl.set_axis_off()
    L = 96
    pal = {"tanh_standard": (INK, "standard"), "tanh_finiteN": (GREY, r"$\sigma_c^2(N)$"),
           "tanh_sweet:1": (ORANGE, r"edge of chaos, $\gamma=1$"), "tanh_sweetN:1": (RED, r"finite-width edge, $\gamma=1$"),
           "orth_tanh_standard": (LIGHT_BLUE, "orthogonal, standard"), "orth_tanh_sweet:1": (DEEP_BLUE, r"orthogonal, edge, $\gamma=1$"),
           "leaky:1": (BLUE, r"shaped leaky ReLU, $\gamma=1$"), "relu_he": (DEEP_RED, "ReLU, He"), "linear": ("#7A9A7A", "linear")}
    for n, (c, lab) in pal.items():
        runs = [r for k, r in R.items() if k[1] == L and k[0] == n and not r["diverged"]]
        if not runs:
            continue
        m = min(len(r["loss"]) for r in runs)
        step = np.array(runs[0]["step"][:m])
        loss = np.mean([r["loss"][:m] for r in runs], axis=0)
        k = 5
        sm = np.convolve(loss, np.ones(k) / k, mode="valid")
        ax[0].plot(step[k - 1:], sm, "-", color=c, lw=1.0, label=lab)
    ax[0].set_yscale("log")
    ax[0].set_yticks([0.1, 0.2, 0.5, 1, 2])
    ax[0].set_yticklabels(["0.1", "0.2", "0.5", "1", "2"])
    ax[0].minorticks_off()
    ax[0].set_xlabel("step")
    ylab(ax[0], "training loss")
    h, lab = ax[0].get_legend_handles_labels()
    axl.legend(h, lab, fontsize=6.2, loc="center left", bbox_to_anchor=(-0.25, 0.5))
    # phase diagram
    txt = open(os.path.join(D, "mnist", "phase_L48_N64.txt")).read()
    sw2 = np.array([0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.7])
    rows = re.findall(r"sigma_b\^2 = ([\d.e-]+) \[([^\]]+)\]", txt)
    sb2 = np.array([float(r[0]) for r in rows])
    A = np.array([[float(t) for t in r[1].split()] for r in rows])
    im = ax[1].pcolormesh(sw2, np.arange(len(sb2)), A, cmap=CMAP_BLUE, shading="nearest", vmin=0.1, vmax=1)
    qq = np.logspace(-4, 0.5, 200)
    pts = np.array([eoc_point("tanh", q) for q in qq])
    ok = (pts[:, 1] <= sb2.max()) & (pts[:, 0] <= sw2.max())
    ax[1].plot(pts[ok, 0], np.interp(pts[ok, 1], sb2, np.arange(len(sb2))), "--", color=DEEP_RED, lw=1.2, label="edge of chaos")
    q, s_w, s_b = sweet_spot("tanh", 48, 1.0)
    ax[1].plot([s_w], [np.interp(s_b, sb2, np.arange(len(sb2)))], "*", ms=11, color=ORANGE, mec="k", mew=0.4,
               label=r"$\gamma=1$, $L=48$")
    ax[1].set_yticks(np.arange(len(sb2)))
    ax[1].set_yticklabels([f"{v:g}" for v in sb2])
    ax[1].set_xlabel(r"$\sigma_w^2$")
    ylab(ax[1], r"$\sigma_b^2$")
    cb = fig.colorbar(im, ax=ax[1], fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_visible(False)
    ax[1].legend(fontsize=6, loc="upper left")
    panel(ax[0], "a", x=-0.2, y=1.02)
    panel(ax[1], "b", x=-0.2, y=1.02)
    save(fig, "figS_mnist")


def fig_finiteN():
    R = pickle.load(open(os.path.join(D, "mnist", "finiteN.pkl"), "rb"))

    def acc(r):
        return r["test_acc"][-1] if (r["test_acc"] and not r["diverged"]) else np.nan

    fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.2))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.2, wspace=0.35)
    xt, xl, x0 = [], [], 0
    for N in [64, 32]:
        for a, b, nm in [("tanh_standard", "tanh_finiteN", "tanh"), ("relu_he", "relu_finiteN", "ReLU")]:
            seeds = sorted({k[3] for k in R if k[0] == a and k[2] == N})
            A = np.array([acc(R[(a, 96, N, s)]) for s in seeds])
            B = np.array([acc(R[(b, 96, N, s)]) for s in seeds])
            for u, v in zip(A, B):
                ax[0].plot([x0, x0 + 1], [u, v], "-", color="#BBBBBB", lw=0.7, zorder=1)
            dots(ax[0], np.full(A.size, x0), A, color=RED, size=16, label="standard" if x0 == 0 else None)
            dots(ax[0], np.full(B.size, x0 + 1), B, color=BLUE, size=16, label=r"$\sigma_c^2(N)$" if x0 == 0 else None)
            xt.append(x0 + 0.5)
            xl.append(f"{nm}\n$N={N}$")
            x0 += 3
    ax[0].set_xticks(xt)
    ax[0].set_xticklabels(xl, fontsize=7)
    ax[0].set_ylim(0, 1)
    ylab(ax[0], "test accuracy")
    ax[0].legend(loc="upper right", fontsize=6.3)
    N = 32
    ks = sorted({float(k[0].split(":")[1]) for k in R if k[0].startswith("tanh_k")})
    m, se = [], []
    for kk in ks:
        v = np.array([acc(r) for k, r in R.items() if k[0] == f"tanh_k:{kk:g}" and k[2] == N])
        ax[1].scatter(np.full(v.size, 1 + kk / N), v, s=8, color=BLUE, alpha=0.35, edgecolors="none")
        m.append(np.nanmean(v))
        se.append(np.nanstd(v) / math.sqrt(np.sum(~np.isnan(v))))
    ax[1].errorbar(1 + np.array(ks) / N, m, yerr=se, fmt="o", ms=4, color=DEEP_BLUE, mec="k", mew=0.4, capsize=1.5, lw=0.7,
                   label="mean of 6 seeds")
    ax[1].axvline((N / 2) * math.exp(-digamma(N / 2)), color=ORANGE, ls="--", lw=1.0, label=r"$\sigma_c^2(N)$")
    ax[1].set_xscale("log")
    ax[1].set_xticks([1, 1.1, 1.25, 1.5])
    ax[1].set_xticklabels(["1", "1.1", "1.25", "1.5"])
    ax[1].minorticks_off()
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel(r"$\sigma_w^2$")
    ylab(ax[1], "test accuracy")
    ax[1].legend(loc="lower right", fontsize=6.3)
    panel(ax[0], "a", x=-0.2, y=1.02)
    panel(ax[1], "b", x=-0.2, y=1.02)
    save(fig, "figS_finiteN")


if __name__ == "__main__":
    which = sys.argv[1:] or ["erf", "spread", "mnist", "finiteN"]
    if "erf" in which:
        fig_erf()
    if "spread" in which:
        fig_spread()
    if "mnist" in which:
        fig_mnist_sm()
    if "finiteN" in which:
        fig_finiteN()


def fig_signal():
    sw = np.load(os.path.join(D, "signal_walk.npz"))
    L = sw["tanh_meanfield"].size
    ell = np.arange(1, L + 1)
    sel = np.unique(np.round(np.logspace(0, np.log10(L), 26)).astype(int)) - 1
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.0))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.2, wspace=0.42)
    for N, (cs, cf) in [(32, (DEEP_RED, DEEP_BLUE)), (64, (RED, LIGHT_BLUE))]:
        a, b = sw[f"tanh_{N}_std"], sw[f"tanh_{N}_fin"]
        dots(ax[0], ell[sel], a[0, sel], color=cs, size=10, label=rf"$\sigma_w^2=1$, $N={N}$")
        dots(ax[0], ell[sel], b[0, sel], color=cf, size=10, label=rf"$\sigma_c^2(N)$, $N={N}$")
        ax[0].plot(ell, a[3], "-", color=cs, lw=0.7)
        ax[0].plot(ell, b[3], "-", color=cf, lw=0.7)
    ax[0].plot(ell, sw["tanh_meanfield"], "--", color=INK, lw=0.9, label="mean field")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    for N, (cs, cf) in [(32, (DEEP_RED, DEEP_BLUE)), (64, (ORANGE, LIGHT_BLUE))]:
        a, b = sw[f"relu_{N}_std"], sw[f"relu_{N}_fin"]
        dots(ax[1], ell[sel], a[0, sel], color=cs, size=10, label=rf"He, $N={N}$")
        dots(ax[1], ell[sel], b[0, sel], color=cf, size=10, label=rf"$2e^{{5/(2N)}}$, $N={N}$")
        ax[1].plot(ell, np.exp(-2.5 * (ell - 1) / N), "-", color=cs, lw=0.7)
    ax[1].axhline(1.0, color=INK, ls="--", lw=0.9, label="mean field")
    ax[1].set_yscale("log")
    for N, (cs, cf) in [(32, (DEEP_RED, DEEP_BLUE)), (64, (RED, LIGHT_BLUE))]:
        a, b = sw[f"orth_tanh_{N}_std"], sw[f"orth_tanh_{N}_fin"]
        dots(ax[2], ell[sel], a[0, sel], color=cs, size=10, label=rf"$\sigma_w^2=1$, $N={N}$")
        dots(ax[2], ell[sel], b[0, sel], color=cf, size=10, label=rf"$\sigma_c^2(N)$, $N={N}$")
    ax[2].plot(ell, sw["tanh_meanfield"], "--", color=INK, lw=0.9, label="mean field")
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    for a, l in zip(ax, "abc"):
        a.set_xlabel(r"$\ell$")
        ylab(a, r"$e^{\langle\ln q_\ell\rangle}$")
        a.legend(loc="lower left", fontsize=5.6)
        panel(a, l, x=-0.27, y=1.02)
    save(fig, "figS_signal")


if __name__ == "__main__" and "signal" in sys.argv[1:]:
    fig_signal()
