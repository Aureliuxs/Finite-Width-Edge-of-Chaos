"""Figure 5: training from the finite-width edge of chaos (large-statistics comparison, data/mnist/big*.pkl).
(a) test accuracy against training step for the tanh rules at L=96, (b) the same for the ReLU family,
(c) the hundred final accuracies of the standard rules and of Crinit.  Also figS_gamma (the scan of the
compression budget, data/mnist/gamma.pkl) and figS_curves (the learning curves at L=32)."""
import glob
import math
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eocfw import style  # noqa: E402
from eocfw.style import BLUE, DEEP_BLUE, DEEP_RED, GREY, INK, LIGHT_BLUE, ORANGE, RED, dots, panel, ylab  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "data", "mnist")
FIG = os.path.join(HERE, "..", "figures")
style.apply(8.5)
GAMMAS = [0.25, 0.5, 1, 2.3026, 3.5]

TANH = [("tanh_standard", "LeCun", GREY, "-"), ("tanh_eoc:1", "mean-field edge of chaos", ORANGE, "--"),
        ("tanh_crinit:1", "Crinit", RED, "-")]
TANH_ALL = [("tanh_standard", "LeCun", GREY, "-"), ("tanh_chaotic", "He", "#B0B0B0", "--"),
            ("tanh_eoc:1", "mean-field edge of chaos", ORANGE, "--"), ("tanh_crinit:1", "Crinit", RED, "-"),
            ("orth_tanh_standard", "orthogonal", LIGHT_BLUE, "-"), ("orth_tanh_crinit:1", "Crinit, orthogonal", DEEP_BLUE, "-")]
RELU = [("relu_he", "ReLU, He", GREY, "-"), ("leaky_he", r"leaky ReLU, He", "#B0B0B0", "--"),
        ("leaky_crinit:1", "Crinit, leaky ReLU", RED, "-")]


def load_big():
    R = {}
    for f in sorted(glob.glob(os.path.join(D, "big*.pkl"))):
        R.update(pickle.load(open(f, "rb")))
    return R


def curves(ax, R, rules, L, N=64):
    for name, lab, col, ls in rules:
        r = R.get((name, L, N))
        if r is None or r["curve_step"].size == 0:
            continue
        acc = np.where(r["diverged"][None, :], np.nan, r["curve_acc"])
        m = np.nanmean(acc, 1)
        ax.plot(r["curve_step"], m, ls, color=col, lw=1.0, label=lab)
    ax.set_xlabel("training step")
    ylab(ax, "test accuracy")


def main():
    R = load_big()
    fig, ax = plt.subplots(1, 3, figsize=(7.06, 1.80))
    fig.subplots_adjust(left=0.06, right=0.995, top=0.865, bottom=0.215, wspace=0.30)
    for a, rules, title, anchor in [(ax[0], TANH, r"$\mathrm{tanh}$", (1.0, 0.02)), (ax[1], RELU, "ReLU family", (1.0, 0.535))]:
        curves(a, R, rules, 96)
        a.set_ylim(0.0, 1.0)
        a.set_xlim(0, 950)
        a.legend(loc="lower right", fontsize=6.0, handlelength=1.8, bbox_to_anchor=anchor, labelspacing=0.3, borderaxespad=0.0)
        a.text(0.03, 0.97, title + ", $L=96$", transform=a.transAxes, ha="left", va="top", fontsize=7)
    # (c) strip plot of the final accuracies of the hundred networks at L = 96
    pairs = [("tanh_standard", "LeCun", GREY), ("tanh_crinit:1", "Crinit", RED),
             ("relu_he", "He", GREY), ("leaky_crinit:1", "Crinit", RED)]
    rng = np.random.default_rng(0)
    xs = [0.0, 1.0, 2.6, 3.6]
    for x0, (name, lab, col) in zip(xs, pairs):
        r = R.get((name, 96, 64))
        if r is None:
            continue
        a = r["test_acc"]
        a = a[~np.isnan(a)]
        ax[2].scatter(x0 + rng.uniform(-0.2, 0.2, a.size), a, s=5, color=col, alpha=0.6, lw=0, zorder=2)
        ax[2].plot([x0 - 0.32, x0 + 0.32], [a.mean()] * 2, "-", color=INK, lw=1.3, zorder=3)
    ax[2].set_xticks(xs)
    ax[2].set_xticklabels([p[1] for p in pairs], fontsize=7)
    for xc, lab in [(0.5, r"$\tanh$"), (3.1, "ReLU family")]:
        ax[2].text(xc, -0.24, lab, transform=ax[2].get_xaxis_transform(), ha="center", va="top", fontsize=7)
    ax[2].set_xlim(-0.6, 4.2)
    ax[2].set_ylim(0.0, 1.0)
    ylab(ax[2], "test accuracy")
    ax[2].text(0.97, 0.03, "$L=96$", transform=ax[2].transAxes, ha="right", va="bottom", fontsize=7)
    for a, l in zip(ax, "abc"):
        panel(a, l, x=-0.27, y=1.04)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig5.{ext}"))
    print("saved fig5")
    # SM: learning curves at L = 32
    fig, ax = plt.subplots(1, 2, figsize=(5.0, 2.05))
    fig.subplots_adjust(left=0.1, right=0.99, top=0.86, bottom=0.2, wspace=0.4)
    curves(ax[0], R, TANH_ALL, 32)
    curves(ax[1], R, RELU, 32)
    for a in ax:
        a.set_ylim(0.6, 1.0)
        a.set_xlim(0, 950)
        a.legend(loc="lower right", fontsize=6.0, handlelength=1.8)
    for a, l in zip(ax, "ab"):
        panel(a, l, x=-0.27, y=1.02)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"figS_curves.{ext}"))
    print("saved figS_curves")


def acc(r):
    return r["test_acc"][-1] if (r["test_acc"] and not r["diverged"]) else np.nan


def scan(R, base, L, N):
    g, m, se = [], [], []
    for gg in GAMMAS:
        v = np.array([acc(r) for k, r in R.items() if k[0] == f"{base}:{gg}" and k[1] == L and k[2] == N], float)
        v = v[~np.isnan(v)]
        if v.size == 0:
            continue
        g.append(gg)
        m.append(v.mean())
        se.append(v.std() / math.sqrt(v.size) if v.size > 1 else 0.0)
    return np.array(g), np.array(m), np.array(se)


def gamma_figure():
    """The scan of the compression budget (formerly Fig. 5(a,b)), now in the SM."""
    R = pickle.load(open(os.path.join(D, "gamma.pkl"), "rb"))
    fig, ax = plt.subplots(1, 2, figsize=(5.0, 2.05))
    fig.subplots_adjust(left=0.1, right=0.99, top=0.86, bottom=0.2, wspace=0.4)
    for base, col, lab in [("tanh_sweet", BLUE, r"$\tanh$, edge of chaos"), ("leaky", RED, "shaped leaky ReLU")]:
        g, m, se = scan(R, base, 96, 64)
        ax[0].errorbar(g, m, yerr=se, fmt="o-", ms=4, color=col, mec="k", mew=0.4, capsize=1.5, lw=0.9, label=lab)
    ax[0].axvline(1.0, color=GREY, ls=":", lw=0.7)
    ax[0].set_ylim(0.3, 1.0)
    ax[0].legend(loc="lower left", fontsize=6.3)
    grid = [(16, 64), (32, 64), (64, 64), (96, 64)]
    cols = [INK, DEEP_BLUE, BLUE, LIGHT_BLUE]
    labs = ["$N/L=4$", "$N/L=2$", "$N/L=1$", "$N/L=2/3$"]
    for (L, N), col, lab in zip(grid, cols, labs):
        g, m, se = scan(R, "tanh_sweet", L, N)
        if g.size == 0:
            continue
        ax[1].errorbar(g, m, yerr=se, fmt="o-", ms=3.5, color=col, mec="k", mew=0.3, capsize=1.2, lw=0.9, label=lab)
    ax[1].set_ylim(0.7, 0.97)
    ax[1].legend(loc="lower left", fontsize=6.3)
    for a in ax:
        a.set_xscale("log")
        a.set_xticks([0.25, 0.5, 1, 2, 4])
        a.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
        a.minorticks_off()
        a.set_xlabel(r"$\gamma$ (nats)")
        ylab(a, "test accuracy")
    for a, l in zip(ax, "ab"):
        panel(a, l, x=-0.27, y=1.02)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"figS_gamma.{ext}"))
    print("saved figS_gamma")


if __name__ == "__main__":
    if glob.glob(os.path.join(D, "big*.pkl")):
        main()
    if os.path.exists(os.path.join(D, "gamma.pkl")):
        gamma_figure()
