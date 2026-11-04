"""Figure 4: information about input similarity along the edge of chaos (data/info/*.npy)."""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eocfw import style  # noqa: E402
from eocfw.info import comp_mf  # noqa: E402
from eocfw.meanfield import kappa  # noqa: E402
from eocfw.style import BLUE, DEEP_BLUE, DEEP_RED, GREY, INK, LIGHT_BLUE, ORANGE, RED, dots, panel, ylab  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "data", "info")
FIG = os.path.join(HERE, "..", "figures")
style.apply(8.5)


def load(L, N):
    return np.load(os.path.join(D, f"info_L{L}_N{N}.npy")), np.load(os.path.join(D, f"ref_L{L}_N{N}.npy"))


def main():
    fig, ax = plt.subplots(1, 3, figsize=(7.06, 1.88))
    fig.subplots_adjust(left=0.06, right=0.995, top=0.88, bottom=0.18, wspace=0.30)
    cols = {64: LIGHT_BLUE, 128: BLUE, 256: DEEP_BLUE, 512: INK}
    L = 40
    for N, col in cols.items():
        a, r = load(L, N)
        ax[0].errorbar(a[:, 0], a[:, 1], yerr=a[:, 2], fmt="o", ms=2.8, color=col, capsize=1.2, lw=0.6,
                       label=f"$N={N}$")
        ax[0].plot([5e-5, 3.0], [r[0, 0]] * 2, ":", color=col, lw=0.8)
        ax[0].plot([5e-5, 3.0], [r[1, 0]] * 2, "--", color=col, lw=0.6, alpha=0.6)
    for N, col in cols.items():
        _, r = load(L, N)
        ax[0].text(4.0, r[1, 0], f"$N={N}$", color=col, fontsize=6.5, va="center", ha="left")
    ax[0].set_xscale("log")
    ax[0].set_xlim(5e-5, 40)
    ax[0].set_xticks([1e-4, 1e-2, 1e0])
    ax[0].set_xlabel(r"$q^*$")
    ylab(ax[0], r"$I$ (bits)")
    # (b) compression collapse
    qfine = np.logspace(-4, 0.3, 40)
    for Lx, col in [(10, ORANGE), (40, RED), (160, DEEP_RED)]:
        for N in [512]:
            a, _ = load(Lx, N)
            kl = np.array([kappa("tanh", x) for x in a[:, 0]]) * Lx
            dots(ax[1], kl, a[:, 6], color=col, size=10, label=f"$L={Lx}$")
        cm = np.array([comp_mf(x, Lx) for x in qfine])
        ax[1].plot(np.array([kappa("tanh", x) for x in qfine]) * Lx, cm, "-", color=col, lw=0.9)
    ax[1].axhline(1.0, color=GREY, ls=":", lw=0.7)
    ax[1].set_xscale("log")
    ax[1].set_xlim(1e-4, 1e3)
    ax[1].set_xlabel(r"$\kappa(q^*)\,L$")
    ylab(ax[1], r"$\gamma$ (nats)")
    ax[1].legend(loc="upper left", fontsize=6)
    # (c) plateau vs N/L
    for Lx, mk in [(10, "o"), (40, "s"), (160, "D")]:
        for N in [64, 128, 256, 512]:
            try:
                a, r = load(Lx, N)
            except FileNotFoundError:
                continue
            plateau = np.mean(a[a[:, 0] < 0.02, 1])
            ax[2].scatter([N / Lx], [plateau], marker=mk, s=16, color=BLUE, edgecolors="k", linewidths=0.4,
                          label=f"$L={Lx}$" if N == 128 else None, zorder=3)
            ax[2].scatter([N / Lx], [r[0, 0]], marker=mk, s=16, color=RED, edgecolors="k", linewidths=0.4, zorder=3)
    x = np.logspace(-0.2, 1.8, 50)
    ax[2].plot(x, 0.5 * np.log2(1 + 1.2 * x), "--", color=INK, lw=0.8, label=r"$\frac{1}{2}\log_2(1+1.2N/L)$")
    ax[2].scatter([], [], marker="o", s=16, color=RED, edgecolors="k", linewidths=0.4, label="ReLU")
    ax[2].set_xscale("log")
    ax[2].set_xlabel(r"$N/L$")
    ylab(ax[2], r"$I$ (bits)")
    ax[2].legend(loc="upper left", fontsize=6)
    for a, l in zip(ax, "abc"):
        panel(a, l, x=-0.26, y=1.02)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig4.{ext}"))
    print("saved fig4")


if __name__ == "__main__":
    main()
