"""Figure 3: typical versus mean propagation for N = 16 and 64 over 300 layers.
(a) two inputs at finite separation (data/two_input_chain_v2.npz); (b) typical signal of one input
(data/signal_walk_v2.npz).  Data from scripts/run_fig3_data.py."""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eocfw import style  # noqa: E402
from eocfw.style import BLUE, INK, RED, dots, panel, ylab  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "data")
FIG = os.path.join(HERE, "..", "figures")
style.apply(8.5)
NS = [16, 64]
COLS = {16: RED, 64: BLUE}
TICKS = [1, 10, 100]


def main():
    fig, ax = plt.subplots(1, 2, figsize=(3.42, 1.9))
    fig.subplots_adjust(left=0.13, right=0.995, top=0.86, bottom=0.19, wspace=0.5)
    # (a) two inputs
    tc = np.load(os.path.join(D, "two_input_chain_v2.npz"))
    mf = tc["meanfield"]
    ell = np.arange(mf.size)
    for N in NS:
        m, t = tc[f"N{N}_mean"], tc[f"N{N}_typical"]
        l = np.arange(m.size)
        ok = tc[f"N{N}_mean_se"] / m < 0.1          # the sample mean is dominated by a few networks beyond this
        cut = np.argmin(ok[1:]) + 1 if not ok[1:].all() else m.size
        ax[0].plot(l[1:cut], m[1:cut], "--", color=COLS[N], lw=0.9, label=f"mean, $N={N}$")
        ax[0].plot(l[1:], t[1:], "-", color=COLS[N], lw=0.9, label=f"typical, $N={N}$")
    ax[0].plot(ell[1:], mf[1:], "-", color=INK, lw=1.0, label="mean field")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlim(1, 400)
    ax[0].set_ylim(1e-9, 1)
    ax[0].set_xticks(TICKS)
    ax[0].set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1])
    ax[0].set_xlabel(r"$\ell$")
    ylab(ax[0], r"$1-C_\ell$")
    ax[0].legend(loc="lower left", fontsize=5.8, handlelength=1.6)
    # (b) typical signal
    sw = np.load(os.path.join(D, "signal_walk_v2.npz"))
    L = sw["tanh_meanfield"].size
    l = np.arange(1, L + 1)
    sel = 2 ** np.arange(0, int(np.log2(L)) + 1) - 1          # markers at l = 1, 2, 4, ...: evenly spaced on a log axis
    for N in NS:
        a, b = sw[f"tanh_{N}_std"], sw[f"tanh_{N}_fin"]
        ax[1].plot(l, a[0], "-", color=COLS[N], lw=0.9, label=rf"$\sigma_w^2=1$, $N={N}$")
        ax[1].plot(l, b[0], "--", color=COLS[N], lw=0.9, label=rf"$\sigma_c^2(N)$, $N={N}$")
        dots(ax[1], l[sel], a[0, sel], color=COLS[N], size=10)
        ax[1].scatter(l[sel], b[0, sel], s=10, facecolors="white", edgecolors=COLS[N], linewidths=0.7, zorder=3)
    ax[1].plot(l, sw["tanh_meanfield"], "-", color=INK, lw=1.0, label="mean field")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlim(1, 400)
    ax[1].set_ylim(1e-9, 2)
    ax[1].set_xticks(TICKS)
    ax[1].set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1])
    ax[1].set_xlabel(r"$\ell$")
    ylab(ax[1], r"$\exp\mathbb{E}[\ln Q_\ell]$")
    ax[1].legend(loc="lower left", fontsize=5.8, handlelength=1.6)
    panel(ax[0], "a", x=-0.42, y=1.1)
    panel(ax[1], "b", x=-0.42, y=1.1)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig3.{ext}"))
    print("saved fig3")


if __name__ == "__main__":
    main()
