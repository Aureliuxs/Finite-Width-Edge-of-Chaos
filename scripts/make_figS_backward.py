"""SM figure and table for Sec. S5: the backward gradient walk against the forward tangent walk, through the same
explicit weight matrices.  Data: data/backward/summary_*.npz, made by summarise_backward.py from the runs of
backward_walk.py.  Outputs figures/figS_backward.pdf and latex/tableS_backward.tex."""
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "crinit"))
from eocfw import style  # noqa: E402
from eocfw.style import BLUE, DEEP_BLUE, DEEP_RED, GREY, RED  # noqa: E402

DATA = os.path.join(HERE, "..", "data", "backward")
FIG = os.path.join(HERE, "..", "figures")
TEX = os.path.join(HERE, "..", "..", "latex")
BURN, BLOCK = 100, 100
A1, B1 = -1.6783517585718526, 3.9936818560014156      # tanh, sigma_b^2 = 0.05, mean-field edge of chaos


def se(x):
    return x.std(ddof=1) / np.sqrt(len(x))


def walk_stats(d):
    """Width times the stationary drift and the long-run variance, forward and backward."""
    N = int(d["N"])
    out = {}
    for key in ("f", "b"):
        per, bs = d["mean_" + key], d["blocks_" + key]
        out[key] = (N * per.mean(), N * se(per), N * bs.var(ddof=1) / BLOCK)
    return out


def load(name):
    return dict(np.load(os.path.join(DATA, name)))


def main():
    runs = {}
    for f in glob.glob(os.path.join(DATA, "summary_N*_L400_sw1.761.npz")):
        d = load(os.path.basename(f))
        runs[int(d["N"])] = d
    Ns = sorted(runs)
    stats = {N: walk_stats(runs[N]) for N in Ns}

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    # (a) distribution of the two sums at N = 64
    ax = axes[0]
    d = runs[64]
    X, Y = d["X"], d["Y"]
    bins = np.linspace(-30, 8, 45)
    ax.hist(X, bins, histtype="step", color=BLUE, lw=1.2, label=r"forward, $X_L$")
    ax.hist(Y, bins, histtype="step", color=RED, lw=1.1, ls="--", label=r"backward, $Y_L$")
    ax.axvline(A1 * float(d["L"]) / float(d["N"]), color=GREY, lw=0.7)
    ax.set_yticks([])
    ax.set_xlabel(r"$X_L$, $Y_L$")
    style.ylab(ax, "networks")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    ax.legend(loc="upper left", fontsize=5.8)
    style.panel(ax, "a")
    # (b) N times the drift against N
    ax = axes[1]
    ax.axhline(A1, color=GREY, lw=0.8)
    ax.text(Ns[0] / 1.15, A1 + 0.012, r"$a_1$", fontsize=7, color=GREY, ha="left", va="bottom")
    for N in Ns:
        fm, fe, _ = stats[N]["f"]
        bm, be, _ = stats[N]["b"]
        ax.errorbar(N / 1.05, fm, fe, fmt="o", color=BLUE, mec=DEEP_BLUE, mew=0.4, ms=4, capsize=0)
        ax.errorbar(N * 1.05, bm, be, fmt="o", color=RED, mec=DEEP_RED, mew=0.4, ms=4, capsize=0)
    ax.scatter([], [], color=BLUE, edgecolors=DEEP_BLUE, s=14, label="forward")
    ax.scatter([], [], color=RED, edgecolors=DEEP_RED, s=14, label="backward")
    ax.set_xscale("log")
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xticks([], minor=True)
    ax.set_ylim(-1.9, -1.45)
    ax.set_xlabel(r"$N$")
    style.ylab(ax, r"$N\mu$")
    ax.legend(loc="upper left", fontsize=6)
    style.panel(ax, "b")
    # (c) gradient norm k layers below the output, N = 64, mean-field and finite-width lines
    ax = axes[2]
    k = np.arange(0, 97)
    for f, col, lab in (("summary_N64_L400_sw1.761.npz", BLUE, r"mean field, $\sigma_w^2=1.761$"),
                        ("summary_N64_L400_sw1.884.npz", RED, r"finite width, $\sigma_w^2=1.884$")):
        Yk = load(f)["Yk"].astype(float)
        ax.fill_between(k, np.exp(np.quantile(Yk, 0.25, 0)), np.exp(np.quantile(Yk, 0.75, 0)), color=col, alpha=0.15, lw=0)
        ax.plot(k, np.exp(np.median(Yk, 0)), color=col, lw=1.2, label=lab)
        ax.plot(k, np.exp(Yk).mean(0), color=col, lw=0.9, ls="--")
    ax.plot(k, np.exp(A1 * k / 64), color=GREY, lw=0.7, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel(r"$k$")
    style.ylab(ax, r"$\|\mathbf{g}^{(L-k)}\|^2/\|\mathbf{g}^{(L)}\|^2$")
    ax.legend(loc="lower left", fontsize=5.8)
    style.panel(ax, "c")
    fig.tight_layout(w_pad=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"figS_backward.{ext}"))

    # table
    lines = [r"\begin{tabular}{rcccccc}", r"\toprule",
             r"$N$ & networks & $N\mu$, forward & $N\mu$, backward & $N\sigma^2$, forward & $N\sigma^2$, backward \\",
             r"\midrule"]
    for N in Ns:
        fm, fe, fv = stats[N]["f"]
        bm, be, bv = stats[N]["b"]
        lines.append(f"{N} & {int(runs[N]['M'])} & ${fm:.3f}\\pm{fe:.3f}$ & ${bm:.3f}\\pm{be:.3f}$ & ${fv:.2f}$ & ${bv:.2f}$ \\\\")
    lines += [r"\midrule", f"theory & & ${A1:.3f}$ & ${A1:.3f}$ & ${B1:.2f}$ & ${B1:.2f}$ \\\\", r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TEX, "tableS_backward.tex"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    # numbers quoted in the text
    for f in ("summary_N64_L400_sw1.761.npz", "summary_N64_L400_sw1.884.npz"):
        dd = load(f)
        for kk in (32, 96):
            Yk = dd["Yk"][:, kk].astype(float)
            print(f, kk, "median", round(float(np.exp(np.median(Yk))), 3), "mean", round(float(np.exp(Yk).mean()), 2),
                  "quartiles", np.round(np.exp(np.quantile(Yk, [0.25, 0.75])), 3))
        print(f, "N mu forward, backward", walk_stats(dd))

if __name__ == "__main__":
    main()
