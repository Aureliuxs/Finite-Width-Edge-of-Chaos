"""Figure 1: the model.  (a) Network schematic with two inputs.  (b) Phase diagram of the tanh network:
mean-field edge of chaos and its finite-width shift.  Output: figures/fig1.pdf"""
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eocfw import style  # noqa: E402
from eocfw.finite_width import eoc_shift  # noqa: E402
from eocfw.meanfield import chi1, eoc_at_bias, fixed_point  # noqa: E402
from eocfw.style import BLUE, DEEP_BLUE, DEEP_RED, GREY, INK, ORANGE, RED, panel, ylab  # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
style.apply(8.5)
SCHEM_X, SCHEM_Y = (-0.62, 14.75), (-1.0, 5.3)       # data extent of the schematic (it is drawn at equal aspect)


def brace(ax, x, y0, y1, width=0.22, color=GREY, lw=0.7):
    """Vertical curly brace spanning y0..y1 with its back at x and its tip at x + 2 width (pointing right)."""
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    ym, w = 0.5 * (y0 + y1), width
    verts = [(x, y0), (x + w, y0), (x + w, y0 + w), (x + w, ym - w), (x + w, ym), (x + 2 * w, ym),
             (x + w, ym), (x + w, ym + w), (x + w, y1 - w), (x + w, y1), (x, y1)]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3]
    ax.add_patch(PathPatch(Path(verts, codes), fill=False, edgecolor=color, lw=lw, zorder=2))


def bubble(ax, xc, yc, w, h, pts, labels, dlabel, color_pts):
    """A small 'state space' ellipse with two points (the two inputs) joined by their distance."""
    from matplotlib.patches import Ellipse
    ax.add_patch(Ellipse((xc, yc), w, h, facecolor="#F7F7F7", edgecolor=GREY, lw=0.6, zorder=2))
    (x1, y1), (x2, y2) = pts
    ax.plot([x1, x2], [y1, y2], "-", color=RED, lw=1.0, zorder=3)
    for (x, y), lab, c, dx in zip(pts, labels, color_pts, (-0.13, 0.13)):
        ax.add_patch(Circle((x, y), 0.11, facecolor=c, edgecolor=INK, lw=0.4, zorder=4))
        ax.text(x + dx, y + 0.02, lab, ha="right" if dx < 0 else "left", va="center", fontsize=7.2, zorder=5)
    xm, ym = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    ax.text(xm, ym + 0.2, dlabel, ha="center", va="bottom", fontsize=7.6, color=DEEP_RED, zorder=5)


def schematic(ax):
    ax.set_axis_off()
    ax.set_xlim(*SCHEM_X)
    ax.set_ylim(*SCHEM_Y)
    ax.set_aspect("equal")
    cols_x = [0.0, 1.9, 3.8, 7.3, 9.3]            # input, layer 1, layer 2, ..., layer L, output
    nodes_y = {0: [0.5, 1.5, 2.5, 3.5], 1: [0.0, 1.0, 2.0, 3.0, 4.0], 2: [0.0, 1.0, 2.0, 3.0, 4.0],
               3: [0.0, 1.0, 2.0, 3.0, 4.0], 4: [1.0, 2.0, 3.0]}
    r = 0.21
    for k in range(len(cols_x) - 1):
        x0, x1 = cols_x[k], cols_x[k + 1]
        if k == 2:
            continue
        for ya in nodes_y[k]:
            for yb in nodes_y[k + 1]:
                ax.plot([x0 + r, x1 - r], [ya, yb], color="#D5D5D5", lw=0.45, zorder=1)
    for yb in nodes_y[2]:
        ax.plot([cols_x[3] - 0.8, cols_x[3] - r], [yb, yb], color="#D5D5D5", lw=0.45, zorder=1)
    ax.text(cols_x[3] - 1.55, 2.0, r"$\cdots$", ha="center", va="center", fontsize=12, color=INK)
    for k, x in enumerate(cols_x):
        for y in nodes_y[k]:
            fc = "white" if k in (0, 4) else "#EAF1F9"
            ec = INK if k in (0, 4) else DEEP_BLUE
            ax.add_patch(Circle((x, y), r, facecolor=fc, edgecolor=ec, lw=0.7, zorder=3))
    ax.text(cols_x[0], -0.72, "input", ha="center", va="center", fontsize=7.5)
    ax.text(cols_x[1], -0.72, r"$\ell=1$", ha="center", va="center", fontsize=7.5)
    ax.text(cols_x[2], -0.72, r"$\ell=2$", ha="center", va="center", fontsize=7.5)
    ax.text(cols_x[3], -0.72, r"$\ell=L$", ha="center", va="center", fontsize=7.5)
    ax.text(cols_x[4], -0.72, "output", ha="center", va="center", fontsize=7.5)
    brace(ax, cols_x[2] + 0.4, -0.22, 4.22, width=0.2)
    ax.text(cols_x[2] + 0.95, 2.0, r"$N$", ha="left", va="center", fontsize=8.5, color=INK)
    # the two inputs x and x' and their representations z, z' (two colours, the layer-L column labelled)
    ax.text(cols_x[0], 4.45, r"$\mathbf{x},\ \mathbf{x}'$", ha="center", va="bottom", fontsize=8, color=INK)
    ax.text(cols_x[3], 4.95, r"$\mathbf{z}^{(L)},\ \mathbf{z}'^{(L)}$", ha="center", va="bottom", fontsize=8, color=INK)
    # the four definitions, evenly spaced; the last sits just above the line of the "output" label
    ys = np.linspace(4.3, -0.3, 4)
    ax.text(10.7, ys[0], r"$W_{ij}\sim\mathcal{N}(0,\sigma_w^2/N)$", ha="left", va="center", fontsize=7.2)
    ax.text(10.7, ys[1], r"$b_i\sim\mathcal{N}(0,\sigma_b^2)$", ha="left", va="center", fontsize=7.2)
    ax.text(10.7, ys[2], r"$q_\ell=\frac{1}{N}\|\mathbf{z}^{(\ell)}\|^2$", ha="left", va="center", fontsize=7.5)
    ax.text(10.7, ys[3], r"$d_\ell^2=\frac{1}{N}\|\mathbf{z}^{(\ell)}-\mathbf{z}'^{(\ell)}\|^2$", ha="left", va="center",
            fontsize=7.5)


def slow_line(sb2_list):
    """sigma_w^2 at which the slope F' of the variance map at its fixed point is maximal, for each sigma_b^2."""
    from scipy.optimize import minimize_scalar
    from eocfw.gauss import moments
    out = []
    for s in sb2_list:
        if s == 0.0:
            out.append(1.0)
            continue

        def negFp(sw2):
            q = fixed_point("tanh", sw2, s)
            return -sw2 * moments("tanh", q)["dV"]

        out.append(minimize_scalar(negFp, bounds=(0.5, 6.0), method="bounded", options=dict(xatol=1e-6)).x)
    return np.array(out)


def finite_width_line(N, q_grid=None):
    """The finite-width edge of chaos traced by its steady signal q, as Crinit places a network on it:
    sigma_w^2 = exp(-a_1(q)/N) / P(q), sigma_b^2 = q - sigma_w^2 V(q); kept where sigma_b^2 >= 0.
    It meets the zero-bias axis at the zero-bias point (sigma_w^2 = 1.31 for N = 64)."""
    from scipy.optimize import brentq
    from eocfw.finite_width import a1_b1
    from eocfw.gauss import moments

    def point(q):
        m = moments("tanh", q)
        sw2 = math.exp(-a1_b1("tanh", q)["a1"] / N) / m["P"]
        return sw2, q - sw2 * m["V"]
    q0 = brentq(lambda q: point(q)[1], 1e-4, 2.0)          # zero-bias point
    qs = np.concatenate([[q0], np.geomspace(q0 * 1.0001, 6.0, 300)])
    pts = np.array([point(q) for q in qs])
    return pts[:, 0], np.clip(pts[:, 1], 0.0, None)


def phase_diagram(ax, N=64):
    sb2 = np.concatenate([[0.0], np.logspace(-4, np.log10(0.55), 60)])
    mf = np.array([1.0 if s == 0.0 else eoc_at_bias("tanh", s)[0] for s in sb2])
    fn_w, fn_b = finite_width_line(N)
    ax.fill_betweenx(sb2, mf, 3.2, color="#FBEBE9", zorder=0)
    ax.fill_betweenx(sb2, 0.6, mf, color="#EEF4FA", zorder=0)
    ax.plot(mf, sb2, "-", color=INK, lw=1.2, label="mean-field edge of chaos")
    ax.plot(fn_w, fn_b, "--", color=RED, lw=1.2, label=f"finite-width edge of chaos,\n$N={N}$")
    ax.plot([1.0], [0.0], "o", ms=4.5, color=INK, zorder=5)
    ax.plot([fn_w[0]], [0.0], "o", ms=4.0, color=RED, mec=DEEP_RED, mew=0.5, zorder=5, clip_on=False)   # zero-bias point
    ax.set_xlim(0.6, 3.2)
    ax.set_ylim(0.0, 0.52)
    ax.set_xlabel(r"$\sigma_w^2$")
    ylab(ax, r"$\sigma_b^2$")
    ax.text(1.1, 0.21, "ordered\n$d_\\ell\\to0$", ha="center", va="center", fontsize=8, color=DEEP_BLUE)
    ax.text(2.75, 0.1, "chaotic\n$d_\\ell\\to d^*$", ha="center", va="center", fontsize=8, color=DEEP_RED)
    ax.text(0.96, 0.012, r"$(1,0)$", fontsize=7.5, ha="right", va="bottom")
    # transparent legend in the ordered corner, clear of both lines
    ax.legend(loc="upper left", fontsize=6.4, frameon=False, handlelength=1.4, borderaxespad=0.1, labelspacing=0.3)


if __name__ == "__main__":
    # explicit layout in inches: the schematic fills the left part at equal aspect, the phase diagram the right part
    W = 7.06                                        # REVTeX text width
    wa = 4.45                                       # width of the schematic
    ha = wa * (SCHEM_Y[1] - SCHEM_Y[0]) / (SCHEM_X[1] - SCHEM_X[0])
    H = ha + 0.2                                    # room for the panel label above
    fig = plt.figure(figsize=(W, H))
    ax0 = fig.add_axes([0.0, 0.0, wa / W, ha / H])
    bx0, bw, by0, bh = 4.95, 2.08, 0.36, H - 0.36 - 0.2
    ax1 = fig.add_axes([bx0 / W, by0 / H, bw / W, bh / H])
    schematic(ax0)
    phase_diagram(ax1)
    fig.text(0.0, 1.0, "(a)", fontsize=9, fontweight="bold", va="top", ha="left")
    panel(ax1, "b", x=-0.25, y=1.03)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig1.{ext}"))
    print("saved fig1", round(W, 2), round(H, 2))
