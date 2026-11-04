"""House style of the figures: no grid, variable-only axis labels, horizontal y-label above the axis,
circular markers, a small validated palette of blues, reds and oranges."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BLUE, RED, ORANGE, DEEP_BLUE, DEEP_RED, LIGHT_BLUE, DEEP_ORANGE = \
    "#4F86C6", "#E0645C", "#E8A33C", "#2D5F95", "#B8433D", "#9CC3E6", "#C77A2A"
PALETTE = [BLUE, RED, ORANGE, DEEP_BLUE, DEEP_RED, LIGHT_BLUE, DEEP_ORANGE]
INK = "#2B2B2B"
GREY = "#8A8A8A"
CMAP = LinearSegmentedColormap.from_list("warm", ["#FBF7F2", "#F8D9B0", "#F2A46B", "#E0645C", "#8E2F2B"])
CMAP_BLUE = LinearSegmentedColormap.from_list("cool", ["#F6F9FC", "#CFE0F3", "#8FB8E3", "#4F86C6", "#1F4470"])


def apply(font_size=8.5):
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "font.size": font_size, "axes.titlesize": font_size, "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 0.5, "ytick.labelsize": font_size - 0.5,
        "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.linewidth": 0.6, "legend.frameon": False, "legend.fontsize": font_size - 1.5,
        "legend.handlelength": 1.6, "legend.handletextpad": 0.5, "legend.borderaxespad": 0.2,
        "axes.prop_cycle": mpl.cycler(color=PALETTE), "lines.markersize": 4, "lines.linewidth": 1.1,
        "mathtext.fontset": "cm", "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"],
        "pdf.fonttype": 42,
    })


def ylab(ax, s, dx=-0.02, dy=1.02):
    ax.set_ylabel(s, rotation=0, ha="left", va="bottom")
    ax.yaxis.set_label_coords(dx, dy)


def panel(ax, letter, x=-0.18, y=1.06):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def dots(ax, x, y, color=BLUE, label=None, size=14, zorder=3, **kw):
    r, g, b = mpl.colors.to_rgb(color)
    return ax.scatter(x, y, s=size, color=color, edgecolors=(0.75 * r, 0.75 * g, 0.75 * b), linewidths=0.4,
                      label=label, zorder=zorder, **kw)


apply()
