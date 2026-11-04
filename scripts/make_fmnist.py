"""
Fashion-MNIST with tuned learning rate and gain (SM Sec. S8): tables and figure from data/fmnist/tuned_L*.pkl.

  ../latex/tableS_fmnist_summary.tex the three comparisons: standard rules at their usual variance, standard rules
                                     with tuned gain, Crinit, Crinit with tuned compression gamma
  ../latex/tableS_fmnist.tex         every rule: test accuracy after 2 and 10 epochs, steps to 80% training accuracy
  ../latex/tableS_fmnist_choice.tex  the gain, slope or gamma and learning rate chosen for every rule and budget
  figures/figS_fmnist.pdf            (a) validation accuracy against sigma_w^2: zero-bias gain scan and Crinit's
                                     gamma line; (b) validation accuracy against gamma for the three Crinit rules;
                                     (c, d) test accuracy against the epoch at L = 96
Also prints the headline comparisons.
"""
import math
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eocfw import style  # noqa: E402
from eocfw.style import BLUE, DEEP_RED, GREY, INK, ORANGE, RED, panel, ylab  # noqa: E402
from run_fmnist_tuned import CHANCE, E_LONG, E_SHORT, configs  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "data", "fmnist")
FIG = os.path.join(HERE, "..", "figures")
TEX = os.path.join(HERE, "..", "..", "latex")
DEPTHS = (32, 96)
TARGET = 0.8
ZB_FW = 1.3108          # zero-bias point of the finite-width edge of chaos of tanh at N = 64 (crinit)
GROUPS = [("tanh", r"$\tanh$, Gaussian weights"), ("orth", r"$\tanh$, orthogonal weights"),
          ("relu", "ReLU family, Gaussian weights")]
CRINIT = {"tanh": "Crinit", "orth": "Crinit, orthogonal", "relu": "Crinit, leaky ReLU"}
TUNED_CRINIT = {"tanh": "Crinit, tuned $\\gamma$", "orth": "Crinit, orthogonal, tuned $\\gamma$",
                "relu": "Crinit, leaky ReLU, tuned $\\gamma$"}
STD1 = {"tanh": "Crinit, standard first layer", "orth": "Crinit, orthogonal, standard first layer",
        "relu": "Crinit, leaky ReLU, standard first layer"}
EXTRA = {"tanh": ["edge of chaos, mean field", "edge of chaos, mean field, input-scaled first layer",
                  "Crinit, input-scaled first layer", "zero bias, tuned gain, steady first layer"],
         "orth": ["Crinit, orthogonal, input-scaled first layer"], "relu": ["Crinit, leaky ReLU, input-scaled first layer"]}
STANDARD = {"tanh": ["PyTorch default", "LeCun", "He", "Xavier, gain 5/3", "typical-preserving"],
            "orth": ["orthogonal"], "relu": ["ReLU, He", "ReLU, typical-preserving", "leaky ReLU (0.01), He"]}
TUNED = {"tanh": ["best standard rule, tuned gain"], "orth": ["orthogonal, tuned gain"],
         "relu": ["ReLU, tuned gain", "leaky ReLU, tuned slope"]}
TEX_LABEL = {"PyTorch default": r"PyTorch default ($\sw=\frac13$)", "LeCun": r"LeCun ($\sw=1$)",
             "He": r"He ($\sw=2$)", "Xavier, gain 5/3": r"Xavier, gain $\frac53$ ($\sw=\frac{25}9$)",
             "typical-preserving": r"typical-preserving $\sigma_c^2(N)$",
             "best standard rule, tuned gain": r"zero bias, tuned $\sw$",
             "edge of chaos, mean field": "edge of chaos, mean field", "Crinit": "Crinit",
             "orthogonal": r"orthogonal ($\sw=1$)", "orthogonal, tuned gain": r"orthogonal, tuned $\sw$",
             "Crinit, orthogonal": "Crinit, orthogonal", "ReLU, He": r"ReLU, He ($\sw=2$)",
             "ReLU, typical-preserving": "ReLU, typical-preserving", "ReLU, tuned gain": r"ReLU, tuned $\sw$",
             "leaky ReLU (0.01), He": r"leaky ReLU ($\alpha=0.01$), He",
             "leaky ReLU, tuned slope": r"leaky ReLU, He, tuned $\alpha$", "Crinit, leaky ReLU": "Crinit, leaky ReLU",
             "Crinit, tuned $\\gamma$": r"Crinit, tuned $\gamma$",
             "Crinit, orthogonal, tuned $\\gamma$": r"Crinit, orthogonal, tuned $\gamma$",
             "Crinit, leaky ReLU, tuned $\\gamma$": r"Crinit, leaky ReLU, tuned $\gamma$",
             "Crinit, standard first layer": "Crinit, standard first layer",
             "Crinit, orthogonal, standard first layer": "Crinit, orthogonal, standard first layer",
             "Crinit, leaky ReLU, standard first layer": "Crinit, leaky ReLU, standard first layer",
             "Crinit, input-scaled first layer": "Crinit, input-scaled first layer (earlier runs)",
             "Crinit, orthogonal, input-scaled first layer": "Crinit, orthogonal, input-scaled first layer (earlier runs)",
             "Crinit, leaky ReLU, input-scaled first layer": "Crinit, leaky ReLU, input-scaled first layer (earlier runs)",
             "edge of chaos, mean field, input-scaled first layer": "edge of chaos, mean field, input-scaled first layer (earlier runs)",
             "zero bias, tuned gain, steady first layer": r"zero bias, tuned $\sw$, Crinit's first layer"}


def labels(R):
    """Rule labels in table order (from the selection stored with the results)."""
    L = min(R)
    return list(R[L]["select"][E_SHORT].keys())


def family(R, label):
    fams = [f for f in ("tanh", "orth", "relu") if label in STANDARD[f] + TUNED[f] + EXTRA[f] + [CRINIT[f], TUNED_CRINIT[f], STD1[f]]]
    return fams[0]


def load():
    R = {}
    for L in DEPTHS:
        p = os.path.join(D, f"tuned_L{L}.pkl")
        if os.path.exists(p):
            R[L] = pickle.load(open(p, "rb"))
    return R


def final_run(R, L, label, E):
    """The final networks of rule `label` for budget E: (summary, key, lr_mult)."""
    r = R[L]
    k, m, _ = r["select"][E][label]
    f = r["final"].get((k, m))
    if f is None:
        return None, k, m
    s = f["short"] if E == E_SHORT else f["long"]
    return s, k, m


def acc_at(s, E):
    a = s["test"][E - 1].astype(float)
    a = a[np.isfinite(a)]
    return a


def steps_to(s, target=TARGET):
    j = int(np.argmin(np.abs(s["targets"] - target)))
    st = s["first"][:, j]
    return st[st > 0], st.size


def stats(R, L, label, E):
    s, k, m = final_run(R, L, label, E)
    if s is None:
        return None
    a = acc_at(s, E)
    st, n = steps_to(s)
    a_all = np.where(np.isfinite(s["test"][E - 1]), s["test"][E - 1], CHANCE).astype(float)
    return dict(mean=a.mean(), se=a.std(ddof=1) / math.sqrt(a.size), sd=a.std(ddof=1), n=s["test"].shape[1],
                mean_all=a_all.mean(), se_all=a_all.std(ddof=1) / math.sqrt(a_all.size), median=float(np.median(a)),
                nfail=int(s["diverged"].sum() + (a < 0.5).sum()),
                ndiv=int(s["diverged"].sum()), steps=st.mean() if st.size else np.nan,
                steps_se=st.std(ddof=1) / math.sqrt(st.size) if st.size > 1 else np.nan, nreach=st.size, key=k, lr=m,
                acc=a, curve=np.array([np.nanmean(x) for x in s["test"]]))


def fmt(v, se, bold=False, box=False):
    """0.8346(6): mean with its standard error in units of the last digit."""
    if se == 0:
        s = f"{v:.3f}"                                    # networks that do not train: all at chance
    elif se < 0.00095:
        s = f"{v:.4f}({max(1, round(se * 1e4))})"
    else:
        s = f"{v:.3f}({max(1, round(se * 1e3))})"
    if bold:
        s = r"\textbf{" + s + "}"
    if box:
        s = r"\fbox{" + s + "}"
    return s


def table(R):
    labs_all = labels(R)
    lines = []
    head = (r"\begin{tabular}{l" + "ccc" * len(R) + "}\n\\hline\\hline\n"
            + " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{$L={L}$}}" for L in R) + r"\\" + "\n"
            + "rule" + r" & 2 epochs & 10 epochs & steps to $80\%$" * len(R) + r"\\" + "\n\\hline\n")
    for fam, title in GROUPS:
        labs = [l for l in labs_all if family(R, l) == fam]
        best = {}
        for L in R:
            for E in (E_SHORT, E_LONG):
                vals = {l: stats(R, L, l, E) for l in labs}
                best[(L, E)] = max((v["mean"] for v in vals.values() if v is not None), default=None)
            vals = {l: stats(R, L, l, E_LONG) for l in labs}
            sv = [v["steps"] for v in vals.values() if v is not None and v["nreach"] >= 0.5 * v["n"]]
            best[(L, "steps")] = min(sv) if sv else None
        lines.append(rf"\multicolumn{{{1 + 3 * len(R)}}}{{l}}{{\emph{{{title}}}}}\\")
        for l in labs:
            cells = [TEX_LABEL[l]]
            for L in R:
                for E in (E_SHORT, E_LONG, "steps"):
                    if E == "steps":
                        v = stats(R, L, l, E_LONG)
                        if v is None or v["nreach"] == 0:
                            cells.append("--")
                            continue
                        s = f"{v['steps']:.0f}"
                        if best[(L, 'steps')] is not None and abs(v["steps"] - best[(L, 'steps')]) < 0.5 and v["nreach"] >= 0.5 * v["n"]:
                            s = r"\textbf{" + s + "}"
                        if v["nreach"] < v["n"]:
                            s += f" ({v['nreach']})"
                        cells.append(s)
                        continue
                    v = stats(R, L, l, E)
                    if v is None:
                        cells.append("")
                        continue
                    c = fmt(v["mean"], v["se"], bold=abs(v["mean"] - best[(L, E)]) < 1e-12, box=(l in (CRINIT[fam], TUNED_CRINIT[fam])))
                    if v["nfail"] and v["nfail"] < v["n"]:
                        c += rf"$^{{{v['nfail']}}}$"          # networks that diverged or stayed below 0.5
                    cells.append(c)
            lines.append(" & ".join(cells) + r"\\")
    out = head + "\n".join(lines) + "\n\\hline\\hline\n\\end{tabular}\n"
    open(os.path.join(TEX, "tableS_fmnist.tex"), "w").write(out)


def choice_label(R, L, k):
    c = dict(R[L]["pilot"][k]["cfg"])
    if k.startswith(("tanh_crinit", "orth_crinit", "leaky_crinit")):
        if "gamma" not in c:
            c["gamma"] = configs(L)[k]["gamma"]
        g = rf"$\gamma={c['gamma']:.2f}$"
        if c["act"] == "leaky":
            return g + rf", $\alpha={c['alpha']:.2f}$"
        return g + rf", ${c['sw2']:.2f}$, ${c['sb2']:.3f}$"
    if c["act"] == "leaky" and c.get("alpha", 0) > 0:
        return rf"$\alpha={c['alpha']:.2f}$"
    if k in ("tanh_eoc", "tanh_typ"):
        return rf"${c['sw2']:.2f}$, ${c['sb2']:.3f}$"
    return rf"${c['sw2']:.2f}$"


def choice_table(R):
    labs_all = labels(R)
    head = (r"\begin{tabular}{l" + "cc" * len(R) + "}\n\\hline\\hline\n"
            + " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$L={L}$}}" for L in R) + r"\\" + "\n"
            + "rule" + r" & 2 epochs & 10 epochs" * len(R) + r"\\" + "\n\\hline\n")
    lines = []
    for fam, title in GROUPS:
        lines.append(rf"\multicolumn{{{1 + 2 * len(R)}}}{{l}}{{\emph{{{title}}}}}\\")
        for l in labs_all:
            if family(R, l) != fam:
                continue
            cells = [TEX_LABEL[l]]
            for L in R:
                for E in (E_SHORT, E_LONG):
                    k, m, _ = R[L]["select"][E][l]
                    show = (l in TUNED[fam] or l in EXTRA[fam] or l in (CRINIT[fam], TUNED_CRINIT[fam], STD1[fam]))
                    lab = choice_label(R, L, k) if show else ""
                    mm = f"{m:g}" if m >= 1 else r"\frac1{" + f"{1 / m:g}" + "}"
                    cells.append((lab + ", " if lab else "") + r"$\times" + mm + "$")
            lines.append(" & ".join(cells) + r"\\")
    out = head + "\n".join(lines) + "\n\\hline\\hline\n\\end{tabular}\n"
    open(os.path.join(TEX, "tableS_fmnist_choice.tex"), "w").write(out)


def comparisons(R):
    """For every family, depth and budget: the best standard rule at its usual variance (i), the best standard rule
    with tuned gain (ii), Crinit (iii) and Crinit with tuned gamma (iv).  Returns rows and win/tie/loss counts
    (tie: difference within two standard errors)."""
    rows, wtl = [], {"iii-i": [0, 0, 0], "iii-ii": [0, 0, 0], "iv-ii": [0, 0, 0], "v-ii": [0, 0, 0]}
    def best(L, E, labs):
        vs = [(stats(R, L, l, E), l) for l in labs]
        vs = [(v, l) for v, l in vs if v is not None]
        return max(vs, key=lambda t: t[0]["mean"])
    def comp(a, b, key):
        z = (a["mean"] - b["mean"]) / math.hypot(a["se"], b["se"])
        wtl[key][0 if z >= 2 else (2 if z <= -2 else 1)] += 1
        return z
    for fam, _ in GROUPS:
        for L in R:
            for E in (E_SHORT, E_LONG):
                i, li = best(L, E, STANDARD[fam])
                ii, lii = best(L, E, STANDARD[fam] + TUNED[fam])
                iii = stats(R, L, CRINIT[fam], E)
                iv = stats(R, L, TUNED_CRINIT[fam], E)
                v = stats(R, L, STD1[fam], E)
                rows.append(dict(fam=fam, L=L, E=E, i=(i, li), ii=(ii, lii), iii=iii, iv=iv, v=v,
                                 z_iii_i=comp(iii, i, "iii-i"), z_iii_ii=comp(iii, ii, "iii-ii"),
                                 z_iv_ii=comp(iv, ii, "iv-ii"), z_v_ii=comp(v, ii, "v-ii")))
    return rows, wtl


def summary_table(R):
    rows, wtl = comparisons(R)
    name = {"tanh": r"$\tanh$", "orth": r"$\tanh$, orth.", "relu": "ReLU family"}
    head = ("\\begin{tabular}{llccccc}\n\\hline\\hline\n"
            r" & & \multicolumn{2}{c}{standard rules} & \multicolumn{3}{c}{Crinit}\\" "\n"
            r"family & $L$, epochs & usual variance & tuned gain & default & tuned $\gamma$ & standard first layer\\"
            "\n\\hline\n")
    lines = []
    for r in rows:
        vals = [r["i"][0], r["ii"][0], r["iii"], r["iv"], r["v"]]
        top = max(v["mean"] for v in vals)
        cells = [name[r["fam"]], f"${r['L']}$, ${r['E']}$"]
        for v in vals:
            c = fmt(v["mean"], v["se"], bold=abs(v["mean"] - top) < 1e-12)
            if v["nfail"] and v["nfail"] < v["n"]:
                c += rf"$^{{{v['nfail']}}}$"
            cells.append(c)
        lines.append(" & ".join(cells) + r"\\")
    t = lambda k: "/".join(str(x) for x in wtl[k])
    foot = ("\\hline\n" + r"\multicolumn{3}{l}{wins/ties/losses against tuned gain} & & "
            + f"{t('iii-ii')} & {t('iv-ii')} & {t('v-ii')}" + r"\\" + "\n"
            + r"\multicolumn{3}{l}{wins/ties/losses against usual variance} & & " + f"{t('iii-i')} & &" + r"\\" + "\n")
    out = head + "\n".join(lines) + "\n" + foot + "\\hline\\hline\n\\end{tabular}\n"
    open(os.path.join(TEX, "tableS_fmnist_summary.tex"), "w").write(out)
    return rows, wtl


def best_lr_point(R, L, k, E=E_SHORT):
    """Pilot validation accuracy of configuration k at its best learning rate (mean, standard error)."""
    p = R[L]["pilot"][k]
    val, lrs = (p["stage1"]["val"][E - 1], p["lr1"]) if E == E_SHORT else (p["stage2"]["val"][E - 1], p["lr2"])
    best = max(np.unique(lrs), key=lambda m: np.mean(np.where(np.isnan(val[lrs == m]), CHANCE, val[lrs == m])))
    v = np.where(np.isnan(val[lrs == best]), CHANCE, val[lrs == best])
    return v.mean(), v.std(ddof=1) / math.sqrt(v.size)


def scan(R, L, keys, x, E=E_SHORT):
    pts = sorted((x(k), *best_lr_point(R, L, k, E)) for k in keys if k in R[L]["pilot"])
    return np.array(pts).T if pts else (np.array([]),) * 3


def figure(R):
    style.apply(8.5)
    fig, ax = plt.subplots(2, 2, figsize=(5.6, 4.0))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.11, wspace=0.38, hspace=0.55)
    ax = ax.ravel()
    # (a) zero-bias gain scan and Crinit's gamma line, tanh with Gaussian weights
    a = ax[0]
    a.axvline(1.0, color=GREY, lw=0.6, ls=":")
    a.axvline(ZB_FW, color=GREY, lw=0.6, ls="--")
    for L, col, tx in [(32, BLUE, (2.3, 0.842)), (96, RED, (1.95, 0.715))]:
        if L not in R:
            continue
        P = R[L]["pilot"]
        sw = lambda k: P[k]["cfg"]["sw2"]
        x, y, e = scan(R, L, [k for k in P if k.startswith("tanh_zb:")], sw)
        a.errorbar(x, y, e, fmt="-o", color=col, ms=2.8, lw=0.9, elinewidth=0.6, capsize=0)
        x, y, e = scan(R, L, [k for k in P if k == "tanh_crinit" or k.startswith("tanh_crinit:")], sw)
        a.errorbar(x, y, e, fmt="--*", color=col, ms=6, mec=INK, mew=0.3, lw=0.8, elinewidth=0.6, capsize=0, zorder=5)
        for key, mk in (("tanh_crinit_std1", "D"), ("tanh_crinit@in", "^")):
            if key in P:
                x, y, e = scan(R, L, [key], sw)
                a.errorbar(x, y, e, fmt=mk, color="white", mec=col, ms=4, mew=1.0, elinewidth=0.6, ecolor=col,
                           capsize=0, zorder=6)
        a.text(*tx, rf"$L={L}$", color=col, fontsize=7, ha="left", va="center")
    a.set_xscale("log", base=2)
    a.set_xticks([0.5, 1, 2, 4])
    a.set_xticklabels(["0.5", "1", "2", "4"])
    a.minorticks_off()
    a.set_xlim(0.45, 4.4)
    a.set_ylim(0.6, 0.875)
    a.set_xlabel(r"$\sigma_w^2$")
    ylab(a, "validation accuracy")
    # (b) Crinit's gamma scan for the three Crinit rules
    a = ax[1]
    for base, col, name in [("tanh_crinit", RED, r"$\tanh$"), ("orth_crinit", BLUE, r"$\tanh$, orth."),
                            ("leaky_crinit", ORANGE, "leaky ReLU")]:
        for L, ls, mk in [(32, "-", "o"), (96, "--", "s")]:
            if L not in R:
                continue
            P = R[L]["pilot"]
            C = configs(L)
            keys = [k for k in P if (k == base or k.startswith(base + ":")) and "@" not in k and "steady" not in k]
            x, y, e = scan(R, L, keys, lambda k: P[k]["cfg"].get("gamma", C[k]["gamma"] if k in C else 1.0))
            a.errorbar(x, y, e, fmt=ls + mk, color=col, ms=2.6, lw=0.9, elinewidth=0.6, capsize=0,
                       label=name if L == 32 else None)
    a.axvline(1.0, color=GREY, lw=0.6, ls=":")
    a.set_xscale("log", base=2)
    a.set_xticks([0.25, 0.5, 1, 2])
    a.set_xticklabels(["0.25", "0.5", "1", "2"])
    a.minorticks_off()
    a.set_xlabel(r"$\gamma$")
    ylab(a, "validation accuracy")
    a.legend(loc="lower left", fontsize=6.0)
    # (c, d) test accuracy against the epoch at L = 96
    for a, labs, names, title in [
            (ax[2], ("LeCun", "best standard rule, tuned gain", "Crinit", "Crinit, input-scaled first layer"),
             ("LeCun", "tuned gain", "Crinit", "Crinit, earlier runs"), r"$\tanh$, $L=96$"),
            (ax[3], ("ReLU, He", "leaky ReLU, tuned slope", "Crinit, leaky ReLU", TUNED_CRINIT["relu"]),
             ("ReLU, He", "tuned slope", "Crinit", r"Crinit, tuned $\gamma$"), r"ReLU family, $L=96$")]:
        L = 96 if 96 in R else max(R)
        for lab, col, ls, name in zip(labs, (GREY, ORANGE, RED, DEEP_RED), ("-", "--", "-", ":"), names):
            v = stats(R, L, lab, E_LONG)
            if v is None:
                continue
            ep = np.arange(1, len(v["curve"]) + 1)
            a.plot(ep, v["curve"], ls, color=col, lw=1.0, marker="o", ms=2.2, label=name)
        a.set_xlabel("epoch")
        ylab(a, "test accuracy")
        a.set_xlim(0.5, E_LONG + 0.5)
        a.set_xticks([2, 4, 6, 8, 10])
        a.text(1.0, 1.04, title, transform=a.transAxes, ha="right", va="bottom", fontsize=7)
        a.legend(loc="lower right", fontsize=5.8, bbox_to_anchor=(1.0, 0.02))
    for a, l in zip(ax, "abcd"):
        panel(a, l, x=-0.28, y=1.06)
    fig.savefig(os.path.join(FIG, "figS_fmnist.pdf"))


FIRST_ROWS = [("std", r"standard: $q_1=\sw$, $W_1\sim\mathcal N(0,\sw/784)$", "1"),
              ("in", r"steady signal by input scaling: $q_1=\qs$, input $\times s$ (earlier runs)", "$s$"),
              ("weight", r"steady signal by weight scaling: $q_1=\qs$, $W_1\sim\mathcal N(0,(\qs-\sbb)/784)$ (Crinit)", "$1/s$"),
              ("in+lr", r"input scaling, first-layer learning rate $\times1/s$", "1"),
              ("std-lr", r"standard, first-layer learning rate $\times s$", "$s$"),
              ("weight*lr0.5", r"weight scaling, first-layer learning rate $\times\frac12$", "$1/(2s)$"),
              ("weight*lr2", r"weight scaling, first-layer learning rate $\times2$", "$2/s$")]


def first_layer_table():
    """SM table of the small first-layer test (data/fmnist/first_small_L*.pkl): best-learning-rate validation accuracy."""
    F = {}
    for L in DEPTHS:
        p = os.path.join(D, f"first_small_L{L}.pkl")
        if os.path.exists(p):
            F[L] = pickle.load(open(p, "rb"))
    if not F:
        return
    def cell(L, key):
        d = F.get(L, {}).get(key)
        if d is None:
            return "--"
        best = max(d["res"], key=lambda m: d["res"][m]["val"][-1].mean())
        v = d["res"][best]["val"][-1]
        return fmt(v.mean(), v.std(ddof=1) / math.sqrt(v.size))
    head = (r"\begin{tabular}{p{8.6cm}ccc}" "\n\\hline\\hline\n"
            r"first layer & relative step of $W_1$ & $L=32$, 2 epochs & $L=96$, 4 epochs\\" "\n\\hline\n")
    lines = [r"\multicolumn{4}{l}{\emph{Crinit point ($\gamma=1$)}}\\"]
    for key, lab, step in FIRST_ROWS:
        lines.append(f"{lab} & {step} & {cell(32, 'crinit:' + key)} & {cell(96, 'crinit:' + key)}" + r"\\")
    lines.append(r"\multicolumn{4}{l}{\emph{zero bias, $\sw=1.41$ (the tuned gain at $L=32$)}}\\")
    for key, lab, step in FIRST_ROWS[:5]:
        lines.append(f"{lab} & {step} & {cell(32, 'zb1.41:' + key)} & --" + r"\\")
    out = head + "\n".join(lines) + "\n\\hline\\hline\n\\end{tabular}\n"
    open(os.path.join(TEX, "tableS_first.tex"), "w").write(out)


def headline(R):
    for L in R:
        for E in (E_SHORT, E_LONG):
            print(f"\nL={L}, {E} epochs: test accuracy (mean +- se, sd, n, diverged; counting diverged as chance) "
                  f"and steps to 80%")
            for l in labels(R):
                v = stats(R, L, l, E)
                if v is None:
                    continue
                print(f"  {l:40s} {v['mean']:.4f} +- {v['se']:.4f}  sd {v['sd']:.3f}  n {v['n']:3d} div {v['ndiv']:2d} "
                      f"[all {v['mean_all']:.4f} median {v['median']:.4f} fail {v['nfail']}]  steps {v['steps']:.0f} ({v['nreach']})  [{v['key']}, lr x{v['lr']}]")


def complete(R):
    return all(all(((k, m) in r["final"]) and ("short" if E == E_SHORT else "long") in r["final"][(k, m)]
                   for E in (E_SHORT, E_LONG) for (k, m, _) in r["select"][E].values()) for r in R.values())


if __name__ == "__main__":
    first_layer_table()
    R = load()
    R = {L: r for L, r in R.items() if "final" in r and "select" in r}
    headline(R)
    if R and complete(R):
        rows, wtl = summary_table(R)
        for r in rows:
            print(f"{r['fam']:5s} L={r['L']:3d} E={r['E']:2d}: (i) {r['i'][0]['mean']:.4f} [{r['i'][1]}]  "
                  f"(ii) {r['ii'][0]['mean']:.4f} [{r['ii'][1]}]  (iii) {r['iii']['mean']:.4f}  (iv) {r['iv']['mean']:.4f}  "
                  f"(v) {r['v']['mean']:.4f}  z(iii-i) {r['z_iii_i']:+.1f} z(iii-ii) {r['z_iii_ii']:+.1f} "
                  f"z(iv-ii) {r['z_iv_ii']:+.1f} z(v-ii) {r['z_v_ii']:+.1f}")
        print("wins/ties/losses", wtl)
        table(R)
        choice_table(R)
        figure(R)
        print("tables and figure written")
