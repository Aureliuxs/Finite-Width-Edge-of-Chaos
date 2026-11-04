"""Tables of the paper and the Supplemental Material, written as LaTeX fragments into latex/."""
import glob
import json
import math
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eocfw.finite_width import a1_b1  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "data")
TEX = os.path.join(HERE, "..", "..", "latex")

PRETTY = {"tanh_standard": "standard ($\\sigma_w^2=1$)", "tanh_finiteN": "typical-preserving $\\sigma_c^2(N)$",
          "tanh_sweet:1": "edge of chaos, $\\gamma=1$", "tanh_sweetN:1": "edge of chaos, $\\gamma=1$, $+\\Delta\\sigma_w^2$",
          "tanh_chaotic": "chaotic ($\\sigma_w^2=2$)", "relu_he": "ReLU, He", "relu_finiteN": "ReLU, $2e^{5/(2N)}$",
          "leaky:1": "shaped leaky ReLU, $\\gamma=1$", "leaky:2.3026": "shaped leaky ReLU, $\\gamma=2.3$",
          "linear": "linear", "orth_tanh_standard": "orthogonal, standard", "orth_tanh_sweet:1": "orthogonal, edge, $\\gamma=1$",
          "orth_linear": "orthogonal, linear"}
ORDER = ["tanh_standard", "tanh_finiteN", "tanh_sweet:1", "tanh_sweetN:1", "tanh_chaotic", "orth_tanh_standard",
         "orth_tanh_sweet:1", "relu_he", "relu_finiteN", "leaky:1", "leaky:2.3026", "linear", "orth_linear"]


def acc(r):
    return r["test_acc"][-1] if (r["test_acc"] and not r["diverged"]) else np.nan


def stats(R, name, L, N):
    rows = [r for k, r in R.items() if k[0] == name and k[1] == L and k[2] == N]
    a = np.array([acc(r) for r in rows], float)
    s = np.array([r["steps_to_target"] if r["steps_to_target"] is not None else np.nan for r in rows], float)
    lr = sorted({round(r["lr"] * L / 32 * 1000, 2) for r in rows})
    return a, s, lr


def fmt_acc(a):
    ok = a[~np.isnan(a)]
    if ok.size == 0:
        return "--"
    return f"${ok.mean():.3f}\\pm{ok.std() / math.sqrt(ok.size):.3f}$"


def fmt_steps(s):
    n = int(np.sum(~np.isnan(s)))
    return f"{np.nanmean(s):.0f} ({n}/{s.size})" if n else f"-- (0/{s.size})"


def mnist_tables():
    R = pickle.load(open(os.path.join(D, "mnist", "main.pkl"), "rb"))
    main, full = [], []
    for name in ORDER:
        if not any(k[0] == name for k in R):
            continue
        cells = [PRETTY[name]]
        fcells = [PRETTY[name]]
        for L in (32, 96):
            a, s, lr = stats(R, name, L, 64)
            cells.append(fmt_acc(a))
            fcells += [fmt_acc(a), fmt_steps(s), ", ".join(f"{v:g}" for v in lr)]
        main.append(" & ".join(cells) + r" \\")
        full.append(" & ".join(fcells) + r" \\")
    open(os.path.join(TEX, "table_mnist.tex"), "w").write("\n".join(main) + "\n")
    head = (r"\begin{tabular}{@{}l ccc ccc@{}}" "\n" r"\toprule" "\n"
            r"& \multicolumn{3}{c}{$L=32$} & \multicolumn{3}{c}{$L=96$}\\" "\n"
            r"\cmidrule(lr){2-4}\cmidrule(l){5-7}" "\n"
            r"initialisation & test acc. & steps to 85\% & $\eta_0$ & test acc. & steps to 85\% & $\eta_0$\\" "\n" r"\midrule" "\n")
    open(os.path.join(TEX, "tableS_mnist.tex"), "w").write(head + "\n".join(full) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    print("\n".join(full))


def gamma_table():
    path = os.path.join(D, "mnist", "gamma.pkl")
    if not os.path.exists(path):
        return
    R = pickle.load(open(path, "rb"))
    grid = [(16, 64), (32, 64), (64, 64), (96, 64), (96, 32)]
    gammas = [0.25, 0.5, 1, 2.3026, 3.5]
    lines = [r"\begin{tabular}{@{}ll" + "c" * len(gammas) + "@{}}", r"\toprule",
             r"$(L,N)$ & rule & " + " & ".join(f"$\\gamma={g:g}$" if g != 2.3026 else "$\\gamma=2.3$" for g in gammas) + r"\\", r"\midrule"]
    for (L, N) in grid:
        for base, lab in [("tanh_sweet", "$\\tanh$, edge of chaos"), ("leaky", "shaped leaky ReLU")]:
            cells = [f"$({L},{N})$" if base == "tanh_sweet" else "", lab]
            for g in gammas:
                a, _, _ = stats(R, f"{base}:{g}", L, N)
                cells.append(fmt_acc(a) if a.size else "--")
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TEX, "tableS_gamma.tex"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def info_table():
    lines = [r"\begin{tabular}{@{}cccccc@{}}", r"\toprule",
             r"$L$ & $N$ & plateau ($\qs<0.02$) & best $\qs$ & ReLU & linear\\", r"\midrule"]
    for L in [10, 40, 160]:
        for N in [64, 128, 256, 512]:
            f = os.path.join(D, "info", f"info_L{L}_N{N}.npy")
            if not os.path.exists(f):
                continue
            a = np.load(f)
            r = np.load(os.path.join(D, "info", f"ref_L{L}_N{N}.npy"))
            pl = a[a[:, 0] < 0.02, 1]
            best = a[np.argmax(a[:, 1]), 0]
            lines.append(f"{L} & {N} & ${pl.mean():.2f}\\pm{pl.std() / math.sqrt(pl.size):.2f}$ & ${best:.2g}$ & "
                         f"${r[0, 0]:.2f}\\pm{r[0, 1]:.2f}$ & ${r[1, 0]:.2f}\\pm{r[1, 1]:.2f}$\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TEX, "tableS_info.tex"), "w").write("\n".join(lines) + "\n")


def walk_table():
    """Table S1: drift and long-run variance of ln D against a_1 and b_1 (tanh at N = 16, 64, 256 from
    distance_walk_v2.json; erf at N = 800 and orthogonal tanh at N = 200 from distance_walk.json)."""
    dw = json.load(open(os.path.join(D, "distance_walk.json")))
    v2 = json.load(open(os.path.join(D, "distance_walk_v2.json"))) if os.path.exists(os.path.join(D, "distance_walk_v2.json")) else None
    lines = [r"\begin{tabular}{@{}llrccccccc@{}}", r"\toprule",
             r"$\phi$ & $\qs$ & $N$ & $a_1$ & $N\mu_d$ & $b_1$ & $N\sigma_d^2$ & $N(1-F')$ & $a_1+\frac12b_1$ & $N\mu_{\rm mean}$\\",
             r"\midrule"]

    def row(lab, q, th, sim):
        return (f"{lab} & {q:g} & {sim['N']} & ${th['a1']:.3f}$ & ${sim['N_mu']:.3f}$ & ${th['b1']:.2f}$ & ${sim['N_var_long']:.2f}$ & "
                f"${sim['N'] * (1 - th['Fp']):.1f}$ & ${th['a1'] + th['b1'] / 2:.2f}$ & ${sim['N_mu_mean']:.2f}$\\\\")
    if v2 is not None:
        for r in v2["gaussian"]["tanh"]:
            for sim in r["sims"]:
                lines.append(row("tanh" if sim["N"] == 16 else "", r["q"], r["theory"], sim))
    lines.append(r"\midrule")
    for r in dw["gaussian"].get("erf", []):
        for sim in r["sims"]:
            if sim["N"] == 800:
                lines.append(row("erf", r["q"], r["theory"], sim))
    lines.append(r"\midrule")
    for r in dw["orthogonal"].get("tanh", []):
        for sim in r["sims"]:
            lines.append(row("tanh, orth.", r["q"], r["theory"], sim))
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TEX, "tableS_walk.tex"), "w").write("\n".join(lines) + "\n")


# ---------------------------------------------------------------- large-statistics comparison (Table I, S4) --
BIG_ROWS = [  # (rule name, label, group header or None, boxed); the best value is marked within each group
    ("tanh_default", r"PyTorch ($\sw=\frac13$)", r"$\tanh$, Gaussian weights", False),
    ("tanh_standard", r"LeCun ($\sw=1$)", None, False),
    ("tanh_chaotic", r"He ($\sw=2$)", None, False),
    ("tanh_finiteN", r"typical-preserving", None, False),
    ("tanh_eoc:1", r"mean-field edge of chaos", None, False),
    ("tanh_crinit:1", r"Crinit", None, True),
    ("orth_tanh_standard", r"orthogonal ($\sw=1$)", r"$\tanh$, orthogonal weights", False),
    ("orth_tanh_eoc:1", r"mean-field edge of chaos", None, False),
    ("orth_tanh_crinit:1", r"Crinit, orthogonal", None, True),
    ("relu_he", r"ReLU, He ($\sw=2$)", r"ReLU family, Gaussian weights", False),
    ("relu_crinit", r"ReLU, typical-preserving", None, False),
    ("leaky_he", r"leaky ReLU, He", None, False),
    ("leaky_crinit:1", r"Crinit, leaky ReLU", None, True),
]
BIG_ROWS_LABELS = [(r[0], r[1]) for r in BIG_ROWS]
SM_EXTRA_ROWS = [  # rows of the SM table only (the "@in" runs and the fixed-bias shift used the input-scaled first layer)
    ("tanh_eoc:1@in", r"edge of chaos, mean field, input-scaled first layer"),
    ("tanh_crinit:1@in", r"Crinit, input-scaled first layer"),
    ("orth_tanh_crinit:1@in", r"Crinit, orthogonal, input-scaled first layer"),
    ("relu_crinit@in", r"ReLU, typical-preserving, input-scaled first layer"),
    ("leaky_crinit:1@in", r"Crinit, leaky ReLU, input-scaled first layer"),
    ("leaky_crinit:1@in@0.3", r"Crinit, leaky ReLU, input-scaled first layer, $\eta_0=1.5$ fixed"),
    ("leaky_crinit:1@0.3", r"Crinit, leaky ReLU, $\eta_0=1.5$ fixed"),
    ("tanh_crinit:1@0.3", r"Crinit, $\eta_0=1.5$ fixed"),
    ("tanh_sweetN:1", r"shift at fixed bias (Eq.~10), input-scaled first layer"),
    ("orth_tanh_sweetN:1", r"shift at fixed bias, orthogonal, input-scaled first layer"),
    ("tanh_sweetN:1@1.0", r"shift at fixed bias, input-scaled first layer, $\eta_0=5$ fixed"),
]


def load_big():
    R = {}
    for f in sorted(glob.glob(os.path.join(D, "mnist", "big*.pkl"))):
        R.update(pickle.load(open(f, "rb")))
    return R


def big_stats(r):
    a = r["test_acc"]
    ok = ~np.isnan(a)
    st = r["steps_to_target"]
    reached = st > 0
    return dict(acc=a[ok].mean(), acc_se=a[ok].std(ddof=1) / math.sqrt(ok.sum()), n_ok=int(ok.sum()), n=a.size,
                steps=st[reached].mean() if reached.any() else np.nan,
                steps_se=st[reached].std(ddof=1) / math.sqrt(reached.sum()) if reached.sum() > 1 else np.nan,
                n_reached=int(reached.sum()), lr=float(np.asarray(r["lr"]).ravel()[0]), lr_mult=r.get("lr_multiplier"))


def fmt_err(mean, se):
    """mean(se) with the error in units of the last printed digit: 0.9517(4), 0.903(7), 0.713(21)."""
    if not np.isfinite(se) or se <= 0:
        return f"{mean:.3f}"
    dec = max(3, -int(math.floor(math.log10(se))))      # decimals such that the error has one significant digit
    dec = min(dec, 4)
    e = int(round(se * 10 ** dec))
    return f"{mean:.{dec}f}({e})" if e > 0 else f"{mean:.{dec}f}"


# rows of Table I whose L = 32 entry is the mean over one thousand paired networks (seeds 0-999; data/mnist/big_paired.pkl)
PAIRED = ("tanh_crinit:1", "tanh_eoc:1")


def merged(R, name, L, N, seed0s=range(100, 1000, 100)):
    """The main run (seeds 0-99) together with the extra paired runs (keys name#s<seed0>), or None."""
    keys = [(name, L, N)] + [(f"{name}#s{s}", L, N) for s in seed0s]
    if not all(k in R for k in keys):
        return None
    out = dict(R[(name, L, N)])
    for f in ("test_acc", "steps_to_target", "diverged", "seeds"):
        out[f] = np.concatenate([np.asarray(R[k][f]) for k in keys])
    return out


def paired_difference(R, a, b, L=32, N=64):
    """Paired difference (a - b) of accuracy and steps over the thousand common seeds."""
    A, B = merged(R, a, L, N), merged(R, b, L, N)
    assert np.array_equal(A["seeds"], B["seeds"])
    d = A["test_acc"] - B["test_acc"]
    ds = (A["steps_to_target"] - B["steps_to_target"]).astype(float)
    se = lambda x: x.std(ddof=1) / math.sqrt(x.size)
    return dict(n=d.size, acc=d.mean(), acc_se=se(d), higher=int((d > 0).sum()), lower=int((d < 0).sum()),
                steps=ds.mean(), steps_se=se(ds), faster=int((ds < 0).sum()), slower=int((ds > 0).sum()))


def big_tables(Ls=(32, 96), N=64):
    R = load_big()
    S = {(name, L): dict(big_stats(R[(name, L, N)]), name=name) for name, *_ in BIG_ROWS + SM_EXTRA_ROWS for L in Ls
         if (name, L, N) in R}
    S1000 = {}
    for name in PAIRED:
        m = merged(R, name, 32, N)
        if m is not None:
            S1000[name] = dict(big_stats(m), name=name)
    if len(S1000) == 2:
        pdiff = paired_difference(R, *PAIRED)
        print("paired Crinit - mean field, L=32, 1000 networks:", pdiff)
    S100 = dict(S)                                     # Table I and the SM table use the hundred-network runs;
    # the thousand-network runs appear only as extra rows of the SM table (see below)
    # every row of the main-text table is the run at the learning rate chosen by its own pilot (one protocol)
    group_of, g = {}, None
    for name, label, group, boxed in BIG_ROWS:
        g = group if group is not None else g
        group_of[name] = g
    best_acc, best_steps = {}, {}
    for gname in set(group_of.values()):
        names = [n for n in group_of if group_of[n] == gname]
        for L in Ls:
            vals = [S[(n, L)] for n in names if (n, L) in S]
            best_acc[(gname, L)] = max(vals, key=lambda v: v["acc"])
            fin = [v for v in vals if np.isfinite(v["steps"])]
            best_steps[(gname, L)] = min(fin, key=lambda v: v["steps"]) if fin else None

    def within(v, best, key, se_key):
        """The best value, or one not significantly different from it: the difference is smaller than twice its
        standard error (the combined standard error of the two means)."""
        if best is None or not np.isfinite(v[key]):
            return False
        if v["name"] == best["name"]:
            return True
        se = math.sqrt(v[se_key] ** 2 + best[se_key] ** 2) if np.isfinite(v[se_key]) and np.isfinite(best[se_key]) else 0.0
        return abs(v[key] - best[key]) < 2 * se

    def cell_acc(v, L):
        t = fmt_err(v["acc"], v["acc_se"])
        if within(v, best_acc[(group_of[v["name"]], L)], "acc", "acc_se"):
            t = r"\textbf{" + t + "}"
        if v["n_ok"] < v["n"]:
            t += f"$^{{{v['n_ok']}}}$"
        return t

    def cell_steps(v, L):
        if not np.isfinite(v["steps"]):
            return "--"
        t = f"{v['steps']:.0f}"
        if within(v, best_steps[(group_of[v["name"]], L)], "steps", "steps_se"):
            t = r"\textbf{" + t + "}"
        if v["n_reached"] < v["n"]:
            t += f"$^{{{v['n_reached']}}}$"
        return t

    # main-text table
    ncol = 1 + 2 * len(Ls)
    lines = [r"\begin{tabular}{l" + "cc" * len(Ls) + "}",
             " & " + " & ".join(r"\multicolumn{2}{c}{$L=%d$}" % L for L in Ls) + r"\\",
             "initialisation & " + " & ".join("accuracy & steps" for _ in Ls) + r"\\", r"\hline"]
    box_open = False
    for i, (name, label, group, boxed) in enumerate(BIG_ROWS):
        if group is not None:
            if box_open:
                lines.append(r"\cline{1-%d}" % ncol)
                box_open = False
            lines.append(r"\multicolumn{%d}{l}{\textit{%s}}\\" % (ncol, group))
        if not all((name, L) in S for L in Ls):
            continue
        cells = []
        for L in Ls:
            v = S[(name, L)]
            cells += [cell_acc(v, L), cell_steps(v, L)]
        next_boxed = i + 1 < len(BIG_ROWS) and BIG_ROWS[i + 1][3] and BIG_ROWS[i + 1][2] is None
        if boxed and not box_open:
            lines.append(r"\cline{1-%d}" % ncol)
            box_open = True
        if boxed:
            row = r"\multicolumn{1}{|l}{%s} & " % label + " & ".join(cells[:-1]) + r" & \multicolumn{1}{c|}{%s}\\" % cells[-1]
        else:
            row = label + " & " + " & ".join(cells) + r"\\"
        lines.append(row)
        if boxed and not next_boxed:
            lines.append(r"\cline{1-%d}" % ncol)
            box_open = False
    if box_open:
        lines.append(r"\cline{1-%d}" % ncol)
    lines.append(r"\end{tabular}")
    open(os.path.join(TEX, "table_mnist.tex"), "w").write("\n".join(lines) + "\n")
    # SM table with errors and learning rates
    head = (r"\begin{tabular}{@{}l" + "cccc" * len(Ls) + "@{}}" + "\n" + r"\toprule" + "\n" +
            " & " + " & ".join(r"\multicolumn{4}{c}{$L=%d$}" % L for L in Ls) + r"\\" + "\n" +
            " ".join(r"\cmidrule(lr){%d-%d}" % (2 + 4 * k, 5 + 4 * k) for k in range(len(Ls))) + "\n" +
            "initialisation & " + " & ".join(r"test accuracy & steps to 85\% & reached & $\eta_0$" for _ in Ls) + r"\\" + "\n" + r"\midrule" + "\n")
    rows = []
    T = dict(S100)
    extra = []
    for name in PAIRED:
        if name in S1000:
            T[(name + "#1000", 32)] = S1000[name]
            extra.append((name + "#1000", dict(BIG_ROWS_LABELS)[name] + r", one thousand networks", None, False))
    for name, label, *_ in BIG_ROWS + [(n, l, None, False) for n, l in SM_EXTRA_ROWS] + extra:
        if not any((name, L) in T for L in Ls):
            continue
        if name == "orth_tanh_eoc:1":
            label = r"mean-field edge of chaos, orthogonal"
        cells = [label]
        for L in Ls:
            if (name, L) not in T:
                cells += ["--", "--", "--", "--"]
                continue
            v = T[(name, L)]
            da = 5 if v["acc_se"] < 0.0002 else 4
            ds = 1 if np.isfinite(v["steps_se"]) and v["steps_se"] < 1 else 0
            acc_cell = f"${v['acc']:.{da}f}\\pm{v['acc_se']:.{da}f}$" + (f" ({v['n_ok']} nets)" if v["n_ok"] < v["n"] else "")
            cells += [acc_cell, f"${v['steps']:.{ds}f}\\pm{v['steps_se']:.{ds}f}$" if np.isfinite(v["steps"]) else "--",
                      f"{v['n_reached']}/{v['n']}", f"{v['lr'] * L / 32 * 1000:g}"]
        if name == SM_EXTRA_ROWS[0][0] or name.endswith("#1000") and name.startswith(PAIRED[0]):
            rows.append(r"\midrule")
        rows.append(" & ".join(cells) + r"\\")
    open(os.path.join(TEX, "tableS_mnist.tex"), "w").write(head + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")
    print("\n".join(lines))
    print("\n".join(rows))
    return S


N32_ROWS = [("tanh_standard", r"LeCun ($\sw=1$)"), ("tanh_finiteN", r"typical-preserving"),
            ("tanh_eoc:1", r"mean-field edge of chaos"), ("tanh_crinit:1", r"Crinit"),
            ("leaky_he", r"leaky ReLU, He"), ("leaky_crinit:1", r"Crinit, leaky ReLU")]


def n32_table(N=32, Ls=(32, 96)):
    """SM table of the narrower networks (data/mnist/big_n32.pkl), same format as the SM MNIST table."""
    R = load_big()
    rows = []
    for name, label in N32_ROWS:
        if not any((name, L, N) in R for L in Ls):
            continue
        cells = [label]
        for L in Ls:
            if (name, L, N) not in R:
                cells += ["--", "--", "--", "--"]
                continue
            v = big_stats(R[(name, L, N)])
            acc_cell = f"${v['acc']:.4f}\\pm{v['acc_se']:.4f}$" + (f" ({v['n_ok']} nets)" if v["n_ok"] < v["n"] else "")
            cells += [acc_cell, f"${v['steps']:.0f}\\pm{v['steps_se']:.0f}$" if np.isfinite(v["steps"]) else "--",
                      f"{v['n_reached']}/{v['n']}", f"{v['lr'] * L / 32 * 1000:g}"]
        rows.append(" & ".join(cells) + r"\\")
    if not rows:
        return
    head = (r"\begin{tabular}{@{}l" + "cccc" * len(Ls) + "@{}}" + "\n" + r"\toprule" + "\n" +
            " & " + " & ".join(r"\multicolumn{4}{c}{$L=%d$}" % L for L in Ls) + r"\\" + "\n" +
            " ".join(r"\cmidrule(lr){%d-%d}" % (2 + 4 * k, 5 + 4 * k) for k in range(len(Ls))) + "\n" +
            "initialisation & " + " & ".join(r"test accuracy & steps to 85\% & reached & $\eta_0$" for _ in Ls) + r"\\" + "\n" + r"\midrule" + "\n")
    open(os.path.join(TEX, f"tableS_mnist_n{N}.tex"), "w").write(head + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")
    print("\n".join(rows))


def orders_table():
    """Table S2: Crinit for tanh at first order and at all orders (exact chain), data/crinit_orders.json."""
    f = os.path.join(D, "crinit_orders.json")
    if not os.path.exists(f):
        return
    rows = json.load(open(f))
    lines = [r"\begin{tabular}{@{}rrcccccc@{}}", r"\toprule",
             r"$L$ & $N$ & $\qs$ & $\sbb$ & $\sw$ mean field & $\sw$ first order & $\sw$ all orders & difference / correction\\",
             r"\midrule"]
    seen = set()
    for r in rows:
        key = (r["N"], round(r["sw2_1"], 6))
        if key in seen:                      # deep and shallow give the same zero-bias point at small N
            continue
        seen.add(key)
        corr = r["sw2_1"] - r["mf"]
        mf_e = r["mf"] if r["sb2_1"] > 0 else None
        lines.append(f"{r['L']} & {r['N']} & {r['q1']:.3f} & {r['sb2_1']:.4f} & {r['mf']:.4f} & {r['sw2_1']:.4f} & "
                     f"${r['sw2_e']:.4f}\\pm{r['se']:.4f}$ & ${(r['sw2_e'] - r['sw2_1']) / corr * 100:+.0f}\\%$\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TEX, "tableS_orders.tex"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    orders_table()
    if glob.glob(os.path.join(D, "mnist", "big*.pkl")):
        big_tables()
        n32_table()
    else:
        mnist_tables()
    gamma_table()
    info_table()
    if os.path.exists(os.path.join(D, "distance_walk.json")):
        walk_table()
    print("tables written")
