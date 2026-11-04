"""Figure for the backward-propagation note: forward tangent walk against backward gradient walk."""
import sys, glob, pickle, re
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/claude/paper/code'); sys.path.insert(0, '/home/claude/crinit')
from eocfw import style
from eocfw.style import BLUE, RED, ORANGE, DEEP_BLUE, DEEP_RED, LIGHT_BLUE, GREY, INK
import crinit
from crinit.finite_width import a1_b1

DATA = '/home/claude/paper/code/data/backward/'
A1 = a1_b1('tanh', 0.5700478816408142)
sw2, q = crinit.eoc_at_bias('tanh', 0.05)
act = crinit.get('tanh'); m = act.moments(q)
NEr2 = sw2 * m['EphiDphi2'] / (q * m['P'] * (1 - A1['Fp']**2))
A1_GI = A1['a1'] - 2 * q * (m['dP'] / m['P']) * NEr2
BURN = 100


def se(x): return x.std(ddof=1) / np.sqrt(len(x))


def drift_explicit():
    rows = []
    for f in sorted(glob.glob(DATA + 'walk_N*_L400_sw1.761.pkl')):
        d = pickle.load(open(f, 'rb')); N, L = d['N'], d['L']
        fw = d['steps_f'][:, BURN:].mean(1); bw = d['steps_b'][:, :L-BURN].mean(1); gi = d['steps_g'][:, :L-BURN].mean(1)
        rows.append((N, N*fw.mean(), N*se(fw), N*bw.mean(), N*se(bw), N*gi.mean(), N*se(gi)))
    return sorted(rows)


def drift_chain():
    rows = {}
    for f in sorted(glob.glob(DATA + 'chain_*_L400_sw1.761.pkl')):
        d = pickle.load(open(f, 'rb')); N, L = d['N'], d['L']
        s = d['steps'][:, :L-BURN].mean(1)
        rows.setdefault(N, {})[d['mode']] = (N*s.mean(), N*se(s))
    return rows


def main():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    # (a) totals: forward against backward at N=64
    ax = axes[0]
    d = pickle.load(open(DATA + 'walk_N64_L400_sw1.761.pkl', 'rb'))
    XL, YL, ZL = d['steps_f'].sum(1), d['steps_b'].sum(1), d['steps_g'].sum(1)
    bins = np.linspace(-30, 8, 50)
    ax.hist(XL, bins, histtype='step', color=BLUE, lw=1.1, label='forward, $\\ln\\|J\\mathbf{v}\\|^2$')
    ax.hist(YL, bins, histtype='step', color=RED, lw=1.1, ls='--', label='backward, $\\ln\\|J^{\\top}\\mathbf{u}\\|^2$')
    ax.hist(ZL, bins, histtype='step', color=ORANGE, lw=1.0, ls=':', label='backward, independent $\\tilde W$')
    ax.axvline(A1['a1'] * d['L'] / d['N'], color=GREY, lw=0.6)
    ax.set_xlabel('$X_L$'); style.ylab(ax, 'networks'); ax.set_yticks([])
    ax.legend(loc='upper left', fontsize=6)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    style.panel(ax, 'a')
    # (b) N * drift against N
    ax = axes[1]
    ex = drift_explicit(); ch = drift_chain()
    Ns = sorted(set([r[0] for r in ex]) | set(ch))
    ax.axhline(A1['a1'], color=BLUE, lw=0.8, label='$a_1$ (forward theory)')
    ax.axhline(A1_GI, color=ORANGE, lw=0.8, ls=':', label='independent $\\tilde W$')
    for N, fm, fe, bm, be, gm, ge in ex:
        ax.errorbar(N * 0.93, fm, fe, fmt='o', color=BLUE, ms=4, mec=DEEP_BLUE, mew=0.4, capsize=0)
        ax.errorbar(N * 1.0, bm, be, fmt='o', color=RED, ms=4, mec=DEEP_RED, mew=0.4, capsize=0)
        ax.errorbar(N * 1.0, gm, ge, fmt='o', color=ORANGE, ms=4, mec='#C77A2A', mew=0.4, capsize=0)
    for N, modes in ch.items():
        if 'exact' in modes:
            ax.errorbar(N * 1.07, modes['exact'][0], modes['exact'][1], fmt='s', color=LIGHT_BLUE, ms=4, mec=DEEP_BLUE, mew=0.4, capsize=0)
        if 'gi' in modes:
            ax.errorbar(N * 1.07, modes['gi'][0], modes['gi'][1], fmt='s', color='#F3CFA0', ms=4, mec='#C77A2A', mew=0.4, capsize=0)
    ax.scatter([], [], marker='o', color=BLUE, edgecolors=DEEP_BLUE, s=14, label='forward, matrices')
    ax.scatter([], [], marker='o', color=RED, edgecolors=DEEP_RED, s=14, label='backward, matrices')
    ax.scatter([], [], marker='s', color=LIGHT_BLUE, edgecolors=DEEP_BLUE, s=14, label='backward, exact chain')
    ax.set_xscale('log'); ax.set_xlabel('$N$'); style.ylab(ax, '$N\\mu$')
    ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns])
    ax.xaxis.set_minor_formatter(plt.NullFormatter()); ax.set_xticks([], minor=True)
    ax.set_ylim(-2.05, -0.55); ax.set_yticks([-2.0, -1.75, -1.5, -1.25, -1.0])
    ax.legend(loc='upper center', fontsize=5.5, ncol=2, columnspacing=0.8)
    style.panel(ax, 'b')
    # (c) gradient-norm profile over 96 layers at N=64, from the exact chain
    ax = axes[2]
    Lp = 96
    for f, col, lab in ((DATA + 'chain_exact_N64_L400_sw1.761.pkl', BLUE, 'mean-field edge of chaos'),
                        (DATA + 'chain_exact_N64_L400_sw1.884.pkl', RED, 'finite-width edge of chaos (Crinit)')):
        d = pickle.load(open(f, 'rb')); L = d['L']
        s = d['steps'][:, L-Lp:][:, ::-1]           # steps from the top, downwards
        Y = np.cumsum(s, axis=1)                    # ln(G_l / G_L), l counted from the top
        k = np.arange(1, Lp + 1)
        ax.plot(k, np.exp(np.median(Y, 0)), color=col, lw=1.2, label=lab)
        ax.plot(k, np.exp(Y).mean(0), color=col, lw=0.9, ls='--')
        ax.fill_between(k, np.exp(np.quantile(Y, 0.25, 0)), np.exp(np.quantile(Y, 0.75, 0)), color=col, alpha=0.15, lw=0)
    ax.set_yscale('log'); ax.set_xlabel('layers below the output'); style.ylab(ax, '$\\|\\mathbf{g}^{(L-k)}\\|^2/\\|\\mathbf{g}^{(L)}\\|^2$')
    ax.legend(loc='lower left', fontsize=6)
    ax.text(0.03, 0.95, '$N=64$\nsolid: median, dashed: mean', transform=ax.transAxes, ha='left', va='top', fontsize=6.5)
    style.panel(ax, 'c')
    fig.tight_layout(w_pad=1.2)
    fig.savefig('/home/claude/paper/code/figures/fig_backward.pdf')
    fig.savefig('/home/claude/paper/code/figures/fig_backward.png', dpi=200)
    print('A1', A1['a1'], 'A1_GI', A1_GI)
    for r in ex: print('explicit', r)
    for N, mo in sorted(ch.items()): print('chain', N, mo)


if __name__ == '__main__':
    main()
