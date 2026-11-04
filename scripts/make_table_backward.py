"""Table of drifts for the backward note: explicit matrices (forward, backward, independent copies) and the
backward chain (exact, independent, projection only) at each width, against the theory."""
import sys, glob, pickle
import numpy as np

sys.path.insert(0, '/home/claude/crinit')
import crinit
from crinit.finite_width import a1_b1

DATA = '/home/claude/paper/code/data/backward/'
BURN = 100
sw2, q = crinit.eoc_at_bias('tanh', 0.05)
act = crinit.get('tanh'); m = act.moments(q); r = a1_b1(act, q)
P, V, M, dP = m['P'], m['V'], m['EphiDphi2'], m['dP']
NEr2 = sw2 * M / (q * P * (1 - r['Fp']**2))
a1 = r['a1']; a1_gi = a1 - 2 * q * (dP / P) * NEr2; a1_proj = a1_gi - M / (V * P)


def fmt(mean, err):
    return f"${mean:.3f}({int(round(err*1000)):d})$"


def main():
    rows = {}
    for f in glob.glob(DATA + 'walk_N*_L400_sw1.761.pkl'):
        d = pickle.load(open(f, 'rb')); N, L = d['N'], d['L']
        for key, s in (('fwd', d['steps_f'][:, BURN:]), ('bwd', d['steps_b'][:, :L-BURN]), ('gi_m', d['steps_g'][:, :L-BURN])):
            per = s.mean(1); rows.setdefault(N, {})[key] = (N * per.mean(), N * per.std(ddof=1) / np.sqrt(len(per)))
        rows[N]['M'] = d['M']
    for f in glob.glob(DATA + 'chain_*_L400_sw1.761.pkl'):
        d = pickle.load(open(f, 'rb')); N, L = d['N'], d['L']
        per = d['steps'][:, :L-BURN].mean(1)
        rows.setdefault(N, {})[d['mode']] = (N * per.mean(), N * per.std(ddof=1) / np.sqrt(len(per)))
        if d['mode'] == 'exact':
            rows[N]['ahat'] = N * (d['ahat'][:, :L-BURN]**2).mean()
    lines = [r"\begin{tabular}{lcccccccc}", r"\toprule",
             r"$N$ & forward, matrices & backward, matrices & backward, chain & $N\E[\hat a^2]$ & independent $\tilde W$, matrices & independent $\tilde W$, chain & projection only, chain \\",
             r"\midrule"]
    for N in sorted(rows):
        d = rows[N]
        c = [str(N)]
        for key in ('fwd', 'bwd', 'exact'):
            c.append(fmt(*d[key]) if key in d else '--')
        c.append(f"${d['ahat']:.3f}$" if 'ahat' in d else '--')
        for key in ('gi_m', 'gi', 'proj'):
            c.append(fmt(*d[key]) if key in d else '--')
        lines.append(' & '.join(c) + r' \\')
    lines += [r"\midrule", f"theory & ${a1:.3f}$ & ${a1:.3f}$ & ${a1:.3f}$ & ${NEr2:.3f}$ & ${a1_gi:.3f}$ & ${a1_gi:.3f}$ & ${a1_proj:.3f}$ \\\\",
              r"\bottomrule", r"\end{tabular}"]
    open('/home/claude/paper/latex/notes/table_backward.tex', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
