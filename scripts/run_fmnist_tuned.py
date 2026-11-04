"""
Fully tuned comparison of initialisation rules on Fashion-MNIST (SM Sec. S8).

Same networks and optimiser as Table I of the paper (width N = 64, depth L = 32 and 96, SGD with momentum 0.9,
batch 128, no normalisation, no skip connections), but every rule gets its best learning rate AND its best
initialisation gain, chosen on a held-out validation set, separately for a short (2 epochs) and a long
(10 epochs) training budget.  Crinit gets only the learning-rate search: its variances come from the theory.

Search space (per depth):
  learning rate      lr = 0.005 * 32/L * m,  m in {1/8, 1/4, 1/2, 1, 2, 4, 8}            (all rules)
  tanh, Gaussian     zero-bias sigma_w^2 in {1/3, 1/2, 2^-1/2, 2^-1/4, 1, 2^1/4, 2^1/2, 2^3/4, 2, 25/9, 4}
                     -- this grid contains PyTorch's nn.Linear default (1/3), LeCun/Glorot (1), He (2) and
                     Xavier with PyTorch's tanh gain 5/3 (25/9 = (5/3)^2), finely spaced where the best values lie;
                     the typical-preserving sigma_c^2(N); the mean-field edge of chaos at the Crinit point; Crinit
  tanh, orthogonal   zero-bias sigma_w^2 in {0.71, ..., 2}; orthogonal Crinit
  ReLU family        ReLU with sigma_w^2 = 2 f, f in {0.9, 0.95, 1, e^(5/(2N)), 1.1} (He and typical-preserving);
                     leaky ReLU at He's variance 2/(1+a^2) with slope a in {0.01, 0.1, 0.2, 0.3, 0.5, 0.7};
                     Crinit's shaped leaky ReLU
  Crinit, std. 1st   Crinit with its first layer initialised like the hidden layers (as the standard rules do) instead
                     of rescaled so that the first hidden layer starts at q* (added after the diagnostic runs of
                     run_fmnist_firstlayer.py; learning-rate search only)
  Crinit, tuned      Crinit's own parameter, the compression gamma over the depth, in {0.25, 0.35, 0.5, 0.7, 1,
                     1.4, 2} for each of the three Crinit rules (points that coincide are run once: below the
                     zero-bias point of the finite-width line the tanh rule stays at that point)

First layer: Crinit's rules (and the mean-field edge-of-chaos rule) start the first hidden layer at the steady
signal q*.  From this version on this is done as the crinit package does it, by drawing the first-layer weights with
variance (q* - sigma_b^2)/d_in ('weight'); the earlier runs rescaled the input instead ('input'), which gives the
same forward pass but a smaller gradient for the first layer.  The earlier runs are kept under the keys "...@in".

Three comparisons are read off the same runs: standard rules at their usual variance against Crinit (learning rate
tuned for all); standard rules with tuned gain against Crinit; standard rules with tuned gain against Crinit with
tuned gamma.

Protocol (successive halving, then fresh networks):
  stage 1   every (configuration, learning rate) with 6 networks for 2 epochs; validation accuracy after
            every epoch (a diverged network counts as chance, 0.1)
  stage 2   for every configuration the two best learning rates continue to 10 epochs (configurations with
            validation accuracy below 0.3 continue only with their best rate)
  extend    a configuration that trains and whose best rate is at an end of the grid gets the next rate beyond
            it (6 networks, 10 epochs), repeated until the best rate is inside the grid
  confirm   for each budget and rule the two best settings of the pilot get 6 more networks (same length), so
            that rare divergence is seen before the choice
  select    for each budget E in (2, 10) and each rule, of these two the setting with the best mean validation
            accuracy over its 12 networks at epoch E (a diverged network counting as 0.1)
  final     100 new networks (seeds 0-99) at the 2-epoch choice, trained for 2 epochs; 50 networks at the
            10-epoch choice trained for 10 epochs (the first 50 of the 100 are continued when the two choices
            coincide).  Test accuracy after every epoch, learning curve, steps to 60-90% training accuracy.

Results: data/fmnist/tuned_L{L}.pkl.  The script resumes where it stopped.

Usage:  python run_fmnist_tuned.py [--threads 2] [--depths 32,96] [--stage all|pilot|extend|confirm|final]
"""
import argparse
import math
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eocfw.finite_width import sigma_c2_smooth  # noqa: E402
from eocfw.mnist import _crinit  # noqa: E402
from eocfw.tuned import Batch, load_fashion  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "data", "fmnist")
RAW = os.path.join(HERE, "..", "data", "fmnist_raw")
N = 64
LR0 = 0.005
LR_MULT = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
PILOT = 6
E_SHORT, E_LONG = 2, 10
N_FINAL_SHORT, N_FINAL_LONG = 100, 50
CHANCE = 0.1
TANH_GRID = sorted({1 / 3, 0.5, 2 ** -0.5, 2 ** -0.25, 1.0, 2 ** 0.25, 2 ** 0.5, 2 ** 0.75, 2.0, 25 / 9, 4.0})
ORTH_GRID = [round(2 ** (k / 4), 4) for k in range(-2, 5)]                               # 0.71 ... 2
RELU_GAIN = (0.9, 0.95, 1.0, math.exp(5 / (2 * N)), 1.1)
LEAKY_SLOPES = (0.01, 0.1, 0.2, 0.3, 0.5, 0.7)
CRINIT_GAMMAS = (0.25, 0.35, 0.5, 0.7, 1.4, 2.0)                  # besides the default gamma = 1
CONFIRM = 6


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def configs(L):
    """All initialisation configurations at depth L: key -> dict(family, act, alpha, sw2, sb2, q1, ens)."""
    c = _crinit()
    C = {}
    for s in TANH_GRID:
        C[f"tanh_zb:{s:.4f}"] = dict(family="tanh", act="tanh", sw2=s, sb2=0.0, q1=s, ens="gaussian")
    C["tanh_typ"] = dict(family="tanh", act="tanh", sw2=sigma_c2_smooth(N), sb2=0.0, q1=1.0, ens="gaussian")
    r = c.crinit("tanh", depth=L, width=N, finite_width=False)
    C["tanh_eoc"] = dict(family="tanh", act="tanh", sw2=r.sigma_w2, sb2=r.sigma_b2, q1=r.q_star, ens="gaussian")
    r = c.crinit("tanh", depth=L, width=N)
    C["tanh_crinit"] = dict(family="tanh", act="tanh", sw2=r.sigma_w2, sb2=r.sigma_b2, q1=r.q_star, ens="gaussian")
    for s in ORTH_GRID:
        C[f"orth_zb:{s:.4f}"] = dict(family="orth", act="tanh", sw2=s, sb2=0.0, q1=s, ens="orthogonal")
    r = c.crinit("tanh", depth=L, width=N, ensemble="orthogonal")
    C["orth_crinit"] = dict(family="orth", act="tanh", sw2=r.sigma_w2, sb2=r.sigma_b2, q1=r.q_star, ens="orthogonal")
    for f in RELU_GAIN:
        C[f"relu:{f:.4f}"] = dict(family="relu", act="leaky", alpha=0.0, sw2=2 * f, sb2=0.0, q1=2 * f, ens="gaussian")
    for a in LEAKY_SLOPES:
        s = 2 / (1 + a * a)
        C[f"leaky_he:{a:.2f}"] = dict(family="relu", act="leaky", alpha=a, sw2=s, sb2=0.0, q1=s, ens="gaussian")
    r = c.crinit("leaky_relu", depth=L, width=N)
    C["leaky_crinit"] = dict(family="relu", act="leaky", alpha=r.alpha, sw2=r.sigma_w2, sb2=0.0, q1=1.0, ens="gaussian")
    C["tanh_crinit"]["gamma"] = c.crinit("tanh", depth=L, width=N).gamma
    C["orth_crinit"]["gamma"] = 1.0
    C["leaky_crinit"]["gamma"] = 1.0
    # Crinit with the first layer initialised like the hidden layers (first hidden signal sigma_w^2 + sigma_b^2, as for
    # the standard rules) instead of rescaled to start at q*: crinit's first_layer=False
    for base in ("tanh_crinit", "orth_crinit", "leaky_crinit"):
        C[base + "_std1"] = dict(C[base], q1=C[base]["sw2"] + C[base]["sb2"])
    # Crinit with its compression gamma as a tuned parameter; a point that coincides with one already present is
    # not added again (for tanh, every gamma below that of the zero-bias point gives the zero-bias point)
    for base, fam, act, kw in (("tanh_crinit", "tanh", "tanh", dict(activation="tanh")),
                               ("orth_crinit", "orth", "tanh", dict(activation="tanh", ensemble="orthogonal")),
                               ("leaky_crinit", "relu", "leaky", dict(activation="leaky_relu"))):
        for g in CRINIT_GAMMAS:
            r = c.crinit(kw["activation"], depth=L, width=N, gamma=g, **{k: v for k, v in kw.items() if k != "activation"})
            if act == "leaky":
                cfg = dict(family=fam, act="leaky", alpha=r.alpha, sw2=r.sigma_w2, sb2=0.0, q1=1.0, ens="gaussian")
            else:
                cfg = dict(family=fam, act="tanh", sw2=r.sigma_w2, sb2=r.sigma_b2, q1=r.q_star, ens=kw.get("ensemble", "gaussian"))
            cfg["gamma"] = r.gamma
            same = lambda a, b: a["ens"] == b["ens"] and a["act"] == b["act"] and all(
                abs(a.get(x, 0.0) - b.get(x, 0.0)) < 1e-6 for x in ("sw2", "sb2", "q1", "alpha"))
            if not any(same(cfg, C[k]) for k in C if k == base or k.startswith(base + ":")):
                C[f"{base}:{g:g}"] = cfg
    # the tuned zero-bias rule with Crinit's first layer (first hidden layer at the steady signal of that rule)
    for sw2 in (2 ** 0.25, 2 ** 0.5):
        k = f"tanh_zb:{sw2:.4f}"
        C[k + ":steady"] = dict(C[k], q1=c.fixed_point("tanh", sw2, 0.0))
    # every rule whose first layer is rescaled, as implemented in the earlier runs (input scaling)
    # (these "@in" entries are bookkeeping for runs that already exist; they are never run anew)
    for k in [k for k in C if abs(C[k]["q1"] - C[k]["sw2"] - C[k]["sb2"]) > 0.05 and ":steady" not in k]:
        C[k + "@in"] = dict(C[k], first_layer="input")
    return C


def reported_rules(L=None):
    """The rows of the results table: label -> (family, list of configuration keys searched over).
    A rule with one key gets only the learning-rate search; one with several keys also gets the gain search
    (standard rules) or the search over the compression gamma (Crinit, tuned).  The keys of the tuned Crinit rows
    depend on the depth L (coinciding points are run once); without L they are left out."""
    tuned_crinit = {}
    if L is not None:
        C = configs(L)
        for lab, base, fam in (("Crinit, tuned $\\gamma$", "tanh_crinit", "tanh"),
                               ("Crinit, orthogonal, tuned $\\gamma$", "orth_crinit", "orth"),
                               ("Crinit, leaky ReLU, tuned $\\gamma$", "leaky_crinit", "relu")):
            tuned_crinit[lab] = (fam, [base] + [k for k in C if k.startswith(base + ":") and "@" not in k])
    zb = lambda v: f"tanh_zb:{v:.4f}"
    t_all = [f"tanh_zb:{s:.4f}" for s in TANH_GRID] + ["tanh_typ"]
    o_all = [f"orth_zb:{s:.4f}" for s in ORTH_GRID]
    r_all = [f"relu:{f:.4f}" for f in RELU_GAIN]
    l_all = [f"leaky_he:{a:.2f}" for a in LEAKY_SLOPES]
    rules = {
        "PyTorch default": ("tanh", [zb(1 / 3)]),
        "LeCun": ("tanh", [zb(1.0)]),
        "He": ("tanh", [zb(2.0)]),
        "Xavier, gain 5/3": ("tanh", [zb(25 / 9)]),
        "typical-preserving": ("tanh", ["tanh_typ"]),
        "best standard rule, tuned gain": ("tanh", t_all),
        "edge of chaos, mean field": ("tanh", ["tanh_eoc"]),
        "edge of chaos, mean field, input-scaled first layer": ("tanh", ["tanh_eoc@in"]),
        "Crinit": ("tanh", ["tanh_crinit"]),
        "orthogonal": ("orth", ["orth_zb:1.0000"]),
        "orthogonal, tuned gain": ("orth", o_all),
        "Crinit, orthogonal": ("orth", ["orth_crinit"]),
        "ReLU, He": ("relu", ["relu:1.0000"]),
        "ReLU, typical-preserving": ("relu", [f"relu:{math.exp(5 / (2 * N)):.4f}"]),
        "ReLU, tuned gain": ("relu", r_all),
        "leaky ReLU (0.01), He": ("relu", ["leaky_he:0.01"]),
        "leaky ReLU, tuned slope": ("relu", l_all),
        "Crinit, leaky ReLU": ("relu", ["leaky_crinit"]),
    }
    if tuned_crinit:
        out = {}
        for lab, v in rules.items():
            out[lab] = v
            if lab == "Crinit":
                out["Crinit, tuned $\\gamma$"] = tuned_crinit["Crinit, tuned $\\gamma$"]
                out["Crinit, standard first layer"] = ("tanh", ["tanh_crinit_std1"])
                out["Crinit, input-scaled first layer"] = ("tanh", ["tanh_crinit@in"])
                out["zero bias, tuned gain, steady first layer"] = ("tanh", ["tanh_zb:1.1892:steady", "tanh_zb:1.4142:steady"])
            if lab == "Crinit, orthogonal":
                out["Crinit, orthogonal, tuned $\\gamma$"] = tuned_crinit["Crinit, orthogonal, tuned $\\gamma$"]
                out["Crinit, orthogonal, standard first layer"] = ("orth", ["orth_crinit_std1"])
                out["Crinit, orthogonal, input-scaled first layer"] = ("orth", ["orth_crinit@in"])
            if lab == "Crinit, leaky ReLU":
                out["Crinit, leaky ReLU, tuned $\\gamma$"] = tuned_crinit["Crinit, leaky ReLU, tuned $\\gamma$"]
                out["Crinit, leaky ReLU, standard first layer"] = ("relu", ["leaky_crinit_std1"])
                out["Crinit, leaky ReLU, input-scaled first layer"] = ("relu", ["leaky_crinit@in"])
        rules = out
    return rules


def score(val):
    """Mean validation accuracy over networks, a diverged network counting as chance."""
    return np.mean(np.where(np.isnan(val), CHANCE, val))


def members_for(key, cfg, L, lr_mults, seeds):
    out = []
    for m in lr_mults:
        for s in seeds:
            d = dict(cfg)
            d.update(key=key, lr_mult=m, lr=LR0 * 32 / L * m, seed=s)
            out.append(d)
    return out


def pilot(data, L, R, path, threads):
    """Stages 1 and 2 for every configuration not yet in R['pilot']."""
    C = configs(L)
    R.setdefault("pilot", {})
    maxK = 252 if L <= 32 else 126                                   # members per batch (memory)
    per = len(LR_MULT) * PILOT
    todo = [k for k in C if k not in R["pilot"] and not k.endswith("@in")]
    acts = {}
    for k in todo:
        acts.setdefault(C[k]["act"], []).append(k)
    for act, keys in acts.items():
        for i in range(0, len(keys), max(1, maxK // per)):
            chunk = keys[i:i + max(1, maxK // per)]
            t0 = time.time()
            mem = []
            for j, k in enumerate(chunk):
                mem += members_for(k, C[k], L, LR_MULT, [1000 + 10 * j + s for s in range(PILOT)])
            b = Batch(data, L, N, mem)
            for _ in range(E_SHORT):
                b.run_epoch(val=True, test=False)
            s1 = b.summary()
            # stage 2: the two best learning rates of each configuration continue
            keep = []
            for k in chunk:
                sc = []
                for m in LR_MULT:
                    idx = [n for n, mm in enumerate(b.members) if mm["key"] == k and mm["lr_mult"] == m]
                    sc.append((score(s1["val"][E_SHORT - 1][idx]), m, idx))
                sc.sort(key=lambda t: -t[0])
                nkeep = 2 if sc[0][0] >= 0.3 else 1
                for t in sc[:nkeep]:
                    keep += t[2]
            b.keep(keep)
            for _ in range(E_LONG - E_SHORT):
                b.run_epoch(val=True, test=False)
            s2 = b.summary()
            for k in chunk:
                i1 = [n for n, mm in enumerate(s1["members"]) if mm["key"] == k]
                i2 = [n for n, mm in enumerate(s2["members"]) if mm["key"] == k]
                cut = lambda S, ii: dict(val=S["val"][:, ii], loss=S["loss"][:, ii], first=S["first"][ii],
                                         diverged=S["diverged"][ii], seed=np.array([S["members"][n]["seed"] for n in ii]))
                R["pilot"][k] = dict(cfg=C[k], targets=s1["targets"], steps_per_epoch=s1["steps_per_epoch"],
                                     stage1=cut(s1, i1), lr1=np.array([s1["members"][n]["lr_mult"] for n in i1]),
                                     stage2=cut(s2, i2), lr2=np.array([s2["members"][n]["lr_mult"] for n in i2]))
                best = max(LR_MULT, key=lambda m: score(s1["val"][E_SHORT - 1][[n for n in i1 if s1["members"][n]["lr_mult"] == m]]))
                log(f"L={L} {k}: 2-epoch val {score(s1['val'][E_SHORT - 1][[n for n in i1 if s1['members'][n]['lr_mult'] == best]]):.4f} "
                    f"at lr x{best}")
            pickle.dump(R, open(path, "wb"))
            log(f"L={L} chunk {chunk[0]}..{chunk[-1]} done ({time.time() - t0:.0f}s, {len(mem)} members)")


def extend(data, L, R, path):
    """If a configuration that trains (validation accuracy >= 0.3) has its best learning rate at an end of the
    grid, for either budget, train the next rate beyond it (6 networks, 10 epochs) and repeat until the best rate
    is inside the grid."""
    C = configs(L)
    while True:
        changed = False
        for k, p in R["pilot"].items():
            lrs = sorted(set(p["lr1"].tolist()))
            sc1 = {m: score(p["stage1"]["val"][E_SHORT - 1][p["lr1"] == m]) for m in lrs}
            sc2 = {m: score(p["stage2"]["val"][E_LONG - 1][p["lr2"] == m]) for m in set(p["lr2"].tolist())}
            new = set()
            for sc in (sc1, sc2):
                b = max(sc, key=sc.get)
                if sc[b] < 0.3:
                    continue
                if b == lrs[0]:
                    new.add(b / 2)
                if b == lrs[-1]:
                    new.add(b * 2)
            new = sorted(m for m in new if m not in lrs)
            if not new:
                continue
            t0 = time.time()
            seeds = [5000 + 10 * len(lrs) + s for s in range(PILOT)]
            b = Batch(data, L, N, members_for(k, C[k], L, new, seeds))
            for _ in range(E_LONG):
                b.run_epoch(val=True, test=False)
            S = b.summary()
            lr_new = np.array([mm["lr_mult"] for mm in S["members"]])
            seed_new = np.array([mm["seed"] for mm in S["members"]])
            for st, E in (("stage1", E_SHORT), ("stage2", E_LONG)):
                q = p[st]
                q["val"] = np.concatenate([q["val"], S["val"][:E]], 1)
                q["loss"] = np.concatenate([q["loss"], S["loss"][:E]], 1)
                q["first"] = np.concatenate([q["first"], S["first"]], 0)
                q["diverged"] = np.concatenate([q["diverged"], S["diverged"]])
                q["seed"] = np.concatenate([q["seed"], seed_new])
            p["lr1"] = np.concatenate([p["lr1"], lr_new])
            p["lr2"] = np.concatenate([p["lr2"], lr_new])
            p.setdefault("extended", []).extend(new)
            pickle.dump(R, open(path, "wb"))
            log(f"L={L} {k}: extended learning rates {new}: 2-epoch val "
                f"{[round(float(score(S['val'][E_SHORT - 1][lr_new == m])), 4) for m in new]}, 10-epoch val "
                f"{[round(float(score(S['val'][E_LONG - 1][lr_new == m])), 4) for m in new]} ({time.time() - t0:.0f}s)")
            changed = True
        if not changed:
            break


def pilot_scores(R, k, E):
    """Pilot validation score at epoch E of every learning rate of configuration k: {m: (score, n networks)}."""
    p = R["pilot"][k]
    val, lrs = (p["stage1"]["val"][E - 1], p["lr1"]) if E == E_SHORT else (p["stage2"]["val"][E - 1], p["lr2"])
    return {float(m): (score(val[lrs == m]), int((lrs == m).sum())) for m in np.unique(lrs)}


def candidates(R, L, E, n=2):
    """For every rule, its n best (key, m) at budget E by pilot score; rules whose best setting does not train
    (score below 0.3) get no candidates."""
    out = {}
    for label, (fam, keys) in reported_rules(L).items():
        sc = sorted(((v[0], k, m) for k in keys for m, v in pilot_scores(R, k, E).items()), reverse=True)
        if sc and sc[0][0] >= 0.3:
            out[label] = [(k, m) for _, k, m in sc[:n]]
    return out


def confirm(data, L, R, path):
    """Six more networks (seeds 7000-7005) for every candidate setting, trained for the longest budget at which it
    is a candidate.  Stored in R['confirm'][(key, m)] = dict(val (E, 6), diverged, epochs)."""
    C = configs(L)
    R.setdefault("confirm", {})
    need = {}
    for E in (E_SHORT, E_LONG):
        for label, cands in candidates(R, L, E).items():
            for km in cands:
                need[km] = max(need.get(km, 0), E)
    todo = [(k, m, E) for (k, m), E in need.items() if R["confirm"].get((k, m), {}).get("epochs", 0) < E]
    maxK = 252 if L <= 32 else 126
    groups = {}
    for k, m, E in todo:
        groups.setdefault((C[k]["act"], E), []).append((k, m))
    for (act, E), items in groups.items():
        per = max(1, maxK // CONFIRM)
        for i in range(0, len(items), per):
            chunk = items[i:i + per]
            t0 = time.time()
            mem = []
            for k, m in chunk:
                mem += members_for(k, C[k], L, [m], [7000 + s for s in range(CONFIRM)])
            b = Batch(data, L, N, mem)
            for _ in range(E):
                b.run_epoch(val=True, test=False)
            S = b.summary()
            for k, m in chunk:
                idx = [n for n, mm in enumerate(S["members"]) if mm["key"] == k and mm["lr_mult"] == m]
                R["confirm"][(k, m)] = dict(val=S["val"][:, idx], diverged=S["diverged"][idx], epochs=E)
            pickle.dump(R, open(path, "wb"))
            log(f"L={L} confirmed {len(chunk)} settings for {E} epochs ({act}; {time.time() - t0:.0f}s): " + ", ".join(
                f"{k} x{m}: {score(R['confirm'][(k, m)]['val'][E - 1]):.4f}"
                + (f" ({int(R['confirm'][(k, m)]['diverged'].sum())} div)" if R['confirm'][(k, m)]['diverged'].any() else "")
                for k, m in chunk))


def combined_score(R, k, m, E):
    """Validation score at epoch E over the pilot and the confirmation networks of setting (k, m)."""
    p = R["pilot"][k]
    val, lrs = (p["stage1"]["val"][E - 1], p["lr1"]) if E == E_SHORT else (p["stage2"]["val"][E - 1], p["lr2"])
    v = [val[lrs == m]]
    c = R.get("confirm", {}).get((k, m))
    if c is not None and c["epochs"] >= E:
        v.append(c["val"][E - 1])
    v = np.concatenate(v)
    return score(v), v.size


def select(R, E, L=None):
    """For budget E: rule label -> (key, lr_mult, validation score).  Of the two best pilot settings of each rule,
    the one with the higher score over its pilot and confirmation networks."""
    out = {}
    cands = candidates(R, L, E)
    for label, (fam, keys) in reported_rules(L).items():
        if label in cands:
            best = max(((*combined_score(R, k, m, E), k, m) for k, m in cands[label]), key=lambda t: t[0])
            out[label] = (best[2], best[3], best[0])
        else:                                       # does not train: the best pilot setting
            sc = max(((v[0], k, m) for k in keys for m, v in pilot_scores(R, k, E).items()))
            out[label] = (sc[1], sc[2], sc[0])
    return out


def final(data, L, R, path):
    C = configs(L)
    R.setdefault("final", {})
    sel = {E: select(R, E, L) for E in (E_SHORT, E_LONG)}
    R["select"] = sel
    pickle.dump(R, open(path, "wb"))
    jobs = {}
    for E in (E_SHORT, E_LONG):
        for label, (k, m, _) in sel[E].items():
            jobs.setdefault((k, m), set()).add(E)
    for (k, m), Es in jobs.items():
        have = R["final"].get((k, m), {})
        need_short = E_SHORT in Es and "short" not in have
        need_long = E_LONG in Es and "long" not in have
        if not (need_short or need_long):
            continue
        t0 = time.time()
        cfg = C[k]
        # a rule whose pilot networks do not train (validation accuracy below 0.3 at the budget) gets 20 / 10 networks
        trains = {E: max(sv for (kk, mm, sv) in sel[E].values() if kk == k and mm == m) >= 0.3 for E in Es}
        n_short = N_FINAL_SHORT if trains.get(E_SHORT, False) else 20
        n_long = N_FINAL_LONG if trains.get(E_LONG, False) else 10
        n0 = n_short if need_short else n_long
        b = Batch(data, L, N, members_for(k, cfg, L, [m], range(n0)))
        res = dict(have)
        for e in range(E_SHORT):
            b.run_epoch(val=False, test=True, curve_every=50 if need_short else 0)
        if need_short:
            res["short"] = b.summary()
        if need_long:
            if len(b.members) > n_long:
                b.keep(range(n_long))
            for e in range(E_LONG - E_SHORT):
                b.run_epoch(val=False, test=True)
            res["long"] = b.summary()
        R["final"][(k, m)] = res
        pickle.dump(R, open(path, "wb"))
        s = res["short"] if need_short else res["long"]
        acc = s["test"][E_SHORT - 1]
        log(f"L={L} final {k} lr x{m} ({'2' if need_short else ''}{',' if need_short and need_long else ''}"
            f"{'10' if need_long else ''} epochs): 2-epoch test {np.nanmean(acc):.4f} +- "
            f"{np.nanstd(acc) / np.sqrt(np.isfinite(acc).sum()):.4f}"
            + (f", 10-epoch {np.nanmean(res['long']['test'][-1]):.4f} ({int(res['long']['diverged'].sum())} div)"
               if need_long else "") + f" ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--depths", default="32,96")
    ap.add_argument("--stage", default="all", choices=["all", "pilot", "extend", "confirm", "final"])
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_flush_denormal(True)
    os.makedirs(D, exist_ok=True)
    data = load_fashion(RAW)
    for L in [int(x) for x in a.depths.split(",")]:
        path = os.path.join(D, f"tuned_L{L}.pkl")
        R = pickle.load(open(path, "rb")) if os.path.exists(path) else dict(L=L, N=N)
        if a.stage in ("all", "pilot"):
            pilot(data, L, R, path, a.threads)
        if a.stage in ("all", "pilot", "extend"):
            extend(data, L, R, path)
        if a.stage in ("all", "confirm"):
            confirm(data, L, R, path)
        if a.stage in ("all", "final"):
            final(data, L, R, path)
    log("done")
