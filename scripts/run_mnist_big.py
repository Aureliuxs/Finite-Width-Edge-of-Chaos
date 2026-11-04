"""
Large-statistics comparison of initialisation rules on MNIST (Table I and Fig. 5 of the paper).

For every rule and depth: a pilot of 12 networks at each of three learning rates (0.3, 1, 3) x 0.005 x 32/L
selects the learning rate with the lowest mean final training loss; then 100 independent networks (seeds
0..99: initialisation and data order) are trained for two epochs at that rate.  All members of a job are
trained at once with eocfw.mnist_batched.  Results go to data/mnist/big.pkl as {(name, L, N): result}.

Usage:  python run_mnist_big.py [--threads 2] [--seeds 100] [--only name1,name2] [--grid 32,96] [--out big.pkl]
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eocfw.mnist import load_mnist  # noqa: E402
from eocfw.mnist_batched import train_batched  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "mnist")
os.makedirs(D, exist_ok=True)

RULES = ["tanh_standard", "tanh_crinit:1", "tanh_eoc:1", "orth_tanh_standard", "orth_tanh_crinit:1",
         "relu_he", "leaky_crinit:1", "leaky_he", "tanh_chaotic", "tanh_default", "tanh_finiteN", "relu_crinit"]
MULT = (0.3, 1.0, 3.0)
LR0 = 0.005


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=0, help="first seed of the main run (extra paired networks: 100, 200, ...)")
    ap.add_argument("--pilot", type=int, default=12)
    ap.add_argument("--only", default=None)
    ap.add_argument("--grid", default="32,96")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--out", default="big.pkl", help="results file in data/mnist (several files are merged by make_tables)")
    ap.add_argument("--lr_mult", type=float, default=None, help="skip the pilot and use this learning-rate multiplier")
    ap.add_argument("--tag", default="", help="suffix added to the rule name in the results key (for fixed-rate runs)")
    ap.add_argument("--data", default=os.path.join(D, "..", "mnist_raw"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_flush_denormal(True)
    data = load_mnist(a.data)
    path = os.path.join(D, a.out)
    R = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
    rules = RULES if a.only is None else a.only.split(",")

    def finished(key):
        """A configuration already present in any results file of the folder (several processes may share the work)."""
        import glob
        for f in glob.glob(os.path.join(D, "big*.pkl")):
            try:
                Q = pickle.load(open(f, "rb"))
            except Exception:
                continue
            if key in Q and Q[key]["test_acc"].size >= a.seeds:
                return True
        return False

    for L in [int(x) for x in a.grid.split(",")]:
        N = a.N
        for name in rules:
            key = (name + a.tag, L, N)
            if finished(key):
                continue
            t0 = time.time()
            if a.lr_mult is None:
                # pilot: 3 learning rates x `pilot` seeds in one batched run
                lrs = np.repeat([LR0 * m * 32 / L for m in MULT], a.pilot)
                seeds = [1000 + i for i in range(len(lrs))]
                pil = train_batched(data, name, L, N, seeds, lrs, epochs=a.epochs, log=lambda s: None)
                fl = pil["final_loss"].reshape(len(MULT), a.pilot)
                div = pil["diverged"].reshape(len(MULT), a.pilot)
                score = np.array([np.mean(f[~d]) if (~d).sum() >= a.pilot / 2 else np.inf for f, d in zip(fl, div)])
                best = int(np.argmin(score))
                mult = MULT[best]
                log(f"{name} L={L}: pilot losses {np.round(score, 4)} -> lr multiplier {mult} ({time.time() - t0:.0f}s)")
            else:
                mult, pil = a.lr_mult, None
            lr = LR0 * mult * 32 / L
            # main run
            res = train_batched(data, name, L, N, list(range(a.seed0, a.seed0 + a.seeds)), lr, epochs=a.epochs, eval_every=50,
                                eval_subset=2000, log=log)
            res["pilot"] = None if pil is None else dict(lrs=lrs, final_loss=pil["final_loss"], diverged=pil["diverged"],
                                                         test_acc=pil["test_acc"], steps_to_target=pil["steps_to_target"])
            res["lr_multiplier"] = mult
            res["name"] = name + a.tag
            R[key] = res
            pickle.dump(R, open(path, "wb"))
            acc = res["test_acc"]
            ok = ~np.isnan(acc)
            st = res["steps_to_target"]
            log(f"{name} L={L} N={N}: acc {np.mean(acc[ok]):.4f} +- {np.std(acc[ok]) / np.sqrt(ok.sum()):.4f} "
                f"(diverged {int((~ok).sum())}), steps to 85% {np.mean(st[st > 0]) if (st > 0).any() else float('nan'):.0f} "
                f"({int((st > 0).sum())}/{st.size}), {time.time() - t0:.0f}s")
    log("done")
