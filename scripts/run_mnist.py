"""
MNIST training runs.  Usage:  python run_mnist.py <job> [--threads 1]

Jobs
  main        the comparison of Table I / Fig. 5: all rules at (L, N) = (32, 64) and (96, 64), 3 seeds,
              learning rate chosen from 3 values by the final training loss
  gamma       scan of the compression gamma for the tanh sweet spot and the shaped leaky ReLU at five aspect
              ratios N/L (2 seeds, 2 learning rates)
  finiteN     standard against finite-N critical point at L = 96 for N = 64 and 32 (6 seeds) and the scan
              sigma_w^2 = 1 + k/N (tanh, N = 32)
Results are appended to data/mnist/<job>.pkl as a dict {(name, L, N, seed): log}.
"""
import argparse
import os
import pickle
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eocfw.mnist import load_mnist, train_best  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "mnist")
os.makedirs(D, exist_ok=True)

JOBS = {
    "main": dict(
        grid=[(L, 64) for L in (32, 96)],
        inits=["tanh_standard", "tanh_finiteN", "tanh_sweet:1", "tanh_sweetN:1", "tanh_chaotic", "relu_he",
               "relu_finiteN", "leaky:1", "leaky:2.3026", "linear", "orth_tanh_standard", "orth_tanh_sweet:1",
               "orth_linear"],
        seeds=[0, 1, 2], lr_multipliers=(0.3, 1.0, 3.0), epochs=2),
    "gamma": dict(
        grid=[(16, 64), (32, 64), (64, 64), (96, 64), (96, 32)],
        inits=[f"{b}:{g}" for b in ("tanh_sweet", "leaky") for g in (0.25, 0.5, 1, 2.3026, 3.5)],
        seeds=[0, 1], lr_multipliers=(0.3, 1.0), epochs=2),
    "finiteN": dict(
        grid=[(96, 64), (96, 32)],
        inits=["tanh_standard", "tanh_finiteN", "relu_he", "relu_finiteN"] + [f"tanh_k:{k}" for k in (1, 2, 4, 8, 16)],
        seeds=[10, 11, 12, 13, 14, 15], lr_multipliers=(0.3, 1.0), epochs=2),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=list(JOBS))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--data", default=os.path.join(D, "..", "mnist_raw"))
    ap.add_argument("--only", default=None, help="comma-separated subset of init names to run")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    job = JOBS[a.job]
    data = load_mnist(a.data)
    path = os.path.join(D, f"{a.job}.pkl")
    R = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
    inits = job["inits"] if a.only is None else a.only.split(",")
    for (L, N) in job["grid"]:
        for name in inits:
            if a.job == "finiteN" and name.startswith("tanh_k") and N != 32:
                continue
            for seed in job["seeds"]:
                key = (name, L, N, seed)
                if key in R:
                    continue
                t = time.time()
                try:
                    r = train_best(data, name, L, N, seed, lr_multipliers=job["lr_multipliers"], epochs=job["epochs"])
                except ValueError as e:          # compression target not reachable at this depth
                    print(f"skip {name} at L={L}: {e}", flush=True)
                    break
                R[key] = r
                acc = r["test_acc"][-1] if r["test_acc"] else float("nan")
                print(f"L={L:3d} N={N:3d} {name:22s} seed={seed} lr={r['lr']:.4f}: acc {acc:.4f} steps {r['steps_to_target']} "
                      f"{'DIV ' if r['diverged'] else ''}({time.time() - t:.0f}s)", flush=True)
                pickle.dump(R, open(path, "wb"))
    print("done", a.job)
