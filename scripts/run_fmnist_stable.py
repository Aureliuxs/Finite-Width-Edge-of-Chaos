"""
Fashion-MNIST, L = 96, ReLU family: the 10-epoch runs of Crinit's leaky ReLU and of the leaky ReLU with tuned slope
repeated at half the selected learning rate (x0.5 instead of x1), where the networks do not diverge (SM Sec. S8).
At the selected rate 20 of 50 Crinit and 8 of 50 tuned-slope networks diverged in the first epoch, although none of
the 6 pilot networks of either did.  Adds R['final'][(key, 0.5)]['long'] to data/fmnist/tuned_L96.pkl.
"""
import os
import pickle
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_fmnist_tuned as r  # noqa: E402
from eocfw.tuned import Batch, load_fashion  # noqa: E402

if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.set_flush_denormal(True)
    L = 96
    data = load_fashion(r.RAW)
    path = os.path.join(r.D, f"tuned_L{L}.pkl")
    C = r.configs(L)
    for k in ("leaky_crinit", "leaky_he:0.70"):
        R = pickle.load(open(path, "rb"))
        if "long" in R["final"].get((k, 0.5), {}):
            continue
        t0 = time.time()
        b = Batch(data, L, r.N, r.members_for(k, C[k], L, [0.5], range(r.N_FINAL_LONG)))
        for _ in range(r.E_LONG):
            b.run_epoch(val=False, test=True)
        R = pickle.load(open(path, "rb"))
        R["final"].setdefault((k, 0.5), {})["long"] = b.summary()
        pickle.dump(R, open(path, "wb"))
        s = b.summary()
        r.log(f"L=96 {k} lr x0.5, 10 epochs: test {s['test'][-1][~s['diverged']].mean():.4f}, "
              f"diverged {int(s['diverged'].sum())} ({time.time() - t0:.0f}s)")
