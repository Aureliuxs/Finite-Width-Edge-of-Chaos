"""
Fashion-MNIST: is the first-layer scale the difference between Crinit and the tuned zero-bias rules?
The standard rules give the first hidden layer the signal q_1 = sigma_w^2 (+ sigma_b^2); Crinit rescales the input so
that q_1 = q*, the steady signal.  Each job swaps this one choice at otherwise fixed settings (the learning rate of the
rule it is compared with, seeds 0-99 or 0-49) and is compared with the runs of run_fmnist_tuned.py (SM Sec. S8).

  python run_fmnist_firstlayer.py 32     L = 32, two epochs: Crinit (gamma = 0.7 and 1) with q_1 = sigma_w^2 + sigma_b^2,
                                         and the tuned zero-bias sigma_w^2 = 1.41 with q_1 = q*
  python run_fmnist_firstlayer.py 96     L = 96: Crinit with q_1 = sigma_w^2 (ten epochs, 50 networks) and orthogonal
                                         Crinit with q_1 = sigma_w^2 + sigma_b^2 (two epochs, 100 networks)
Results: data/fmnist/firstlayer_L{L}.pkl.
"""
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_fmnist_tuned as r  # noqa: E402
from eocfw.tuned import Batch, load_fashion  # noqa: E402

if __name__ == "__main__":
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
    torch.set_flush_denormal(True)
    data = load_fashion(r.RAW)
    C = r.configs(L)
    std_first = lambda c: dict(c, q1=c["sw2"] + c["sb2"])
    if L == 32:
        b = C["tanh_zb:1.4142"]
        jobs = {"crinit_0.7_q1=sw2": (std_first(C["tanh_crinit:0.7"]), 1.0, r.E_SHORT, 100),
                "zb_1.41_q1=q*": (dict(b, q1=r._crinit().fixed_point("tanh", b["sw2"], 0.0)), 1.0, r.E_SHORT, 100),
                "crinit_q1=sw2": (std_first(C["tanh_crinit"]), 1.0, r.E_SHORT, 100)}
    else:
        jobs = {"crinit_q1=sw2": (std_first(C["tanh_crinit"]), 0.5, r.E_LONG, 50),
                "orth_crinit_q1=sw2": (std_first(C["orth_crinit"]), 2.0, r.E_SHORT, 100)}
    path = os.path.join(r.D, f"firstlayer_L{L}.pkl")
    out = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
    for name, (cfg, m, E, n) in jobs.items():
        if name in out:
            continue
        t0 = time.time()
        bt = Batch(data, L, r.N, r.members_for(name, cfg, L, [m], range(n)))
        for _ in range(E):
            bt.run_epoch(val=False, test=True)
        s = bt.summary()
        out[name] = dict(cfg=cfg, lr_mult=m, epochs=E, summary=s)
        pickle.dump(out, open(path, "wb"))
        acc = s["test"][-1][~s["diverged"]]
        r.log(f"L={L} {name}: q1 = {cfg['q1']:.3f}, lr x{m}, {E} epochs: test {acc.mean():.4f} +- "
              f"{acc.std(ddof=1) / np.sqrt(acc.size):.4f} ({int(s['diverged'].sum())} div, {time.time() - t0:.0f}s)")
