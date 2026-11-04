"""Gradient walk with a real loss (SM, Sec. S5.4).  tanh networks with explicit Gaussian matrices, N = 64,
L = 400 hidden layers, sigma_b^2 = 0.05, on the mean-field edge of chaos (sigma_w^2 = 1.761) and on the
finite-width edge of chaos at the same bias (1.884).  Each network gets a first layer from the 784 standardised
MNIST pixels with variance (q* - sigma_b^2)/784 and a linear read-out of variance 1/N.  For each network and
each of K MNIST images, the cross-entropy gradient u = W_out^T (p - onehot) at the output is backpropagated
through the same matrices, and so is an isotropic Gaussian probe.  Recorded per layer: ln ||g^(l)||^2.
Output: data/backward/crossentropy.json.  A few minutes on one core."""
import gzip
import json
import os
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from eocfw.finite_width import a1_b1  # noqa: E402
from eocfw.meanfield import fixed_point  # noqa: E402

RAW = os.path.join(HERE, "..", "data", "mnist_raw", "MNIST", "raw")
OUT = os.path.join(HERE, "..", "data", "backward", "crossentropy.json")
N, L, SB2, BURN, BLOCK = 64, 400, 0.05, 100, 100


def load_images(n):
    with gzip.open(os.path.join(RAW, "train-images-idx3-ubyte.gz"), "rb") as f:
        _, num, r, c = struct.unpack(">IIII", f.read(16))
        X = np.frombuffer(f.read(), np.uint8).reshape(num, r * c).astype(np.float64)
    with gzip.open(os.path.join(RAW, "train-labels-idx1-ubyte.gz"), "rb") as f:
        f.read(8)
        y = np.frombuffer(f.read(), np.uint8).astype(np.int64)
    X = (X - X.mean()) / X.std()                       # standardised as in the training runs
    return X[:n], y[:n]


def run(sw2, nets, K, rng, X, y):
    q = fixed_point("tanh", sw2, SB2)
    steps_ce, steps_iso = [], []                       # per-layer steps below the output, after burn-in
    blocks_ce, blocks_iso = [], []
    k96_ce, k96_iso = [], []
    for _ in range(nets):
        idx = rng.choice(X.shape[0], K, replace=False)
        x, lab = X[idx], y[idx]
        W1 = rng.normal(0, np.sqrt(max(q - SB2, 1e-12) / 784), (N, 784))
        Ws = [rng.normal(0, np.sqrt(sw2 / N), (N, N)) for _ in range(L - 1)]
        bs = [rng.normal(0, np.sqrt(SB2), N) for _ in range(L)]
        Wout = rng.normal(0, np.sqrt(1.0 / N), (10, N))
        # forward: keep the slopes of every hidden layer
        z = x @ W1.T + bs[0]
        slopes = [1 - np.tanh(z) ** 2]
        for l in range(L - 1):
            z = np.tanh(z) @ Ws[l].T + bs[l + 1]
            slopes.append(1 - np.tanh(z) ** 2)
        logits = np.tanh(z) @ Wout.T
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        p[np.arange(K), lab] -= 1.0                     # dL/dlogits = p - onehot
        u_ce = (p @ Wout) * slopes[-1]                  # gradient w.r.t. the last pre-activations
        u_iso = rng.normal(0, 1, (K, N)) * slopes[-1]   # isotropic probe at the same place
        for u, steps, blocks, k96 in ((u_ce, steps_ce, blocks_ce, k96_ce), (u_iso, steps_iso, blocks_iso, k96_iso)):
            g = u / np.linalg.norm(u, axis=1, keepdims=True)
            walk = np.zeros((K, L))
            for k in range(1, L):                       # k layers below the output
                g = (g @ Ws[L - 1 - k]) * slopes[L - 1 - k]
                nv = np.linalg.norm(g, axis=1)
                walk[:, k] = walk[:, k - 1] + 2 * np.log(nv)
                g /= nv[:, None]
            st = np.diff(walk[:, BURN:], axis=1)
            steps.append(st.mean(1))
            blocks.append(walk[:, BURN + BLOCK] - walk[:, BURN])
            k96.append(walk[:, 96])
    out = {}
    for name, steps, blocks, k96 in (("crossentropy", steps_ce, blocks_ce, k96_ce), ("isotropic", steps_iso, blocks_iso, k96_iso)):
        st, bl, kk = np.concatenate(steps), np.concatenate(blocks), np.concatenate(k96)
        out[name] = dict(N_mu=float(N * st.mean()), N_mu_se=float(N * st.std(ddof=1) / np.sqrt(st.size)),
                         N_var_long=float(N * bl.var(ddof=1) / BLOCK), median_ratio_96=float(np.exp(np.median(kk))),
                         mean_ratio_96=float(np.exp(kk).mean()), quartiles_96=[float(v) for v in np.exp(np.quantile(kk, [0.25, 0.75]))],
                         n=int(st.size))
    th = a1_b1("tanh", q, sw2)
    out["theory"] = dict(q=q, a1=th["a1"], b1=th["b1"], chi1=th["chi1"], N_mu=float(N * np.log(th["chi1"]) + th["a1"]))
    return out


if __name__ == "__main__":
    nets = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    K = 4
    X, y = load_images(2000)
    rng = np.random.default_rng(3)
    res = {}
    for label, sw2 in (("meanfield", 1.7609546396067868), ("finitewidth", 1.884)):
        t0 = time.time()
        res[label] = dict(sw2=sw2, nets=nets, inputs_per_net=K, **run(sw2, nets, K, rng, X, y))
        print(label, json.dumps(res[label], indent=1), f"{time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
