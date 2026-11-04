"""
How much input geometry survives L layers of a finite random network.

K pairs of inputs with decorrelations eps_0 = 1 - C_0 spread log-uniformly over [1e-3, 1] are sent through the
same network; after L layers eps_L = 1 - C_L is measured for every pair.  Within one network ln eps_L is fitted
as a smooth function f of ln eps_0, and the Gaussian-channel information

        I = 1/2 log2(1 + Var[f] / Var[ln eps_L - f])      (bits per pair)

measures how precisely the input similarity can be read off the output similarity.  comp = loss of log-range of
similarities (mean-field compression); logspread = spread of ln q_L across independent inputs (finite width).
"""
import math

import numpy as np

from .finite_width import a1_b1
from .gauss import moments
from .meanfield import eoc_point

RELU = lambda x: np.maximum(x, 0)
LIN = lambda x: x


def make_pairs(K, N, q_in, rng):
    eps0 = np.logspace(-3, 0, K)
    C0 = 1 - eps0
    g1 = rng.standard_normal((K, N))
    g2 = rng.standard_normal((K, N))
    x1 = math.sqrt(q_in) * g1
    x2 = math.sqrt(q_in) * (C0[:, None] * g1 + np.sqrt(1 - C0 ** 2)[:, None] * g2)
    return eps0, np.concatenate([x1, x2], 0).astype(np.float32)


def forward(X, L, sw2, sb2, rng, act=np.tanh):
    N = X.shape[1]
    z = X
    for _ in range(L):
        W = (rng.standard_normal((N, N)) * math.sqrt(sw2 / N)).astype(np.float32)
        b = (rng.standard_normal(N) * math.sqrt(sb2)).astype(np.float32)
        z = act(z) @ W.T + b
    return z


def info_bits(eps0, z, K):
    z1, z2 = z[:K].astype(np.float64), z[K:].astype(np.float64)
    C = np.sum(z1 * z2, 1) / np.sqrt(np.sum(z1 ** 2, 1) * np.sum(z2 ** 2, 1))
    y = np.log(np.clip(1 - C, 1e-14, None))
    x = np.log(eps0)
    f = np.polyval(np.polyfit(x, y, 3), x)
    res = np.var(y - f)
    sig = np.var(f)
    comp = (x[-1] - x[0]) - (f[-1] - f[0])
    return 0.5 * np.log2(1 + sig / max(res, 1e-30)), sig, res, comp


def run_point(N, L, q, n_nets=4, K=128, seed=0, act=np.tanh, sw2=None, sb2=None):
    if sw2 is None:
        sw2, sb2 = eoc_point("tanh", q)
    rng = np.random.default_rng(seed)
    I, S, R, CP, CV, QR = [], [], [], [], [], []
    for _ in range(n_nets):
        eps0, X = make_pairs(K, N, q, rng)
        z = forward(X, L, sw2, sb2, rng, act)
        i, s, r, cp = info_bits(eps0, z, K)
        I.append(i); S.append(s); R.append(r); CP.append(cp)
        qh = np.mean(z[:K].astype(np.float64) ** 2, 1)
        CV.append(np.std(np.log(qh)))
        QR.append(np.mean(qh) / q)
    return dict(I=np.mean(I), I_se=np.std(I) / math.sqrt(n_nets), sig=np.mean(S), res=np.mean(R), Qratio=np.mean(QR),
                comp=np.mean(CP), logspread=np.mean(CV), sw2=sw2, sb2=sb2)


def logspread_theory(q, L, N):
    """std of ln q_L across inputs: AR(1) with coefficient F' and one-layer noise sw2^2 (V4 - V^2)/(N q^2), + 2/N."""
    m = moments("tanh", q)
    sw2 = 1 / m["P"]
    Fp = sw2 * m["dV"]
    s2 = sw2 ** 2 * (m["V4"] - m["V"] ** 2) / (N * q ** 2)
    return math.sqrt(s2 * (1 - Fp ** (2 * L)) / (1 - Fp ** 2) + 2.0 / N)


def comp_mf(q, L, eps_lo=1e-3, eps_hi=1.0):
    from .meanfield import dist_map
    sw2, _ = eoc_point("tanh", q)
    lo, hi = eps_lo, eps_hi
    for _ in range(L):
        lo = dist_map("tanh", lo, q, sw2)
        hi = dist_map("tanh", min(hi, 1.999), q, sw2)
    return (math.log(eps_hi) - math.log(eps_lo)) - (math.log(hi) - math.log(lo))
