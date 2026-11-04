"""
Finite-width simulations.

Two kinds of simulation are used.

* The *exact chain*.  For Gaussian weights, the pre-activations of layer l+1 given the activations of layer l are
  exactly Gaussian, independent across neurons, with covariance set by the empirical second moments of layer l.
  The law of (one or two) inputs through a random network is therefore a Markov chain on N-vectors that needs no
  weight matrices.  This is used for the drift/diffusion measurements and the avalanches.

* *Explicit matrices* (Gaussian or Haar-orthogonal), used to check the chain and for the orthogonal ensemble,
  for which the chain does not apply.
"""
import math

import numpy as np

from .gauss import get


def _haar(rng, B, N):
    """B independent Haar-orthogonal N x N matrices (QR with sign correction)."""
    A = rng.standard_normal((B, N, N))
    Q, R = np.linalg.qr(A)
    d = np.sign(np.einsum("bii->bi", R))
    return Q * d[:, None, :]


# ----------------------------------------------------------------------------------------------------------------
def distance_walk(name, N, sw2, sb2, B=1000, burn=400, L=2000, block=100, seed=0, orthogonal=False):
    """Random walk of the log squared distance D between two inputs in the tangent limit.

    Returns dict(mu, var_long, var_step, mu_mean): mu = E[step of ln D], var_long = long-run variance per step
    (batch means), var_step = one-step variance, mu_mean = ln of the mean growth factor of D (annealed drift).
    With orthogonal=False the exact chain is used; with orthogonal=True explicit Haar matrices (slower).
    """
    phi, d1, _ = get(name)
    rng = np.random.default_rng(seed)
    z = 0.5 * rng.standard_normal((B, N))
    dl = rng.standard_normal((B, N))
    Dprev = np.mean(dl ** 2, 1)
    steps = np.empty((L, B))
    for l in range(burn + L):
        y, dy = phi(z), d1(z) * dl
        if orthogonal:
            # one Haar matrix per layer, shared by the B samples (each sample has its own pre-activations)
            W = _haar(rng, 1, N)[0]
            z = math.sqrt(sw2) * y @ W.T + math.sqrt(sb2) * rng.standard_normal((B, N))
            dl = math.sqrt(sw2) * dy @ W.T
            Dn = np.mean(dl ** 2, 1)
        else:
            Qn = sw2 * np.mean(y ** 2, 1) + sb2
            Dn = sw2 * np.mean(dy ** 2, 1)
            Kn = sw2 * np.mean(y * dy, 1)
            z = np.sqrt(Qn)[:, None] * rng.standard_normal((B, N))
            dl = (Kn / Qn)[:, None] * z + np.sqrt(np.maximum(Dn - Kn ** 2 / Qn, 0))[:, None] * rng.standard_normal((B, N))
        if l >= burn:
            steps[l - burn] = np.log(Dn / Dprev)
        s = np.sqrt(np.mean(dl ** 2, 1))               # renormalise: the tangent process is linear in delta
        dl = dl / s[:, None]
        Dprev = Dn / s ** 2
    mu = steps.mean()
    nb = steps.shape[0] // block
    bm = steps[: nb * block].reshape(nb, block, B).sum(1)
    var_long = bm.var() / block
    # annealed (mean) growth rate: ln E[prod] per step from the block sums (cumulant estimate)
    mu_mean = np.log(np.mean(np.exp(bm - bm.mean()))) / block + bm.mean() / block
    return dict(mu=mu, var_long=var_long, var_step=steps.var(), mu_mean=mu_mean)


# ----------------------------------------------------------------------------------------------------------------
def signal_walk(name, N, sw2, L, B=2000, q1=1.0, seed=0, orthogonal=False, sb2=0.0):
    """Typical signal exp(E ln q_l) of one input through L layers (explicit matrices).  Returns (typical, se, mean)."""
    phi = get(name)[0]
    rng = np.random.default_rng(seed)
    z = math.sqrt(q1) * rng.standard_normal((B, N))
    logq = np.empty((L, B))
    logq[0] = np.log(np.mean(z ** 2, 1))
    for l in range(1, L):
        if orthogonal:
            W = _haar(rng, B, N) * math.sqrt(sw2)
        else:
            W = rng.standard_normal((B, N, N)) * math.sqrt(sw2 / N)
        z = np.einsum("bij,bj->bi", W, phi(z)) + math.sqrt(sb2) * rng.standard_normal((B, N))
        logq[l] = np.log(np.mean(z ** 2, 1))
    return np.exp(logq.mean(1)), logq.std(1) / math.sqrt(B), np.exp(logq).mean(1)


def typical_recursion(name, sw2, N, L, q1=1.0, sb2=0.0):
    """ln Q_{l+1} = ln(sw2 V(Q_l) + sb2) - (V4/V^2 - 1)/(2N): typical signal ignoring the spread across networks."""
    from .gauss import gavg
    phi = get(name)[0]
    out = [q1]
    for _ in range(L - 1):
        q = out[-1]
        Vq = gavg(lambda u: phi(u) ** 2, q)
        V4 = gavg(lambda u: phi(u) ** 4, q)
        out.append((sw2 * Vq + sb2) * math.exp(-sw2 ** 2 * (V4 - Vq ** 2) / (2 * N * (sw2 * Vq + sb2) ** 2)))
    return np.array(out)


# ----------------------------------------------------------------------------------------------------------------
def two_input_chain(name, N, sw2, sb2, c0, L, B=2000, q1=None, seed=0):
    """Exact chain for two inputs at finite separation.  Returns rho[l] = 1 - C_l (cosine) for every sample,
    shape (L+1, B).  q1: input signal (default: the mean-field fixed point is reached by a burn-in instead)."""
    phi = get(name)[0]
    rng = np.random.default_rng(seed)
    if q1 is None:
        q1 = 1.0
    g1 = rng.standard_normal((B, N))
    g2 = rng.standard_normal((B, N))
    z1 = math.sqrt(q1) * g1
    z2 = math.sqrt(q1) * (c0 * g1 + math.sqrt(1 - c0 ** 2) * g2)
    rho = np.empty((L + 1, B))

    def cos(a, b):
        return np.sum(a * b, 1) / np.sqrt(np.sum(a * a, 1) * np.sum(b * b, 1))

    rho[0] = 1 - cos(z1, z2)
    for l in range(1, L + 1):
        y1, y2 = phi(z1), phi(z2)
        Q1 = sw2 * np.mean(y1 ** 2, 1) + sb2
        Q2 = sw2 * np.mean(y2 ** 2, 1) + sb2
        Q12 = sw2 * np.mean(y1 * y2, 1) + sb2
        g1 = rng.standard_normal((B, N))
        g2 = rng.standard_normal((B, N))
        z1 = np.sqrt(Q1)[:, None] * g1
        c = Q12 / np.sqrt(Q1 * Q2)
        z2 = np.sqrt(Q2)[:, None] * (c[:, None] * g1 + np.sqrt(np.maximum(1 - c * c, 0))[:, None] * g2)
        rho[l] = 1 - cos(z1, z2)
    return rho


# ----------------------------------------------------------------------------------------------------------------
def survival(name, N, sw2, sb2, B=20000, L=400, burn=300, seed=5):
    """Avalanches of the distance: X_l = ln(d_l^2/d_0^2); returns durations T (first return below 0) and areas."""
    phi, d1, _ = get(name)
    rng = np.random.default_rng(seed)
    z = 0.7 * rng.standard_normal((B, N))
    for _ in range(burn):
        Q = sw2 * np.mean(phi(z) ** 2, 1) + sb2
        z = np.sqrt(Q)[:, None] * rng.standard_normal((B, N))
    dl = rng.standard_normal((B, N))
    logd0 = np.log(np.mean(dl ** 2, 1))
    logscale = np.zeros(B)
    alive = np.arange(B)
    T = np.zeros(B, int)
    area = np.zeros(B)
    for l in range(1, L + 1):
        y, dy = phi(z), d1(z) * dl
        Qn = sw2 * np.mean(y ** 2, 1) + sb2
        Dn = sw2 * np.mean(dy ** 2, 1)
        Kn = sw2 * np.mean(y * dy, 1)
        z = np.sqrt(Qn)[:, None] * rng.standard_normal(z.shape)
        dl = (Kn / Qn)[:, None] * z + np.sqrt(np.maximum(Dn - Kn ** 2 / Qn, 0))[:, None] * rng.standard_normal(z.shape)
        s = np.sqrt(np.mean(dl ** 2, 1))
        dl = dl / s[:, None]
        logscale = logscale + 2 * np.log(s)
        X = logscale - logd0[alive]
        up = X >= 0
        T[alive[up]] = l
        area[alive[up]] += X[up]
        alive, z, dl, logscale = alive[up], z[up], dl[up], logscale[up]
        if alive.size == 0:
            break
    return T, area, B
