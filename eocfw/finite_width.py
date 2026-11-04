"""
Finite-width theory: the random walks of the signal size and of the distance between two inputs.

Signal size (one input).  ln q_l performs a random walk; the typical signal is stationary at
    sigma_c^2(N) = (1/a^2) (N/2) exp(-psi(N/2))              (smooth phi, a = phi'(0))
    sigma_c^2(N) = (N/2) exp(-E_K psi(K/2)),  K ~ Bin(N, 1/2) conditioned on K >= 1      (ReLU, exact)
    sigma_c^2(N) = 2/(1+alpha^2) exp(C_alpha/(2N))                (leaky ReLU, leading order)

Distance between two nearby inputs.  X_l = ln D_l performs a random walk with
    drift  mu_d = ln chi_1 + a_1/N,     diffusion  sigma_d^2 = b_1/N,
with a_1, b_1 explicit Gaussian averages (a1_b1).  The finite-width edge of chaos is mu_d = 0.
"""
import math

import numpy as np
from scipy.special import comb, digamma

from .gauss import moments
from .meanfield import eoc_at_bias, fixed_point


# ---------------------------------------------------------------- signal size --------------------------------
def sigma_c2_smooth(N, a2=1.0):
    return (N / 2) * math.exp(-digamma(N / 2)) / a2


def sigma_c2_leaky(N, alpha=0.0):
    C = (5 - 2 * alpha ** 2 + 5 * alpha ** 4) / (1 + alpha ** 2) ** 2
    return 2 / (1 + alpha ** 2) * math.exp(C / (2 * N))


def sigma_c2_relu_exact(N):
    """Exact typical-preserving sigma_w^2 of ReLU: the signal is a chi^2 with K active neurons, K ~ Bin(N, 1/2)
    conditioned on K >= 1 (the all-dead event kills the signal outright)."""
    K = np.arange(1, N + 1)
    p = comb(N, K) / (2.0 ** N - 1)
    return (N / 2) * math.exp(-float(np.sum(p * digamma(K / 2))))


# ---------------------------------------------------------------- distance walk ------------------------------
def a1_b1(name, q, sw2=None, orthogonal=False):
    """O(1/N) drift coefficient a_1 and long-run variance coefficient b_1 of ln D at the steady signal q.

    sw2 = None puts the point on the mean-field edge of chaos (chi_1 = 1).  With orthogonal=True the per-layer
    variances are replaced by their orthogonal-ensemble ('projected') values: the parts of each per-neuron
    fluctuation that lie in the span of {z^2, z delta, delta^2} are fixed exactly by norm preservation.
    Returns a dict with the separate contributions.
    """
    m = moments(name, q)
    if sw2 is None:
        sw2 = 1.0 / m["P"]
    P, dP, d2P, V, dV = m["P"], m["dP"], m["d2P"], m["V"], m["dV"]
    Fp, Fpp = sw2 * dV, sw2 * m["d2V"]
    chi = sw2 * P
    # one-layer relative variances of the three per-neuron averages
    var_w = 3 * m["P4"] / P ** 2 - 1                       # Var[phi'^2 eta^2] / P^2  (eta independent of z)
    var_Q = sw2 ** 2 * (m["V4"] - V ** 2)                  # N Var of one-layer noise in Q
    var_r = sw2 * m["EphiDphi2"] / (q * P)                 # N Var of one-layer noise in r
    cov_wQ = sw2 * (m["EphiDphi2"] - P * V) / P            # N Cov(eps, eta): shared neuron fluctuations
    if orthogonal:
        var_w -= (2 * P ** 2 + 2 * q ** 2 * dP ** 2) / P ** 2
        var_Q -= sw2 ** 2 * 2 * q ** 2 * dV ** 2
        var_r -= sw2 * q * dV ** 2 / (q * P)
        cov_wQ -= sw2 * 2 * q ** 2 * dP * dV / P
    NEx2 = var_Q / (1 - Fp ** 2)                           # N * stationary Var(Q - q*)
    NEx = 0.5 * Fpp * NEx2 / (1 - Fp)                      # N * stationary mean shift of Q
    NEr2 = var_r / (1 - Fp ** 2 / chi)                     # N * stationary E[r^2]
    a_jensen = -0.5 * var_w
    a_r = 2 * q * (dP / P) * NEr2
    a_mean = (dP / P) * NEx
    a_curv = 0.5 * (d2P / P - (dP / P) ** 2) * NEx2
    b_white = var_w
    b_Q = (dP / P) ** 2 * var_Q / (1 - Fp) ** 2
    b_cross = 2 * (dP / P) * cov_wQ / (1 - Fp)
    return dict(q=q, sw2=sw2, chi1=chi, Fp=Fp, a1=a_jensen + a_r + a_mean + a_curv, a_jensen=a_jensen, a_r=a_r,
                a_mean=a_mean, a_curv=a_curv, b1=b_white + b_Q + b_cross, b_white=b_white, b_Q=b_Q, b_cross=b_cross)


def dlnchi_dsw2(name, q, sw2):
    """d ln chi_1 / d sigma_w^2 at fixed sigma_b^2 (q* moves with sigma_w^2)."""
    m = moments(name, q)
    Fp = sw2 * m["dV"]
    return 1 / sw2 + (m["dP"] / m["P"]) * m["V"] / (1 - Fp)


def eoc_shift(name, sb2, N, orthogonal=False):
    """Finite-width edge of chaos at bias sb2: returns (sw2_mf, q*, Delta sigma_w^2) with
    Delta sigma_w^2 = -a_1 / (N d ln chi_1/d sigma_w^2)."""
    sw2, q = eoc_at_bias(name, sb2)
    a1 = a1_b1(name, q, sw2, orthogonal=orthogonal)["a1"]
    return sw2, q, -a1 / (N * dlnchi_dsw2(name, q, sw2))


def eoc_factor_at_q(name, q, N):
    """Multiplicative factor 1 + Delta sigma_w^2/sigma_w^2 that moves the EOC point with steady signal q
    (bias fixed) to the finite-width edge of chaos."""
    m = moments(name, q)
    sw2 = 1 / m["P"]
    a1 = a1_b1(name, q, sw2)["a1"]
    return 1 + (-a1 / N) / (dlnchi_dsw2(name, q, sw2) * sw2)
