"""
Mean-field (infinite-width) theory: variance map, edge of chaos, correlation map and shaping rules.

    q_{l+1} = sigma_w^2 V(q_l) + sigma_b^2            (variance map, slope F' = sigma_w^2 V')
    chi_1   = sigma_w^2 P(q*)                         (slope of the correlation map at C = 1)

The edge of chaos (EOC) is chi_1 = 1.  Parametrised by the steady signal q* it reads
    sigma_w^2 = 1/P(q*),    sigma_b^2 = q* - V(q*)/P(q*).
"""
import math

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import brentq

from .gauss import gavg, get, moments

# 2-D Gauss-Hermite grid for the distance map
_X2, _W2 = hermegauss(80)
_W2 = _W2 / _W2.sum()
_ZA, _ZB = np.meshgrid(_X2, _X2, indexing="ij")
_WAB = np.outer(_W2, _W2)


def fixed_point(name, sw2, sb2, q0=1.0, iters=5000, tol=1e-14):
    """Steady signal q* of the variance map."""
    phi = get(name)[0]
    q = q0
    for _ in range(iters):
        qn = sw2 * gavg(lambda u: phi(u) ** 2, q) + sb2
        if abs(qn - q) < tol or not np.isfinite(qn):
            break
        q = qn
    return q


def eoc_point(name, q):
    """(sigma_w^2, sigma_b^2) of the edge of chaos with steady signal q."""
    m = moments(name, q)
    sw2 = 1.0 / m["P"]
    return sw2, q - sw2 * m["V"]


def eoc_at_bias(name, sb2, lo=0.3, hi=6.0):
    """(sigma_w^2, q*) of the edge of chaos at a given bias variance."""
    def chi_minus_1(sw2):
        q = fixed_point(name, sw2, sb2)
        return sw2 * moments(name, q)["P"] - 1.0
    sw2 = brentq(chi_minus_1, lo, hi, xtol=1e-12)
    return sw2, fixed_point(name, sw2, sb2)


def chi1(name, sw2, sb2):
    q = fixed_point(name, sw2, sb2)
    return sw2 * moments(name, q)["P"]


def kappa(name, q, sw2=None):
    """Quadratic coefficient of the correlation map 1-C' = chi_1(1-C) - kappa(1-C)^2 (smooth activations):
    kappa = sigma_w^2 q E[phi''^2]/2 = 1/beta_q of Hayou et al."""
    d2 = get(name)[2]
    if sw2 is None:
        sw2 = 1.0 / moments(name, q)["P"]
    return sw2 * q / 2 * gavg(lambda u: d2(u) ** 2, q)


def dist_map(name, eps, q, sw2):
    """One layer of the mean-field decorrelation map eps = 1 - C at fixed signal q.
    Uses the midpoint/difference representation, which has no cancellation at small eps."""
    phi = get(name)[0]
    d2 = 2 * q * eps
    m = math.sqrt(max(q - d2 / 4, 0.0)) * _ZA
    dl = math.sqrt(d2) * _ZB
    return sw2 * float(np.sum(_WAB * (phi(m + dl / 2) - phi(m - dl / 2)) ** 2)) / (2 * q)


def compression(name, sw2, q, L, eps0=1.0):
    """comp = ln(eps_0/eps_L): loss of log-range of input similarities over L layers (eps_0 = 1: orthogonal inputs)."""
    eps = eps0
    for _ in range(L):
        eps = dist_map(name, min(eps, 1.999), q, sw2)
    return -math.log(max(eps, 1e-300)) + math.log(eps0)


def sweet_spot(name, L, gamma, q_lo=1e-5, q_hi=20.0):
    """Steady signal q* on the edge of chaos whose compression over L layers equals gamma; returns (q*, sw2, sb2)."""
    def f(lq):
        q = math.exp(lq)
        sw2, _ = eoc_point(name, q)
        return compression(name, sw2, q, L) - gamma
    q = math.exp(brentq(f, math.log(q_lo), math.log(q_hi), xtol=1e-10))
    sw2, sb2 = eoc_point(name, q)
    return q, sw2, sb2


def leaky_cmap(c, alpha):
    """Closed-form correlation map of the leaky ReLU (bias 0): C -> f(C)."""
    a, b = (1 + alpha) / 2, (1 - alpha) / 2
    c = min(c, 1.0)
    return (a * a * c + 2 * b * b / math.pi * (math.sqrt(max(1 - c * c, 0.0)) + c * math.asin(c))) / (a * a + b * b)


def leaky_compression(alpha, L, c0=0.0):
    c = c0
    for _ in range(L):
        c = leaky_cmap(c, alpha)
    return -math.log(max(1 - c, 1e-300))


def shaped_leaky_alpha(L, gamma):
    """Negative slope alpha of the depth-shaped leaky ReLU: compression of orthogonal inputs over L layers = gamma."""
    return brentq(lambda a: leaky_compression(a, L) - gamma, 0.0, 0.9999, xtol=1e-10)


def leaky_kappa(alpha):
    """Coefficient of the (1-C)^{3/2} term of the leaky-ReLU correlation map, sigma_w^2 = 2/(1+alpha^2)."""
    return math.sqrt(2) * (1 - alpha) ** 2 / (3 * math.pi * (1 + alpha ** 2)) * 2
