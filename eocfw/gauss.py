"""
Gaussian averages of activation functions.

All mean-field quantities of a deep random network are Gaussian averages of the activation phi and its
derivatives at a given pre-activation variance q:

    V(q) = E[phi(sqrt(q) z)^2],   P(q) = E[phi'(sqrt(q) z)^2],   z ~ N(0, 1).

They are computed with Gauss-Hermite quadrature (160 nodes), which is exact to machine precision for the
smooth activations and accurate to ~1e-8 for the kinked ones.
"""
import math

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import erf

_X, _W = hermegauss(160)
_W = _W / _W.sum()

_SQ2PI = math.sqrt(2 / math.pi)


def _gelu(u):
    return 0.5 * u * (1 + erf(u / math.sqrt(2)))


def _dgelu(u):
    return 0.5 * (1 + erf(u / math.sqrt(2))) + u * np.exp(-u * u / 2) / math.sqrt(2 * math.pi)


def _d2gelu(u):
    return (2 - u * u) * np.exp(-u * u / 2) / math.sqrt(2 * math.pi)


ACTIVATIONS = {
    # name: (phi, phi', phi'')
    "tanh": (np.tanh, lambda u: 1 / np.cosh(u) ** 2, lambda u: -2 * np.tanh(u) / np.cosh(u) ** 2),
    "erf": (lambda u: erf(u / math.sqrt(2)), lambda u: _SQ2PI * np.exp(-u * u / 2),
            lambda u: -_SQ2PI * u * np.exp(-u * u / 2)),
    "gelu": (_gelu, _dgelu, _d2gelu),
    "linear": (lambda u: u, lambda u: np.ones_like(u), lambda u: np.zeros_like(u)),
    "relu": (lambda u: np.maximum(u, 0.0), lambda u: (u > 0).astype(float), lambda u: np.zeros_like(u)),
}


def leaky(alpha):
    """(phi, phi', phi'') of the leaky ReLU with negative slope alpha."""
    return (lambda u: np.where(u > 0, u, alpha * u), lambda u: np.where(u > 0, 1.0, alpha),
            lambda u: np.zeros_like(u))


def get(name):
    """Return (phi, phi', phi'') for a named activation or 'leaky:alpha'."""
    if name.startswith("leaky"):
        return leaky(float(name.split(":")[1]))
    return ACTIVATIONS[name]


def gavg(f, q):
    """E[f(sqrt(q) z)] for z ~ N(0, 1)."""
    return float(np.sum(_W * f(math.sqrt(q) * _X)))


def V(name, q):
    phi = get(name)[0]
    return gavg(lambda u: phi(u) ** 2, q)


def P(name, q):
    d1 = get(name)[1]
    return gavg(lambda u: d1(u) ** 2, q)


def moments(name, q, h=None):
    """All Gaussian averages needed by the finite-width theory at variance q.

    Returns a dict with P, V, their first two q-derivatives (central differences), the fourth moments
    P4 = E[phi'^4] and V4 = E[phi^4], and the mixed moments E[(phi phi')^2] and E[phi^2 phi'^2].
    """
    phi, d1, _ = get(name)
    if h is None:
        h = 1e-4 * max(q, 1e-3)
    Pq = lambda s: gavg(lambda u: d1(u) ** 2, s)
    Vq = lambda s: gavg(lambda u: phi(u) ** 2, s)
    return dict(
        q=q, P=Pq(q), V=Vq(q),
        dP=(Pq(q + h) - Pq(q - h)) / (2 * h), d2P=(Pq(q + h) - 2 * Pq(q) + Pq(q - h)) / h ** 2,
        dV=(Vq(q + h) - Vq(q - h)) / (2 * h), d2V=(Vq(q + h) - 2 * Vq(q) + Vq(q - h)) / h ** 2,
        P4=gavg(lambda u: d1(u) ** 4, q), V4=gavg(lambda u: phi(u) ** 4, q),
        EphiDphi2=gavg(lambda u: (phi(u) * d1(u)) ** 2, q),
    )


def hermite_coefficients(name, q, nmax=12):
    """Coefficients c_n of phi(sqrt(q) z) = sum_n c_n He_n(z) (probabilists' Hermite polynomials)."""
    phi = get(name)[0]
    x = _X
    He = [np.ones_like(x), x.copy()]
    for n in range(1, nmax):
        He.append(x * He[n] - n * He[n - 1])
    f = phi(math.sqrt(q) * x)
    return np.array([np.sum(_W * f * He[n]) / math.factorial(n) for n in range(nmax + 1)])
