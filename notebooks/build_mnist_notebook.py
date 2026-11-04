"""Builds mnist_criticality.ipynb (run: python build_mnist_notebook.py)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip("\n")))


md(r"""
# Criticality-based initialisation on MNIST

This notebook tests, on a real task, the initialisation rules that come out of our theory of signal propagation in
deep random networks. Every rule is a statement about the two numbers that describe an initialisation, the weight
variance $\sigma_w^2=N\,\mathrm{Var}(W_{ij})$ and the bias variance $\sigma_b^2$, or about the activation function.

| name | rule | where it comes from |
|---|---|---|
| **standard** | Xavier/LeCun $\sigma_w^2=1$ (tanh), He $\sigma_w^2=2$ (ReLU) | mean-field critical point, $N\to\infty$ |
| **finite-N critical** | $\sigma_w^2=\sigma_c^2(N)=\frac{1}{a^2}\frac N2e^{-\psi(N/2)}\approx\sigma_{w,c}^2(1+\frac1N)$; ReLU $2e^{5/(2N)}$ | zero *typical* (log) growth of the signal at width $N$ |
| **EOC sweet spot** | a point $(\sigma_w^2,\sigma_b^2)$ on the edge of chaos with steady signal $q^*$ chosen so that the correlation map compresses input similarities by a fixed amount $\gamma$ over the whole depth $L$; `tanh_eoc_sweet` also moves $\sigma_w^2$ to the finite-width edge of chaos, `tanh_eoc_sweet_mf` does not | information/nonlinearity trade-off |
| **shaped leaky ReLU** | leaky ReLU whose negative slope $\alpha$ is chosen with the same depth rule | correlation map with a kink, $\kappa\propto(1-\alpha)^2$ |
| **linear** | identity activation, $\sigma_w^2=1$ | "critical everywhere", but no nonlinearity |

The questions:
1. Does correcting the critical point for finite width, $\sigma_w^2\to\sigma_c^2(N)$, change training?
2. Does moving *along* the edge of chaos to the sweet spot (a non-zero signal, but only as much nonlinearity as the depth can carry) beat sitting at the critical point $q^*=0$?
3. Does a depth-shaped activation help very deep, narrow networks?
4. How far does the "critical everywhere" linear network get? (It cannot beat logistic regression.)

**How to use it.** Edit the configuration cell, then *Run All*. The default configuration runs in roughly 10–15
minutes on a laptop CPU; set `device` to `"mps"` on Apple silicon or `"cuda"` if available (for these small networks
the CPU is often as fast). MNIST is downloaded by
torchvision on first use (with a GitHub mirror as fall-back).
""")

md(r"""
## 1. Configuration
All tunable settings are in this one cell.
""")
code(r'''
CONFIG = dict(
    # network
    depths=[32, 96],            # L: number of hidden layers
    widths=[64],                # N: neurons per hidden layer
    # which initialisations / activations to compare (see the table above and INIT_RULES below)
    inits=["tanh_standard", "tanh_finiteN", "tanh_eoc_sweet_mf", "tanh_eoc_sweet", "tanh_chaotic",
           "relu_he", "relu_finiteN", "leaky_shaped", "linear"],
    gamma=1.0,                  # sweet-spot rule: compression of input similarities over the whole depth (nats)
    finiteN_eoc_correction=True,  # also shift the sweet-spot sigma_w^2 to the finite-N edge of chaos
    # training
    optimizer="sgd",            # "sgd" or "adam"
    lr=0.005, momentum=0.9,     # base learning rate (for L = 32)
    lr_depth_scaling=True,      # use lr * 32 / L: with SGD the effective step grows with depth
    lr_multipliers=[1.0],       # e.g. [0.3, 1, 3] (used in the research note): train each init at several lrs and keep
                                # the one with the lowest training loss (fairer, 3x slower)
    grad_clip=None,             # e.g. 5.0 to clip the gradient norm (off by default: it hides init effects)
    batch_size=128,
    epochs=2,
    seeds=[0, 1, 2],
    train_subset=None,          # e.g. 10000 to use only part of the training set (faster)
    target_acc=0.85,            # report the number of steps to reach this running training accuracy
    # optional experiments
    run_diagnostics=True,       # signal/correlation profiles at initialisation vs theory
    run_phase_diagram=False,    # accuracy after a few hundred steps over a (sigma_w^2, sigma_b^2) grid
    phase_grid=dict(L=48, N=64, steps=300, sw2=[0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.7], sb2=[0.0, 1e-3, 1e-2, 0.05, 0.2]),
    # system
    device="cpu",               # "cpu", "mps" or "cuda"
    data_dir="./data",
    results_file=None,          # e.g. "mnist_results.pkl" to save all training logs
)
''')

md(r"""
## 2. Imports and data
Images are flattened and standardised (zero mean, unit variance over the training set). The *scale* of the input is
then set per initialisation so that the first hidden layer starts at the intended signal strength $q_1$ (see §4).
""")
code(r'''
import math, time, gzip, os, urllib.request
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.special import digamma
from scipy.optimize import brentq
from numpy.polynomial.hermite_e import hermegauss

torch.set_num_threads(max(1, os.cpu_count() or 1))
DEV = torch.device(CONFIG["device"])

# plot style: no grid, horizontal y-label above the axis, pastel colours
PAL = ["#4F86C6", "#E0645C", "#E8A33C", "#2D5F95", "#B8433D", "#6FA3DC", "#C77A2A", "#7A7A7A", "#F2A488", "#9CC3E6"]
plt.rcParams.update({"axes.grid": False, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
def ylabel_top(ax, s):
    ax.set_ylabel(s, rotation=0, ha="left", va="bottom"); ax.yaxis.set_label_coords(-0.02, 1.02)

def load_mnist(root):
    """torchvision MNIST if possible, otherwise the raw files from a GitHub mirror."""
    try:
        from torchvision import datasets
        tr = datasets.MNIST(root, train=True, download=True); te = datasets.MNIST(root, train=False, download=True)
        return (tr.data.numpy(), tr.targets.numpy()), (te.data.numpy(), te.targets.numpy())
    except Exception as e:
        print("torchvision download failed (", e, "); using the GitHub mirror")
    base = "https://raw.githubusercontent.com/fgnt/mnist/master/"
    raw = os.path.join(root, "MNIST", "raw"); os.makedirs(raw, exist_ok=True)
    def get(name, offset):
        path = os.path.join(raw, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(base + name, path)
        with gzip.open(path) as f:
            return np.frombuffer(f.read(), np.uint8, offset=offset)
    Xtr = get("train-images-idx3-ubyte.gz", 16).reshape(-1, 28, 28); ytr = get("train-labels-idx1-ubyte.gz", 8)
    Xte = get("t10k-images-idx3-ubyte.gz", 16).reshape(-1, 28, 28); yte = get("t10k-labels-idx1-ubyte.gz", 8)
    return (Xtr, ytr), (Xte, yte)

(Xtr, ytr), (Xte, yte) = load_mnist(CONFIG["data_dir"])
Xtr = Xtr.reshape(len(Xtr), -1).astype(np.float32) / 255.0
Xte = Xte.reshape(len(Xte), -1).astype(np.float32) / 255.0
mu, sd = Xtr.mean(), Xtr.std()
Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd          # mean 0, mean square 1 per pixel
if CONFIG["train_subset"]:
    Xtr, ytr = Xtr[: CONFIG["train_subset"]], ytr[: CONFIG["train_subset"]]
Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.long)
Xte_t, yte_t = torch.tensor(Xte), torch.tensor(yte, dtype=torch.long)
print("train", Xtr_t.shape, "test", Xte_t.shape)
''')

md(r"""
## 3. The theory we need, in a few functions

**Gaussian averages.** For an input of variance $q$, $u=\sqrt q\,\zeta$ with $\zeta\sim\mathcal N(0,1)$:
$$V(q)=\mathbb E[\phi(u)^2],\qquad P(q)=\mathbb E[\phi'(u)^2].$$

**Variance map and fixed point.** $q_{\ell+1}=\sigma_w^2V(q_\ell)+\sigma_b^2$; the steady signal $q^*$ solves
$q^*=\sigma_w^2V(q^*)+\sigma_b^2$.

**Edge of chaos**, parametrised by the steady signal: $\chi_1=\sigma_w^2P(q^*)=1$ gives
$$\sigma_w^2=\frac1{P(q^*)},\qquad \sigma_b^2=q^*-\frac{V(q^*)}{P(q^*)} .$$

**Compression of input similarities.** Two inputs with decorrelation $\varepsilon=1-C$ at variance $q$ are mapped by one
layer to
$$\varepsilon'=\frac{\sigma_w^2}{2q}\,\mathbb E\big[(\phi(m+\tfrac{\delta}{2})-\phi(m-\tfrac{\delta}{2}))^2\big],\quad
m\sim\mathcal N(0,q-\tfrac{d^2}{4}),\ \delta\sim\mathcal N(0,d^2),\ d^2=2q\varepsilon .$$
On the edge of chaos small distances are kept ($\varepsilon'\approx\varepsilon$) but large ones shrink. We call
$\mathrm{comp}(L)=\ln\varepsilon_0-\ln\varepsilon_L$ for orthogonal inputs ($\varepsilon_0=1$) the *compression* after $L$
layers: $0$ for a linear network, large when the network maps every input to nearly the same direction.

**Sweet spot.** Choose $q^*$ (for tanh) or the negative slope $\alpha$ (for leaky ReLU) so that $\mathrm{comp}(L)=\gamma$:
the network is as nonlinear as possible while the input geometry still survives the full depth. For deep tanh networks
the sweet spot decreases roughly as $q^*\propto L^{-1/2}$ (e.g. $q^*\approx0.31$ at $L=32$, $0.14$ at $L=96$ for $\gamma=1$).

**Finite width.** A small signal is multiplied each layer by $\sigma_w^2a^2\chi^2_N/N$ ($a=\phi'(0)$); its logarithm is
a random walk with drift $\mu=\ln(\sigma_w^2a^2)+\psi(N/2)-\ln(N/2)$. The *typical* signal is preserved when $\mu=0$:
$$\sigma_c^2(N)=\frac1{a^2}\frac N2e^{-\psi(N/2)}\approx\frac{1}{a^2}\Big(1+\frac1N+\frac{5}{6N^2}\Big),\qquad
\text{leaky ReLU: }\ \sigma_c^2(N)=\frac{2}{1+\alpha^2}\,e^{C_\alpha/(2N)},\ C_\alpha=\frac{5-2\alpha^2+5\alpha^4}{(1+\alpha^2)^2}.$$
The same reasoning for the distance between two nearby inputs moves the edge of chaos to
$\ln\chi_1=-a_1/N$, with $a_1$ given by Gaussian averages (research note, Sec. 4).
""")
code(r'''
GX, GW = hermegauss(120); GW = GW / GW.sum()
G2X, G2W = hermegauss(70); G2W = G2W / G2W.sum()
ZA, ZB = np.meshgrid(G2X, G2X, indexing="ij"); WAB = np.outer(G2W, G2W)

def np_act(name, alpha=0.0):
    """activation and its derivative (numpy)."""
    if name == "tanh":   return np.tanh, lambda u: 1 / np.cosh(u) ** 2
    if name == "relu":   return (lambda u: np.maximum(u, 0)), (lambda u: (u > 0).astype(float))
    if name == "leaky":  return (lambda u: np.where(u > 0, u, alpha * u)), (lambda u: np.where(u > 0, 1.0, alpha))
    if name == "linear": return (lambda u: u), (lambda u: np.ones_like(u))
    raise ValueError(name)

def gavg(f, q):
    return float(np.sum(GW * f(math.sqrt(q) * GX)))

def V(phi, q): return gavg(lambda u: phi(u) ** 2, q)
def P(dphi, q): return gavg(lambda u: dphi(u) ** 2, q)

def fixed_point(phi, sw2, sb2, q=1.0, iters=5000):
    for _ in range(iters):
        qn = sw2 * V(phi, q) + sb2
        if abs(qn - q) < 1e-13 or not np.isfinite(qn): break
        q = qn
    return q

def eoc_point(name, q, alpha=0.0):
    phi, dphi = np_act(name, alpha)
    sw2 = 1.0 / P(dphi, q)
    return sw2, q - sw2 * V(phi, q)

def compression(name, sw2, q, L, alpha=0.0):
    """ln(eps_0/eps_L) for orthogonal inputs (eps_0 = 1) after L layers of the mean-field correlation map."""
    phi, _ = np_act(name, alpha)
    eps = 1.0
    for _ in range(L):
        d2 = 2 * q * eps
        m = math.sqrt(max(q - d2 / 4, 0)) * ZA; dl = math.sqrt(d2) * ZB
        eps = sw2 * float(np.sum(WAB * (phi(m + dl / 2) - phi(m - dl / 2)) ** 2)) / (2 * q)
    return -math.log(max(eps, 1e-300))

def sweet_spot_tanh(L, gamma):
    """steady signal q* on the edge of chaos whose compression over L layers equals gamma."""
    def f(log_q):
        q = math.exp(log_q)
        sw2, _ = eoc_point("tanh", q)
        return compression("tanh", sw2, q, L) - gamma
    q = math.exp(brentq(f, math.log(1e-5), math.log(5.0)))
    sw2, sb2 = eoc_point("tanh", q)
    return q, sw2, sb2

def shaped_leaky_alpha(L, gamma):
    f = lambda a: compression("leaky", 2 / (1 + a * a), 1.0, L, alpha=a) - gamma
    return brentq(f, 0.0, 0.999)

def sigma_c2_smooth(N, a2=1.0):
    return (N / 2) * math.exp(-digamma(N / 2)) / a2

def sigma_c2_leaky(N, alpha=0.0):
    Ca = (5 - 2 * alpha ** 2 + 5 * alpha ** 4) / (1 + alpha ** 2) ** 2
    return 2 / (1 + alpha ** 2) * math.exp(Ca / (2 * N))

def finiteN_eoc_factor(q, N, h=1e-4):
    """factor 1 + Delta sigma_w^2 / sigma_w^2 that moves sigma_w^2 (at fixed sigma_b^2) to the finite-N edge of chaos, tanh."""
    phi, dphi = np_act("tanh")
    ddphi = lambda u: -2 * np.tanh(u) / np.cosh(u) ** 2
    Pq = lambda s: P(dphi, s); Vq = lambda s: V(phi, s)
    sw2 = 1 / Pq(q)
    dP = (Pq(q + h) - Pq(q - h)) / (2 * h); d2P = (Pq(q + h) - 2 * Pq(q) + Pq(q - h)) / h ** 2
    dV = (Vq(q + h) - Vq(q - h)) / (2 * h); d2V = (Vq(q + h) - 2 * Vq(q) + Vq(q - h)) / h ** 2
    P0, V0 = Pq(q), Vq(q); Fp, Fpp = sw2 * dV, sw2 * d2V
    P4 = gavg(lambda u: dphi(u) ** 4, q); V4 = gavg(lambda u: phi(u) ** 4, q)
    Epp2 = gavg(lambda u: (phi(u) * dphi(u)) ** 2, q)
    NEx2 = sw2 ** 2 * (V4 - V0 ** 2) / (1 - Fp ** 2); NEx = 0.5 * Fpp * NEx2 / (1 - Fp)
    NEr2 = sw2 * Epp2 / (q * P0) / (1 - Fp ** 2)
    a1 = (-0.5 * (3 * P4 / P0 ** 2 - 1) + 2 * q * (dP / P0) * NEr2 + (dP / P0) * NEx
          + 0.5 * (d2P / P0 - (dP / P0) ** 2) * NEx2)
    dlnchi = 1 / sw2 + (dP / P0) * V0 / (1 - Fp)          # d ln chi_1 / d sigma_w^2 at fixed sigma_b^2
    return 1 + (-a1 / N) / (dlnchi * sw2)

print("sigma_c^2(N=64) =", sigma_c2_smooth(64), "  ReLU:", sigma_c2_leaky(64))
for L in CONFIG["depths"]:
    q, sw2, sb2 = sweet_spot_tanh(L, CONFIG["gamma"])
    print(f"L={L}: tanh sweet spot q*={q:.4f}, sigma_w^2={sw2:.4f}, sigma_b^2={sb2:.2e};  "
          f"shaped leaky alpha={shaped_leaky_alpha(L, CONFIG['gamma']):.3f}")
''')

md(r"""
## 4. Initialisation rules

Each rule returns the activation, $(\sigma_w^2,\sigma_b^2)$ for the hidden layers, and the signal strength $q_1$ at
which the first hidden layer should start. The input scale is set so that
$\sigma_w^2\,\overline{x^2}+\sigma_b^2=q_1$ in the first layer. For the sweet spot $q_1=q^*$, so the network starts at its
fixed point; for the others $q_1=1$ (the usual choice for standardised data).
""")
code(r'''
def INIT_RULES(name, L, N):
    g = CONFIG["gamma"]
    if name == "tanh_standard":   return dict(act="tanh", sw2=1.0, sb2=0.0, q1=1.0)
    if name == "tanh_finiteN":    return dict(act="tanh", sw2=sigma_c2_smooth(N), sb2=0.0, q1=1.0)
    if name == "tanh_chaotic":    return dict(act="tanh", sw2=2.0, sb2=0.0, q1=1.0)
    if name == "tanh_ordered":    return dict(act="tanh", sw2=1 / 3, sb2=0.0, q1=1.0)
    if name in ("tanh_eoc_sweet", "tanh_eoc_sweet_mf"):
        q, sw2, sb2 = sweet_spot_tanh(L, g)
        if name == "tanh_eoc_sweet" and CONFIG["finiteN_eoc_correction"]:
            sw2 = sw2 * finiteN_eoc_factor(q, N)
        return dict(act="tanh", sw2=sw2, sb2=sb2, q1=q)
    if name == "relu_he":         return dict(act="relu", sw2=2.0, sb2=0.0, q1=1.0)
    if name == "relu_finiteN":    return dict(act="relu", sw2=sigma_c2_leaky(N, 0.0), sb2=0.0, q1=1.0)
    if name == "leaky_shaped":
        a = shaped_leaky_alpha(L, g)
        return dict(act="leaky", alpha=a, sw2=sigma_c2_leaky(N, a), sb2=0.0, q1=1.0)
    if name == "linear":          return dict(act="linear", sw2=1.0, sb2=0.0, q1=1.0)
    raise ValueError(name)

def torch_act(rule):
    a = rule.get("alpha", 0.0)
    return {"tanh": torch.tanh, "relu": F.relu, "linear": (lambda x: x),
            "leaky": (lambda x: F.leaky_relu(x, a))}[rule["act"]]

class MLP(nn.Module):
    """L hidden layers of width N, Gaussian weights with Var = sigma_w^2/fan_in, Gaussian biases with Var = sigma_b^2."""
    def __init__(self, L, N, rule, d_in=784, d_out=10):
        super().__init__()
        self.rule, self.act = rule, torch_act(rule)
        self.layers = nn.ModuleList([nn.Linear(d_in if i == 0 else N, N) for i in range(L)])
        self.head = nn.Linear(N, d_out)
        with torch.no_grad():
            for lin in self.layers:
                lin.weight.normal_(0, math.sqrt(rule["sw2"] / lin.in_features))
                lin.bias.normal_(0, math.sqrt(rule["sb2"])) if rule["sb2"] > 0 else lin.bias.zero_()
            self.head.weight.normal_(0, math.sqrt(1.0 / N)); self.head.bias.zero_()
        # input scale so that the first layer starts at q1: sw2 * scale^2 * mean(x^2) + sb2 = q1 (mean(x^2) = 1)
        self.in_scale = math.sqrt(max(rule["q1"] - rule["sb2"], 1e-12) / rule["sw2"])

    def forward(self, x, return_hidden=False):
        h = self.layers[0](x * self.in_scale)
        hidden = [h]
        for lin in self.layers[1:]:
            h = lin(self.act(h)); hidden.append(h)
        out = self.head(self.act(h))
        return (out, hidden) if return_hidden else out

for n in CONFIG["inits"]:
    print(n, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in INIT_RULES(n, CONFIG["depths"][0], CONFIG["widths"][0]).items()})
''')

md(r"""
## 5. Diagnostics at initialisation (no training)

Before training we check that each network does what the theory says: the signal strength $\hat q_\ell$ of each layer
(mean square pre-activation, averaged over a batch) and the typical correlation between different images. Standard tanh
decays towards zero as $q_\ell\simeq1/(2\ell)$; the sweet spot stays at $q^*$; ReLU keeps its mean signal but its
correlations are pushed towards 1.
""")
code(r'''
def diagnostics(L, N, names, n_img=256, seed=0):
    torch.manual_seed(seed)
    x = Xte_t[:n_img].to(DEV)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for i, name in enumerate(names):
        rule = INIT_RULES(name, L, N)
        net = MLP(L, N, rule).to(DEV)
        with torch.no_grad():
            _, hid = net(x, return_hidden=True)
        q = [h.pow(2).mean().item() for h in hid]
        C = []
        for h in hid:
            hn = h / h.norm(dim=1, keepdim=True)
            G = hn @ hn.T
            C.append(((G.sum() - G.diag().sum()) / (n_img * (n_img - 1))).item())
        ell = np.arange(1, L + 1)
        ax[0].plot(ell, q, "o", ms=3, color=PAL[i % len(PAL)], label=name)
        ax[1].plot(ell, C, "o", ms=3, color=PAL[i % len(PAL)], label=name)
    ax[0].plot(ell, 1 / (2 * ell), "--", color="#3A3A3A", lw=1, label=r"$1/(2\ell)$")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel(r"$\ell$"); ylabel_top(ax[0], r"$\hat q_\ell$")
    ax[0].set_title(f"signal strength through depth (L={L}, N={N})", pad=22); ax[0].legend(fontsize=7)
    ax[1].set_xscale("log"); ax[1].set_xlabel(r"$\ell$"); ylabel_top(ax[1], r"$\langle C_\ell\rangle$")
    ax[1].set_title("mean correlation between different images", pad=22); ax[1].legend(fontsize=7)
    plt.tight_layout(); plt.show()

if CONFIG["run_diagnostics"]:
    for L in CONFIG["depths"]:
        diagnostics(L, CONFIG["widths"][0], CONFIG["inits"])
''')

md(r"""
## 6. Training

Plain mini-batch training with cross-entropy. We record the training loss every 20 steps, the running training
accuracy, the test accuracy after every epoch, and the number of steps needed to reach the target running accuracy.
SGD is the default because adaptive optimisers (Adam) rescale gradients per parameter and partly hide the effect of the
initialisation; switch `optimizer` to compare. With plain SGD the change of the network function per step grows with
depth, so the learning rate is scaled as $\eta=\eta_0\cdot32/L$ (set `lr_depth_scaling=False` to switch this off). Using
one learning rate for every initialisation can favour some of them; `lr_multipliers=[0.3, 1, 3]` trains each at three
learning rates and keeps the one with the lowest final training loss (the test set is not used for this choice), which
is the fairer comparison.
""")
code(r'''
def evaluate(net, X, y, bs=2000):
    net.eval(); correct = 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            correct += (net(X[i:i + bs].to(DEV)).argmax(1).cpu() == y[i:i + bs]).sum().item()
    net.train(); return correct / len(X)

def train_one(name, L, N, seed, lr_mult=1.0):
    torch.manual_seed(seed); np.random.seed(seed)
    rule = INIT_RULES(name, L, N)
    net = MLP(L, N, rule).to(DEV)
    lr = CONFIG["lr"] * lr_mult * (32 / L if CONFIG["lr_depth_scaling"] else 1.0)
    opt = (torch.optim.SGD(net.parameters(), lr=lr, momentum=CONFIG["momentum"])
           if CONFIG["optimizer"] == "sgd" else torch.optim.Adam(net.parameters(), lr=lr))
    n, bs = len(Xtr_t), CONFIG["batch_size"]
    log = dict(step=[], loss=[], test_acc=[], steps_to_target=None, diverged=False)
    run_acc, step, t0 = 0.0, 0, time.time()
    for ep in range(CONFIG["epochs"]):
        perm = torch.randperm(n)
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i + bs]
            xb, yb = Xtr_t[idx].to(DEV), ytr_t[idx].to(DEV)
            out = net(xb); loss = F.cross_entropy(out, yb)
            if not torch.isfinite(loss):
                log["diverged"] = True; break
            opt.zero_grad(); loss.backward()
            if CONFIG["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(net.parameters(), CONFIG["grad_clip"])
            opt.step(); step += 1
            run_acc = 0.98 * run_acc + 0.02 * (out.argmax(1) == yb).float().mean().item()
            if log["steps_to_target"] is None and run_acc / (1 - 0.98 ** step) >= CONFIG["target_acc"]:
                log["steps_to_target"] = step
            if step % 20 == 0:
                log["step"].append(step); log["loss"].append(loss.item())
        if log["diverged"]: break
        log["test_acc"].append(evaluate(net, Xte_t, yte_t))
    log["time"] = time.time() - t0
    log["lr"] = lr
    log["final_loss"] = float(np.mean(log["loss"][-10:])) if (log["loss"] and not log["diverged"]) else float("inf")
    return log

def train_best(name, L, N, seed):
    """train at every lr multiplier and keep the run with the lowest final training loss (the test set is not used)."""
    runs = [train_one(name, L, N, seed, m) for m in CONFIG["lr_multipliers"]]
    return min(runs, key=lambda r: r["final_loss"])

RESULTS = {}
for L in CONFIG["depths"]:
    for N in CONFIG["widths"]:
        for name in CONFIG["inits"]:
            for s in CONFIG["seeds"]:
                r = train_best(name, L, N, s)
                RESULTS[(L, N, name, s)] = r
                acc = r["test_acc"][-1] if r["test_acc"] else float("nan")
                print(f"L={L:3d} N={N:3d} {name:17s} seed={s} lr={r['lr']:.4f}: test acc {acc:.4f}, "
                      f"steps to {CONFIG['target_acc']:.0%}: {r['steps_to_target']}, "
                      f"{'DIVERGED ' if r['diverged'] else ''}({r['time']:.0f}s)")

if CONFIG["results_file"]:
    import pickle
    with open(CONFIG["results_file"], "wb") as f:
        pickle.dump(dict(config=CONFIG, results=RESULTS), f)
''')

md(r"""
## 7. Results
Mean over seeds (error bars: standard error). Left: test accuracy after the last epoch. Middle: steps to reach the
target running training accuracy (lower is better; missing if never reached). Right: training-loss curves for the
deepest network.
""")
code(r'''
def summarise(L, N, name, key):
    vals = []
    for s in CONFIG["seeds"]:
        r = RESULTS[(L, N, name, s)]
        if key == "acc": vals.append(r["test_acc"][-1] if r["test_acc"] and not r["diverged"] else np.nan)
        else: vals.append(r["steps_to_target"] if r["steps_to_target"] is not None else np.nan)
    vals = np.array(vals, float)
    ok = ~np.isnan(vals)
    if not ok.any():
        return np.nan, np.nan
    return vals[ok].mean(), vals[ok].std() / ok.sum() ** 0.5

for N in CONFIG["widths"]:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    names = CONFIG["inits"]; x = np.arange(len(names)); w = 0.8 / len(CONFIG["depths"])
    for j, L in enumerate(CONFIG["depths"]):
        acc = np.array([summarise(L, N, n, "acc") for n in names])
        stp = np.array([summarise(L, N, n, "steps") for n in names])
        ax[0].bar(x + j * w, acc[:, 0], w, yerr=acc[:, 1], color=PAL[j], label=f"L={L}")
        ax[1].bar(x + j * w, stp[:, 0], w, yerr=stp[:, 1], color=PAL[j], label=f"L={L}")
    for a_, t in zip(ax[:2], ["test accuracy", f"steps to {CONFIG['target_acc']:.0%} train accuracy"]):
        a_.set_xticks(x + w * (len(CONFIG["depths"]) - 1) / 2); a_.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        a_.set_title(f"{t} (N={N})"); a_.legend(fontsize=8)
    ax[0].set_ylim(0.0, 1.0)
    L = max(CONFIG["depths"])
    for i, name in enumerate(names):
        r = RESULTS[(L, N, name, CONFIG["seeds"][0])]
        ax[2].plot(r["step"], r["loss"], "-", lw=1.2, color=PAL[i % len(PAL)], label=name)
    ax[2].set_yscale("log"); ax[2].set_xlabel("step"); ylabel_top(ax[2], "loss")
    ax[2].set_title(f"training loss, L={L}, N={N}", pad=22); ax[2].legend(fontsize=7)
    plt.tight_layout(); plt.show()

print(f"{'init':18s}" + "".join(f"  L={L:<4d} acc      steps" for L in CONFIG["depths"]))
for N in CONFIG["widths"]:
    for name in CONFIG["inits"]:
        row = f"{name:18s}"
        for L in CONFIG["depths"]:
            a, ea = summarise(L, N, name, "acc"); s_, es = summarise(L, N, name, "steps")
            row += f"  {a:.4f}±{ea:.4f} {s_:7.0f}"
        print(row)
''')

md(r"""
## 8. (Optional) Trainability over the $(\sigma_w^2,\sigma_b^2)$ plane

Accuracy of a deep tanh network after a fixed number of steps, over a grid of initialisations, with the mean-field edge
of chaos (dashed) and the sweet spot (star) overlaid. Networks train well in a band around the edge of chaos whose
width shrinks with depth.
""")
code(r'''
def phase_diagram(cfg):
    L, N, steps = cfg["L"], cfg["N"], cfg["steps"]
    A = np.zeros((len(cfg["sb2"]), len(cfg["sw2"])))
    for i, sb2 in enumerate(cfg["sb2"]):
        for j, sw2 in enumerate(cfg["sw2"]):
            torch.manual_seed(0)
            rule = dict(act="tanh", sw2=sw2, sb2=sb2, q1=max(fixed_point(np.tanh, sw2, sb2), 1e-3))
            net = MLP(L, N, rule).to(DEV)
            lr = CONFIG["lr"] * (32 / L if CONFIG["lr_depth_scaling"] else 1.0)
            opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=CONFIG["momentum"])
            for k in range(steps):
                idx = torch.randint(0, len(Xtr_t), (CONFIG["batch_size"],))
                loss = F.cross_entropy(net(Xtr_t[idx].to(DEV)), ytr_t[idx].to(DEV))
                if not torch.isfinite(loss): break
                opt.zero_grad(); loss.backward(); opt.step()
            A[i, j] = evaluate(net, Xte_t[:5000], yte_t[:5000])
        print("sigma_b^2 =", sb2, np.round(A[i], 3))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.pcolormesh(cfg["sw2"], range(len(cfg["sb2"])), A, cmap="Blues", shading="nearest", vmin=0.1, vmax=1)
    qq = np.logspace(-4, 0.5, 200); pts = np.array([eoc_point("tanh", q) for q in qq])
    yy = np.interp(pts[:, 1], cfg["sb2"], range(len(cfg["sb2"])))
    ok = (pts[:, 1] <= max(cfg["sb2"])) & (pts[:, 0] <= max(cfg["sw2"]))
    ax.plot(pts[ok, 0], yy[ok], "--", color="#B8433D", label="edge of chaos (mean field)")
    q, sw2, sb2 = sweet_spot_tanh(L, CONFIG["gamma"])
    ax.plot([sw2], [np.interp(sb2, cfg["sb2"], range(len(cfg["sb2"])))], "*", ms=14, color="#E8A33C", label="sweet spot")
    ax.set_yticks(range(len(cfg["sb2"]))); ax.set_yticklabels(cfg["sb2"])
    ax.set_xlabel(r"$\sigma_w^2$"); ylabel_top(ax, r"$\sigma_b^2$")
    ax.set_title(f"test accuracy after {steps} steps (L={L}, N={N})", pad=22)
    plt.colorbar(im, ax=ax); ax.legend(fontsize=8, loc="upper left"); plt.tight_layout(); plt.show()
    return A

if CONFIG["run_phase_diagram"]:
    PHASE = phase_diagram(CONFIG["phase_grid"])
''')

md(r"""
## 9. What to look for

* **Finite-N correction.** The effect of $\sigma_c^2(N)$ versus the standard value grows with the aspect ratio $L/N$: the
  typical signal of a standard-initialised network shrinks by about $e^{-L/N}$ (tanh) or $e^{-5L/(2N)}$ (ReLU). Expect no
  difference for $L\ll N$ and a visible one for $L\gtrsim N$. (Research note: at $L=96$ it raised tanh test accuracy by
  about 0.08 for $N=64$ and 0.22 for $N=32$; a scan of $\sigma_w^2=1+k/N$ showed a threshold near $k\approx1$–$2$ rather
  than a sharp optimum; for ReLU, which is positively homogeneous, it had no effect.)
* **Sweet spot.** Standard tanh at $q^*=0$ is linear deep down (its signal decays as $1/(2\ell)$); the sweet spot keeps a
  steady non-zero signal and a controlled amount of nonlinearity. Vary `gamma` (e.g. 0.3, 1, 2.3) to see the trade-off.
  (Research note: about 25% fewer steps at $L=32$, and 0.89 against 0.70 test accuracy at $L=96$, $N=64$.)
* **Linear.** Trains fast but saturates at logistic-regression accuracy (about 92%): being critical everywhere is not
  enough.
* **ReLU / shaped leaky ReLU.** Plain ReLU compresses input similarities strongly with depth; the shaped slope keeps
  the compression at $\gamma$. (Research note: the shaped leaky ReLU was the best rule at both depths, 0.92 at $L=96$.)
""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "mnist_criticality.ipynb")
print("wrote mnist_criticality.ipynb with", len(C), "cells")
