"""
MNIST training of deep fully connected networks from the initialisations of the paper.

Rules (INIT_RULES): each returns the activation, the hidden-layer variances (sigma_w^2, sigma_b^2), the input
signal q_1 of the first hidden layer, and the weight ensemble ('gaussian' or 'orthogonal').

  tanh_default         tanh, sigma_w^2 = 1/3 (the PyTorch default of nn.Linear)
  tanh_standard        tanh, sigma_w^2 = 1 (LeCun; Glorot for square layers)
  tanh_finiteN         tanh, sigma_w^2 = sigma_c^2(N) (typical-preserving critical point)
  tanh_sweet:gamma     tanh on the mean-field edge of chaos at the steady signal q* with compression gamma
                       (alias tanh_eoc:gamma)
  tanh_sweetN:gamma    same, sigma_w^2 moved by the fixed-bias shift of Eq. (10) (lands on the finite-width line
                       at a larger q*; the SM rows "shift at fixed bias")
  tanh_crinit:gamma    the Crinit rule: the point of the finite-width edge of chaos with the steady signal of the
                       budget (computed with the crinit package)
  tanh_chaotic         tanh, sigma_w^2 = 2 (He)
  relu_he              ReLU, sigma_w^2 = 2
  relu_finiteN         ReLU, sigma_w^2 = 2 exp(5/(2N))
  relu_crinit          ReLU, the exact typical-preserving sigma_c^2(N) of the crinit package
  leaky_he             leaky ReLU with the usual slope 0.01, sigma_w^2 = 2/(1 + 0.01^2)
  leaky:gamma          leaky ReLU shaped to compression gamma over the depth, sigma_w^2 = sigma_c^2(N, alpha)
                       (alias leaky_crinit:gamma, computed with the crinit package)
  linear               identity, sigma_w^2 = 1
  orth_<rule>          any of the above with Haar-orthogonal hidden weights (orth_tanh_crinit uses the
                       orthogonal-ensemble shift)
"""
import gzip
import math
import os
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .finite_width import eoc_factor_at_q, sigma_c2_leaky, sigma_c2_smooth
from .meanfield import shaped_leaky_alpha, sweet_spot

MIRROR = "https://raw.githubusercontent.com/fgnt/mnist/master/"
FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]


def load_mnist(root):
    """Flattened, standardised MNIST as float32 tensors (downloads from a GitHub mirror if needed)."""
    raw = os.path.join(root, "MNIST", "raw")
    os.makedirs(raw, exist_ok=True)
    for f in FILES:
        p = os.path.join(raw, f)
        if not os.path.exists(p):
            urllib.request.urlretrieve(MIRROR + f, p)

    def img(f):
        with gzip.open(os.path.join(raw, f)) as g:
            return np.frombuffer(g.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255

    def lab(f):
        with gzip.open(os.path.join(raw, f)) as g:
            return np.frombuffer(g.read(), np.uint8, offset=8).astype(np.int64)

    Xtr, ytr, Xte, yte = img(FILES[0]), lab(FILES[1]), img(FILES[2]), lab(FILES[3])
    mu, sd = Xtr.mean(), Xtr.std()
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    return (torch.tensor(Xtr), torch.tensor(ytr), torch.tensor(Xte), torch.tensor(yte))


def _crinit():
    """The crinit package (sibling folder of the code folder, or installed)."""
    try:
        import crinit
    except ImportError:
        import sys
        here = os.path.dirname(os.path.abspath(__file__))
        for rel in ("..", "../..", "../../.."):            # crinit/ next to this code folder or one level up
            cand = os.path.normpath(os.path.join(here, rel, "crinit"))
            if os.path.isdir(os.path.join(cand, "crinit")):
                sys.path.insert(0, cand)
                break
        import crinit
    return crinit


def INIT_RULES(name, L, N):
    ens = "gaussian"
    if name.startswith("orth_"):
        ens, name = "orthogonal", name[5:]
    base, _, arg = name.partition(":")
    g = float(arg) if arg else 1.0
    if base == "tanh_crinit":                                # the Crinit rule, from the package
        c = _crinit().crinit("tanh", depth=L, width=N, gamma=g, ensemble=ens)
        r = dict(act="tanh", sw2=c.sigma_w2, sb2=c.sigma_b2, q1=c.q_star)
    elif base == "tanh_eoc":                                 # mean-field edge of chaos at the same point
        c = _crinit().crinit("tanh", depth=L, width=N, gamma=g, ensemble=ens, finite_width=False)
        r = dict(act="tanh", sw2=c.sigma_w2, sb2=c.sigma_b2, q1=c.q_star)
    elif base == "leaky_crinit":
        c = _crinit().crinit("leaky_relu", depth=L, width=N, gamma=g, ensemble=ens)
        r = dict(act="leaky", alpha=c.alpha, sw2=c.sigma_w2, sb2=0.0, q1=1.0)
    elif base == "relu_crinit":
        c = _crinit().crinit("relu", depth=L, width=N, ensemble=ens)
        r = dict(act="relu", sw2=c.sigma_w2, sb2=0.0, q1=1.0)
    elif base == "leaky_he":
        r = dict(act="leaky", alpha=0.01, sw2=2 / (1 + 0.01 ** 2), sb2=0.0, q1=2 / (1 + 0.01 ** 2))
    elif base == "tanh_default":
        r = dict(act="tanh", sw2=1.0 / 3.0, sb2=0.0, q1=1.0 / 3.0)
    elif base == "tanh_standard":
        r = dict(act="tanh", sw2=1.0, sb2=0.0, q1=1.0)
    elif base == "tanh_finiteN":
        r = dict(act="tanh", sw2=sigma_c2_smooth(N), sb2=0.0, q1=1.0)
    elif base == "tanh_chaotic":
        r = dict(act="tanh", sw2=2.0, sb2=0.0, q1=2.0)
    elif base == "tanh_k":                                   # sigma_w^2 = 1 + k/N
        r = dict(act="tanh", sw2=1 + g / N, sb2=0.0, q1=1.0)
    elif base in ("tanh_sweet", "tanh_sweetN"):                # tanh_sweetN: the shift of Eq. (10) at fixed bias
        q, sw2, sb2 = sweet_spot("tanh", L, g)
        if base == "tanh_sweetN":
            c = _crinit().crinit("tanh", depth=L, width=N, gamma=g, ensemble=ens, finite_width=False)
            sw2 = c.sigma_w2 + _crinit().shift("tanh", c.q_star, N, c.sigma_w2, orthogonal=(ens == "orthogonal"))[0]
        r = dict(act="tanh", sw2=sw2, sb2=sb2, q1=q)
    elif base == "relu_he":
        r = dict(act="relu", sw2=2.0, sb2=0.0, q1=2.0)
    elif base == "relu_finiteN":
        r = dict(act="relu", sw2=sigma_c2_leaky(N, 0.0), sb2=0.0, q1=1.0)
    elif base == "leaky":
        a = shaped_leaky_alpha(L, g)
        r = dict(act="leaky", alpha=a, sw2=sigma_c2_leaky(N, a) if ens == "gaussian" else 2 / (1 + a * a), sb2=0.0, q1=1.0)
    elif base == "linear":
        r = dict(act="linear", sw2=1.0, sb2=0.0, q1=1.0)
    else:
        raise ValueError(name)
    r["ensemble"] = ens
    return r


def torch_act(rule):
    a = rule.get("alpha", 0.0)
    return {"tanh": torch.tanh, "relu": F.relu, "linear": (lambda x: x), "leaky": (lambda x: F.leaky_relu(x, a))}[rule["act"]]


class MLP(nn.Module):
    """L hidden layers of width N; Gaussian weights with variance sigma_w^2/fan_in, or orthogonal weights scaled
    so that every hidden layer preserves the norm by the factor sigma_w^2 exactly; biases N(0, sigma_b^2)."""

    def __init__(self, L, N, rule, d_in=784, d_out=10):
        super().__init__()
        self.rule, self.act = rule, torch_act(rule)
        self.layers = nn.ModuleList([nn.Linear(d_in if i == 0 else N, N) for i in range(L)])
        self.head = nn.Linear(N, d_out)
        with torch.no_grad():
            for i, lin in enumerate(self.layers):
                if rule["ensemble"] == "orthogonal" and i > 0:
                    nn.init.orthogonal_(lin.weight, gain=math.sqrt(rule["sw2"]))
                else:
                    lin.weight.normal_(0, math.sqrt(rule["sw2"] / lin.in_features))
                if rule["sb2"] > 0:
                    lin.bias.normal_(0, math.sqrt(rule["sb2"]))
                else:
                    lin.bias.zero_()
            self.head.weight.normal_(0, math.sqrt(1.0 / N))
            self.head.bias.zero_()
        self.in_scale = math.sqrt(max(rule["q1"] - rule["sb2"], 1e-12) / rule["sw2"])

    def forward(self, x):
        h = self.layers[0](x * self.in_scale)
        for lin in self.layers[1:]:
            h = lin(self.act(h))
        return self.head(self.act(h))


def evaluate(net, X, y, bs=2000):
    net.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            correct += (net(X[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
    net.train()
    return correct / len(X)


def train_one(data, name, L, N, seed, lr, epochs=2, batch_size=128, momentum=0.9, target_acc=0.85, train_subset=None):
    Xtr, ytr, Xte, yte = data
    if train_subset:
        Xtr, ytr = Xtr[:train_subset], ytr[:train_subset]
    torch.manual_seed(seed)
    np.random.seed(seed)
    rule = INIT_RULES(name, L, N)
    net = MLP(L, N, rule)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    n = len(Xtr)
    log = dict(name=name, L=L, N=N, seed=seed, lr=lr, rule=rule, step=[], loss=[], test_acc=[], steps_to_target=None,
               diverged=False)
    run_acc, step, t0 = 0.0, 0, time.time()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size]
            out = net(Xtr[idx])
            loss = F.cross_entropy(out, ytr[idx])
            if not torch.isfinite(loss):
                log["diverged"] = True
                break
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            run_acc = 0.98 * run_acc + 0.02 * (out.argmax(1) == ytr[idx]).float().mean().item()
            if log["steps_to_target"] is None and run_acc / (1 - 0.98 ** step) >= target_acc:
                log["steps_to_target"] = step
            if step % 20 == 0:
                log["step"].append(step)
                log["loss"].append(loss.item())
        if log["diverged"]:
            break
        log["test_acc"].append(evaluate(net, Xte, yte))
    log["time"] = time.time() - t0
    log["final_loss"] = float(np.mean(log["loss"][-10:])) if (log["loss"] and not log["diverged"]) else float("inf")
    return log


def train_best(data, name, L, N, seed, lr0=0.005, lr_multipliers=(0.3, 1.0, 3.0), **kw):
    """Train at lr = lr0 * m * 32/L for every multiplier m and keep the run with the lowest final training loss."""
    runs = [train_one(data, name, L, N, seed, lr0 * m * 32 / L, **kw) for m in lr_multipliers]
    return min(runs, key=lambda r: r["final_loss"])
