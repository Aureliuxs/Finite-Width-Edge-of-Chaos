"""
Fully tuned comparison of initialisation rules (Fashion-MNIST, SM Sec. S8).

Many independent networks are trained at once with batched matrix products, as in eocfw.mnist_batched, but here
every member of a batch can have its own initialisation (sigma_w^2, sigma_b^2, input signal q_1, weight
ensemble, leaky slope) and its own learning rate, so that a whole grid of learning rates and gains is one batch.
A batch can be trained epoch by epoch, reduced to a subset of its members, and saved to disk and resumed, which
is what the successive-halving search of scripts/run_fmnist_tuned.py needs.

Every member has its own seed for the weights and for the data order and its own momentum buffer, so the members
are exactly independent.  The optimiser is SGD with momentum 0.9 and a constant learning rate, batch size 128,
no normalisation and no skip connections (the protocol of Table I of the paper).
"""
import gzip
import math
import os
import time
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F

FASHION = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
TARGETS = (0.6, 0.7, 0.75, 0.8, 0.85, 0.9)      # training accuracies whose first passage is recorded


def load_fashion(root, n_val=10_000):
    """Fashion-MNIST as standardised float32 tensors: (Xtr, ytr, Xval, yval, Xte, yte).  The last n_val images of
    the official training set are held out for validation (used only to choose hyperparameters); the official
    test set is used only for the reported accuracies."""
    os.makedirs(root, exist_ok=True)
    for f in FILES:
        p = os.path.join(root, f)
        if not os.path.exists(p):
            urllib.request.urlretrieve(FASHION + f, p)

    def img(f):
        with gzip.open(os.path.join(root, f)) as g:
            return np.frombuffer(g.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255

    def lab(f):
        with gzip.open(os.path.join(root, f)) as g:
            return np.frombuffer(g.read(), np.uint8, offset=8).astype(np.int64)

    X, y, Xte, yte = img(FILES[0]), lab(FILES[1]), img(FILES[2]), lab(FILES[3])
    Xtr, ytr, Xval, yval = X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]
    mu, sd = Xtr.mean(), Xtr.std()
    t = lambda a: torch.tensor(a)
    return t((Xtr - mu) / sd), t(ytr), t((Xval - mu) / sd), t(yval), t((Xte - mu) / sd), t(yte)


def _orthogonal(gen, n):
    a = torch.randn(n, n, generator=gen)
    q, r = torch.linalg.qr(a)
    return q * torch.sign(torch.diagonal(r))


class Batch:
    """K independent networks with L hidden layers of width N.  `members` is a list of dicts with keys
    act ('tanh' | 'relu' | 'leaky', the same for all members), alpha, sw2, sb2, q1, ens ('gaussian' |
    'orthogonal'), lr, seed (and any labels, which are kept).  Optional keys: first_layer, how the first hidden
    layer is brought to the signal q1 ('weight': the first-layer weights have variance (q1 - sb2)/d_in, as the
    crinit package does; the default; 'input': the weights have variance sw2/d_in and the input is multiplied by
    sqrt((q1 - sb2)/sw2) instead, as in the earlier runs; the forward pass at initialisation is the same, the
    gradient of the first layer is not), and lr1_mult, a factor on the learning rate of the first-layer weights
    (default 1)."""

    def __init__(self, data, L, N, members, batch_size=128, momentum=0.9, targets=TARGETS, d_in=784, d_out=10):
        self.data, self.L, self.N = data, L, N
        self.members = [dict(m) for m in members]
        acts = {m["act"] for m in members}
        assert len(acts) == 1, "one activation family per batch"
        self.act = acts.pop()
        self.bs, self.momentum, self.targets = batch_size, momentum, np.asarray(targets)
        K = len(members)
        f = lambda key: torch.tensor([float(m.get(key, 0.0)) for m in members])
        sw, sb = f("sw2").sqrt(), f("sb2").clamp(min=0).sqrt()
        self.lr = f("lr")
        self.alpha = f("alpha").view(K, 1, 1)
        self.in_scale = ((f("q1") - f("sb2")).clamp(min=1e-12) / f("sw2")).sqrt().view(K, 1, 1)
        self.lr1 = torch.tensor([float(m.get("lr1_mult", 1.0)) for m in members])
        by_weight = torch.tensor([m.get("first_layer", "weight") == "weight" for m in members])
        Ws = [torch.empty(K, d_in, N)] + [torch.empty(K, N, N) for _ in range(L - 1)] + [torch.empty(K, N, d_out)]
        bs = [torch.zeros(K, N) for _ in range(L)] + [torch.zeros(K, d_out)]
        for k, m in enumerate(members):
            gen = torch.Generator().manual_seed(int(m["seed"]))
            Ws[0][k] = torch.randn(d_in, N, generator=gen) * (sw[k] / math.sqrt(d_in))
            for i in range(1, L):
                if m["ens"] == "orthogonal":
                    Ws[i][k] = _orthogonal(gen, N) * sw[k]
                else:
                    Ws[i][k] = torch.randn(N, N, generator=gen) * (sw[k] / math.sqrt(N))
            if sb[k] > 0:
                for i in range(L):
                    bs[i][k] = torch.randn(N, generator=gen) * sb[k]
            Ws[L][k] = torch.randn(N, d_out, generator=gen) / math.sqrt(N)
        if by_weight.any():
            Ws[0][by_weight] *= self.in_scale[by_weight]
            self.in_scale = torch.where(by_weight.view(K, 1, 1), torch.ones_like(self.in_scale), self.in_scale)
        self.params = Ws + bs
        for p in self.params:
            p.requires_grad_(True)
        self.bufs = [torch.zeros_like(p) for p in self.params]
        self.gens = [torch.Generator().manual_seed(int(m["seed"]) + 10_000_019) for m in members]
        self.step = 0
        self.epoch = 0
        self.run_acc = torch.zeros(K)
        self.first = np.full((K, len(self.targets)), -1, int)          # first step at each training accuracy
        self.diverged = np.zeros(K, bool)
        self.hist = dict(val=[], test=[], loss=[])                    # per epoch, (K,) each
        self.curve_step, self.curve_acc = [], []

    # -- network --------------------------------------------------------------------------------------------
    def _act(self, h):
        if self.act == "tanh":
            return torch.tanh(h)
        # leaky ReLU with a slope per member: PReLU over the member axis (one fused kernel)
        return F.prelu(h.transpose(0, 1), self.alpha.view(-1)).transpose(0, 1)

    def forward(self, x):
        L = self.L
        Ws, bs = self.params[:L + 1], self.params[L + 1:]
        h = torch.baddbmm(bs[0].unsqueeze(1), x * self.in_scale, Ws[0])
        for W, b in zip(Ws[1:], bs[1:]):
            h = torch.baddbmm(b.unsqueeze(1), self._act(h), W)
        return h

    @torch.no_grad()
    def accuracy(self, X, y, chunk=1000):
        K = len(self.members)
        correct = torch.zeros(K)
        for i in range(0, len(X), chunk):
            xb = X[i:i + chunk].unsqueeze(0).expand(K, -1, -1)
            correct += (self.forward(xb).argmax(-1) == y[i:i + chunk]).float().sum(1)
        acc = (correct / len(X)).numpy()
        acc[self.diverged] = np.nan
        return acc

    # -- training -------------------------------------------------------------------------------------------
    def run_epoch(self, val=True, test=False, curve_every=0, curve_n=1000):
        Xtr, ytr, Xval, yval, Xte, yte = self.data
        K, B = len(self.members), self.bs
        n = len(Xtr)
        perms = torch.stack([torch.randperm(n, generator=g) for g in self.gens])
        losses = []
        lr = self.lr
        for i in range(n // B):
            idx = perms[:, i * B:(i + 1) * B]
            xb, yb = Xtr[idx], ytr[idx]
            out = self.forward(xb)
            lossk = F.cross_entropy(out.reshape(-1, out.shape[-1]), yb.reshape(-1), reduction="none").view(K, -1).mean(1)
            bad = ~torch.isfinite(lossk.detach())
            if bad.any():
                self.diverged |= bad.numpy()
                lossk = torch.where(bad, torch.zeros_like(lossk), lossk)
            for p in self.params:
                p.grad = None
            lossk.sum().backward()
            with torch.no_grad():
                dead = torch.as_tensor(self.diverged)
                for j, (p, v) in enumerate(zip(self.params, self.bufs)):
                    g = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                    v.mul_(self.momentum).add_(g)
                    v[dead] = 0.0
                    lr_p = lr * self.lr1 if j == 0 else lr
                    p.sub_(lr_p.view((K,) + (1,) * (p.dim() - 1)) * v)
                self.step += 1
                hit = (out.argmax(-1) == yb).float().mean(1)
                self.run_acc.mul_(0.98).add_(0.02 * hit)
                ra = (self.run_acc / (1 - 0.98 ** self.step)).numpy()
                newly = (ra[:, None] >= self.targets[None, :]) & (self.first < 0) & ~self.diverged[:, None]
                self.first[newly] = self.step
            losses.append(lossk.detach().numpy())
            if curve_every and self.step % curve_every == 0:
                self.curve_step.append(self.step)
                self.curve_acc.append(self.accuracy(Xte[:curve_n], yte[:curve_n]))
        self.epoch += 1
        loss = np.mean(losses, 0)
        loss[self.diverged] = np.inf
        self.hist["loss"].append(loss)
        self.hist["val"].append(self.accuracy(Xval, yval) if val else np.full(K, np.nan))
        self.hist["test"].append(self.accuracy(Xte, yte) if test else np.full(K, np.nan))

    # -- bookkeeping ----------------------------------------------------------------------------------------
    def keep(self, idx):
        """Reduce the batch to the members idx (in this order)."""
        idx = np.asarray(idx, int)
        t = torch.as_tensor(idx)
        with torch.no_grad():
            self.params = [p.detach()[t].clone().requires_grad_(True) for p in self.params]
            self.bufs = [v[t].clone() for v in self.bufs]
        self.members = [self.members[i] for i in idx]
        self.gens = [self.gens[i] for i in idx]
        self.lr, self.alpha, self.in_scale, self.lr1 = self.lr[t], self.alpha[t], self.in_scale[t], self.lr1[t]
        self.run_acc = self.run_acc[t]
        self.first, self.diverged = self.first[idx], self.diverged[idx]
        self.hist = {k: [h[idx] for h in v] for k, v in self.hist.items()}
        self.curve_acc = [c[idx] for c in self.curve_acc]

    def state(self):
        return dict(L=self.L, N=self.N, members=self.members, act=self.act, bs=self.bs, momentum=self.momentum,
                    targets=self.targets, params=[p.detach() for p in self.params], bufs=self.bufs,
                    gens=[g.get_state() for g in self.gens], step=self.step, epoch=self.epoch, run_acc=self.run_acc,
                    first=self.first, diverged=self.diverged, hist=self.hist, curve_step=self.curve_step,
                    curve_acc=self.curve_acc, lr=self.lr, alpha=self.alpha, in_scale=self.in_scale, lr1=self.lr1)

    @classmethod
    def from_state(cls, data, s):
        self = cls.__new__(cls)
        self.data = data
        for k in ("L", "N", "members", "act", "bs", "momentum", "targets", "bufs", "step", "epoch", "run_acc", "first",
                  "diverged", "hist", "curve_step", "curve_acc", "lr", "alpha", "in_scale"):
            setattr(self, k, s[k])
        self.lr1 = s.get("lr1", torch.ones(len(self.members)))
        self.params = [p.clone().requires_grad_(True) for p in s["params"]]
        self.gens = []
        for st in s["gens"]:
            g = torch.Generator()
            g.set_state(st)
            self.gens.append(g)
        return self

    def summary(self):
        """Per-member results as plain arrays (what is stored in the results files)."""
        return dict(members=self.members, val=np.array(self.hist["val"]), test=np.array(self.hist["test"]),
                    loss=np.array(self.hist["loss"]), first=self.first.copy(), targets=self.targets,
                    diverged=self.diverged.copy(), steps_per_epoch=len(self.data[0]) // self.bs,
                    curve_step=np.array(self.curve_step), curve_acc=np.array(self.curve_acc))
