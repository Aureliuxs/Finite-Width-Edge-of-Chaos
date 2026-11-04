"""
Many independent networks trained at once (the large-statistics comparison of Table I).

K networks of the same architecture and initialisation rule are stacked along a leading axis and trained with
batched matrix products.  Every member has its own initialisation seed, its own data order and its own
momentum buffers, so the K runs are exactly independent; stacking them only removes the Python overhead of
tiny 64 x 64 layers.  The optimiser is plain SGD with momentum, written out so that the learning rate can
differ between members (used for the learning-rate selection).
"""
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .mnist import INIT_RULES


def _orthogonal(gen, n, m):
    """Haar-orthogonal n x m matrix (n >= m), as nn.init.orthogonal_ does it."""
    a = torch.randn(n, m, generator=gen)
    q, r = torch.linalg.qr(a)
    return q * torch.sign(torch.diagonal(r))


def init_members(K, L, N, rule, seeds, d_in=784, d_out=10, first_layer="weight"):
    """Stacked parameters, one leaf tensor per layer with the member as leading axis:
    weights [W0 (K, d_in, N), W1..W_{L-1} (K, N, N), Wout (K, N, d_out)] and biases [b1..bL (K, N), bout (K, d_out)].
    first_layer: 'weight' gives the first-layer weights the variance (q1 - sb2)/d_in, so that the first hidden
    layer has the signal q1 of the rule (what the crinit package does); 'input' gives them sw2/d_in and leaves the
    rescaling to the input (the earlier implementation)."""
    sw, sb = math.sqrt(rule["sw2"]), math.sqrt(max(rule["sb2"], 0.0))
    s1 = math.sqrt(max(rule["q1"] - rule["sb2"], 1e-12)) if first_layer == "weight" else sw
    Ws = [torch.empty(K, d_in, N)] + [torch.empty(K, N, N) for _ in range(L - 1)] + [torch.empty(K, N, d_out)]
    bs = [torch.zeros(K, N) for _ in range(L)] + [torch.zeros(K, d_out)]
    for k, seed in enumerate(seeds):
        gen = torch.Generator().manual_seed(int(seed))
        Ws[0][k] = torch.randn(d_in, N, generator=gen) * (s1 / math.sqrt(d_in))
        for i in range(1, L):
            if rule["ensemble"] == "orthogonal":
                Ws[i][k] = _orthogonal(gen, N, N) * sw
            else:
                Ws[i][k] = torch.randn(N, N, generator=gen) * (sw / math.sqrt(N))
        if sb > 0:
            for i in range(L):
                bs[i][k] = torch.randn(N, generator=gen) * sb
        Ws[L][k] = torch.randn(N, d_out, generator=gen) / math.sqrt(N)
    params = Ws + bs
    for p in params:
        p.requires_grad_(True)
    return Ws, bs


def act_fn(rule):
    a = rule.get("alpha", 0.0)
    return {"tanh": torch.tanh, "relu": F.relu, "linear": (lambda x: x), "leaky": (lambda x: F.leaky_relu(x, a))}[rule["act"]]


def forward(Ws, bs, x, act, in_scale):
    """x: (K, B, d_in) -> logits (K, B, d_out)."""
    h = torch.baddbmm(bs[0].unsqueeze(1), x * in_scale, Ws[0])
    for W, b in zip(Ws[1:], bs[1:]):
        h = torch.baddbmm(b.unsqueeze(1), act(h), W)
    return h


@torch.no_grad()
def evaluate(Ws, bs, act, in_scale, X, y, chunk=1000):
    """Test accuracy of every member, (K,)."""
    K = Ws[0].shape[0]
    correct = torch.zeros(K)
    for i in range(0, len(X), chunk):
        xb = X[i:i + chunk].unsqueeze(0).expand(K, -1, -1)
        correct += (forward(Ws, bs, xb, act, in_scale).argmax(-1) == y[i:i + chunk]).float().sum(1)
    return (correct / len(X)).numpy()


def train_batched(data, name, L, N, seeds, lr, epochs=2, batch_size=128, momentum=0.9, target_acc=0.85,
                  eval_every=0, eval_subset=2000, log=print, first_layer="weight"):
    """Train len(seeds) independent networks of rule `name`.  `lr` is a float or an array of per-member rates.
    Returns per-member arrays: test_acc (final), test_acc_epoch (epochs, K), steps_to_target (-1 if never),
    final_loss, diverged, and the learning curve (curve_step, curve_acc) if eval_every > 0."""
    Xtr, ytr, Xte, yte = data
    K = len(seeds)
    rule = INIT_RULES(name, L, N)
    act = act_fn(rule)
    in_scale = math.sqrt(max(rule["q1"] - rule["sb2"], 1e-12) / rule["sw2"]) if first_layer == "input" else 1.0
    Ws, bs = init_members(K, L, N, rule, seeds, first_layer=first_layer)
    params = Ws + bs
    lr = torch.as_tensor(np.broadcast_to(np.asarray(lr, dtype=np.float32), (K,)).copy())
    bufs = [torch.zeros_like(p) for p in params]
    n = len(Xtr)
    steps_per_epoch = n // batch_size
    gens = [torch.Generator().manual_seed(int(s) + 10_000_019) for s in seeds]
    run_acc = torch.zeros(K)
    steps_to_target = np.full(K, -1, int)
    diverged = np.zeros(K, bool)
    acc_epoch, curve_step, curve_acc, loss_hist = [], [], [], []
    step, t0 = 0, time.time()
    for ep in range(epochs):
        perms = torch.stack([torch.randperm(n, generator=g) for g in gens])        # (K, n)
        for i in range(steps_per_epoch):
            idx = perms[:, i * batch_size:(i + 1) * batch_size]                      # (K, B)
            xb, yb = Xtr[idx], ytr[idx]                                               # (K, B, 784), (K, B)
            out = forward(Ws, bs, xb, act, in_scale)
            lossk = F.cross_entropy(out.reshape(-1, out.shape[-1]), yb.reshape(-1), reduction="none").view(K, -1).mean(1)
            bad = ~torch.isfinite(lossk.detach())
            if bad.any():
                diverged |= bad.numpy()
                lossk = torch.where(bad, torch.zeros_like(lossk), lossk)
            for p in params:
                p.grad = None
            lossk.sum().backward()
            with torch.no_grad():
                for p, v in zip(params, bufs):
                    g = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                    v.mul_(momentum).add_(g)
                    p.sub_(lr.view((K,) + (1,) * (p.dim() - 1)) * v)
                step += 1
                hit = (out.argmax(-1) == yb).float().mean(1)
                run_acc.mul_(0.98).add_(0.02 * hit)
                reached = (run_acc / (1 - 0.98 ** step) >= target_acc).numpy() & (steps_to_target < 0) & ~diverged
                steps_to_target[reached] = step
            loss_hist.append(lossk.detach().numpy())
            if eval_every and step % eval_every == 0:
                curve_step.append(step)
                curve_acc.append(evaluate(Ws, bs, act, in_scale, Xte[:eval_subset], yte[:eval_subset]))
        acc_epoch.append(evaluate(Ws, bs, act, in_scale, Xte, yte))
        log(f"  {name} L={L} N={N} K={K} epoch {ep + 1}: mean test acc "
            f"{np.nanmean(np.where(diverged, np.nan, acc_epoch[-1])):.4f} ({time.time() - t0:.0f}s)")
    lh = np.array(loss_hist)
    final_loss = lh[-10:].mean(0)
    final_loss[diverged] = np.inf
    acc = np.array(acc_epoch[-1])
    acc[diverged] = np.nan
    return dict(name=name, L=L, N=N, rule=rule, seeds=np.array(seeds), lr=lr.numpy(), epochs=epochs,
                test_acc=acc, test_acc_epoch=np.array(acc_epoch), steps_to_target=steps_to_target,
                final_loss=final_loss, diverged=diverged, curve_step=np.array(curve_step),
                curve_acc=np.array(curve_acc), time=time.time() - t0)
