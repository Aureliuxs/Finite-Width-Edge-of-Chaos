"""
Forward tangent walk against backward gradient walk in random tanh networks.

For one network with pre-activations z^{(l+1)} = W^{(l+1)} phi(z^{(l)}) + b^{(l+1)} and Jacobian
J = prod_l W^{(l+1)} D_l, D_l = diag(phi'(z^{(l)})), three walks are followed:

  forward   X_L = ln ||J v||^2 / ||v||^2            v random at layer 0 (tangent vector, nearby inputs)
  backward  Y_L = ln ||J^T u||^2 / ||u||^2          u random at layer L (gradient of a random linear loss)
  GI        Z_L = same as backward but with fresh matrices W~ in place of W (gradient independence)

The exact backward chain of the note (conditioned Gaussians, no weight matrices) is in backward_chain.py.
Usage: python backward_walk.py N L M sw2 [seed]
"""
import sys, pickle, time
import numpy as np

sys.path.insert(0, '/home/claude/crinit')


def run(N, L, M, sw2, sb2=0.05, q0=0.5700478816408142, seed=0, batch=500, store=None):
    rng = np.random.default_rng(seed)
    steps_f = np.zeros((M, L))      # forward tangent steps s_l = ln(D_{l+1}/D_l)
    steps_b = np.zeros((M, L))      # backward steps t_l = ln(G_l/G_{l+1}), index l = layer whose gradient is formed
    steps_g = np.zeros((M, L))      # gradient-independence steps
    Qs = np.zeros((M, L + 1))       # signal sizes
    cross = np.zeros((M, L))        # backward signal-gradient correlation rho_l = (c.g)/(|c||g|) at layer l+1
    for m0 in range(0, M, batch):
        mb = min(batch, M - m0)
        seeds = rng.integers(0, 2**63, size=L)
        # forward pass, storing z
        z = np.sqrt(q0) * rng.standard_normal((mb, N))
        Z = np.empty((L + 1, mb, N)); Z[0] = z
        B = np.empty((L, mb, N))
        v = rng.standard_normal((mb, N))
        for l in range(L):
            r_l = np.random.default_rng(seeds[l])
            W = np.sqrt(sw2 / N) * r_l.standard_normal((mb, N, N))
            b = np.sqrt(sb2) * r_l.standard_normal((mb, N))
            y = np.tanh(z)
            dphi = 1.0 - y**2
            # tangent
            vn = np.einsum('mij,mj->mi', W, dphi * v)
            steps_f[m0:m0+mb, l] = np.log((vn**2).mean(1) / (v**2).mean(1))
            v = vn / np.sqrt((vn**2).mean(1, keepdims=True))
            z = np.einsum('mij,mj->mi', W, y) + b
            Z[l + 1] = z; B[l] = b
            Qs[m0:m0+mb, l + 1] = (z**2).mean(1)
        Qs[m0:m0+mb, 0] = (Z[0]**2).mean(1)
        # backward pass, regenerating W
        g = rng.standard_normal((mb, N))
        gg = g.copy()
        for l in range(L - 1, -1, -1):
            r_l = np.random.default_rng(seeds[l])
            W = np.sqrt(sw2 / N) * r_l.standard_normal((mb, N, N))
            _ = r_l.standard_normal((mb, N))
            y = np.tanh(Z[l]); dphi = 1.0 - y**2
            c = Z[l + 1] - B[l]
            cross[m0:m0+mb, l] = (c * g).sum(1) / np.sqrt((c**2).sum(1) * (g**2).sum(1))
            gn = dphi * np.einsum('mji,mj->mi', W, g)
            steps_b[m0:m0+mb, l] = np.log((gn**2).mean(1) / (g**2).mean(1))
            g = gn / np.sqrt((gn**2).mean(1, keepdims=True))
            Wt = np.sqrt(sw2 / N) * rng.standard_normal((mb, N, N))
            ggn = dphi * np.einsum('mji,mj->mi', Wt, gg)
            steps_g[m0:m0+mb, l] = np.log((ggn**2).mean(1) / (gg**2).mean(1))
            gg = ggn / np.sqrt((ggn**2).mean(1, keepdims=True))
    out = dict(N=N, L=L, M=M, sw2=sw2, sb2=sb2, steps_f=steps_f, steps_b=steps_b, steps_g=steps_g, Q=Qs, cross=cross)
    if store:
        with open(store, 'wb') as f:
            pickle.dump(out, f)
    return out


def summarise(out, burn=100):
    N, L = out['N'], out['L']
    sf, sb, sg = out['steps_f'], out['steps_b'], out['steps_g']
    XL, YL, ZL = sf.sum(1), sb.sum(1), sg.sum(1)
    M = len(XL)
    def se(x): return x.std(ddof=1) / np.sqrt(len(x))
    print(f"N={N} L={L} M={M} sw2={out['sw2']:.4f}")
    print(f"  total   forward X_L  mean {XL.mean():9.4f} +- {se(XL):.4f}   var {XL.var(ddof=1):8.3f}")
    print(f"  total   backward Y_L mean {YL.mean():9.4f} +- {se(YL):.4f}   var {YL.var(ddof=1):8.3f}")
    print(f"  total   GI Z_L       mean {ZL.mean():9.4f} +- {se(ZL):.4f}   var {ZL.var(ddof=1):8.3f}")
    # stationary per-step drift: forward after burn-in layers; backward after burn-in layers from the top
    f_st = sf[:, burn:]; b_st = sb[:, :L - burn]; g_st = sg[:, :L - burn]
    for name, s in (('forward', f_st), ('backward', b_st), ('GI', g_st)):
        per_net = s.mean(1)
        mu = per_net.mean(); e = se(per_net)
        print(f"  drift {name:9s} N*mu = {N*mu:8.4f} +- {N*e:.4f}   (per step {mu:.5f})")
    # long-run variance from block sums
    blk = 100
    for name, s in (('forward', f_st), ('backward', b_st)):
        nb = s.shape[1] // blk
        bs = s[:, :nb*blk].reshape(len(s), nb, blk).sum(2)
        print(f"  longrun var {name:9s} N*sigma^2 = {N*bs.var(ddof=1)/blk:8.4f}")
    # quantiles of totals
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    print("  quantiles forward ", np.round(np.quantile(XL, qs), 3))
    print("  quantiles backward", np.round(np.quantile(YL, qs), 3))
    print("  mean of exp (mean distance/gradient growth): fwd", np.log(np.exp(XL).mean()), " back", np.log(np.exp(YL).mean()))
    print("  rho (backward signal-gradient corr) N*E[rho^2] =", N * (out['cross'][:, :L-burn]**2).mean())


if __name__ == '__main__':
    N, L, M = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    sw2 = float(sys.argv[4]); seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    t = time.time()
    out = run(N, L, M, sw2, seed=seed, store=f'/home/claude/paper/code/data/backward/walk_N{N}_L{L}_sw{sw2:.3f}.pkl')
    print(f"{time.time()-t:.0f} s")
    summarise(out)
