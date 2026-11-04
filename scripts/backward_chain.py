"""
Exact backward chain: the gradient of a random linear loss through a random tanh network of width N,
without drawing weight matrices.

Forward (exact chain for one input): given z^{(l)}, y = phi(z^{(l)}), Vhat = mean(y^2),
    Q_{l+1} = sw2 Vhat + sb2,   z^{(l+1)}_i ~ N(0, Q_{l+1}) independent.

Backward: conditioned on the forward trajectory, the pair (W^{(l+1)}, b^{(l+1)}) is Gaussian with the
constraint W y + b = z^{(l+1)}.  Drawing b | z^{(l+1)} and then W | (c = z^{(l+1)} - b, y):
    b_i | z_i   ~ N(z_i sb2 / Q,  sb2 sw2 Vhat / Q),
    W           = c y^T / |y|^2 + W~ (I - y y^T / |y|^2),   W~ fresh N(0, sw2/N) entries,
so that with the gradient g = g^{(l+1)} of the layer above, G = mean(g^2),
    W^T g = y (c.g)/|y|^2 + sqrt(sw2 G) [xi - y (y.xi)/|y|^2],   xi ~ N(0, I_N),
    g^{(l)} = phi'(z^{(l)}) * W^T g .
Only the first term (the feedback) and the projection in the second distinguish this from the gradient
independence assumption, which takes W^T g = sqrt(sw2 G) xi.

Usage: python backward_chain.py N L M sw2 [seed]
"""
import sys, pickle, time
import numpy as np


def run(N, L, M, sw2, sb2=0.05, q0=0.5700478816408142, seed=0, mode='exact'):
    rng = np.random.default_rng(seed)
    Z = np.empty((L + 1, M, N))
    Z[0] = np.sqrt(q0) * rng.standard_normal((M, N))
    Q = np.empty((L + 1, M)); Q[0] = (Z[0]**2).mean(1)
    for l in range(L):
        y = np.tanh(Z[l])
        Q[l + 1] = sw2 * (y**2).mean(1) + sb2
        Z[l + 1] = np.sqrt(Q[l + 1])[:, None] * rng.standard_normal((M, N))
    steps = np.zeros((M, L)); ahat = np.zeros((M, L))
    g = rng.standard_normal((M, N))
    g /= np.sqrt((g**2).mean(1, keepdims=True))          # G = 1
    for l in range(L - 1, -1, -1):
        y = np.tanh(Z[l]); dphi = 1.0 - y**2
        yy = (y**2).sum(1, keepdims=True)                 # |y|^2
        Vh = yy[:, 0] / N
        xi = rng.standard_normal((M, N))
        if mode == 'exact':
            b = Z[l + 1] * (sb2 / Q[l + 1])[:, None] + np.sqrt(sb2 * sw2 * Vh / Q[l + 1])[:, None] * rng.standard_normal((M, N))
            c = Z[l + 1] - b
            a = (c * g).sum(1, keepdims=True) / yy
            ahat[:, l] = a[:, 0] / np.sqrt(sw2)
            wtg = y * a + np.sqrt(sw2) * (xi - y * (y * xi).sum(1, keepdims=True) / yy)
        elif mode == 'gi':
            wtg = np.sqrt(sw2) * xi
        elif mode == 'proj':                               # projection without feedback
            wtg = np.sqrt(sw2) * (xi - y * (y * xi).sum(1, keepdims=True) / yy)
        gn = dphi * wtg
        Gn = (gn**2).mean(1)
        steps[:, l] = np.log(Gn)
        g = gn / np.sqrt(Gn)[:, None]
    return dict(N=N, L=L, M=M, sw2=sw2, sb2=sb2, mode=mode, steps=steps, ahat=ahat, Q=Q.T)


def summarise(out, burn=100):
    N, L, s = out['N'], out['L'], out['steps']
    st = s[:, :L - burn]
    per = st.mean(1)
    print(f"{out['mode']:6s} N={N} L={L} M={out['M']} sw2={out['sw2']:.4f}:  N*drift = {N*per.mean():8.4f} +- {N*per.std(ddof=1)/np.sqrt(len(per)):.4f}", end='')
    blk = 100; nb = st.shape[1] // blk
    bs = st[:, :nb*blk].reshape(len(st), nb, blk).sum(2)
    print(f"   N*longrun var = {N*bs.var(ddof=1)/blk:7.4f}", end='')
    if out['mode'] == 'exact':
        print(f"   N E[ahat^2] = {N*(out['ahat'][:, :L-burn]**2).mean():.4f}", end='')
    print()


if __name__ == '__main__':
    N, L, M = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    sw2 = float(sys.argv[4]); seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    for mode in ('exact', 'gi', 'proj'):
        t = time.time()
        out = run(N, L, M, sw2, seed=seed, mode=mode)
        summarise(out)
        with open(f'/home/claude/paper/code/data/backward/chain_{mode}_N{N}_L{L}_sw{sw2:.3f}.pkl', 'wb') as f:
            pickle.dump(out, f)
