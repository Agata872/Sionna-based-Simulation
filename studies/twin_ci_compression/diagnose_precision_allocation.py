#!/usr/bin/env python3
"""Stage-6 diagnostic (negative result): precision allocation variants on CI dumped by
explore_twin_cellfree.py --dump. Compares urts_code (18-bit coefficients), urts_vq (twin-set precision)
and a precoder-aware variant (errors priced by their leakage into the other UEs' RZF beams, beams from
an earlier reconstruction). Usage: diagnose_precision_allocation.py <dump dir> [<dump dir> ...].
Independent exploration script (oran/ untouched)."""
import math, sys, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import explore_twin_cellfree as m

def rzf_w(G, K):
    W = G.conj().T@np.linalg.inv(G@G.conj().T+K*np.eye(K))
    return W/np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-30)/np.sqrt(K)

def proxy2(E, D, Lam, K, F, Nt):
    a = (Nt-K+1)/(Nt*F*K)
    return np.log2(1+np.maximum(E-np.maximum(D, 0), 0)*a/(1+np.maximum(Lam, 0)))

def urts_lvq(Hn, bases, lam, W, nvar, ru_bits, sp, fd, bmax=12, res_thresh=1.):
    """Bits per KLT dimension and residual coefficients chosen by greedy gain per bit on
    SINR_k = a(E_k - D_k)/(1 + Lambda_k), Lambda_k = sum over sent/unsent error components of their
    energy times their leakage weight sum_f sum_{j != k} |v^H w_j|^2 / F (W: previous precoders [F, Nt, K]).
    KLT part at the O-DU from the twin (as urts_vq, with leakage weights); residual at each O-RU."""
    K, L, A, F = Hn.shape
    N, Nt = A*F, L*A
    idx = math.ceil(math.log2(N)); rcost = 18+idx
    tau2 = res_thresh*2*math.log(N)*nvar
    # leakage weight of a unit-norm space-frequency vector v (a*F+f order) on link l for UE k
    Wl = [W[:, l*A:(l+1)*A, :] for l in range(L)]           # [F, A, K]
    def weights(V, k, l):                                   # V [N, n] -> [n]
        Vf = V.reshape(A, F, -1).transpose(1, 0, 2)         # [F, A, n]
        P = np.abs(np.einsum('fan,faj->fnj', Vf.conj(), Wl[l]))**2   # [F, n, K]
        return (P.sum(2)-P[:, :, k]).sum(0)/F
    ATOMS = np.stack([np.kron(sp[:, j//F], fd[:, j % F]) for j in range(N)], 1)   # [N, N], index a*F+f
    smax = max(len(x) for r in lam for x in r if x is not None)
    lp = np.zeros((K, L, smax)); om = np.zeros((K, L, smax))
    for k in range(K):
        for l in range(L):
            if lam[k][l] is not None:
                n = len(lam[k][l]); lp[k, l, :n] = lam[k][l]; om[k, l, :n] = weights(bases[k][l], k, l)
    def err(lv, bb):
        return np.where(bb == 0, lv, nvar+m.Q_MSE[np.minimum(bb, bmax)]*(lv+nvar))
    E = lp.sum((1, 2)); D = E.copy(); Lam = (om*lp).sum((1, 2))
    b = np.zeros((K, L, smax), int); left = np.full(L, float(ru_bits))
    def gains(k):
        dd = err(lp[k], b[k])-err(lp[k], b[k]+1)
        g = proxy2(E[k], D[k]-dd, Lam[k]-om[k]*dd, K, F, Nt)-proxy2(E[k], D[k], Lam[k], K, F, Nt)
        return np.where((lp[k] > 0) & (b[k] < bmax) & (left[:, None] >= 2), g, -np.inf)
    G = np.stack([gains(k) for k in range(K)])
    while True:
        k, l, i = np.unravel_index(int(np.argmax(G)), G.shape)
        if not G[k, l, i] > 0: break
        dd = err(lp[k, l, i], b[k, l, i])-err(lp[k, l, i], b[k, l, i]+1)
        D[k] -= dd; Lam[k] -= om[k, l, i]*dd; b[k, l, i] += 1; left[l] -= 2
        G[k] = gains(k)
        if left[l] < 2: G[:, l] = -np.inf
    Hhat = np.zeros_like(Hn); bits = 2*int(b.sum())+10*K*L
    for l in range(L):
        cand, gs, Ek, Dk, Lk = [], [], E.copy(), D.copy(), Lam.copy()
        for k in range(K):
            y = Hn[k, l].reshape(-1); r = np.zeros_like(y)
            if bases[k][l] is not None:
                bb = b[k, l, :bases[k][l].shape[1]]; use = np.flatnonzero(bb)
                if len(use):
                    Q = bases[k][l][:, use]; c = Q.conj().T@y
                    step = m.Q_STEP[bb[use]]*np.sqrt((lam[k][l][use]+nvar)/2); half = 2.**(bb[use]-1)
                    qz = lambda v: (np.clip(np.floor(v/step), -half, half-1)+.5)*step
                    r = Q@(qz(c.real)+1j*qz(c.imag))
            Hhat[k, l] = r.reshape(A, F)
            z = (y-r).reshape(A, F); g = (sp.conj().T@z)@fd.conj(); gs.append(g)
            e = np.abs(g.reshape(-1))**2
            w_all = weights(ATOMS, k, l)                      # leakage weight of every angle-delay atom
            # the O-RU replaces the twin's view of its own link by what it measures
            used = int((b[k, l] > 0).sum()) if lam[k][l] is not None else 0
            n_ = len(lam[k][l]) if lam[k][l] is not None else 0
            tw_err = err(lp[k, l, :n_], b[k, l, :n_]) if n_ else np.zeros(0)
            Dk[k] += max(np.linalg.norm(z)**2-(N-used)*nvar, 0.)+used*nvar-tw_err.sum()
            Lk[k] += (w_all*np.maximum(e-nvar, 0)).sum()-(om[k, l, :n_]*tw_err).sum()
            Ek[k] += max(np.linalg.norm(y)**2-N*nvar, 0.)-lp[k, l].sum()
            o = np.argsort(e)[::-1]; o = o[e[o] > tau2]
            cand.append((o, e, w_all[o]))
        ptr = np.zeros(K, int); sel = [[] for _ in range(K)]
        while left[l] >= rcost:
            best, arg = 0., None
            for k in range(K):
                o, e, w_at = cand[k]
                if ptr[k] >= len(o): continue
                dd = e[o[ptr[k]]]-nvar
                gain = proxy2(Ek[k], Dk[k]-dd, Lk[k]-w_at[ptr[k]]*dd, K, F, Nt)-proxy2(Ek[k], Dk[k], Lk[k], K, F, Nt)
                if gain > best: best, arg = gain, k
            if arg is None: break
            k = arg; o, e, w_at = cand[k]; j = o[ptr[k]]; dd = e[j]-nvar
            sel[k].append(j); Dk[k] -= dd; Lk[k] -= w_at[ptr[k]]*dd; ptr[k] += 1; left[l] -= rcost
        for k in range(K):
            if sel[k]:
                mm = np.zeros(N, complex); mm[sel[k]] = gs[k].reshape(-1)[sel[k]]
                Hhat[k, l] += sp@(mm.reshape(A, F)@fd.T); bits += rcost*len(sel[k])
    return Hhat, bits

from pathlib import Path
for dump in sys.argv[1:]:
    for path in sorted(Path(dump).glob('drop*.npz')):
        drop = path.stem
        z = np.load(path)
        H, Hn = z['H'], z['Hn']; K, L, A, F = Hn.shape
        sp, fd = m.ad_bases(4, 4, F); Ht = H.reshape(K, -1, F)
        noisy = m.rzf_se(Ht, Hn.reshape(K, -1, F)).sum()
        for sigma in ['0', '2']:
            bases = [[z[f'{sigma}_klt_{k}_{l}'] if f'{sigma}_klt_{k}_{l}' in z.files else None for l in range(L)] for k in range(K)]
            lam = [[z[f'{sigma}_lam_{k}_{l}'] if f'{sigma}_lam_{k}_{l}' in z.files else None for l in range(L)] for k in range(K)]
            prev = m.urts_code(Hn, bases, lam, 1., K*32*18, sp, fd, gamma=2.7)[0]      # stand-in for the previous occasion
            W = np.stack([rzf_w(prev[:, :, :, f].reshape(K, -1), K) for f in range(F)])   # [F, Nt, K]
            line = []
            for B in [4, 16, 64, 128]:
                cells = []
                for name in ['code', 'vq', 'lvq']:
                    if name == 'lvq':
                        Hh, bits = urts_lvq(Hn, bases, lam, W, 1., K*B*18, sp, fd)
                    else:
                        r = (m.urts_code if name == 'code' else m.urts_vq)(Hn, bases, lam, 1., K*B*18, sp, fd, gamma=2.7)
                        Hh, bits = r[0], r[2]
                    cells.append(f'{bits/L/1e3:4.1f}k {100*m.rzf_se(Ht, Hh.reshape(K,-1,F)).sum()/noisy:3.0f}%')
                line.append(f'B{B}: ' + '/'.join(cells))
            print(f'{dump} {drop} s{sigma}: ' + ' | '.join(line), flush=True)
