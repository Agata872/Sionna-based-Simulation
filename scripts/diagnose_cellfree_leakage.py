#!/usr/bin/env python3
"""Stage-5 diagnostics on CI dumped by explore_twin_cellfree.py --dump DIR (drop*.npz).

Why does compressed CI cost more sum SE than white CI noise of the same energy?
  leak      interference of each UE vs the isotropic-error prediction, and the share of
            its CI error that lies in the span of the other UEs' reconstructed channels
  parts     the same for the pieces of the plain region-KLT error: truncation inside the
            region span, outside it, and CI noise projected into the sent subspace
  gamma     fit the leakage factor gamma of the first-order SE model
            SE_k ~ log2(1 + a (E-D) / (1 + gamma c D)) per UE (see explore_twin_cellfree.py)
  weighted  nulling-aware KLT (twin covariance weighted by the other UEs' precoder
            subspaces) vs plain KLT, same number of coefficients per link
  gamma_pred  per-UE leakage factor of the plain region KLT predicted from the twin alone
            (error covariance = unsent KLT dimensions + CI noise in the sent ones; leakage =
            its overlap with the other UEs' unit-trace per-group twin covariances, weighted by
            their twin power split over O-RUs) vs the measured one

Independent exploration script (oran/ untouched).
"""
import argparse
from pathlib import Path

import numpy as np


def rzf_w(G, K):
    W = G.conj().T@np.linalg.inv(G@G.conj().T+K*np.eye(K))
    return W/np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-30)/np.sqrt(K)


def rzf_se(Ht, Hh):
    K, F = Ht.shape[0], Ht.shape[-1]
    Ht, Hh = Ht.reshape(K, -1, F), Hh.reshape(K, -1, F)
    se = np.zeros(K)
    for f in range(F):
        M = np.abs(Ht[:, :, f]@rzf_w(Hh[:, :, f], K))**2
        se += np.log2(1+np.diag(M)/(M.sum(1)-np.diag(M)+1))
    return se/F


def recon_keys(z, budgets=('16', '32')):
    return [k for k in z.files if k.endswith(('_task', '_uniform')) and k.split('_')[1] in budgets]


def leak(drops):
    agg = {}
    for p in drops:
        z = np.load(p)
        H = z['H']
        K, L, A, F = H.shape
        Nt = L*A
        for key in ['Hn']+recon_keys(z):
            Hh = z[key]
            r = agg.setdefault(key, dict(I=[], Iiso=[], frac=[], sig=[], sig0=[]))
            for f in range(F):
                Ht, Gh = H[:, :, :, f].reshape(K, Nt), Hh[:, :, :, f].reshape(K, Nt)
                M, M0 = np.abs(Ht@rzf_w(Gh, K))**2, np.abs(Ht@rzf_w(Ht, K))**2
                E = Gh-Ht
                for k in range(K):
                    D = np.linalg.norm(E[k])**2
                    Q, _ = np.linalg.qr(np.delete(Gh, k, 0).conj().T)
                    r['I'].append(M[k].sum()-M[k, k])
                    r['Iiso'].append(D*(K-1)/(K*Nt))
                    r['frac'].append(np.linalg.norm(Q.conj().T@E[k])**2/max(D, 1e-30))
                    r['sig'].append(M[k, k])
                    r['sig0'].append(M0[k, k])
    print(f"{'CI':28s} {'I/I_iso':>8s} {'err in span(others)':>20s} {'signal loss dB':>15s}   (isotropic share {(K-1)/Nt:.3f})")
    for key, r in agg.items():
        a = {k: np.array(v) for k, v in r.items()}
        print(f"{key:28s} {a['I'].mean()/a['Iiso'].mean():8.2f} {a['frac'].mean():20.3f} "
              f"{10*np.log10(a['sig'].mean()/a['sig0'].mean()):15.2f}")


def parts(drops):
    for sigma in ['0', '2']:
        for d in [8, 32]:
            acc = {}
            for p in drops:
                z = np.load(p)
                H, Hn = z['H'], z['Hn']
                K, L, A, F = H.shape
                Nt = L*A
                pc = {x: np.zeros_like(H) for x in ['t_in', 't_out', 'noise']}
                Hh = np.zeros_like(H)
                for k in range(K):
                    for l in range(L):
                        key = f'{sigma}_klt_{k}_{l}'
                        if key not in z.files:
                            pc['t_out'][k, l] = -H[k, l]
                            continue
                        U = z[key]
                        Q = U[:, :d]
                        h, n = H[k, l].reshape(-1), (Hn[k, l]-H[k, l]).reshape(-1)
                        PQh, PUh = Q@(Q.conj().T@h), U@(U.conj().T@h)
                        pc['t_in'][k, l] = (PQh-PUh).reshape(A, F)
                        pc['t_out'][k, l] = (PUh-h).reshape(A, F)
                        pc['noise'][k, l] = (Q@(Q.conj().T@n)).reshape(A, F)
                        Hh[k, l] = (Q@(Q.conj().T@(h+n))).reshape(A, F)
                for f in range(F):
                    W = rzf_w(Hh[:, :, :, f].reshape(K, Nt), K)
                    for x, e_ in pc.items():
                        e = e_[:, :, :, f].reshape(K, Nt)
                        M = np.abs(e.conj()@W)**2
                        s = acc.setdefault(x, [0., 0., 0.])
                        s[0] += (M.sum(1)-np.diag(M)).sum()
                        s[1] += (np.linalg.norm(e, axis=1)**2*(K-1)/(K*Nt)).sum()
                        s[2] += (np.linalg.norm(e, axis=1)**2).sum()
            print(f'sigma={sigma} d={d}: '+'  '.join(f'{x}: energy {10*np.log10(v[2]):5.1f} dB, leakage x{v[0]/v[1]:4.1f} isotropic'
                                                       for x, v in acc.items()))


def gamma(drops):
    recs = []
    for p in drops:
        z = np.load(p)
        H = z['H']
        K, L, A, F = H.shape
        Nt = L*A
        a, c = (Nt-K+1)/(Nt*F*K), (K-1)/(K*F*Nt)
        E = np.linalg.norm(H.reshape(K, -1), axis=1)**2
        se0 = rzf_se(H, H)
        for key in ['Hn']+[k for k in z.files if k.endswith(('_task', '_uniform'))]:
            D = np.linalg.norm((z[key]-H).reshape(K, -1), axis=1)**2
            se = rzf_se(H, z[key])
            name = 'noisy' if key == 'Hn' else key.split('_', 2)[-1]
            recs += [(name, E[k], D[k], se0[k]-se[k], a, c) for k in range(K)]
    names = [r[0] for r in recs]
    E, D, loss, a, c = (np.array([r[i] for r in recs]) for i in range(1, 6))

    def pred(g):
        return np.log2(1+a*E)-np.log2(1+a*np.maximum(E-D, 0)/(1+g*c*D))
    gs = np.linspace(.5, 8, 76)
    for label, m in [('white CI noise', np.array([n == 'noisy' for n in names])),
                     ('compressed CI', np.array([n != 'noisy' for n in names]))]:
        err = [np.sqrt(np.mean((pred(g)[m]-loss[m])**2)) for g in gs]
        g = gs[int(np.argmin(err))]
        print(f'{label:15s}: gamma = {g:.1f}, per-UE rmse {min(err):.2f} bit (gamma = 1: {err[int(np.argmin(np.abs(gs-1)))]:.2f}),'
              f' mean loss {loss[m].mean():.2f}')
    # the same model evaluated per PRB group (no averaging of E and D over groups before the log)
    recs = []
    for p in drops:
        z = np.load(p)
        H = z['H']
        K, L, A, F = H.shape
        Nt = L*A
        Ef = np.linalg.norm(H.reshape(K, L*A, F), axis=1)**2                   # [K, F]
        se0 = rzf_se(H, H)
        for key in ['Hn']+[k for k in z.files if k.endswith(('_task', '_uniform'))]:
            Df = np.linalg.norm((z[key]-H).reshape(K, L*A, F), axis=1)**2
            se = rzf_se(H, z[key])
            recs += [(key == 'Hn', Ef[k], Df[k], se0[k]-se[k]) for k in range(K)]
    a1, c1 = (Nt-K+1)/(Nt*K), (K-1)/(K*Nt)

    def pred_f(g, sel):
        return np.array([np.mean(np.log2(1+a1*e)-np.log2(1+a1*np.maximum(e-d, 0)/(1+g*c1*d))) for w, e, d, _ in recs if sel(w)])
    for label, sel in [('white CI noise', lambda w: w), ('compressed CI', lambda w: not w)]:
        lo = np.array([x[3] for x in recs if sel(x[0])])
        err = [np.sqrt(np.mean((pred_f(g, sel)-lo)**2)) for g in gs]
        print(f'{label:15s} (per PRB group): gamma = {gs[int(np.argmin(err))]:.1f}, per-UE rmse {min(err):.2f} bit')


def weighted(drops, ds=(4, 8, 16, 32)):
    for sigma in ['0', '2']:
        acc = {}
        for p in drops:
            z = np.load(p)
            H, Hn = z['H'], z['Hn']
            K, L, A, F = H.shape
            U = {(k, l): z[f'{sigma}_klt_{k}_{l}'] for k in range(K) for l in range(L) if f'{sigma}_klt_{k}_{l}' in z.files}
            lam = {kl: z[f'{sigma}_lam_{kl[0]}_{kl[1]}'] for kl in U}
            Ekl = np.array([[lam[k, l].sum() if (k, l) in lam else 0. for l in range(L)] for k in range(K)])
            E = Ekl.sum(1)
            rho = Ekl/np.maximum(E[:, None], 1e-30)            # twin-predicted precoder power split
            C = {}
            for kl, Ukl in U.items():                           # unit-trace spatial covariance per PRB group
                blocks = np.stack([(Ukl[np.arange(A)*F+f]*lam[kl])@Ukl[np.arange(A)*F+f].conj().T for f in range(F)])
                C[kl] = blocks/np.maximum(np.trace(blocks, axis1=1, axis2=2).real, 1e-30)[:, None, None]

            def apply(M_, x):                                   # block-diagonal (per group) operator on a*F+f vectors
                xs = x.reshape(A, F, -1).transpose(1, 0, 2)
                return np.einsum('fab,fbn->fan', M_, xs).transpose(1, 0, 2).reshape(A*F, -1)
            for mode in ['plain', 'weighted']:
                for d in ds:
                    Hh = np.zeros_like(H)
                    for (k, l), Ukl in U.items():
                        y = Hn[k, l].reshape(-1)
                        if mode == 'plain':
                            Q = Ukl[:, :d]
                            Hh[k, l] = (Q@(Q.conj().T@y)).reshape(A, F)
                            continue
                        # Omega = I/E_k + (1/F) blockdiag_f sum_{j != k} rho_jl/K C_jl(f)
                        Om = sum(rho[j, l]/K*C[j, l] for j in range(K) if j != k and (j, l) in C)/F+np.eye(A)[None]/E[k]
                        w, V = np.linalg.eigh(Om)
                        w = np.maximum(w, 1e-30)
                        Oh = np.einsum('fab,fb,fcb->fac', V, np.sqrt(w), V.conj())
                        Oih = np.einsum('fab,fb,fcb->fac', V, 1/np.sqrt(w), V.conj())
                        Qw = np.linalg.svd(apply(Oh, Ukl*np.sqrt(lam[k, l])[None]), full_matrices=False)[0][:, :d]
                        Hh[k, l] = apply(Oih, Qw@(Qw.conj().T@apply(Oh, y[:, None]))).reshape(A, F)
                    acc.setdefault((mode, d), []).append(rzf_se(H, Hh).sum())
        print(f'sigma={sigma}: '+'  '.join(f'd={d}: plain {np.mean(acc["plain", d]):.1f} / weighted {np.mean(acc["weighted", d]):.1f}' for d in ds))


def twin_blocks(U, lam, A, F, unit=True):
    """Per-PRB-group spatial covariance blocks [F, A, A] of U diag(lam) U^H (vector order a*F+f)."""
    out = np.stack([(U[np.arange(A)*F+f]*lam)@U[np.arange(A)*F+f].conj().T for f in range(F)])
    if unit:
        out = out/np.maximum(np.trace(out, axis1=1, axis2=2).real, 1e-30)[:, None, None]
    return out


def gamma_pred(drops, ds=(8, 16, 32), nvar=1.):
    pred, meas, predw, predt = [], [], [], []
    for p in drops:
        z = np.load(p)
        H, Hn = z['H'], z['Hn']
        K, L, A, F = H.shape
        Nt = L*A
        for sigma in ['0', '2']:
            U = {(k, l): z[f'{sigma}_klt_{k}_{l}'] for k in range(K) for l in range(L) if f'{sigma}_klt_{k}_{l}' in z.files}
            lam = {kl: z[f'{sigma}_lam_{kl[0]}_{kl[1]}'] for kl in U}
            Ekl = np.array([[lam[k, l].sum() if (k, l) in lam else 0. for l in range(L)] for k in range(K)])
            rho = Ekl/np.maximum(Ekl.sum(1, keepdims=True), 1e-30)
            C = {kl: twin_blocks(U[kl], lam[kl], A, F) for kl in U}
            for d in ds:
                Hh = np.zeros_like(H)
                for (k, l), Ukl in U.items():
                    Q = Ukl[:, :d]
                    Hh[k, l] = (Q@(Q.conj().T@Hn[k, l].reshape(-1))).reshape(A, F)
                # measured: leakage caused by the CI error (minus RZF's own residual interference with
                # ideal CI) over the isotropic prediction, per UE
                I, Iiso = np.zeros(K), np.zeros(K)
                for f in range(F):
                    Ht = H[:, :, :, f].reshape(K, Nt)
                    M = np.abs(Ht@rzf_w(Hh[:, :, :, f].reshape(K, Nt), K))**2
                    M0 = np.abs(Ht@rzf_w(Ht, K))**2
                    I += (M.sum(1)-np.diag(M))-(M0.sum(1)-np.diag(M0))
                    Iiso += np.linalg.norm((Hh-H)[:, :, :, f].reshape(K, Nt), axis=1)**2*(K-1)/(K*Nt)
                # predicted with the actual precoders (known at the O-DU) and the twin error covariance
                Ws = [rzf_w(Hh[:, :, :, f].reshape(K, Nt), K) for f in range(F)]
                # twin only: precoders from twin proxy channels (first KLT component of every link)
                proxy = np.zeros_like(H)
                for (k_, l_), U_ in U.items():
                    proxy[k_, l_] = (U_[:, 0]*np.sqrt(lam[k_, l_][0])).reshape(A, F)
                Wt = [rzf_w(proxy[:, :, :, f].reshape(K, Nt), K) for f in range(F)]
                for k in range(K):
                    nump = numt = denp = 0.
                    for l in range(L):
                        if (k, l) not in U:
                            continue
                        Ukl, lk = U[k, l], lam[k, l]
                        Rerr = twin_blocks(Ukl[:, d:], lk[d:], A, F, unit=False)+nvar*twin_blocks(Ukl[:, :d], np.ones(min(d, Ukl.shape[1])), A, F, unit=False)
                        denp += np.trace(Rerr, axis1=1, axis2=2).real.sum()*(K-1)/(K*Nt)
                        for f in range(F):
                            Wl = Ws[f][l*A:(l+1)*A]                     # [A, K] this O-RU's part of the precoder
                            q = np.einsum('aj,ab,bj->j', Wl.conj(), Rerr[f], Wl).real
                            nump += q.sum()-q[k]
                            Wl = Wt[f][l*A:(l+1)*A]
                            q = np.einsum('aj,ab,bj->j', Wl.conj(), Rerr[f], Wl).real
                            numt += q.sum()-q[k]
                    if denp > 0 and Iiso[k] > 0 and I[k] > 0:
                        predw.append(nump/denp)
                        predt.append(numt/denp)
                # predicted from the twin: error covariance per link, overlap with the other UEs' covariances
                for k in range(K):
                    num = den = 0.
                    for l in range(L):
                        if (k, l) not in U:
                            continue
                        Ukl, lk = U[k, l], lam[k, l]
                        Rerr = twin_blocks(Ukl[:, d:], lk[d:], A, F, unit=False)+nvar*twin_blocks(Ukl[:, :d], np.ones(min(d, Ukl.shape[1])), A, F, unit=False)
                        den += np.trace(Rerr, axis1=1, axis2=2).real.sum()*(K-1)/(K*Nt)
                        for j in range(K):
                            if j != k and (j, l) in C:
                                num += rho[j, l]/K*np.einsum('fab,fba->', Rerr, C[j, l]).real
                    if den > 0 and Iiso[k] > 0 and I[k] > 0:
                        pred.append(num/den)
                        meas.append(I[k]/Iiso[k])
    pred, meas, predw, predt = np.array(pred), np.array(meas), np.array(predw), np.array(predt)
    if len(predt) == len(meas):
        lt = np.log10(predt)
        print(f'twin only (RZF on twin proxy channels): predicted median {np.median(predt):.2f}; log-log correlation '
              f'{np.corrcoef(lt, np.log10(meas))[0, 1]:.2f}; within a factor 2: {np.mean(np.abs(lt-np.log10(meas)) < np.log10(2))*100:.0f}%')
    if len(predw) == len(meas):
        lw = np.log10(predw)
        print(f'with the actual precoders: predicted median {np.median(predw):.2f}; log-log correlation '
              f'{np.corrcoef(lw, np.log10(meas))[0, 1]:.2f}; within a factor 2: {np.mean(np.abs(lw-np.log10(meas)) < np.log10(2))*100:.0f}%')
    lp, lm = np.log10(pred), np.log10(meas)
    print(f'{len(pred)} UE cases: measured gamma median {np.median(meas):.2f} (p10 {np.percentile(meas, 10):.2f}, p90 {np.percentile(meas, 90):.2f}); '
          f'predicted median {np.median(pred):.2f}')
    print(f'log-log correlation {np.corrcoef(lp, lm)[0, 1]:.2f}; within a factor 2: {np.mean(np.abs(lp-lm) < np.log10(2))*100:.0f}%')
    for lo, hi in [(0, 1), (1, 2), (2, 4), (4, 100)]:
        m = (pred >= lo) & (pred < hi)
        if m.any():
            print(f'  predicted {lo}-{hi}: n={m.sum():3d}, measured median {np.median(meas[m]):.2f}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('job', choices=['leak', 'parts', 'gamma', 'weighted', 'gamma_pred'])
    p.add_argument('dump', help='directory written by explore_twin_cellfree.py --dump')
    a = p.parse_args()
    drops = sorted(Path(a.dump).glob('drop*.npz'))
    dict(leak=leak, parts=parts, gamma=gamma, weighted=weighted, gamma_pred=gamma_pred)[a.job](drops)


if __name__ == '__main__':
    main()
