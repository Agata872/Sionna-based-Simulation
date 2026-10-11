#!/usr/bin/env python3
"""Stage-5 feasibility: digital-twin (CKM) dictionaries for compressing SRS channel
information (CI) that an O-RU sends to the O-DU, versus generic bases.

Independent exploration script (oran/ untouched). Sionna RT, munich scene, one O-RU
with a 4x8 cross-polarized UPA (64 elements). "Reality" is ray-traced at the true UE
position; the "twin" (shared by O-RU and O-DU) is ray-traced at an estimated position
x+e (horizontal Gaussian error) in the nominal scene. Optional reality mismatch: diffuse
scattering enabled only in reality.

Representations compared at equal numbers of complex coefficients C:
  dft_beam   O-RAN beamspace type II analogue: wideband top-N spatial DFT beams x F
  ad_dft     angle-delay DFT (beam x delay), top-K coefficients (indices charged separately)
  twin_path  one atom per twin path: twin per-antenna response (incl. polarization) x delay
  twin_pol   two atoms per twin path (polarization blocks): steering x delay
  twin_der   twin_pol + first-order mismatch atoms (d/dtheta, d/dphi, d/dtau per polarization)
  twin_res   twin_pol with half the budget + top-K angle-delay DFT on the residual
  twin_srch  (--search-step>0) O-RU picks the best twin entry on a grid around x+e (index bits charged)
  cov_klt    (--stat-samples>0) statistical CKM: space-frequency KLT from ray-traced channels at
             random points within --stat-radius of x+e (a covariance map, no path geometry)
  jac_klt    analytic twin covariance under position uncertainty from ONE twin trace at x+e:
             R = sum_p beta_p [d d^H + s_phi^2 dphi dphi^H + s_theta^2 dth dth^H + s_tau^2 dtau dtau^H],
             KLT of R (eigenvectors in water-filling order), SURE-selected dimension + residual layer
The O-RU estimates a common delay offset between twin and CI (8-bit search) before fitting.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import sionna.rt
from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene
from explore_csirs_precoding import Recorder

C0 = 299792458.


def rotation(yaw, pitch, roll):
    cz, sz, cy, sy, cx, sx = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return rz@ry@rx


class Site:
    def __init__(self, args):
        self.args = args
        self.scene = load_scene(sionna.rt.scene.munich, merge_shapes=True)
        self.scene.frequency = args.fc
        self.lam = C0/args.fc
        self.array = PlanarArray(num_rows=args.rows, num_cols=args.cols, pattern='tr38901', polarization='cross')
        self.scene.tx_array = self.array
        self.scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern='iso', polarization='V')
        self.tx = Transmitter('ru0', position=args.ru_pos, look_at=args.ru_look)
        self.scene.add(self.tx)
        o = np.array(self.tx.orientation).reshape(-1)
        self.rg = (rotation(*o[:3])@np.array(self.array.positions(self.lam))).T      # [S,3] global offsets
        self.solver = PathSolver(deterministic=True)
        self.freqs = (np.arange(args.groups)*args.grp_prb*12+args.grp_prb*6-args.num_prb*6)*args.scs

    def materials(self, scattering):
        for obj in self.scene.objects.values():
            obj.radio_material.scattering_coefficient = scattering

    def trace(self, positions, diffuse=False, chunk=32):
        out = []
        for c0 in range(0, len(positions), chunk):
            out.extend(self._trace(positions[c0:c0+chunk], diffuse))
        return out

    def _trace(self, positions, diffuse):
        for name in list(self.scene.receivers):
            self.scene.remove(name)
        for i, p in enumerate(positions):
            self.scene.add(Receiver(f'ue{i}', position=[float(v) for v in p]))
        paths = self.solver(self.scene, max_depth=self.args.max_depth, samples_per_src=self.args.samples,
                            diffraction=True, diffuse_reflection=diffuse, synthetic_array=True)
        a, tau = paths.cir(normalize_delays=False, out_type='numpy')
        th = np.array(paths.theta_t)[:, 0]
        ph = np.array(paths.phi_t)[:, 0]
        a = a[:, 0, 0, :, :, 0]                       # [N, 64, P]
        tau = tau[:, 0]                               # [N, P]
        out = []
        for i in range(len(positions)):
            pw = (np.abs(a[i])**2).sum(0)
            ok = np.flatnonzero((tau[i] >= 0) & (pw > 0))
            ok = ok[np.argsort(pw[ok])[::-1]]
            out.append(dict(a=a[i][:, ok], tau=tau[i][ok], th=th[i][ok], ph=ph[i][ok], pw=pw[ok]))
        return out

    def steer(self, th, ph):
        """[S, P] steering (one polarization block), validated against Sionna synthetic arrays."""
        u = np.stack([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])      # [3,P]
        return np.exp(1j*2*np.pi/self.lam*(self.rg@u))

    def dsteer(self, th, ph):
        u_t = np.stack([np.cos(th)*np.cos(ph), np.cos(th)*np.sin(ph), -np.sin(th)])
        u_p = np.stack([-np.sin(th)*np.sin(ph), np.sin(th)*np.cos(ph), np.zeros_like(th)])
        s = self.steer(th, ph)
        k = 1j*2*np.pi/self.lam
        return s*(k*(self.rg@u_t)), s*(k*(self.rg@u_p))


def cfr(path, freqs, t0):
    """H[m,f] = sum_p a_mp exp(-j2pi f (tau_p - t0))."""
    e = np.exp(-2j*np.pi*np.outer(path['tau']-t0, freqs))                 # [P,F]
    return path['a']@e


def project(h, atoms):
    """LS fit of vec(h) on atom columns; returns reconstruction and coefficient count."""
    q, _ = np.linalg.qr(atoms)
    keep = np.abs(np.diag(_)) > 1e-9*np.abs(np.diag(_)).max() if atoms.shape[1] else np.zeros(0, bool)
    q = q[:, keep]
    return (q@(q.conj().T@h)), q.shape[1]


def twin_atoms(site, twin, freqs, t0, n_paths, kind):
    s_ = site.steer(twin['th'][:n_paths], twin['ph'][:n_paths])                  # [S,P]
    b = np.exp(-2j*np.pi*np.outer(twin['tau'][:n_paths]-t0, freqs))               # [P,F]
    S = s_.shape[0]
    cols = []
    for p in range(min(n_paths, len(twin['tau']))):
        if kind == 'twin_path':
            cols.append(np.kron(twin['a'][:, p], b[p]))                          # [64*F]
            continue
        for blk in range(2):
            z = np.zeros(2*S, complex)
            z[blk*S:(blk+1)*S] = s_[:, p]
            cols.append(np.kron(z, b[p]))
            if kind == 'twin_der':
                dt, dp = site.dsteer(twin['th'][p:p+1], twin['ph'][p:p+1])
                for d in (dt[:, 0], dp[:, 0]):
                    z2 = np.zeros(2*S, complex)
                    z2[blk*S:(blk+1)*S] = d
                    cols.append(np.kron(z2, b[p]))
                cols.append(np.kron(z, b[p]*(-2j*np.pi*freqs)))
    return np.stack(cols, 1) if cols else np.zeros((2*S*len(freqs), 0), complex)


def pol_gains(site, twin, n):
    """Per-path polarization weights g1, g2 so that twin a[:, p] = [g1 s_p; g2 s_p]."""
    s_ = site.steer(twin['th'][:n], twin['ph'][:n])
    S = s_.shape[0]
    g1 = (s_.conj()*twin['a'][:S, :n]).sum(0)/S
    g2 = (s_.conj()*twin['a'][S:, :n]).sum(0)/S
    return s_, g1, g2


def adaptive_atoms(site, twin, freqs, t0, budget, sigma_pos, resid_tau_frac=.3, nvar=0., kappa=2.):
    """Mismatch-aware greedy dictionary, computable identically at O-RU and O-DU.

    Base unit per path: full twin atom (1 coefficient, value beta_p (1 - eps_p)).
    Refinement unit per path: d/dphi, d/dtheta, d/dtau of the full atom (3 coefficients,
    value ~ beta_p eps_p). eps_p = first-order leakage of the base atom under the known
    position-error statistics: angle error ~ sigma/r_p with r_p = c tau_p, residual delay
    error ~ resid_tau_frac sigma / c after common alignment.
    """
    n = min(len(twin['tau']), budget)
    if n == 0:
        return np.zeros((2*site.rg.shape[0]*len(freqs), 0), complex)
    s_, g1, g2 = pol_gains(site, twin, n)
    dt, dp = site.dsteer(twin['th'][:n], twin['ph'][:n])
    b = np.exp(-2j*np.pi*np.outer(twin['tau'][:n]-t0, freqs))
    beta = twin['pw'][:n]
    r = np.maximum(C0*twin['tau'][:n], 5.)
    sig_ang = max(sigma_pos, .25)/r
    sig_tau = resid_tau_frac*max(sigma_pos, .25)/C0
    base, ref, eps = [], [], []
    for p in range(n):
        d0 = np.kron(twin['a'][:, p], b[p])
        dph = np.kron(np.concatenate([g1[p]*dp[:, p], g2[p]*dp[:, p]]), b[p])
        dth = np.kron(np.concatenate([g1[p]*dt[:, p], g2[p]*dt[:, p]]), b[p])
        dta = np.kron(twin['a'][:, p], b[p]*(-2j*np.pi*freqs))
        u = d0/np.linalg.norm(d0)
        def perp(v):
            return np.linalg.norm(v-u*(u.conj()@v))**2
        e = (sig_ang[p]**2*(perp(dph)+perp(dth))+sig_tau**2*perp(dta))/np.linalg.norm(d0)**2
        eps.append(min(e, 1.))
        base.append(d0)
        ref.append([dph, dth, dta])
    eps = np.array(eps)
    units = [(beta[p]*(1-eps[p]), 1, ('b', p)) for p in range(n)]
    units += [(beta[p]*eps[p]/3, 3, ('r', p)) for p in range(n)]
    units.sort(key=lambda x: -x[0])
    chosen, used, have_base = [], 0, set()
    for value, cost, (kind, p) in units:
        if used+cost > budget or value <= kappa*cost*nvar:     # unit must beat the noise it admits
            continue
        if kind == 'r' and p not in have_base:
            if used+cost+1 > budget:
                continue
            chosen.append(base[p])
            have_base.add(p)
            used += 1
        if kind == 'b':
            if p in have_base:
                continue
            have_base.add(p)
            chosen.append(base[p])
        else:
            chosen.extend(ref[p])
        used += cost
    if not chosen:
        chosen = [base[0]]
    return np.stack(chosen, 1)


def jacobian_basis(site, twin, freqs, t0, sigma_pos, n_paths=64, resid_tau_frac=.3):
    """Orthonormal KLT basis (columns sorted by eigenvalue) of the first-order twin covariance."""
    n = min(len(twin['tau']), n_paths)
    if n == 0:
        return np.zeros((2*site.rg.shape[0]*len(freqs), 0), complex), np.zeros(0)
    s_, g1, g2 = pol_gains(site, twin, n)
    dt, dp = site.dsteer(twin['th'][:n], twin['ph'][:n])
    b = np.exp(-2j*np.pi*np.outer(twin['tau'][:n]-t0, freqs))
    beta = twin['pw'][:n]
    r = np.maximum(C0*twin['tau'][:n], 5.)
    sa = max(sigma_pos, .25)/r
    st = resid_tau_frac*max(sigma_pos, .25)/C0
    cols = []
    for p in range(n):
        w = math.sqrt(beta[p])
        cols.append(w*np.kron(twin['a'][:, p], b[p]))
        cols.append(w*sa[p]*np.kron(np.concatenate([g1[p]*dp[:, p], g2[p]*dp[:, p]]), b[p]))
        cols.append(w*sa[p]*np.kron(np.concatenate([g1[p]*dt[:, p], g2[p]*dt[:, p]]), b[p]))
        cols.append(w*st*np.kron(twin['a'][:, p], b[p]*(-2j*np.pi*freqs)))
    B = np.stack(cols, 1)
    u, sv, _ = np.linalg.svd(B, full_matrices=False)
    return u, sv**2


def dft_bases(site, F):
    """Spatial DFT (I_2 x D_cols x D_rows in Sionna position order) and delay DFT."""
    rows, cols = site.args.rows, site.args.cols
    dc = np.exp(2j*np.pi*np.outer(np.arange(cols), np.arange(cols))/cols)/math.sqrt(cols)
    dr = np.exp(2j*np.pi*np.outer(np.arange(rows), np.arange(rows))/rows)/math.sqrt(rows)
    sp = np.kron(np.eye(2), np.kron(dc, dr))                                     # positions: col-major rows
    fd = np.exp(-2j*np.pi*np.outer(np.arange(F), np.arange(F))/F)/math.sqrt(F)
    return sp, fd


def best_offset(site, twin, h, t0, n_paths, kind):
    """O-RU search of a common delay offset (8 bits over +/-128 ns, 1 ns steps)."""
    best, arg = -1., 0.
    for d in np.arange(-128, 128, 4)*1e-9:
        rec, _ = project(h.reshape(-1), twin_atoms(site, twin, site.freqs, t0+d, n_paths, 'twin_pol'))
        e = np.linalg.norm(rec)**2
        if e > best:
            best, arg = e, d
    for d in arg+np.arange(-4, 5)*1e-9:
        rec, _ = project(h.reshape(-1), twin_atoms(site, twin, site.freqs, t0+d, n_paths, 'twin_pol'))
        e = np.linalg.norm(rec)**2
        if e > best:
            best, arg = e, d
    return arg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--num-ue', type=int, default=120)
    p.add_argument('--seed', type=int, default=20268000)
    p.add_argument('--pos-errors', type=float, nargs='+', default=[0., 1., 2., 5.])
    p.add_argument('--reality', choices=['nominal', 'diffuse'], default='nominal')
    p.add_argument('--scattering', type=float, default=.3)
    p.add_argument('--budgets', type=int, nargs='+', default=[8, 16, 32, 64, 128, 256])
    p.add_argument('--ci-snr-db', type=float, default=25.)
    p.add_argument('--fc', type=float, default=3.5e9)
    p.add_argument('--scs', type=float, default=30e3)
    p.add_argument('--num-prb', type=int, default=52)
    p.add_argument('--grp-prb', type=int, default=2)
    p.add_argument('--rows', type=int, default=4)
    p.add_argument('--cols', type=int, default=8)
    p.add_argument('--max-depth', type=int, default=3)
    p.add_argument('--samples', type=int, default=1_000_000)
    p.add_argument('--ru-pos', type=float, nargs=3, default=[8.5, 21., 30.])
    p.add_argument('--ru-look', type=float, nargs=3, default=[8.5, 171., 1.5])
    p.add_argument('--search-step', type=float, default=0., help='twin grid step [m] for twin_srch (0 disables)')
    p.add_argument('--search-half', type=int, default=2, help='grid half-width in steps (5x5 for 2)')
    p.add_argument('--stat-samples', type=int, default=0, help='samples per statistical-CKM cell (0 disables)')
    p.add_argument('--stat-radius', type=float, default=5.)
    args = p.parse_args()
    args.groups = args.num_prb//args.grp_prb
    rec = Recorder(args.out)
    (rec.path/'experiment_source.py').write_bytes(Path(__file__).read_bytes())
    (rec.path/'metadata.json').write_text(json.dumps(dict(args=vars(args), started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        assumptions=['Single O-RU, munich scene, Sionna RT specular+diffraction (max_depth 3); one UE port (1 Rx antenna).',
                     'Reality: true UE position; optional diffuse scattering with scattering coefficient on all materials.',
                     'Twin: nominal scene at x+e, e ~ N(0, sigma^2 I) horizontally; same scene geometry and materials.',
                     'CI = per-PRG-group CFR at group centres, referenced to the true first arrival; CN estimation error at ci-snr-db.',
                     'Coefficient budgets are complex numbers; quantization/bit allocation is not modelled here.',
                     'ad_dft and twin_res indices: K*log2(64*F) bits charged in the bits column; twin atoms cost 0 index bits.']), indent=2))
    site = Site(args)
    rng = np.random.default_rng(args.seed)
    # candidate UE positions: outdoor, at least one path
    cand = []
    while len(cand) < args.num_ue:
        xy = rng.uniform([-80., 35.], [100., 230.], size=(64, 2))
        pts = np.column_stack([xy, np.full(len(xy), 1.5)])
        tr = site.trace(pts)
        for pt, t in zip(pts, tr):
            if len(t['tau']) and 10*np.log10(t['pw'].sum()) > -125:
                cand.append(pt)
            if len(cand) >= args.num_ue:
                break
    cand = np.array(cand)
    if args.reality == 'diffuse':
        site.materials(args.scattering)
    truth = site.trace(cand, diffuse=(args.reality == 'diffuse'))
    if args.reality == 'diffuse':
        site.materials(0.)
    sp, fd = dft_bases(site, args.groups)
    F, A = args.groups, 2*args.rows*args.cols
    for sigma in args.pos_errors:
        err = rng.normal(0, sigma, size=(len(cand), 2)) if sigma > 0 else np.zeros((len(cand), 2))
        est = cand+np.column_stack([err, np.zeros(len(cand))])
        twins = site.trace(est)
        grid_twins, stat = None, None
        if args.search_step > 0:
            offs = [(i*args.search_step, j*args.search_step) for i in range(-args.search_half, args.search_half+1)
                    for j in range(-args.search_half, args.search_half+1)]
            pts = np.concatenate([est+np.array([dx, dy, 0.]) for dx, dy in offs])
            gt = site.trace(pts)
            grid_twins = [[gt[g*len(cand)+u] for g in range(len(offs))] for u in range(len(cand))]
        if args.stat_samples > 0:
            r = args.stat_radius*np.sqrt(rng.uniform(size=(len(cand), args.stat_samples)))
            a_ = rng.uniform(0, 2*np.pi, size=(len(cand), args.stat_samples))
            pts = (est[:, None, :]+np.stack([r*np.cos(a_), r*np.sin(a_), np.zeros_like(r)], -1)).reshape(-1, 3)
            st = site.trace(pts)
            stat = [st[u*args.stat_samples:(u+1)*args.stat_samples] for u in range(len(cand))]
        for u in range(len(cand)):
            tr_, tw = truth[u], twins[u]
            if not len(tr_['tau']) or not len(tw['tau']):
                continue
            t0 = tr_['tau'].min()
            h0 = cfr(tr_, site.freqs, t0)                                         # [64,F]
            power = np.mean(np.abs(h0)**2)
            noise = (rng.normal(size=h0.shape)+1j*rng.normal(size=h0.shape))*math.sqrt(power/10**(args.ci_snr_db/10)/2)
            h = h0+noise
            vec0 = h0.reshape(-1)
            nrm = np.linalg.norm(vec0)**2
            t0_tw = tw['tau'].min()
            off = best_offset(site, tw, h, t0_tw, 16, 'twin_pol')
            klt = None
            if stat is not None:
                cols = [cfr(x, site.freqs, x['tau'].min()).reshape(-1) for x in stat[u] if len(x['tau'])]
                if cols:
                    klt, _sv, _ = np.linalg.svd(np.stack(cols, 1), full_matrices=False)
            ang = np.einsum('am,mf->af', sp.conj().T, h)                          # beamspace [64,F]
            adg = ang@fd.conj()                                                   # angle-delay [64,F]
            beam_energy = (np.abs(ang)**2).sum(1)
            for C in args.budgets:
                res = {}
                n = max(1, C//F)
                keep = np.argsort(beam_energy)[::-1][:n]
                rec_ang = np.zeros_like(ang)
                rec_ang[keep] = ang[keep]
                res['dft_beam'] = (sp@rec_ang, n*F, 64)
                k = min(C, adg.size)
                idx = np.argsort(np.abs(adg).reshape(-1))[::-1][:k]
                m = np.zeros(adg.size, complex)
                m[idx] = adg.reshape(-1)[idx]
                res['ad_dft'] = (sp@(m.reshape(adg.shape)@fd.T), k, k*math.ceil(math.log2(adg.size)))
                for kind, per in (('twin_path', 1), ('twin_pol', 2), ('twin_der', 8)):
                    npath = max(1, C//per)
                    atoms = twin_atoms(site, tw, site.freqs, t0_tw+off, npath, kind)
                    r, used = project(h.reshape(-1), atoms)
                    res[kind] = (r.reshape(A, F), used, 8)
                # twin_adapt: mismatch-aware greedy dictionary + O-RU operational two-layer split,
                # selected by SURE (the O-RU knows its SRS estimation-noise variance)
                best_adapt = None
                nvar = power/10**(args.ci_snr_db/10)
                for frac, use_res in ((1., False), (.75, True), (.5, True), (.25, True), (.5, False), (.25, False), (.125, False)):
                    ka = max(1, int(round(frac*C)))
                    atoms = adaptive_atoms(site, tw, site.freqs, t0_tw+off, ka, sigma, nvar=nvar)
                    r, used = project(h.reshape(-1), atoms)
                    k2 = max(0, C-used) if use_res else 0
                    rr = r.reshape(A, F)
                    if k2:
                        resid = (h.reshape(-1)-r).reshape(A, F)
                        radg = (sp.conj().T@resid)@fd.conj()
                        flat = np.abs(radg).reshape(-1)**2
                        idx = np.argsort(flat)[::-1][:k2]
                        idx = idx[flat[idx] > 2*math.log(A*F)*nvar]       # universal threshold
                        k2 = len(idx)
                        m = np.zeros(radg.size, complex)
                        m[idx] = radg.reshape(-1)[idx]
                        rr = rr+sp@(m.reshape(radg.shape)@fd.T)
                    fit_err = np.linalg.norm(rr-h)**2-(A*F-2*(used+k2))*nvar   # SURE on its own noisy CI
                    if best_adapt is None or fit_err < best_adapt[0]:
                        best_adapt = (fit_err, rr, used+k2, 8+2+k2*math.ceil(math.log2(A*F)))
                res['twin_adapt'] = best_adapt[1:]
                # jac_klt: analytic twin covariance KLT, SURE-selected dimension (+ thresholded residual)
                U, lam = jacobian_basis(site, tw, site.freqs, t0_tw+off, sigma)
                best_j = None
                for kd in sorted({max(1, min(C, U.shape[1], x)) for x in (C, C*3//4, C//2, C//4, C//8, 4, 2)}):
                    q = U[:, :kd]
                    r = q@(q.conj().T@h.reshape(-1))
                    k2 = C-kd
                    rr = r.reshape(A, F)
                    if k2:
                        resid = (h.reshape(-1)-r).reshape(A, F)
                        radg = (sp.conj().T@resid)@fd.conj()
                        flat = np.abs(radg).reshape(-1)**2
                        idx = np.argsort(flat)[::-1][:k2]
                        idx = idx[flat[idx] > 2*math.log(A*F)*nvar]
                        k2 = len(idx)
                        m = np.zeros(radg.size, complex)
                        m[idx] = radg.reshape(-1)[idx]
                        rr = rr+sp@(m.reshape(radg.shape)@fd.T)
                    sure = np.linalg.norm(rr-h)**2-(A*F-2*(kd+k2))*nvar
                    if best_j is None or sure < best_j[0]:
                        best_j = (sure, rr, kd+k2, 8+3+k2*math.ceil(math.log2(A*F)))
                res['jac_klt'] = best_j[1:]
                atoms = twin_atoms(site, tw, site.freqs, t0_tw+off, max(1, C//4), 'twin_pol')
                r, used = project(h.reshape(-1), atoms)
                resid = (h.reshape(-1)-r).reshape(A, F)
                radg = (sp.conj().T@resid)@fd.conj()
                k2 = max(0, C-used)
                idx = np.argsort(np.abs(radg).reshape(-1))[::-1][:k2]
                m = np.zeros(radg.size, complex)
                m[idx] = radg.reshape(-1)[idx]
                res['twin_res'] = (r.reshape(A, F)+sp@(m.reshape(radg.shape)@fd.T), used+k2,
                                   8+k2*math.ceil(math.log2(radg.size)))
                if grid_twins is not None:
                    best, choice = -1., None
                    for gt_ in grid_twins[u]:
                        if not len(gt_['tau']):
                            continue
                        atoms = twin_atoms(site, gt_, site.freqs, gt_['tau'].min()+off, max(1, C), 'twin_path')
                        r_, used_ = project(h.reshape(-1), atoms)
                        e_ = np.linalg.norm(r_)**2
                        if e_ > best:
                            best, choice = e_, (r_, used_)
                    if choice is not None:
                        res['twin_srch'] = (choice[0].reshape(A, F), choice[1],
                                            8+math.ceil(math.log2(len(grid_twins[u]))))
                if klt is not None:
                    q = klt[:, :min(C, klt.shape[1])]
                    res['cov_klt'] = ((q@(q.conj().T@h.reshape(-1))).reshape(A, F), q.shape[1], 0)
                for name, (hh, used, side_bits) in res.items():
                    e = hh.reshape(-1)-vec0
                    g = np.abs(np.einsum('af,af->f', hh.conj(), h0))**2/(np.linalg.norm(hh, axis=0)**2*np.linalg.norm(h0, axis=0)**2+1e-30)
                    rec.add(dict(ue=u, sigma=sigma, budget=C, scheme=name, coeffs=int(used), side_bits=int(side_bits),
                                 nmse_db=float(10*np.log10(np.linalg.norm(e)**2/nrm+1e-12)),
                                 bf_loss_db=float(-10*np.log10(np.mean(g)+1e-12)),
                                 n_true_paths=len(tr_['tau']), n_twin_paths=len(tw['tau']),
                                 offset_ns=float(off*1e9), pos_err=float(np.linalg.norm(err[u]))))
        print(f'sigma={sigma} rows={len(rec.rows)}', flush=True)
    rec.finish()


if __name__ == '__main__':
    main()
