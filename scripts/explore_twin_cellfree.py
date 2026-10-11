#!/usr/bin/env python3
"""Stage-5 feasibility: twin-aided compression of SRS channel information (CI) on a
cell-free O-RAN fronthaul, judged by centralized precoding performance at the O-DU.

Independent exploration script (oran/ untouched). Sionna RT munich scene; L O-RUs on
outdoor poles, each a 4x4 cross-polarized UPA; K single-antenna UEs per drop. Each
O-RU compresses its CI per UE under a coefficient budget; the O-DU reconstructs all
CI and computes a centralized RZF precoder per PRB group; sum/5%-SE are evaluated on
the true channels. Twin = nominal scene ray-traced at the estimated UE position.

Schemes: ideal, ad_dft (angle-delay DFT top-K per link), twin_path (twin path atoms),
twin_res (twin atoms on half budget + angle-delay residual), urts (uncertainty-region twin
subspace: per-link KLT of twin channels synthesized at --region-samples points of the
position uncertainty region; both ends can build it from the twin/CKM; plain truncation),
urts_code (the full method: the O-DU allocates KLT dimensions across links by greedy
reverse water-filling of the twin eigenvalues on a first-order sum-SE proxy under per-O-RU
bit budgets, deterministically from the twin; each O-RU spends its leftover bits on
angle-delay residual coefficients above the universal threshold 2 ln(N) x noise, chosen
with the same proxy). Allocation: 'uniform' (same budget on every link) or 'twin' (budget
proportional to twin-predicted link power, links 25 dB below the UE's strongest link get
nothing); urts_code always uses its own allocation ('task'). urts_vq is urts_code with
variable-precision KLT coefficients: bits per dimension by greedy reverse water-filling of the
twin eigenvalues (1 bit per real component at a time, optimal uniform Gaussian quantizer with
the twin-predicted variance, up to 12 bits), no signalling of the bit pattern. urts_jvq: as
urts_vq, but each O-RU splits its bits between KLT precision and residual coefficients from what it
measures (per-link prefix of the twin-ordered increments; 10 extra bits per link); tested on
dumped drops, no gain over urts_vq (stage-6 note), kept for reference.
Region: 'disc' (uniform in --region-radius) or 'gauss' (posterior N(x_hat, s^2 I) with
s^2 = sigma^2 + --region-floor^2).
Bits: 2 x 9 bits per complex coefficient (BFP9-like) + index bits for DFT supports.

Paper-1 additions (all opt-in; defaults reproduce the stage-5..7 runs row for row, and every new
random draw uses its own stream so the drops stay paired with those runs):
  --idx-coding comb   index sets of n out of N angle-delay cells cost ceil(log2 C(N, n)) bits per link
                      instead of n ceil(log2 N) (ad_dft, twin_res and the residual layers alike).
  grid_ecb            environment basis of a fixed grid cell (Cai et al., arXiv 2508.05142): per link the
                      KLT of twin channels at the 4 vertices (+ uniform interior points) of the --grid-size
                      cell holding the estimated position; plain truncation like 'urts'.
  grid_vq             the final method (urts_vq) with that grid-cell basis instead of the posterior region
                      (ablation: what the posterior region adds).
  site_klt            one KLT per O-RU from twin channels over the whole area, no position (linear
                      autoencoder / 'global model' baseline); plain truncation.
  --precoder rzf_perru  the O-DU's RZF scaled so that no O-RU exceeds 1/L of the total power.
  twin_ae / twin_aep  per-O-RU autoencoders trained on twin channels over the whole area (scripts/twin_ae.py),
                      without / with the estimated UE position as input (data-driven baselines).

First-order SE proxy (used by urts_code and recorded for every row): for UE k with
channel energy E and CI error energy D (summed over all O-RUs, antennas and PRB groups,
UE noise = 1), SINR ~ a (E - D) / (1 + c D) with a = (Nt-K+1)/(Nt F K) (ZF signal, power
1/K per UE) and c = (K-1)/(K F Nt) (leakage of the other K-1 beams through an isotropic
error). Compression errors leak more than that (gamma, see se_proxy); fitted on dumped drops.
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
import drjit as dr
import mitsuba as mi
from sionna.rt import (ITURadioMaterial, PathSolver, PlanarArray, RadioMaterial, Receiver, SceneObject, Transmitter,
                       load_scene)
from explore_csirs_precoding import Recorder
from explore_twin_ci_compression import rotation, cfr, project

C0 = 299792458.
BOLTZMANN = 1.380649e-23


class MultiSite:
    """Twin = munich scene with merged shapes. Reality = the same scene plus optional diffuse
    scattering (--reality diffuse) and, if any --real-* option is set, a separate unmerged copy
    in which every building is shifted rigidly (--real-geom, std per axis, m), gets a perturbed
    material (--real-material, log-normal std of permittivity and conductivity factors, 16
    variants per ITU material) and parked cars are added (--real-cars, metal, absent in the twin)."""

    def __init__(self, args, ru_pos, rng):
        self.args = args
        self.lam = C0/args.fc
        self.array = PlanarArray(num_rows=args.rows, num_cols=args.cols, pattern='tr38901', polarization='cross')
        self.centre = np.array(args.area_centre+[1.5])
        self.ru_pos = ru_pos
        self.scene = self._setup(load_scene(sionna.rt.scene.munich, merge_shapes=True))
        self.rg = []
        for l in range(len(ru_pos)):
            o = np.array(self.scene.transmitters[f'ru{l}'].orientation).reshape(-1)
            self.rg.append((rotation(*o[:3])@np.array(self.array.positions(self.lam))).T)
        self.solver = PathSolver(deterministic=True)
        self.freqs = (np.arange(args.groups)*args.grp_prb*12+args.grp_prb*6-args.num_prb*6)*args.scs
        p_re = 10**((args.ru_power_dbm-30)/10)/(args.num_prb*12)
        n0 = BOLTZMANN*290*args.scs*10**(args.ue_nf_db/10)
        self.amp = math.sqrt(p_re/n0)
        self.twin_scat = float(getattr(args, 'twin_scattering', 0.))
        if self.twin_scat > 0:                                      # twin that models diffuse scattering too
            self.materials(self.twin_scat)
        self.real, self.cars = None, np.zeros((0, 2))
        if self.perturbed(args):
            self.real = self._setup(load_scene(sionna.rt.scene.munich, merge_shapes=False))
            self._perturb(np.random.default_rng(args.seed+7919))    # own stream: drops stay paired across mismatch types

    @staticmethod
    def perturbed(args):
        return any(getattr(args, k, 0) for k in ('real_geom', 'real_material', 'real_cars'))

    def _setup(self, scene):
        scene.frequency = self.args.fc
        scene.tx_array = self.array
        scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern='iso', polarization='V')
        for l, p in enumerate(self.ru_pos):
            scene.add(Transmitter(f'ru{l}', position=[float(v) for v in p], look_at=[float(v) for v in self.centre]))
        return scene

    def _free(self, xy, z=1.0, r=2.6):
        """Points whose disc of radius r is open sky above height z in the twin (car footprint check)."""
        ang = np.linspace(0, 2*np.pi, 8, endpoint=False)
        ok = np.ones(len(xy), bool)
        for dx, dy in [(0., 0.)]+[(r*math.cos(t), r*math.sin(t)) for t in ang]:
            q = np.asarray(xy, np.float32)+np.float32([dx, dy])
            o = mi.Point3f(mi.Float(q[:, 0]), mi.Float(q[:, 1]), mi.Float(np.full(len(q), z, np.float32)))
            ok &= ~self.scene.mi_scene.ray_intersect(mi.Ray3f(o, mi.Vector3f(0, 0, 1))).is_valid().numpy()
        return ok

    def _perturb(self, rng):
        a, sc = self.args, self.real
        groups = {}
        for name, o in sc.objects.items():
            if name != 'ground':
                groups.setdefault(name.split('-itu_')[0], []).append(o)
        if getattr(a, 'real_geom', 0) > 0:
            for objs in groups.values():
                d = rng.normal(0, a.real_geom, 2)
                for o in objs:
                    o.position = o.position+mi.Point3f(float(d[0]), float(d[1]), 0.)
        if getattr(a, 'real_material', 0) > 0:
            var = {}
            for objs in groups.values():
                for o in objs:
                    m = o.radio_material
                    if m.name not in var:
                        eps, sig, th = (float(np.asarray(v).reshape(-1)[0]) for v in (m.relative_permittivity, m.conductivity, m.thickness))
                        f = np.exp(rng.normal(0, a.real_material, size=(16, 2)))
                        var[m.name] = [RadioMaterial(f'{m.name}_v{i}', thickness=th, relative_permittivity=max(1., eps*f[i, 0]),
                                                     conductivity=sig*f[i, 1]) for i in range(16)]
                    o.radio_material = var[m.name][int(rng.integers(16))]
        if getattr(a, 'real_cars', 0) > 0:
            mat = ITURadioMaterial('car-metal', 'metal', thickness=0.01)
            cars = [SceneObject(fname=sionna.rt.scene.low_poly_car, name=f'car{i}', radio_material=mat) for i in range(a.real_cars)]
            sc.edit(add=cars)
            spots = []
            while len(spots) < a.real_cars:
                c = rng.uniform(a.area_lo, a.area_hi, size=(64, 2))
                for q in c[self._free(c)]:
                    if len(spots) < a.real_cars and all(np.linalg.norm(q-s_) > 6 for s_ in spots):
                        spots.append(q)
            for car, q in zip(cars, spots):
                car.position = mi.Point3f(float(q[0]), float(q[1]), 0.775)
                car.orientation = mi.Point3f(float(rng.uniform(0, 2*np.pi)), 0., 0.)
            self.cars = np.array(spots)
        if a.reality == 'diffuse':
            for m in sc.radio_materials.values():
                m.scattering_coefficient = a.scattering

    def clear_of_cars(self, xy, r=3.):
        if not len(self.cars):
            return np.ones(len(xy), bool)
        return np.linalg.norm(np.asarray(xy)[:, None, :2]-self.cars[None], axis=-1).min(1) > r

    def trace_truth(self, positions):
        """Reality channels: perturbed scene if configured, diffuse scattering if --reality diffuse."""
        diffuse = self.args.reality == 'diffuse'
        if self.real is not None:
            return self.trace(positions, diffuse=diffuse, real=True)
        self.materials(self.args.scattering if diffuse else 0.)
        out = self.trace(positions, diffuse=diffuse)
        self.materials(self.twin_scat)
        return out

    def trace_twin(self, positions):
        return self.trace(positions, diffuse=self.twin_scat > 0)

    def materials(self, scattering):
        for obj in self.scene.objects.values():
            obj.radio_material.scattering_coefficient = scattering

    def trace(self, positions, diffuse=False, chunk=16, real=False):
        scene = self.real if real and self.real is not None else self.scene
        out = []
        for c0 in range(0, len(positions), chunk):
            out.extend(self._trace(scene, positions[c0:c0+chunk], diffuse))
        return out

    def _trace(self, scene, positions, diffuse):
        for name in list(scene.receivers):
            scene.remove(name)
        for i, p in enumerate(positions):
            scene.add(Receiver(f'ue{i}', position=[float(v) for v in p]))
        paths = self.solver(scene, max_depth=self.args.max_depth, samples_per_src=self.args.samples,
                            diffraction=True, diffuse_reflection=diffuse, synthetic_array=True)
        a, tau = paths.cir(normalize_delays=False, out_type='numpy')
        th, ph = np.array(paths.theta_t), np.array(paths.phi_t)
        del paths
        dr.flush_malloc_cache()            # return cached GPU memory: two runs share one T4 (results unchanged)
        out = []
        for i in range(len(positions)):
            per_ru = []
            for l in range(a.shape[2]):
                al = a[i, 0, l, :, :, 0]*self.amp
                pw = (np.abs(al)**2).sum(0)
                ok = np.flatnonzero((tau[i, l] >= 0) & (pw > 0))
                ok = ok[np.argsort(pw[ok])[::-1]]
                per_ru.append(dict(a=al[:, ok], tau=tau[i, l][ok], th=th[i, l][ok], ph=ph[i, l][ok], pw=pw[ok]))
            out.append(per_ru)
        return out


IDX_COMB = False              # --idx-coding comb: combinatorial index sets instead of ceil(log2 N) bits per index
PER_RU_L = 0                  # --precoder rzf_perru: number of O-RUs for the per-O-RU power scaling (0 = total power only)


def idx_bits_total(N, n):
    """Bits for n distinct indices out of N."""
    if n <= 0:
        return 0
    if not IDX_COMB:
        return n*math.ceil(math.log2(N))
    return math.ceil((math.lgamma(N+1)-math.lgamma(n+1)-math.lgamma(N-n+1))/math.log(2))


def idx_bits_next(N, n):
    """Marginal bits of one more index when n are already chosen (greedy cost estimate)."""
    if not IDX_COMB:
        return math.ceil(math.log2(N))
    return max(math.log2((N-n)/(n+1)), .01)


BFP_MANT = 0                  # >0: every coefficient billed as 18 bits is really BFP-quantized (set by --bfp)
BFP_BLOCK, BFP_EXP_BITS = 12, 4
BFP_KAPPA = 0.                # >0: noise-matched BFP (set by --bfp-kappa): per block the narrowest mantissa
BFP_NVAR = 1.                 #     (2..BFP_MANT bits, 3-bit width field) whose step is <= kappa*sqrt(noise)


def _bfp_width(mx):
    return int(np.clip(math.ceil(math.log2(mx/(BFP_KAPPA*math.sqrt(BFP_NVAR))+1))+1, 2, BFP_MANT))


def coef_bits(c):
    """Expected bits of one coefficient under the active quantizer (for greedy cost estimates)."""
    if BFP_MANT and BFP_KAPPA > 0:
        return 2*_bfp_width(max(abs(c.real), abs(c.imag), 1e-30))+(BFP_EXP_BITS+3)/BFP_BLOCK
    return 18+(BFP_EXP_BITS/BFP_BLOCK if BFP_MANT else 0)


def bfpq(c):
    """O-RAN-style block floating point on coefficients in send order: blocks of BFP_BLOCK share an
    exponent (BFP_EXP_BITS bits), each real component gets a BFP_MANT-bit signed mantissa (exponent
    range not limited); noise-matched mode narrows the mantissa per block. Returns (quantized
    coefficients, bits beyond the nominal 18 per coefficient); identity if BFP_MANT == 0."""
    c = np.asarray(c, complex)
    if not BFP_MANT or not len(c):
        return c, 0
    out = np.empty_like(c)
    bits = 0
    for b0 in range(0, len(c), BFP_BLOCK):
        x = c[b0:b0+BFP_BLOCK]
        mx = max(np.abs(x.real).max(), np.abs(x.imag).max(), 1e-30)
        w = _bfp_width(mx) if BFP_KAPPA > 0 else BFP_MANT
        top = 2**(w-1)
        step = 2.**math.ceil(math.log2(mx/(top-1))) if w > 1 else mx
        out[b0:b0+BFP_BLOCK] = (np.clip(np.round(x.real/step), -top, top-1)
                                +1j*np.clip(np.round(x.imag/step), -top, top-1))*step
        bits += BFP_EXP_BITS+(3 if BFP_KAPPA > 0 else 0)+2*w*len(x)
    return out, bits-18*len(c)


def project_q(h, atoms):
    """project() with the coefficients (in the orthonormalized atom basis, atom order) BFP-quantized.
    Returns reconstruction, coefficient count, exponent bits."""
    q, rr = np.linalg.qr(atoms)
    keep = np.abs(np.diag(rr)) > 1e-9*np.abs(np.diag(rr)).max() if atoms.shape[1] else np.zeros(0, bool)
    q = q[:, keep]
    c, xb = bfpq(q.conj().T@h)
    return q@c, q.shape[1], xb


def twin_path_atoms(tw, freqs, t0, n):
    e = np.exp(-2j*np.pi*np.outer(tw['tau'][:n]-t0, freqs))
    return np.stack([np.kron(tw['a'][:, p], e[p]) for p in range(min(n, len(tw['tau'])))], 1) \
        if len(tw['tau']) else np.zeros((tw['a'].shape[0]*len(freqs), 0), complex)


def ad_bases(rows, cols, F):
    dc = np.exp(2j*np.pi*np.outer(np.arange(cols), np.arange(cols))/cols)/math.sqrt(cols)
    dr = np.exp(2j*np.pi*np.outer(np.arange(rows), np.arange(rows))/rows)/math.sqrt(rows)
    sp = np.kron(np.eye(2), np.kron(dc, dr))
    fd = np.exp(-2j*np.pi*np.outer(np.arange(F), np.arange(F))/F)/math.sqrt(F)
    return sp, fd


def topk_ad(h, sp, fd, k):
    """k strongest angle-delay coefficients (sent strongest first). Returns reconstruction, exponent bits."""
    adg = (sp.conj().T@h)@fd.conj()
    idx = np.argsort(np.abs(adg).reshape(-1))[::-1][:k]
    m = np.zeros(adg.size, complex)
    m[idx], xb = bfpq(adg.reshape(-1)[idx])
    return sp@(m.reshape(adg.shape)@fd.T), xb


def offset_search(h, tw, freqs, t0, n=12):
    best, arg = -1., 0.
    for d in np.concatenate([np.arange(-128, 128, 4), [0]])*1e-9:
        r, _ = project(h.reshape(-1), twin_path_atoms(tw, freqs, t0+d, n))
        e = np.linalg.norm(r)**2
        if e > best:
            best, arg = e, d
    return arg


def compress(h, tw, off, freqs, sp, fd, scheme, k):
    """Return reconstruction, complex coefficients sent, side bits. `off` is the
    O-RU's twin delay alignment (computed once per link, 8 bits)."""
    A, F = h.shape
    if k <= 0:
        return np.zeros_like(h), 0, 0
    if scheme == 'ad_dft' or not len(tw['tau']):
        r, xb = topk_ad(h, sp, fd, k)
        return r, k, idx_bits_total(A*F, k)+xb
    t0 = off                                                   # absolute twin reference incl. alignment
    if scheme == 'twin_path':
        r, used, xb = project_q(h.reshape(-1), twin_path_atoms(tw, freqs, t0, k))
        return r.reshape(A, F), used, 8+xb
    r, used, xb = project_q(h.reshape(-1), twin_path_atoms(tw, freqs, t0, max(1, k//2)))
    resid = h-r.reshape(A, F)
    k2 = max(0, k-used)
    rr, xb2 = topk_ad(resid, sp, fd, k2) if k2 else (np.zeros_like(h), 0)
    return r.reshape(A, F)+rr, used+k2, 8+xb+xb2+idx_bits_total(A*F, k2)


TASK_SCHEMES = ('urts_code', 'urts_vq', 'urts_jvq', 'grid_vq')


def rzf_se(H, Hhat, noise=1.):
    """H, Hhat: [K, Nt, F]. Centralized RZF per PRB group, total power Nt-normalized to L RUs.
    With PER_RU_L > 0 the precoder is scaled down until no O-RU exceeds 1/L of the total power."""
    K, Nt, F = H.shape
    se = np.zeros(K)
    for f in range(F):
        Hf, Gf = H[:, :, f], Hhat[:, :, f]
        # RZF on the reconstructed CI; equal power 1/K per UE column (noise 1 at each UE)
        W = Gf.conj().T@np.linalg.inv(Gf@Gf.conj().T+K*noise*np.eye(K))
        W = W/np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-30)/math.sqrt(K)
        if PER_RU_L:
            p_ru = (np.abs(W)**2).reshape(PER_RU_L, Nt//PER_RU_L, K).sum((1, 2))
            W = W*min(1., math.sqrt(1/PER_RU_L/max(p_ru.max(), 1e-30)))
        S = np.abs(Hf@W)**2
        sig = np.diag(S)
        intf = S.sum(1)-sig
        se += np.log2(1+sig/(intf+noise))
    return se/F


def _g(gamma, k):
    """Leakage factor of UE k: a scalar for all UEs or a per-UE array (twin-predicted)."""
    return gamma[k] if np.ndim(gamma) else gamma


def se_proxy(E, D, K, F, Nt, gamma=1.):
    """First-order SE of centralized ZF/RZF for channel energy E and CI error energy D (see module doc).
    gamma: leakage of the error relative to an isotropic one (about 1 for white CI noise, about 2.7
    for compression errors, which sit in the dominant directions shared by the UEs at each O-RU)."""
    a, c = (Nt-K+1)/(Nt*F*K), (K-1)/(K*F*Nt)
    D = np.maximum(D, 0.)                                      # mixed twin/measured bookkeeping can undershoot
    return np.log2(1+np.maximum(E-D, 0.)*a/(1+gamma*c*D))


def klt_cost(lv, nvar):
    """Expected bits of a KLT coefficient of variance lv + nvar under the active quantizer: 18 (BFP9 or
    legacy), or with noise-matched BFP the width a block led by a ~2.5-sigma value would get."""
    if BFP_MANT and BFP_KAPPA > 0:
        amp = 2.5*np.sqrt((np.asarray(lv, float)+nvar)/2)
        w = np.clip(np.ceil(np.log2(amp/(BFP_KAPPA*math.sqrt(BFP_NVAR))+1))+1, 2, BFP_MANT)
        return 2*w+(BFP_EXP_BITS+3)/BFP_BLOCK
    return np.full(np.shape(lv), 18.+(BFP_EXP_BITS/BFP_BLOCK if BFP_MANT else 0.))


def wf_allocate(lam, nvar, ru_bits, F, Nt, cost=18, kappa=1., gamma=1.):
    """O-DU (and, identically, every O-RU): greedy reverse water-filling of the twin KLT
    eigenvalues lam[k][l] (descending, None if no twin path) on the SE proxy, per-O-RU bit
    budget ru_bits, by gain per expected bit (klt_cost). A dimension is worth sending only if its
    eigenvalue exceeds kappa x noise. Only the chosen UE's gains change after a step, so only they
    are recomputed."""
    K, L = len(lam), len(lam[0])
    smax = max((len(x) for r in lam for x in r if x is not None), default=0)
    lp = np.zeros((K, L, smax+1))                              # zero padding = not worth sending
    for k in range(K):
        for l in range(L):
            if lam[k][l] is not None:
                lp[k, l, :len(lam[k][l])] = lam[k][l]
    cp = klt_cost(lp, nvar)
    E = lp.sum((1, 2))
    D = E.copy()
    n = np.zeros((K, L), int)
    left = np.full(L, float(ru_bits))
    ls = np.arange(L)

    def gains(k):
        x, c = lp[k, ls, n[k]], cp[k, ls, n[k]]
        g = se_proxy(E[k], D[k]-x+nvar, K, F, Nt, _g(gamma, k))-se_proxy(E[k], D[k], K, F, Nt, _g(gamma, k))
        return np.where((x > kappa*nvar) & (left >= c), g/c, -np.inf)
    G = np.stack([gains(k) for k in range(K)])
    while True:
        k, l = np.unravel_index(int(np.argmax(G)), G.shape)
        if not G[k, l] > 0:
            return n, E, D, left
        D[k] -= lp[k, l, n[k, l]]-nvar
        left[l] -= cp[k, l, n[k, l]]
        n[k, l] += 1
        G[k] = gains(k)
        G[:, l] = np.where(left[l] >= cp[np.arange(K), l, n[:, l]], G[:, l], -np.inf)


def urts_code(Hn, bases, lam, nvar, ru_bits, sp, fd, kappa=1., gamma=1., res_thresh=1.):
    """Full method. Hn [K, L, A, F] noisy CI at the O-RUs; bases/lam the shared twin KLT.
    Returns reconstruction, coefficients per link, total bits."""
    K, L, A, F = Hn.shape
    N, Nt = A*F, L*A
    idx_bits = math.ceil(math.log2(N))
    n, E, D, left = wf_allocate(lam, nvar, ru_bits, F, Nt, kappa=kappa, gamma=gamma)
    Hhat = np.zeros_like(Hn)
    coeffs = n.copy()
    bits = 18*int(n.sum())+10*K*L                              # KLT coefficients + residual count field per link
    exp_cost = [[float(klt_cost(lam[k][l][:n[k, l]], nvar).sum()) if n[k, l] else 0. for l in range(L)] for k in range(K)]
    tau2 = res_thresh*2*math.log(N)*nvar                       # universal threshold (scaled)
    for l in range(L):
        cand, Ek, Dk, gs = [], E.copy(), D.copy(), []
        for k in range(K):
            y = Hn[k, l].reshape(-1)
            d = n[k, l]
            if d:
                c, xb = bfpq(bases[k][l][:, :d].conj().T@y)
                r = bases[k][l][:, :d]@c
                bits += xb
                left[l] += exp_cost[k][l]-(18*d+xb)             # actual KLT bits replace the allocator's estimate
            else:
                r = np.zeros_like(y)
            Hhat[k, l] = r.reshape(A, F)
            z = (y-r).reshape(A, F)
            g = (sp.conj().T@z)@fd.conj()
            gs.append(g)
            # the O-RU replaces the twin's prediction for its own link by what it measures
            tw_e = lam[k][l].sum() if lam[k][l] is not None else 0.
            tw_d = (lam[k][l][d:].sum() if lam[k][l] is not None else 0.)+d*nvar
            Ek[k] += max(np.linalg.norm(y)**2-N*nvar, 0.)-tw_e
            Dk[k] += max(np.linalg.norm(z)**2-(N-d)*nvar, 0.)+d*nvar-tw_d
            e = np.abs(g.reshape(-1))**2
            o = np.argsort(e)[::-1]
            cand.append((o[e[o] > tau2], e))
        ptr = np.zeros(K, int)
        sel = [[] for _ in range(K)]

        def rcost(k):
            return coef_bits(gs[k].reshape(-1)[cand[k][0][ptr[k]]])+idx_bits_next(N, len(sel[k]))

        def rgain(k):
            o, e = cand[k]
            if ptr[k] >= len(o) or rcost(k) > left[l]:
                return -np.inf
            g = se_proxy(Ek[k], Dk[k]-e[o[ptr[k]]]+nvar, K, F, Nt, _g(gamma, k))-se_proxy(Ek[k], Dk[k], K, F, Nt, _g(gamma, k))
            return g/rcost(k)                                   # gain per bit (costs differ with noise-matched BFP)
        Gr = np.array([rgain(k) for k in range(K)])
        while True:
            k = int(np.argmax(Gr))
            if not Gr[k] > 0:
                break
            j = cand[k][0][ptr[k]]
            left[l] -= rcost(k)
            sel[k].append(j)
            Dk[k] -= cand[k][1][j]-nvar
            ptr[k] += 1
            Gr = np.array([rgain(kk) for kk in range(K)]) if left[l] < 40 else Gr
            Gr[k] = rgain(k)
        for k in range(K):
            if sel[k]:
                m = np.zeros(N, complex)
                m[sel[k]], xb = bfpq(gs[k].reshape(-1)[sel[k]])
                Hhat[k, l] += sp@(m.reshape(A, F)@fd.T)
                coeffs[k, l] += len(sel[k])
                bits += 18*len(sel[k])+idx_bits_total(N, len(sel[k]))+xb
    return Hhat, coeffs, bits


def _uniform_gauss_table(bmax=14):
    """Optimal step (in units of the std) and normalized MSE of a 2^b-level midrise uniform
    quantizer for a Gaussian, b = 1..bmax (numerical; overload maps to the outer levels)."""
    x = np.linspace(-12, 12, 240001)
    w = np.exp(-x**2/2)
    w /= w.sum()
    steps, mses = [np.inf], [1.]
    for b in range(1, bmax+1):
        n = 2**b

        def mse(d):
            q = (np.clip(np.floor(x/d), -n//2, n//2-1)+.5)*d
            return float((w*(x-q)**2).sum())
        lo, hi = 1e-4, 4.
        for _ in range(80):                                     # golden-section search on the step
            m1, m2 = lo+.382*(hi-lo), lo+.618*(hi-lo)
            lo, hi = (lo, m2) if mse(m1) < mse(m2) else (m1, hi)
        steps.append((lo+hi)/2)
        mses.append(mse((lo+hi)/2))
    return np.array(steps), np.array(mses)


Q_STEP, Q_MSE = _uniform_gauss_table()


VQ_MEASURED, VQ_BMAX = False, 11  # urts_vq: twin-set widths with measured BFP scale (set by --vq-scale measured)


VQ_CAP = False                # urts_vq measured scale: also cap each block's width at its noise-matched width (--vq-cap)


def bfp_widths(c, widths):
    """Quantize coefficients (lambda order) with per-coefficient mantissa widths set by the twin: those of
    equal width form blocks of BFP_BLOCK sharing a power-of-two scale from their measured maximum (midrise
    levels spanning +-max). With VQ_CAP (and noise-matched BFP) a block whose measured maximum needs fewer
    bits than the twin gave is sent narrower (3-bit width field). Returns the quantized coefficients and the
    bits actually spent (mantissas, exponents, width fields)."""
    out = np.zeros_like(c)
    bits = 0
    for w0 in np.unique(widths):
        idx = np.flatnonzero(widths == w0)
        for b0 in range(0, len(idx), BFP_BLOCK):
            j = idx[b0:b0+BFP_BLOCK]
            x = c[j]
            mx = max(np.abs(x.real).max(), np.abs(x.imag).max(), 1e-30)
            w = min(w0, _bfp_width(mx)) if (VQ_CAP and BFP_KAPPA > 0) else w0
            step = 2.**math.ceil(math.log2(2*mx/2**w))
            half = 2**(w-1)

            def q(v):
                return (np.clip(np.floor(v/step), -half, half-1)+.5)*step
            out[j] = q(x.real)+1j*q(x.imag)
            bits += 2*w*len(j)+BFP_EXP_BITS+(3 if (VQ_CAP and BFP_KAPPA > 0) else 0)
    return out, bits


def link_scale(c, var):
    """Per-link gain of the twin-set quantizer (like a BFP block exponent): measured over predicted
    coefficient energy of the sent dimensions, quantized to 1.5 dB steps within +-48 dB (6 bits)."""
    r = np.sum(np.abs(c)**2)/max(np.sum(var), 1e-30)
    db = np.clip(np.round(10*np.log10(max(r, 1e-30))/1.5)*1.5, -48, 46.5)
    return 10**(db/20)


def wf_allocate_bits(lam, nvar, ru_bits, F, Nt, gamma=1., bmax=12):
    """Variable-precision version of wf_allocate: greedy over 1-bit-per-real-component increments of
    every (UE, link, KLT dimension). Dimension i with b bits per real component leaves error
    nvar + Q_MSE[b] (lam_i + nvar) (quantizing the noisy coefficient), lam_i if not sent (b = 0).
    Both ends know lam, so the bit pattern needs no signalling. Returns b[k][l] (int arrays)."""
    K, L = len(lam), len(lam[0])
    smax = max((len(x) for r in lam for x in r if x is not None), default=0)
    lp = np.zeros((K, L, smax))
    for k in range(K):
        for l in range(L):
            if lam[k][l] is not None:
                lp[k, l, :len(lam[k][l])] = lam[k][l]
    E = lp.sum((1, 2))
    D = E.copy()
    b = np.zeros((K, L, smax), int)
    left = np.full(L, float(ru_bits))

    def err(lv, bb):
        return np.where(bb == 0, lv, nvar+Q_MSE[np.minimum(bb, bmax)]*(lv+nvar))

    def gains(k):
        dd = err(lp[k], b[k])-err(lp[k], b[k]+1)                   # [L, S] error reduction of one more bit
        g = se_proxy(E[k], D[k]-dd, K, F, Nt, _g(gamma, k))-se_proxy(E[k], D[k], K, F, Nt, _g(gamma, k))
        ok = (lp[k] > 0) & (b[k] < bmax) & (left[:, None] >= 2)
        return np.where(ok, g, -np.inf)
    G = np.stack([gains(k) for k in range(K)])                  # [K, L, S]
    while True:
        k, l, i = np.unravel_index(int(np.argmax(G)), G.shape)
        if not G[k, l, i] > 0:
            break
        D[k] -= err(lp[k, l, i], b[k, l, i])-err(lp[k, l, i], b[k, l, i]+1)
        b[k, l, i] += 1
        left[l] -= 2
        G[k] = gains(k)
        if left[l] < 2:
            G[:, l] = -np.inf
    return b, E, D, left


def urts_vq(Hn, bases, lam, nvar, ru_bits, sp, fd, gamma=1., res_thresh=1., bmax=12, scale=False, measured=None):
    """URTS-CODE with variable-precision KLT coefficients (wf_allocate_bits) and the same residual
    layer. Coefficient i of link (k, l) is quantized per real component with b bits by the optimal
    uniform quantizer for a Gaussian of variance (lam_i + nvar)/2 (step known at both ends)."""
    K, L, A, F = Hn.shape
    N, Nt = A*F, L*A
    idx_bits = math.ceil(math.log2(N))
    if measured is None:
        measured = VQ_MEASURED
    if measured:
        bmax = min(bmax, VQ_BMAX)
    b, E, D, left = wf_allocate_bits(lam, nvar, ru_bits, F, Nt, gamma=gamma, bmax=bmax)
    Hhat = np.zeros_like(Hn)
    coeffs = (b > 0).sum(2)
    bits = 2*int(b.sum())+10*K*L
    tau2 = res_thresh*2*math.log(N)*nvar
    for l in range(L):
        cand, Ek, Dk, gs = [], E.copy(), D.copy(), []
        for k in range(K):
            y = Hn[k, l].reshape(-1)
            r = np.zeros_like(y)
            if bases[k][l] is not None:
                bb = b[k, l, :bases[k][l].shape[1]]
                use = np.flatnonzero(bb)
                if len(use):
                    Q = bases[k][l][:, use]
                    c = Q.conj().T@y
                    if measured:
                        cq, used_bits = bfp_widths(c, bb[use])
                        r = Q@cq
                        bits += used_bits-2*int(bb[use].sum())       # the allocator already billed 2 bits per increment
                        left[l] += 2*int(bb[use].sum())-used_bits    # bits saved by the cap go to the residual layer
                    else:
                        sd = np.sqrt((lam[k][l][use]+nvar)/2)*(link_scale(c, lam[k][l][use]+nvar) if scale else 1.)
                        bits += 6*scale
                        step = Q_STEP[bb[use]]*sd
                        half = 2.**(bb[use]-1)

                        def quant(v):
                            return (np.clip(np.floor(v/step), -half, half-1)+.5)*step
                        r = Q@(quant(c.real)+1j*quant(c.imag))
            Hhat[k, l] = r.reshape(A, F)
            z = (y-r).reshape(A, F)
            g = (sp.conj().T@z)@fd.conj()
            gs.append(g)
            # the O-RU replaces the twin's prediction for its own link by what it measures
            tw_e = lam[k][l].sum() if lam[k][l] is not None else 0.
            Ek[k] += max(np.linalg.norm(y)**2-N*nvar, 0.)-tw_e
            used = int((b[k, l] > 0).sum())
            tw_d = 0. if lam[k][l] is None else float(np.where(b[k, l, :len(lam[k][l])] == 0, lam[k][l],
                                                                nvar+Q_MSE[b[k, l, :len(lam[k][l])]]*(lam[k][l]+nvar)).sum())
            Dk[k] += max(np.linalg.norm(z)**2-(N-used)*nvar, 0.)+used*nvar-tw_d
            e = np.abs(g.reshape(-1))**2
            o = np.argsort(e)[::-1]
            cand.append((o[e[o] > tau2], e))
        ptr = np.zeros(K, int)
        sel = [[] for _ in range(K)]

        def rcost(k):
            return coef_bits(gs[k].reshape(-1)[cand[k][0][ptr[k]]])+idx_bits_next(N, len(sel[k]))

        def rgain(k):
            o, e = cand[k]
            if ptr[k] >= len(o) or rcost(k) > left[l]:
                return -np.inf
            g = se_proxy(Ek[k], Dk[k]-e[o[ptr[k]]]+nvar, K, F, Nt, _g(gamma, k))-se_proxy(Ek[k], Dk[k], K, F, Nt, _g(gamma, k))
            return g/rcost(k)                                   # gain per bit (costs differ with noise-matched BFP)
        Gr = np.array([rgain(k) for k in range(K)])
        while True:
            k = int(np.argmax(Gr))
            if not Gr[k] > 0:
                break
            j = cand[k][0][ptr[k]]
            left[l] -= rcost(k)
            sel[k].append(j)
            Dk[k] -= cand[k][1][j]-nvar
            ptr[k] += 1
            Gr = np.array([rgain(kk) for kk in range(K)]) if left[l] < 40 else Gr
            Gr[k] = rgain(k)
        for k in range(K):
            if sel[k]:
                m = np.zeros(N, complex)
                m[sel[k]], xb = bfpq(gs[k].reshape(-1)[sel[k]])
                Hhat[k, l] += sp@(m.reshape(A, F)@fd.T)
                coeffs[k, l] += len(sel[k])
                bits += 18*len(sel[k])+idx_bits_total(N, len(sel[k]))+xb
    return Hhat, coeffs, bits


def urts_jvq(Hn, bases, lam, nvar, ru_bits, sp, fd, gamma=1., res_thresh=1., bmax=12, lookahead=8, scale=False):
    """URTS-CODE, variable precision, joint split at the O-RU. For every link both ends know, from the
    twin eigenvalues alone, the order in which 1-bit-per-real-component increments would be added
    (largest error reduction first). O-RU l interleaves, by gain per bit on the SE proxy, the next
    increment of each of its links with the best angle-delay residual coefficient of that UE, using
    what it measures (coefficient energies, residual energies), and signals per link only how many
    increments it took (10 bits) plus the residual count (10 bits) and indices."""
    K, L, A, F = Hn.shape
    N, Nt = A*F, L*A
    idx_bits = math.ceil(math.log2(N))
    rcost = 18+idx_bits
    tau2 = res_thresh*2*math.log(N)*nvar
    # twin-only allocation: gives the other links' expected error as seen from each O-RU
    b_tw, E, _, _ = wf_allocate_bits(lam, nvar, ru_bits, F, Nt, gamma=gamma, bmax=bmax)
    smax = b_tw.shape[2]
    lp = np.zeros((K, L, smax))
    for k in range(K):
        for l in range(L):
            if lam[k][l] is not None:
                lp[k, l, :len(lam[k][l])] = lam[k][l]

    def err_tw(lv, bb):
        return np.where(bb == 0, lv, nvar+Q_MSE[np.minimum(bb, bmax)]*(lv+nvar))

    def link_order(lv):
        """Increment order of one link from its eigenvalues: repeatedly the dimension whose next bit
        removes the most error (deterministic, known at both ends)."""
        bb = np.zeros(len(lv), int)
        out = []
        while True:
            dd = np.where(bb < bmax, err_tw(lv, bb)-err_tw(lv, bb+1), -np.inf)
            i = int(np.argmax(dd))
            if not dd[i] > 0:
                return out
            out.append(i)
            bb[i] += 1
    Hhat = np.zeros_like(Hn)
    coeffs = np.zeros((K, L), int)
    bits = 20*K*L
    other = err_tw(lp, b_tw).sum(2)                              # [K, L] twin error of each link after its allocation
    for l in range(L):
        Ek = E.copy()
        Dl = other.sum(1)-other[:, l]
        seqs, cs, cand, gs, ys = [], [], [], [], []
        for k in range(K):
            y = Hn[k, l].reshape(-1)
            ys.append(y)
            own = max(np.linalg.norm(y)**2-N*nvar, 0.)
            Ek[k] += own-lp[k, l].sum()
            Dl[k] += own
            Q = bases[k][l]
            if Q is not None:
                c = Q.conj().T@y
                seqs.append(link_order(lam[k][l]))
            else:
                c = np.zeros(0, complex)
                seqs.append([])
            cs.append(c)
            cand.append(None)
        bu = [np.zeros(len(c), int) for c in cs]
        ptr, rptr = np.zeros(K, int), np.zeros(K, int)
        sel = [[] for _ in range(K)]

        def residual(k, sent_only):
            """Residual candidates of UE k above the universal threshold: w.r.t. the whole twin span
            (phase 1: cannot overlap any KLT dimension, so nothing is counted twice) or w.r.t. the KLT
            dimensions actually sent (phase 2: also picks up in-span energy that was not sent)."""
            Q = bases[k][l]
            if sent_only:
                use = np.flatnonzero(bu[k])
                z = ys[k]-(Q[:, use]@cs[k][use] if len(use) else 0.)
            else:
                z = ys[k]-(Q@cs[k] if Q is not None else 0.)
            e = np.abs(((sp.conj().T@z.reshape(A, F))@fd.conj()).reshape(-1))**2
            o = np.argsort(e)[::-1]
            cand[k] = (o[e[o] > tau2], e)
            rptr[k] = 0
        for k in range(K):
            residual(k, False)
        budget = float(ru_bits)
        W = lookahead
        cum = np.full((K, W), np.nan)                           # error removed by the next 1..W increments

        def refresh(k):
            """Cumulative error reduction of the next n = 1..W increments of UE k on this link (first bits
            of a dimension use its measured energy, later bits the quantizer's error reduction)."""
            cum[k] = np.nan
            dd, tmp = 0., {}
            for n, i in enumerate(seqs[k][ptr[k]:ptr[k]+W]):
                bi = tmp.get(i, bu[k][i])
                lv = lam[k][l][i]
                dd += (max(np.abs(cs[k][i])**2-nvar, 0.)-Q_MSE[1]*(lv+nvar) if bi == 0
                       else (Q_MSE[bi]-Q_MSE[min(bi+1, bmax)])*(lv+nvar))
                tmp[i] = bi+1
                cum[k, n] = dd
        for k in range(K):
            refresh(k)
        nn = 2*np.arange(1, W+1)
        while True:
            base = se_proxy(Ek, Dl, K, F, Nt, gamma)
            ok = ~np.isnan(cum) & (nn[None, :] <= budget)
            gk = np.where(ok, (se_proxy(Ek[:, None], Dl[:, None]-np.nan_to_num(cum), K, F, Nt, np.asarray(gamma)[:, None] if np.ndim(gamma) else gamma)-base[:, None])/nn[None, :], -np.inf)
            rd = np.array([cand[k][1][cand[k][0][rptr[k]]]-nvar if rptr[k] < len(cand[k][0]) else np.nan for k in range(K)])
            gr = np.where(~np.isnan(rd) & (budget >= rcost), (se_proxy(Ek, Dl-np.nan_to_num(rd), K, F, Nt, gamma)-base)/rcost, -np.inf)
            ka, na = np.unravel_index(int(np.argmax(gk)), gk.shape)
            kr = int(np.argmax(gr))
            if not max(gk[ka, na], gr[kr]) > 0:
                break
            if gk[ka, na] >= gr[kr]:
                k, n = ka, na+1
                for i in seqs[k][ptr[k]:ptr[k]+n]:
                    bu[k][i] += 1
                ptr[k] += n
                budget -= 2*n
                Dl[k] -= cum[k, n-1]
                refresh(k)
            else:
                k = kr
                sel[k].append(int(cand[k][0][rptr[k]]))
                rptr[k] += 1
                budget -= rcost
                Dl[k] -= rd[k]
        # phase 2: the bits reserved for residuals (plus leftovers) are re-spent on the residual w.r.t.
        # what is actually sent, by the same proxy gain, starting from the error after the KLT part
        budget += rcost*sum(len(x) for x in sel)
        for k in range(K):
            Dl[k] += sum(cand[k][1][j]-nvar for j in sel[k])
            sel[k] = []
            residual(k, True)
        while budget >= rcost:
            base = se_proxy(Ek, Dl, K, F, Nt, gamma)
            rd = np.array([cand[k][1][cand[k][0][rptr[k]]]-nvar if rptr[k] < len(cand[k][0]) else np.nan for k in range(K)])
            gr = np.where(~np.isnan(rd), se_proxy(Ek, Dl-np.nan_to_num(rd), K, F, Nt, gamma)-base, -np.inf)
            k = int(np.argmax(gr))
            if not gr[k] > 0:
                break
            sel[k].append(int(cand[k][0][rptr[k]]))
            rptr[k] += 1
            budget -= rcost
            Dl[k] -= rd[k]
        for k in range(K):
            r = np.zeros(N, complex)
            use = np.flatnonzero(bu[k])
            if len(use):
                bb = bu[k][use]
                c = cs[k][use]
                step = Q_STEP[bb]*np.sqrt((lam[k][l][use]+nvar)/2)*(link_scale(c, lam[k][l][use]+nvar) if scale else 1.)
                bits += 6*scale
                half = 2.**(bb-1)

                def quant(v):
                    return (np.clip(np.floor(v/step), -half, half-1)+.5)*step
                c = cs[k][use]
                r = bases[k][l][:, use]@(quant(c.real)+1j*quant(c.imag))
                coeffs[k, l] += len(use)
                bits += 2*int(bb.sum())
            if sel[k]:
                g = ((sp.conj().T@(ys[k]-r).reshape(A, F))@fd.conj()).reshape(-1)   # final residual values
                m = np.zeros(N, complex)
                m[sel[k]], xb = bfpq(g[sel[k]])
                r = r+(sp@(m.reshape(A, F)@fd.T)).reshape(-1)
                coeffs[k, l] += len(sel[k])
                bits += rcost*len(sel[k])+xb
            Hhat[k, l] = r.reshape(A, F)
    return Hhat, coeffs, bits


def link_subspaces(traced, n, K, L, freqs):
    """Per (UE, O-RU) KLT of twin channels; `traced` holds n points per UE, consecutive. Each CFR is
    referenced to that point's earliest arrival over all O-RUs (DL timing). Returns bases[k][l] (columns
    by decreasing eigenvalue) and lam[k][l] (average power per dimension); None where no point has a path."""
    bases, lams = [], []
    for k in range(K):
        samples = traced[k*n:(k+1)*n]
        per, per_lam = [], []
        for l in range(L):
            cols = []
            for smp in samples:
                taus = [x['tau'].min() for x in smp if len(x['tau'])]
                if not taus or not len(smp[l]['tau']):
                    continue
                cols.append(cfr(smp[l], freqs, min(taus)).reshape(-1))
            if cols:
                u, sv, _ = np.linalg.svd(np.stack(cols, 1), full_matrices=False)
                per.append(u)
                per_lam.append(sv**2/len(samples))
            else:
                per.append(None)
                per_lam.append(None)
        bases.append(per)
        lams.append(per_lam)
    return bases, lams


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--drops', type=int, default=10)
    p.add_argument('--num-ue', type=int, default=8)
    p.add_argument('--num-ru', type=int, default=4)
    p.add_argument('--seed', type=int, default=20269000)
    p.add_argument('--pos-errors', type=float, nargs='+', default=[0., 2.])
    p.add_argument('--reality', choices=['nominal', 'diffuse'], default='nominal')
    p.add_argument('--scattering', type=float, default=.3)
    p.add_argument('--real-geom', type=float, default=0., help='reality: rigid shift of every building, std per axis, m')
    p.add_argument('--real-material', type=float, default=0., help='reality: log-normal std of permittivity/conductivity factors')
    p.add_argument('--real-cars', type=int, default=0, help='reality: parked metal cars absent in the twin')
    p.add_argument('--twin-scattering', type=float, default=0., help='twin models diffuse scattering with this coefficient (0 = specular/diffraction only)')
    p.add_argument('--budgets', type=int, nargs='+', default=[4, 8, 16, 32, 64])
    p.add_argument('--ci-noise', type=float, default=1., help='SRS CI estimation-noise variance per element in units of the UE thermal noise (channels are scaled so that the latter is 1)')
    p.add_argument('--min-gain-db', type=float, default=-125., help='drop UEs whose best RU path gain is below this')
    p.add_argument('--region-samples', type=int, default=0, help='>0 enables the urts scheme')
    p.add_argument('--region-radius', type=float, default=5.)
    p.add_argument('--region-shape', choices=['disc', 'gauss'], default='disc')
    p.add_argument('--region-floor', type=float, default=1., help='gauss region: std floor in m (twin tolerance at sigma = 0)')
    p.add_argument('--wf-kappa', type=float, default=1., help='urts_code: send a KLT dimension only if its twin eigenvalue exceeds kappa x CI noise')
    p.add_argument('--bfp', type=int, default=0, help='>0: BFP-quantize every 18-bit coefficient with this mantissa width '
                   '(blocks of 12, 4-bit exponent counted); 0 = legacy unquantized accounting')
    p.add_argument('--bfp-kappa', type=float, default=0., help='>0 (with --bfp): noise-matched BFP, mantissa per block just '
                   'wide enough for a step <= kappa x CI noise std (3-bit width field per block)')
    p.add_argument('--vq-cap', action='store_true', help='urts_vq measured scale + noise-matched BFP: cap each block width at '
                   'its noise-matched width (3-bit field), saved bits go to the residual layer')
    p.add_argument('--vq-scale', choices=['twin', 'measured'], default='twin', help='urts_vq: quantizer scale from the twin variance, '
                   'or measured per block (BFP-like, 4-bit exponent; widths still from the twin, capped at 11 bits)')
    p.add_argument('--leak-gamma', type=float, default=2.7, help='urts_code: leakage factor of compression errors in the SE proxy (fitted on dumps)')
    p.add_argument('--res-thresh', type=float, default=1., help='urts_code: residual threshold as a multiple of the universal threshold 2 ln(N) x noise')
    p.add_argument('--idx-coding', choices=['fixed', 'comb'], default='fixed', help='angle-delay index sets: ceil(log2 N) bits '
                   'per index, or ceil(log2 C(N, n)) per link')
    p.add_argument('--precoder', choices=['rzf', 'rzf_perru'], default='rzf', help='rzf_perru: scale the RZF so that no O-RU '
                   'exceeds 1/L of the total power')
    p.add_argument('--grid-size', type=float, default=5., help='grid_ecb/grid_vq: cell size in m (Cai et al. use 5 m)')
    p.add_argument('--grid-samples', type=int, default=64, help='grid_ecb/grid_vq: twin points per cell, the 4 vertices first, '
                   'the rest uniform in the cell (4 = vertices only)')
    p.add_argument('--site-samples', type=int, default=4096, help='site_klt: twin points over the whole area per run')
    p.add_argument('--ae-steps', type=int, default=4000, help='twin_ae/twin_aep: training steps per O-RU model')
    p.add_argument('--ae-pos-noise', type=float, default=5., help='twin_aep: training position noise std ~ U(0, this), m')
    p.add_argument('--dump', default='', help='directory: save true/noisy/reconstructed CI per drop (diagnostics)')
    p.add_argument('--schemes', nargs='+', default=['ad_dft', 'twin_path', 'twin_res', 'twin_all'])
    p.add_argument('--fc', type=float, default=3.5e9)
    p.add_argument('--scs', type=float, default=30e3)
    p.add_argument('--num-prb', type=int, default=52)
    p.add_argument('--grp-prb', type=int, default=2)
    p.add_argument('--rows', type=int, default=4)
    p.add_argument('--cols', type=int, default=4)
    p.add_argument('--max-depth', type=int, default=3)
    p.add_argument('--samples', type=int, default=500_000)
    p.add_argument('--ru-power-dbm', type=float, default=30.)
    p.add_argument('--ue-nf-db', type=float, default=9.)
    p.add_argument('--ru-height', type=float, default=12.)
    p.add_argument('--area-lo', type=float, nargs=2, default=[-80., 35.])
    p.add_argument('--area-hi', type=float, nargs=2, default=[100., 230.])
    p.add_argument('--area-centre', type=float, nargs=2, default=[10., 130.])
    args = p.parse_args()
    args.groups = args.num_prb//args.grp_prb
    global BFP_MANT, BFP_KAPPA, BFP_NVAR, VQ_MEASURED, VQ_CAP, IDX_COMB, PER_RU_L
    BFP_MANT, BFP_KAPPA, BFP_NVAR = args.bfp, args.bfp_kappa, args.ci_noise
    VQ_MEASURED, VQ_CAP = args.vq_scale == 'measured', args.vq_cap
    IDX_COMB = args.idx_coding == 'comb'
    PER_RU_L = args.num_ru if args.precoder == 'rzf_perru' else 0
    rng = np.random.default_rng(args.seed)
    rec = Recorder(args.out)
    (rec.path/'experiment_source.py').write_bytes(Path(__file__).read_bytes())
    (rec.path/'dependency_explore_twin_ci_compression.py').write_bytes(Path(__file__).with_name('explore_twin_ci_compression.py').read_bytes())
    # RU poles: well-separated outdoor points (checked with a single-RU-free probe scene)
    probe = load_scene(sionna.rt.scene.munich, merge_shapes=True)
    import mitsuba as mi

    def outdoor(pts, z):
        pts = np.asarray(pts, dtype=np.float32)
        o = mi.Point3f(mi.Float(pts[:, 0]), mi.Float(pts[:, 1]), mi.Float(np.full(len(pts), z, np.float32)))
        return ~probe.mi_scene.ray_intersect(mi.Ray3f(o, mi.Vector3f(0, 0, 1))).is_valid().numpy()
    ru = []
    while len(ru) < args.num_ru:
        c = rng.uniform(args.area_lo, args.area_hi, size=(256, 2))
        ok = outdoor(c, args.ru_height) & outdoor(c, 1.5)
        for q in c[ok]:
            if all(np.linalg.norm(q-r_[:2]) > 60 for r_ in ru):
                ru.append(np.array([q[0], q[1], args.ru_height]))
            if len(ru) >= args.num_ru:
                break
    ru = np.array(ru)
    site = MultiSite(args, ru, rng)
    (rec.path/'metadata.json').write_text(json.dumps(dict(args=vars(args), ru_positions=ru.tolist(), car_positions=site.cars.tolist(),
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        assumptions=['L O-RUs on outdoor poles, 4x4 cross-pol UPA each, single-antenna UEs; Sionna RT munich.',
                     'Reality: true positions; optional diffuse scattering, building shifts, material perturbation, cars; twin: nominal scene at x+e.',
                     'CI noise: complex Gaussian, variance ci-noise per element in units of the UE thermal noise (weak links relatively noisier).',
                     'O-DU: centralized RZF per PRB group on reconstructed CI, total power normalization, no per-RU power limit.',
                     'Per-link coefficient budgets; bits = 18 per complex coefficient + side bits; no packetization overhead.',
                     'With --bfp > 0 those coefficients are block-floating-point quantized (blocks of 12 in send order, 4-bit exponent counted).']), indent=2))
    sp, fd = ad_bases(args.rows, args.cols, args.groups)
    n_ci = args.ci_noise
    site_basis, ae = None, {}
    if {'site_klt', 'twin_ae', 'twin_aep'} & set(args.schemes):
        # one KLT per O-RU over the whole area from the twin, unit-norm samples (shape only), own stream
        rs_ = np.random.default_rng([args.seed, 11])
        sp_pts = []
        while len(sp_pts) < args.site_samples:
            c = rs_.uniform(args.area_lo, args.area_hi, size=(256, 2))
            sp_pts.extend(c[outdoor(c, 1.5)])
        sp_pts = np.column_stack([np.array(sp_pts[:args.site_samples]), np.full(args.site_samples, 1.5)])
        traced = site.trace_twin(sp_pts)
        site_basis = []
        for l in range(args.num_ru):
            cols = []
            for smp in traced:
                taus = [x['tau'].min() for x in smp if len(x['tau'])]
                if taus and len(smp[l]['tau']):
                    h_ = cfr(smp[l], site.freqs, min(taus)).reshape(-1)
                    cols.append(h_/max(np.linalg.norm(h_), 1e-30))
            site_basis.append(np.linalg.svd(np.stack(cols, 1), full_matrices=False)[0])
        print(f'site_klt: {args.site_samples} twin points, per-RU samples with paths '
              f'{[sum(1 for s in traced if len(s[l]["tau"])) for l in range(args.num_ru)]}', flush=True)
        variants = [v for v in ('twin_ae', 'twin_aep') if v in args.schemes]
        if variants:
            from twin_ae import TwinAE
            log = {}
            for l in range(args.num_ru):
                Hs, ps = [], []
                for q_, smp in zip(sp_pts, traced):
                    taus = [x['tau'].min() for x in smp if len(x['tau'])]
                    if taus and len(smp[l]['tau']):
                        Hs.append(cfr(smp[l], site.freqs, min(taus)))
                        ps.append(q_)
                for v in variants:
                    t_ = time.time()
                    m_ = TwinAE(2*args.rows*args.cols, args.groups, args.ci_noise, v == 'twin_aep', site.centre, args.seed+100*l+(v == 'twin_aep'))
                    loss = m_.fit(np.array(Hs), np.array(ps), steps=args.ae_steps, pos_noise=args.ae_pos_noise, seed=args.seed+l)
                    ae[(v, l)] = m_
                    log[f'{v}_ru{l}'] = dict(samples=len(Hs), final_loss=loss, seconds=round(time.time()-t_, 1))
                    print(f'{v} ru{l}: {len(Hs)} samples, loss {loss:.4f}', flush=True)
            (rec.path/'ae_training.json').write_text(json.dumps(log, indent=1))
            (rec.path/'dependency_twin_ae.py').write_bytes(Path(__file__).with_name('twin_ae.py').read_bytes())
    for drop in range(args.drops):
        ues, truth = [], []
        while len(ues) < args.num_ue:
            c = rng.uniform(args.area_lo, args.area_hi, size=(32, 2))
            pts = np.column_stack([c, np.full(len(c), 1.5)])
            pts = pts[outdoor(pts[:, :2], 1.5) & site.clear_of_cars(pts[:, :2])]
            tr = site.trace_truth(pts)
            for q, t in zip(pts, tr):
                best = max((x['pw'].sum() for x in t if len(x['tau'])), default=0.)
                if best > 0 and 10*np.log10(best/site.amp**2) > args.min_gain_db and len(ues) < args.num_ue:
                    ues.append(q)
                    truth.append(t)
        ues = np.array(ues)
        L = args.num_ru
        # true CFR per UE/RU referenced to the UE's earliest arrival over all RUs (DL timing)
        H = np.zeros((args.num_ue, L, 2*args.rows*args.cols, args.groups), complex)
        t_ref = []
        for k in range(args.num_ue):
            taus = [x['tau'].min() for x in truth[k] if len(x['tau'])]
            t0 = min(taus) if taus else 0.
            t_ref.append(t0)
            for l in range(L):
                if len(truth[k][l]['tau']):
                    H[k, l] = cfr(truth[k][l], site.freqs, t0)
        Hn = H+(rng.normal(size=H.shape)+1j*rng.normal(size=H.shape))*math.sqrt(n_ci/2)
        Htrue = H.reshape(args.num_ue, -1, args.groups)
        se_ideal = rzf_se(Htrue, Htrue)
        se_noisy = rzf_se(Htrue, Hn.reshape(args.num_ue, -1, args.groups))
        noisy_link = 10*np.log10(np.linalg.norm(Hn-H, axis=(2, 3))**2/np.maximum(np.linalg.norm(H, axis=(2, 3))**2, 1e-30))
        dump = dict(H=H, Hn=Hn)
        for sigma in args.pos_errors:
            err = rng.normal(0, sigma, size=(args.num_ue, 2)) if sigma > 0 else np.zeros((args.num_ue, 2))
            est_pos = ues+np.column_stack([err, np.zeros(args.num_ue)])
            twins = site.trace_twin(est_pos)
            # twin-predicted link powers for allocation (no CI needed)
            tw_pow = np.array([[twins[k][l]['pw'].sum() if len(twins[k][l]['tau']) else 0. for l in range(L)]
                               for k in range(args.num_ue)])
            t_tw = [min((x['tau'].min() for x in twins[k] if len(x['tau'])), default=0.) for k in range(args.num_ue)]
            offs = [[t_tw[k]+offset_search(Hn[k, l], twins[k][l], site.freqs, t_tw[k])
                     if len(twins[k][l]['tau']) else 0. for l in range(L)] for k in range(args.num_ue)]
            klt = None
            if args.region_samples > 0:
                est = ues+np.column_stack([err, np.zeros(args.num_ue)])
                u_ = rng.uniform(size=(args.num_ue, args.region_samples))
                aa_ = rng.uniform(0, 2*np.pi, size=(args.num_ue, args.region_samples))
                if args.region_shape == 'disc':
                    rr_ = args.region_radius*np.sqrt(u_)
                else:                                          # Rayleigh radius: 2-D Gaussian posterior
                    rr_ = math.hypot(sigma, args.region_floor)*np.sqrt(-2*np.log1p(-u_))
                pts = (est[:, None, :]+np.stack([rr_*np.cos(aa_), rr_*np.sin(aa_), np.zeros_like(rr_)], -1)).reshape(-1, 3)
                klt, klt_lam = link_subspaces(site.trace_twin(pts), args.region_samples, args.num_ue, L, site.freqs)
            grid = None
            if klt is not None and {'grid_ecb', 'grid_vq'} & set(args.schemes):
                # Cai et al.: the fixed grid cell holding the estimated position, vertices first (own stream)
                rg_ = np.random.default_rng([args.seed, 7, drop, int(round(sigma*100))])
                G, ng = args.grid_size, args.grid_samples
                lo = np.floor(est[:, :2]/G)*G
                corner = np.broadcast_to(np.array([[0., 0.], [G, 0.], [0., G], [G, G]]), (args.num_ue, 4, 2))
                off_ = np.concatenate([corner, rg_.uniform(0, G, size=(args.num_ue, max(ng-4, 0), 2))], 1)[:, :ng]
                gp = lo[:, None, :]+off_
                gp = np.concatenate([gp, np.full(gp.shape[:2]+(1,), 1.5)], -1).reshape(-1, 3)
                grid = link_subspaces(site.trace_twin(gp), ng, args.num_ue, L, site.freqs)
            Nt = L*2*args.rows*args.cols
            E_true = np.linalg.norm(H.reshape(args.num_ue, -1), axis=1)**2
            px_ideal = se_proxy(E_true, 0., args.num_ue, args.groups, Nt).sum()
            px_noisy = se_proxy(E_true, np.linalg.norm((Hn-H).reshape(args.num_ue, -1), axis=1)**2,
                                args.num_ue, args.groups, Nt).sum()
            for budget in args.budgets:
                task = [x for x in TASK_SCHEMES if x in args.schemes] if klt is not None else []
                for alloc in (['uniform', 'twin'] if [x for x in args.schemes if x not in TASK_SCHEMES] or klt is not None
                              else [])+(['task'] if task else []):
                    if alloc == 'uniform':
                        kb = np.full((args.num_ue, L), budget)
                    else:
                        rel = tw_pow/np.maximum(tw_pow.max(1, keepdims=True), 1e-30)
                        w = np.where(10*np.log10(rel+1e-30) > -25, rel**.5, 0.)
                        kb = np.floor(budget*L*w/np.maximum(w.sum(1, keepdims=True), 1e-30)).astype(int)
                    for scheme in [x for x in args.schemes if x not in TASK_SCHEMES]+(['urts'] if klt is not None else [])+(
                            task if alloc == 'task' else []):
                        if scheme == 'twin_all' and (alloc != 'uniform' or budget != args.budgets[0]):
                            continue
                        if (alloc == 'task') != (scheme in TASK_SCHEMES):
                            continue
                        Hhat = np.zeros_like(H)
                        coeffs = bits = 0
                        if scheme == 'urts_code':
                            Hhat, cl, bits = urts_code(Hn, klt, klt_lam, n_ci, args.num_ue*budget*18, sp, fd, args.wf_kappa, args.leak_gamma, args.res_thresh)
                            coeffs = int(cl.sum())
                        if scheme in ('urts_vq', 'urts_jvq', 'grid_vq'):
                            fn = urts_jvq if scheme == 'urts_jvq' else urts_vq
                            bs_, lm_ = grid if scheme == 'grid_vq' else (klt, klt_lam)
                            Hhat, cl, bits = fn(Hn, bs_, lm_, n_ci, args.num_ue*budget*18, sp, fd, args.leak_gamma, args.res_thresh)
                            coeffs = int(cl.sum())
                        for k in range(args.num_ue if scheme not in TASK_SCHEMES else 0):
                            for l in range(L):
                                if scheme == 'urts':
                                    q = klt[k][l]
                                    kk = int(kb[k, l])
                                    if q is None or kk <= 0:
                                        r, used, side = np.zeros_like(Hn[k, l]), 0, 0
                                    else:
                                        q = q[:, :min(kk, q.shape[1])]
                                        c, xb = bfpq(q.conj().T@Hn[k, l].reshape(-1))
                                        r = (q@c).reshape(Hn[k, l].shape)
                                        used, side = q.shape[1], 8+xb
                                elif scheme in ('twin_ae', 'twin_aep'):
                                    r, used, tot = ae[(scheme, l)].code(Hn[k, l], int(kb[k, l]), est_pos[k])
                                    side = tot-18*used
                                elif scheme in ('grid_ecb', 'site_klt'):
                                    q = grid[0][k][l] if scheme == 'grid_ecb' else site_basis[l]
                                    kk = int(kb[k, l])
                                    if q is None or kk <= 0:
                                        r, used, side = np.zeros_like(Hn[k, l]), 0, 0
                                    else:               # both ends know the basis: no side information
                                        q = q[:, :min(kk, q.shape[1])]
                                        c, xb = bfpq(q.conj().T@Hn[k, l].reshape(-1))
                                        r = (q@c).reshape(Hn[k, l].shape)
                                        used, side = q.shape[1], xb
                                elif scheme == 'twin_all':
                                    nb = max(1, len(twins[k][l]['tau']))
                                    r, used, side = compress(Hn[k, l], twins[k][l], offs[k][l], site.freqs, sp, fd, 'twin_path', nb)
                                else:
                                    r, used, side = compress(Hn[k, l], twins[k][l], offs[k][l], site.freqs, sp, fd, scheme, int(kb[k, l]))
                                Hhat[k, l] = r
                                coeffs += used
                                bits += 18*used+side
                        se = rzf_se(Htrue, Hhat.reshape(args.num_ue, -1, args.groups))
                        nmse = float(10*np.log10(np.linalg.norm(Hhat-H)**2/np.linalg.norm(H)**2))
                        link = 10*np.log10(np.linalg.norm(Hhat-H, axis=(2, 3))**2/np.maximum(np.linalg.norm(H, axis=(2, 3))**2, 1e-30))
                        link_nmse_med = float(np.median(link))
                        px = se_proxy(E_true, np.linalg.norm((Hhat-H).reshape(args.num_ue, -1), axis=1)**2,
                                      args.num_ue, args.groups, Nt)
                        px_g = se_proxy(E_true, np.linalg.norm((Hhat-H).reshape(args.num_ue, -1), axis=1)**2,
                                        args.num_ue, args.groups, Nt, args.leak_gamma)
                        if args.dump:
                            dump[f'{sigma:g}_{budget}_{scheme}_{alloc}'] = Hhat
                        rec.add(dict(drop=drop, sigma=sigma, budget=budget, alloc=alloc, scheme=scheme,
                                     coeffs=int(coeffs), kbits_per_ru=float(bits/L/1e3), nmse_db=nmse,
                                     link_nmse_median_db=link_nmse_med,
                                     sum_se=float(se.sum()), min_se=float(se.min()),
                                     proxy_sum_se=float(px.sum()), proxy_sum_se_ideal=float(px_ideal),
                                     proxy_sum_se_noisy=float(px_noisy), proxy_gamma_sum_se=float(px_g.sum()),
                                     sum_se_ideal=float(se_ideal.sum()), sum_se_noisy=float(se_noisy.sum()),
                                     min_se_ideal=float(se_ideal.min()), min_se_noisy=float(se_noisy.min()),
                                     noisy_link_nmse_median_db=float(np.median(noisy_link)),
                                     raw_kbits_per_ru=float(args.num_ue*2*args.rows*args.cols*args.groups*18/1e3),
                                     se_ue=[round(float(x), 5) for x in se], se_ue_noisy=[round(float(x), 5) for x in se_noisy]))
            print(f'drop={drop} sigma={sigma} rows={len(rec.rows)}', flush=True)
            if args.dump and klt is not None:
                for k_ in range(args.num_ue):
                    for l_ in range(L):
                        if klt[k_][l_] is not None:
                            dump[f'{sigma:g}_klt_{k_}_{l_}'] = klt[k_][l_]
                            dump[f'{sigma:g}_lam_{k_}_{l_}'] = klt_lam[k_][l_]
        if args.dump:
            Path(args.dump).mkdir(parents=True, exist_ok=True)
            np.savez(Path(args.dump)/f'drop{drop}.npz', **dump)
    rec.finish()


if __name__ == '__main__':
    main()
