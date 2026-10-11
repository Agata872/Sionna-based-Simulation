#!/usr/bin/env python3
"""Paper 1, experiment E5b: how good can a twin-trained data-driven codec get? Offline study on the
site-wide twin samples of the K = 8 runs (seed 20269300 O-RU poles), so that the autoencoder baseline of
E5 is tuned before it is used in the cell-free comparison.

Step 1 (cached): trace --site-samples twin points over the area (own stream, same as site_klt in
explore_twin_cellfree.py but more points), store per-O-RU CFRs and positions in --cache.
Step 2: per O-RU (--rus), split 90/10 at random, and report held-out NMSE of the clean unit-norm shape
reconstructed from the noisy CI (CI noise nu = 1, the channels' own SNR) versus the latent size d, for:
  pca       site-wide KLT of the unit-norm training shapes, first d coefficients (linear AE optimum)
  ae        the E5 autoencoder (scripts/twin_ae.py), nested dropout, --steps-short and --steps-long
  ae_d      one autoencoder per d (no nested dropout), --steps-long
  pca_res   first d PCA coefficients + an MLP that corrects the PCA reconstruction (cannot be worse
            than pca in training; the strongest data-driven variant we try)
Quantization is not applied here (BFP9 is far below these error levels); bits are the same per d.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from torch import nn
from explore_csirs_precoding import Recorder

D_SET = [4, 8, 16, 32, 64, 128]


def build_cache(args, path):
    import sionna.rt
    import mitsuba as mi
    from sionna.rt import load_scene
    from explore_twin_ci_compression import cfr
    from explore_twin_cellfree import MultiSite
    rng = np.random.default_rng(args.seed)
    probe = load_scene(sionna.rt.scene.munich, merge_shapes=True)

    def outdoor(pts, z):
        pts = np.asarray(pts, dtype=np.float32)
        o = mi.Point3f(mi.Float(pts[:, 0]), mi.Float(pts[:, 1]), mi.Float(np.full(len(pts), z, np.float32)))
        return ~probe.mi_scene.ray_intersect(mi.Ray3f(o, mi.Vector3f(0, 0, 1))).is_valid().numpy()
    ru = []                                                   # identical placement to explore_twin_cellfree.py
    while len(ru) < args.num_ru:
        c = rng.uniform(args.area_lo, args.area_hi, size=(256, 2))
        ok = outdoor(c, args.ru_height) & outdoor(c, 1.5)
        for q in c[ok]:
            if all(np.linalg.norm(q-r_[:2]) > 60 for r_ in ru):
                ru.append(np.array([q[0], q[1], args.ru_height]))
            if len(ru) >= args.num_ru:
                break
    site = MultiSite(args, np.array(ru), rng)
    rs = np.random.default_rng([args.seed, 13])
    pts = []
    while len(pts) < args.site_samples:
        c = rs.uniform(args.area_lo, args.area_hi, size=(256, 2))
        pts.extend(c[outdoor(c, 1.5)])
    pts = np.column_stack([np.array(pts[:args.site_samples]), np.full(args.site_samples, 1.5)])
    traced = site.trace_twin(pts)
    out = {}
    for l in range(args.num_ru):
        H, P = [], []
        for q, smp in zip(pts, traced):
            taus = [x['tau'].min() for x in smp if len(x['tau'])]
            if taus and len(smp[l]['tau']):
                H.append(cfr(smp[l], site.freqs, min(taus)))
                P.append(q)
        out[f'H{l}'] = np.array(H, np.complex64)
        out[f'P{l}'] = np.array(P, np.float32)
    out['ru'] = np.array(ru)
    out['centre'] = site.centre
    np.savez_compressed(path, **out)


def to_real(h):
    return np.concatenate([h.real.reshape(len(h), -1), h.imag.reshape(len(h), -1)], 1).astype(np.float32)


def noisy_shapes(H, rng, nvar=1.):
    n = (rng.normal(size=H.shape)+1j*rng.normal(size=H.shape))*math.sqrt(nvar/2)
    y = H+n
    x_in = y/np.linalg.norm(y.reshape(len(y), -1), axis=1)[:, None, None]
    x_t = H/np.linalg.norm(H.reshape(len(H), -1), axis=1)[:, None, None]
    return to_real(x_in), to_real(x_t)


def nmse_stats(out, tgt):
    e = ((out-tgt)**2).sum(1)/(tgt**2).sum(1)
    return float(10*np.log10(e.mean())), float(np.median(10*np.log10(np.maximum(e, 1e-12))))


class MLP(nn.Module):
    def __init__(self, n_in, n_out, width, depth=2):
        super().__init__()
        layers, w = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(w, width), nn.LeakyReLU(.1)]
            w = width
        layers.append(nn.Linear(w, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_eval(kind, Xtr, Ttr, Xte, Tte, d_list, steps, width, dev, seed, U=None):
    """Returns {d: (mean NMSE dB, median NMSE dB)} on the test set."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = Xtr.shape[1]
    dmax = max(d_list)
    if kind == 'pca_res':
        half = U.shape[1]//2                                  # columns: Re-coefficient block, then Im block
        Ut = torch.as_tensor(U[:, np.concatenate([np.arange(dmax), half+np.arange(dmax)])], device=dev)
        dec = MLP(2*dmax, n, width).to(dev)
        nn.init.zeros_(dec.net[-1].weight)                    # start exactly at the PCA reconstruction
        nn.init.zeros_(dec.net[-1].bias)
        params = list(dec.parameters())
    else:
        enc, dec = MLP(n, 2*dmax, width).to(dev), MLP(2*dmax, n, width).to(dev)
        params = list(enc.parameters())+list(dec.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    Xt, Tt = torch.as_tensor(Xtr, device=dev), torch.as_tensor(Ttr, device=dev)

    def code(x, d):
        mask = torch.zeros(1, 2*dmax, device=dev)
        mask[:, :d] = 1.
        mask[:, dmax:dmax+d] = 1.
        if kind == 'pca_res':
            zr = (x@Ut)*mask                                    # first d complex PCA coefficients (Re, Im)
            base = zr@Ut.T
            return base+dec(zr)
        return dec(enc(x)*mask)
    for it in range(steps):
        i = torch.as_tensor(rng.integers(len(Xtr), size=512), device=dev)
        d = int(rng.choice(d_list))
        loss = ((code(Xt[i], d)-Tt[i])**2).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
    out = {}
    with torch.no_grad():
        Xe = torch.as_tensor(Xte, device=dev)
        for d in d_list:
            out[d] = nmse_stats(code(Xe, d).cpu().numpy(), Tte)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--seed', type=int, default=20269300)
    p.add_argument('--num-ru', type=int, default=4)
    p.add_argument('--site-samples', type=int, default=24576)
    p.add_argument('--rus', type=int, nargs='+', default=[0, 1])
    p.add_argument('--steps-short', type=int, default=4000)
    p.add_argument('--steps-long', type=int, default=16000)
    p.add_argument('--width', type=int, default=1024)
    for k, v in dict(fc=3.5e9, scs=30e3, ru_power_dbm=30., ue_nf_db=9., ru_height=12.).items():
        p.add_argument('--'+k.replace('_', '-'), type=float, default=v)
    p.add_argument('--num-prb', type=int, default=52)
    p.add_argument('--grp-prb', type=int, default=2)
    p.add_argument('--rows', type=int, default=4)
    p.add_argument('--cols', type=int, default=4)
    p.add_argument('--max-depth', type=int, default=3)
    p.add_argument('--samples', type=int, default=500_000)
    p.add_argument('--area-lo', type=float, nargs=2, default=[-80., 35.])
    p.add_argument('--area-hi', type=float, nargs=2, default=[100., 230.])
    p.add_argument('--area-centre', type=float, nargs=2, default=[10., 130.])
    args = p.parse_args()
    args.groups = args.num_prb//args.grp_prb
    args.reality, args.scattering, args.twin_scattering = 'nominal', 0., 0.
    cache = Path(args.cache)
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        build_cache(args, cache)
        print(f'cache built in {time.time()-t0:.0f} s', flush=True)
    data = np.load(cache)
    rec = Recorder(args.out)
    (rec.path/'experiment_source.py').write_bytes(Path(__file__).read_bytes())
    (rec.path/'metadata.json').write_text(json.dumps(dict(args=vars(args), cache=str(cache),
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), indent=2))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    for l in args.rus:
        H = data[f'H{l}']
        rng = np.random.default_rng([args.seed, 17, l])
        perm = rng.permutation(len(H))
        nte = len(H)//10
        te, tr = perm[:nte], perm[nte:]
        Xtr, Ttr = noisy_shapes(H[tr], rng)
        Xte, Tte = noisy_shapes(H[te], rng)
        # PCA in the real embedding of complex shapes: complex KLT of the clean training shapes, written as
        # a real 2N x 2N orthonormal basis [Re; Im] so that the first d complex coefficients are kept
        S = H[tr]/np.linalg.norm(H[tr].reshape(len(tr), -1), axis=1)[:, None, None]
        Uc = np.linalg.svd(S.reshape(len(S), -1).T, full_matrices=False)[0]       # [N, r] complex
        N = Uc.shape[0]
        Ur = np.block([[Uc.real, -Uc.imag], [Uc.imag, Uc.real]]).astype(np.float32)   # [2N, 2r]
        r = Uc.shape[1]
        cols = np.concatenate([np.arange(128), r+np.arange(128)])                      # first 128 complex dims
        U = Ur[:, cols]
        for d in D_SET:
            keep = np.concatenate([np.arange(d), 128+np.arange(d)])
            Ud = U[:, keep]
            rec.add(dict(ru=l, model='pca', d=d, steps=0, n_train=len(tr), n_test=nte,
                         nmse_mean_db=nmse_stats(Xte@Ud@Ud.T, Tte)[0], nmse_median_db=nmse_stats(Xte@Ud@Ud.T, Tte)[1]))
        configs = [('ae', D_SET, args.steps_short), ('ae', D_SET, args.steps_long), ('pca_res', D_SET, args.steps_long)]
        configs += [('ae_d', [d], args.steps_long) for d in (32, 128)]
        for kind, dl, steps in configs:
            t0 = time.time()
            res = train_eval('pca_res' if kind == 'pca_res' else 'ae', Xtr, Ttr, Xte, Tte, dl, steps, args.width, dev,
                             args.seed+l, U=U)
            for d, (m_, md_) in res.items():
                rec.add(dict(ru=l, model=kind, d=d, steps=steps, n_train=len(tr), n_test=nte, nmse_mean_db=m_,
                             nmse_median_db=md_, seconds=round(time.time()-t0, 1)))
            print(f'ru={l} {kind} steps={steps} done in {time.time()-t0:.0f} s: '
                  + ', '.join(f'd={d}: {v[0]:.1f} dB' for d, v in res.items()), flush=True)
    rec.finish()


if __name__ == '__main__':
    main()
