#!/usr/bin/env python3
"""Paper 1, experiment E6 (theory T2): effective dimension of the region twin subspace versus the
position uncertainty s, the array aperture and the bandwidth.

Same site as explore_twin_cellfree.py (Sionna RT munich, O-RU poles from the same seed, twin = nominal
scene). For --centres outdoor UE positions and each region std s (2-D Gaussian, no floor), the twin is
ray-traced at --region-samples points and, per O-RU link, the region KLT is computed on the CFR at each
bandwidth (same per-RE SNR: the 52-PRB amplitude scaling is kept, only the subcarrier grid changes).
Recorded per (array, bandwidth, centre, s, link): eigenvalue spectrum (top 192, npz), the number of
dimensions above the CI noise (lambda > nu), the dimensions for 99 % / 99.9 % of the region energy, and
the twin paths at the centre (power, length, departure angles) for the first-order prediction.
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
import mitsuba as mi
from sionna.rt import load_scene
from explore_csirs_precoding import Recorder
from explore_twin_ci_compression import cfr
from explore_twin_cellfree import MultiSite


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=20269300, help='same O-RU poles as the K = 8 main runs')
    p.add_argument('--num-ru', type=int, default=4)
    p.add_argument('--centres', type=int, default=24)
    p.add_argument('--stds', type=float, nargs='+', default=[0.25, 0.5, 1., 2., 3., 5., 8.])
    p.add_argument('--region-samples', type=int, default=192)
    p.add_argument('--arrays', nargs='+', default=['4x4', '8x8'], help='rows x cols per polarization')
    p.add_argument('--prbs', type=int, nargs='+', default=[52, 104], help='bandwidth variants (PRBs at --scs, groups of --grp-prb)')
    p.add_argument('--ci-noise', type=float, default=1.)
    p.add_argument('--min-gain-db', type=float, default=-115.)
    p.add_argument('--fc', type=float, default=3.5e9)
    p.add_argument('--scs', type=float, default=30e3)
    p.add_argument('--num-prb', type=int, default=52, help='reference bandwidth for the amplitude (per-RE SNR) scaling')
    p.add_argument('--grp-prb', type=int, default=2)
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
    args.reality, args.scattering, args.twin_scattering = 'nominal', 0., 0.
    rng = np.random.default_rng(args.seed)
    rec = Recorder(args.out)
    here = Path(__file__)
    (rec.path/'experiment_source.py').write_bytes(here.read_bytes())
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
    ru = np.array(ru)
    (rec.path/'metadata.json').write_text(json.dumps(dict(args=vars(args), ru_positions=ru.tolist(),
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), script_sha256=hashlib.sha256(here.read_bytes()).hexdigest(),
        assumptions=['Twin only (nominal munich); region = 2-D Gaussian of std s around the centre, no floor.',
                     'Per link: KLT of the region CFRs, CFR referenced to each sample\'s earliest arrival over all O-RUs.',
                     'Bandwidth variants keep the per-RE SNR of the --num-prb reference (only the subcarrier grid changes).']), indent=2))
    spectra = {}
    crng = np.random.default_rng([args.seed, 31])
    centres = None
    for arr in args.arrays:
        rows_, cols_ = (int(x) for x in arr.split('x'))
        args.rows, args.cols = rows_, cols_
        site = MultiSite(args, ru, rng)
        if centres is None:                                   # same centres for every array (chosen with the first)
            centres = []
            while len(centres) < args.centres:
                c = crng.uniform(args.area_lo, args.area_hi, size=(32, 2))
                pts = np.column_stack([c, np.full(len(c), 1.5)])
                pts = pts[outdoor(pts[:, :2], 1.5)]
                for q, t in zip(pts, site.trace_twin(pts)):
                    best = max((x['pw'].sum() for x in t if len(x['tau'])), default=0.)
                    if best > 0 and 10*np.log10(best/site.amp**2) > args.min_gain_db and len(centres) < args.centres:
                        centres.append(q)
            centres = np.array(centres)
        freqs = {P: (np.arange(P//args.grp_prb)*args.grp_prb*12+args.grp_prb*6-P*6)*args.scs for P in args.prbs}
        centre_paths = site.trace_twin(centres)
        for ci, x0 in enumerate(centres):
            for s in args.stds:
                srng = np.random.default_rng([args.seed, 37, ci, int(round(s*100))])
                u = srng.uniform(size=args.region_samples)
                a_ = srng.uniform(0, 2*np.pi, size=args.region_samples)
                r_ = s*np.sqrt(-2*np.log1p(-u))
                pts = x0[None]+np.stack([r_*np.cos(a_), r_*np.sin(a_), np.zeros_like(r_)], -1)
                st = site.trace_twin(pts)
                for l in range(args.num_ru):
                    cp = centre_paths[ci][l]
                    for P, fr in freqs.items():
                        cols = []
                        for smp in st:
                            taus = [x['tau'].min() for x in smp if len(x['tau'])]
                            if taus and len(smp[l]['tau']):
                                cols.append(cfr(smp[l], fr, min(taus)).reshape(-1))
                        if not cols:
                            continue
                        lam = np.linalg.svd(np.stack(cols, 1), compute_uv=False)**2/args.region_samples
                        cum = np.cumsum(lam)/lam.sum()
                        key = f'{arr}_{P}_{ci}_{s:g}_{l}'
                        spectra[key] = lam[:192].astype(np.float32)
                        rec.add(dict(array=arr, prb=P, centre=ci, std=s, link=l, samples_with_paths=len(cols),
                                     energy=float(lam.sum()), centre_energy=float(np.linalg.norm(cfr(cp, fr, cp['tau'].min()))**2) if len(cp['tau']) else 0.,
                                     d_noise=int((lam > args.ci_noise).sum()), d99=int(np.searchsorted(cum, .99)+1),
                                     d999=int(np.searchsorted(cum, .999)+1), n_elements=2*rows_*cols_, n_freq=len(fr),
                                     paths_pw=[float(v) for v in cp['pw'][:64]], paths_len=[float(v)*299792458. for v in cp['tau'][:64]],
                                     paths_th=[float(v) for v in cp['th'][:64]], paths_ph=[float(v) for v in cp['ph'][:64]]))
            print(f'array={arr} centre={ci} rows={len(rec.rows)}', flush=True)
    np.savez_compressed(rec.path/'spectra.npz', **spectra)
    rec.finish()


if __name__ == '__main__':
    main()
