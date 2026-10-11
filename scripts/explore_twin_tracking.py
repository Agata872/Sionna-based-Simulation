#!/usr/bin/env python3
"""Stage-5: where does the UE position come from? Closed-loop O-DU localization on the twin
(CKM) with the region particles of URTS-CODE, feeding the position posterior back to the O-RUs.

Independent exploration script (oran/ untouched). Same cell-free site as
explore_twin_cellfree.py (Sionna RT munich, L O-RUs on poles, K single-antenna UEs, centralized
RZF). UEs are static or move at --speed with a fixed heading between SRS occasions (the O-DU
widens its prediction by --motion-model-sd); every SRS occasion has fresh CI noise.

Occasion t:
  both ends   particles ~ N(c_t, s_t^2 I) (shared seed), twin CI per particle and O-RU,
              per-link KLT (as urts_code in explore_twin_cellfree.py)
  O-RU        URTS-CODE on its noisy CI
  O-DU        reconstruct, centralized RZF; then score every particle by the energy of the
              reconstructed CI captured by that particle's twin path atoms, summed over O-RUs
              (a free complex gain per path, so the score ignores the sub-wavelength phase
              pattern); tempered importance weights w ~ exp((score - max)/(temp x energy));
              c_{t+1} = weighted mean, sigma_{t+1}^2 = weighted variance + particle resolution
              (s_t^2 2 pi / S); s_{t+1}^2 = sigma_{t+1}^2 + floor^2 ('argmax' mode: best particle).
              c_{t+1} goes to the O-RUs (2 x 16 bits per UE).
Occasion 1 starts from an external coarse prior c_1 = x + e0, e0 ~ N(0, sigma0^2 I).
Reference per drop: 'oracle' (particles around the true position, floor only).
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
from sionna.rt import load_scene
from explore_csirs_precoding import Recorder
from explore_twin_ci_compression import cfr, project
from explore_twin_cellfree import MultiSite, ad_bases, rzf_se, twin_path_atoms, urts_code, urts_vq
import explore_twin_cellfree as _cf


def region(site, centers, s_eff, S, rng, L):
    """Particles ~ N(centre, s_eff^2 I) per UE (2-D); twin traces; per-link KLT and eigenvalues."""
    K = len(centers)
    u = rng.uniform(size=(K, S))
    ang = rng.uniform(0, 2*np.pi, size=(K, S))
    r = s_eff[:, None]*np.sqrt(-2*np.log1p(-u))
    pts = centers[:, None, :]+np.stack([r*np.cos(ang), r*np.sin(ang), np.zeros_like(r)], -1)
    st = site.trace_twin(pts.reshape(-1, 3))
    st = [st[k*S:(k+1)*S] for k in range(K)]
    bases, lams = [], []
    for k in range(K):
        per_b, per_l = [], []
        for l in range(L):
            cols = []
            for x in st[k]:
                taus = [y['tau'].min() for y in x if len(y['tau'])]
                if taus and len(x[l]['tau']):
                    cols.append(cfr(x[l], site.freqs, min(taus)).reshape(-1))
            if cols:
                u_, sv, _ = np.linalg.svd(np.stack(cols, 1), full_matrices=False)
                per_b.append(u_)
                per_l.append(sv**2/S)
            else:
                per_b.append(None)
                per_l.append(None)
        bases.append(per_b)
        lams.append(per_l)
    return pts, st, bases, lams


def particle_scores(Hk, st_k, freqs, n_paths, per_link=False):
    """Energy of the CI of one UE (Hk [L, A, F]) captured by each particle's twin path atoms
    (per_link: also the per-O-RU captured energy and the twin-predicted per-O-RU energy)."""
    L = Hk.shape[0]
    cap, pred = np.zeros((len(st_k), L)), np.zeros((len(st_k), L))
    for s, x in enumerate(st_k):
        taus = [y['tau'].min() for y in x if len(y['tau'])]
        if not taus:
            continue
        t0 = min(taus)
        for l, tw in enumerate(x):
            if len(tw['tau']):
                r, _ = project(Hk[l].reshape(-1), twin_path_atoms(tw, freqs, t0, n_paths))
                cap[s, l] = np.linalg.norm(r)**2
                pred[s, l] = np.linalg.norm(cfr(tw, freqs, t0))**2
    return (cap.sum(1), cap, pred) if per_link else cap.sum(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    p.add_argument('--drops', type=int, default=8)
    p.add_argument('--num-ue', type=int, default=8)
    p.add_argument('--num-ru', type=int, default=4)
    p.add_argument('--seed', type=int, default=20269500)
    p.add_argument('--reality', choices=['nominal', 'diffuse'], default='nominal')
    p.add_argument('--scattering', type=float, default=.3)
    p.add_argument('--real-geom', type=float, default=0., help='reality: rigid shift of every building, std per axis, m')
    p.add_argument('--real-material', type=float, default=0., help='reality: log-normal std of permittivity/conductivity factors')
    p.add_argument('--real-cars', type=int, default=0, help='reality: parked metal cars absent in the twin')
    p.add_argument('--twin-scattering', type=float, default=0., help='twin models diffuse scattering with this coefficient (0 = specular/diffraction only)')
    p.add_argument('--sigma0', type=float, nargs='+', default=[5., 10.], help='external coarse prior std per axis, m')
    p.add_argument('--occasions', type=int, default=3)
    p.add_argument('--budgets', type=int, nargs='+', default=[4, 8, 16, 32, 64])
    p.add_argument('--budget-loc', type=int, default=32, help='per-link coefficient budget whose reconstruction the O-DU localizes on')
    p.add_argument('--loc-paths', type=int, default=16, help='twin paths per particle and O-RU in the score')
    p.add_argument('--loc-mode', choices=['argmax', 'posterior'], default='posterior')
    p.add_argument('--loc-temp', type=float, default=1e-3, help='posterior: weights exp((score - max)/(temp x CI energy)); temp ~ twin mismatch level')
    p.add_argument('--dump-scores', action='store_true', help='record particle distances and normalized scores (diagnostics)')
    p.add_argument('--speed', type=float, default=0., help='UE speed, m/s (fixed random heading per UE)')
    p.add_argument('--srs-period', type=float, default=.02, help='time between SRS occasions, s')
    p.add_argument('--motion-model-sd', type=float, default=-1., help='O-DU motion model: extra posterior std per occasion, m '
                   '(-1 = speed x period)')
    p.add_argument('--acq-samples', type=int, default=0, help='particles in the acquisition occasion (t = 1); 0 = --region-samples')
    p.add_argument('--loc-rss-db', type=float, default=0., help='>0: add an RSS fingerprint term (per-O-RU measured vs twin energy, '
                   'common offset removed) with this std in dB, in the acquisition occasion')
    p.add_argument('--codec', choices=['urts_code', 'urts_vq'], default='urts_code', help='CI codec at every occasion '
                   '(urts_vq with the quantizer options below is the final method of explore_twin_cellfree.py)')
    p.add_argument('--bfp', type=int, default=0, help='as explore_twin_cellfree.py (0 = unquantized 18-bit accounting)')
    p.add_argument('--bfp-kappa', type=float, default=0.)
    p.add_argument('--vq-scale', choices=['twin', 'measured'], default='twin')
    p.add_argument('--vq-cap', action='store_true')
    p.add_argument('--idx-coding', choices=['fixed', 'comb'], default='fixed')
    p.add_argument('--region-samples', type=int, default=64)
    p.add_argument('--region-floor', type=float, default=1.)
    p.add_argument('--ci-noise', type=float, default=1.)
    p.add_argument('--min-gain-db', type=float, default=-115.)
    p.add_argument('--leak-gamma', type=float, default=2.7)
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
    _cf.BFP_MANT, _cf.BFP_KAPPA, _cf.BFP_NVAR = args.bfp, args.bfp_kappa, args.ci_noise
    _cf.VQ_MEASURED, _cf.VQ_CAP, _cf.IDX_COMB = args.vq_scale == 'measured', args.vq_cap, args.idx_coding == 'comb'
    rng = np.random.default_rng(args.seed)
    rec = Recorder(args.out)
    here = Path(__file__)
    (rec.path/'experiment_source.py').write_bytes(here.read_bytes())
    for dep in ['explore_twin_cellfree.py', 'explore_twin_ci_compression.py']:
        (rec.path/f'dependency_{dep}').write_bytes(here.with_name(dep).read_bytes())
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
        script_sha256=hashlib.sha256(here.read_bytes()).hexdigest(),
        assumptions=['Same site and channel model as explore_twin_cellfree.py; UEs static or moving at --speed between SRS occasions, fresh CI noise per occasion.',
                     'Particles drawn from N(c_t, s_t^2 I) with a seed both ends share; twin = nominal scene.',
                     'O-DU localizes on the URTS-CODE reconstruction at --budget-loc (tempered particle posterior or argmax); no external positioning after occasion 1.',
                     'Position feedback to the O-RUs: 2 x 16 bits per UE per occasion (not counted in the CI bits).']), indent=2))
    sp, fd = ad_bases(args.rows, args.cols, args.groups)
    L, K, S = args.num_ru, args.num_ue, args.region_samples
    for drop in range(args.drops):
        ues, truth = [], []
        while len(ues) < K:
            c = rng.uniform(args.area_lo, args.area_hi, size=(32, 2))
            pts = np.column_stack([c, np.full(len(c), 1.5)])
            pts = pts[outdoor(pts[:, :2], 1.5) & site.clear_of_cars(pts[:, :2])]
            tr = site.trace_truth(pts)
            for q, t in zip(pts, tr):
                best = max((x['pw'].sum() for x in t if len(x['tau'])), default=0.)
                if best > 0 and 10*np.log10(best/site.amp**2) > args.min_gain_db and len(ues) < K:
                    ues.append(q)
                    truth.append(t)
        ues = np.array(ues)

        def truth_state(pos, tr):
            """True CFR (UE-earliest-arrival reference) and ideal RZF SE at the given true positions."""
            H_ = np.zeros((K, L, 2*args.rows*args.cols, args.groups), complex)
            for k in range(K):
                taus = [x['tau'].min() for x in tr[k] if len(x['tau'])]
                t0 = min(taus) if taus else 0.
                for l in range(L):
                    if len(tr[k][l]['tau']):
                        H_[k, l] = cfr(tr[k][l], site.freqs, t0)
            Ht_ = H_.reshape(K, -1, args.groups)
            return dict(pos=pos, H=H_, Htrue=Ht_, se_ideal=rzf_se(Ht_, Ht_))

        def move(state, heading):
            """Advance every UE by speed x SRS period along its heading (stays put if that would leave the street)."""
            step = args.speed*args.srs_period
            if step <= 0:
                return state
            cand = state['pos']+step*np.column_stack([np.cos(heading), np.sin(heading), np.zeros(K)])
            ok = outdoor(cand[:, :2], 1.5) & site.clear_of_cars(cand[:, :2])
            pos = np.where(ok[:, None], cand, state['pos'])
            return truth_state(pos, site.trace_truth(pos))

        def occasion(state, label, sigma0, t, centres, s_eff):
            H, Htrue, ues, se_ideal = state['H'], state['Htrue'], state['pos'], state['se_ideal']
            Hn = H+(rng.normal(size=H.shape)+1j*rng.normal(size=H.shape))*math.sqrt(args.ci_noise/2)
            se_noisy = rzf_se(Htrue, Hn.reshape(K, -1, args.groups))
            n_part = args.acq_samples if (t == 1 and args.acq_samples) else S
            pts, st, bases, lams = region(site, centres, s_eff, n_part, rng, L)
            centre_err = np.linalg.norm((centres-ues)[:, :2], axis=1)
            Hloc = None
            for budget in args.budgets:
                codec = urts_vq if args.codec == 'urts_vq' else urts_code
                Hhat, cl, bits = codec(Hn, bases, lams, args.ci_noise, K*budget*18, sp, fd, gamma=args.leak_gamma)
                se = rzf_se(Htrue, Hhat.reshape(K, -1, args.groups))
                if budget == args.budget_loc:
                    Hloc = Hhat
                rec.add(dict(drop=drop, setting=label, sigma0=sigma0, occasion=t, budget=budget,
                             kbits_per_ru=float(bits/L/1e3), coeffs=int(cl.sum()),
                             nmse_db=float(10*np.log10(np.linalg.norm(Hhat-H)**2/np.linalg.norm(H)**2)),
                             sum_se=float(se.sum()), min_se=float(se.min()),
                             sum_se_ideal=float(se_ideal.sum()), sum_se_noisy=float(se_noisy.sum()),
                             min_se_noisy=float(se_noisy.min()),
                             centre_err_median=float(np.median(centre_err)), centre_err_mean=float(centre_err.mean()),
                             centre_err_max=float(centre_err.max()), s_eff_mean=float(s_eff.mean())))
            if Hloc is None:
                return None
            # O-DU localization on the reconstruction (and, as a diagnostic, on the uncompressed noisy CI)
            new, post_sd, err_rec, err_arg, err_raw, dumps = np.zeros_like(centres), np.zeros(K), [], [], [], []
            for k in range(K):
                sc, cap, pred = particle_scores(Hloc[k], st[k], site.freqs, args.loc_paths, per_link=True)
                sr = particle_scores(Hn[k], st[k], site.freqs, args.loc_paths)
                e = max(np.linalg.norm(Hloc[k])**2, 1e-30)
                logw = (sc-sc.max())/(args.loc_temp*e)
                if args.loc_rss_db > 0 and t == 1:
                    floor = Hloc.shape[2]*Hloc.shape[3]*args.ci_noise
                    meas = np.linalg.norm(Hloc[k], axis=(1, 2))**2
                    diff = 10*np.log10(np.maximum(meas-floor, floor))[None, :]-10*np.log10(np.maximum(pred, floor))
                    diff -= diff.mean(1, keepdims=True)                 # unknown common gain offset
                    logw -= (diff**2).sum(1)/(2*args.loc_rss_db**2)
                w = np.exp(logw-logw.max())
                w /= w.sum()
                mean = (w[:, None]*pts[k]).sum(0)
                var = (w*np.sum((pts[k, :, :2]-mean[:2])**2, 1)).sum()/2
                arg = pts[k, int(np.argmax(logw))]
                new[k] = arg if args.loc_mode == 'argmax' else mean
                post_sd[k] = math.sqrt(var)
                err_rec.append(float(np.linalg.norm(new[k, :2]-ues[k, :2])))
                err_arg.append(float(np.linalg.norm(arg[:2]-ues[k, :2])))
                err_raw.append(float(np.linalg.norm(pts[k, int(np.argmax(sr)), :2]-ues[k, :2])))
                if args.dump_scores:
                    dumps.append(dict(dist=np.linalg.norm(pts[k, :, :2]-ues[k, :2], axis=1).round(3).tolist(),
                                      score=(sc/e).round(6).tolist(), cap=(cap/e).round(6).tolist(),
                                      pred=pred.tolist(), meas=(np.linalg.norm(Hloc[k], axis=(1, 2))**2).tolist(),
                                      truth_pred=[float(np.linalg.norm(H[k, l])**2) for l in range(L)]))
            nn = np.linalg.norm(pts[:, :, :2]-ues[:, None, :2], axis=-1).min(1)   # best any particle could do
            row = dict(drop=drop, setting=label, sigma0=sigma0, occasion=t, budget=-1, particles=n_part, loc_err=err_rec, loc_err_argmax=err_arg,
                       loc_err_raw_ci=err_raw, posterior_sd=post_sd.round(3).tolist(), nearest_particle_err=nn.tolist(),
                       loc_err_median=float(np.median(err_rec)), loc_err_argmax_median=float(np.median(err_arg)),
                       loc_err_raw_median=float(np.median(err_raw)), nearest_particle_median=float(np.median(nn)))
            if dumps:
                row['scores'] = dumps
            rec.add(row)
            return new, post_sd

        start = truth_state(ues, truth)
        occasion(start, 'oracle', 0., 0, ues.copy(), np.full(K, args.region_floor))
        motion_sd = args.motion_model_sd if args.motion_model_sd >= 0 else args.speed*args.srs_period
        for sigma0 in args.sigma0:
            state = start
            heading = rng.uniform(0, 2*np.pi, size=K)
            centres = ues+np.column_stack([rng.normal(0, sigma0, size=(K, 2)), np.zeros(K)])
            s_eff = np.full(K, math.hypot(sigma0, args.region_floor))
            for t in range(1, args.occasions+1):
                if t > 1:
                    state = move(state, heading)
                new, post_sd = occasion(state, 'tracked', sigma0, t, centres, s_eff)
                n_part = args.acq_samples if (t == 1 and args.acq_samples) else S
                res = s_eff*math.sqrt(2*math.pi/n_part)            # particle resolution of this occasion
                sig = res if args.loc_mode == 'argmax' else np.sqrt(post_sd**2+res**2)
                # predict: the UE may move by about motion_sd before the next occasion
                centres, s_eff = new, np.sqrt(sig**2+motion_sd**2+args.region_floor**2)
            print(f'drop={drop} sigma0={sigma0} rows={len(rec.rows)}', flush=True)
    rec.finish()


if __name__ == '__main__':
    main()
