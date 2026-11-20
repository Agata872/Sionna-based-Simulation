"""第二轮仿真。网络、功率规则、信道样本与第一轮相同（run_round1.draw_g，同一种子规则），可逐样本配对。

  --exp mapping    (a) 按原文最强 UE 规则，用每个 AP 的完整信道生成 AP–UE 映射，A/B/C 共用映射。
                   报告无法让两个 UE 都获得服务的样本比例，三种方案在相同的可行样本上比较。
  --exp precoder   (b)+(c) MRT A/B/C 与 ZF/RZF（两种式 (18) 推广：'ant' 逐天线、'ap' 逐 AP）的 A/C，
                   CSI 误差 NMSE eps ∈ {0, 1e-3, 1e-2, 1e-1}；eps = 0 即 (b) 理想 CSI。
                   估计信道 h_est = h + e，e[u,n,m] ~ CN(0, eps * beta[u,n])，误差样本由
                   default_rng([20261119, s, 1]) 生成（形状 [U, N, 8]，截取前 M 根），各 eps、beta、SNR 共用。
                   选择、B 的天线排序、MRT/ZF/RZF 权值和 ZF 缩放因子都用 h_est，sum SE 用真实 h 评估。

用法：python run_round2.py --exp mapping|precoder [--M 1 2 4 6 8] [--samples 1000] [--workers 48] [--out DIR]
"""

import argparse
import json
import logging
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')        # 多进程并行时每个进程单线程 BLAS，避免线程争用
import platform
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy

import antsel
import antsel_zf
from run_round1 import (AP_UE, BETAS, M_MAX, N, NOISE, P_MAX, P_TOTAL, SEED, SNRS_DB, TOL, U, count_diag,
                        draw_g, git_info, paired)

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = [0.0, 1e-3, 1e-2, 1e-1]
PREC_SCHEMES = ['MRT_A', 'MRT_B', 'MRT_C',
                'ZFant_A', 'ZFant_C', 'ZFap_A', 'ZFap_C',
                'RZFant_A', 'RZFant_C', 'RZFap_A', 'RZFap_C']
PREC_DIFFS = [('MRT_C', 'MRT_A'), ('MRT_C', 'MRT_B'), ('MRT_B', 'MRT_A'),
              ('ZFant_C', 'ZFant_A'), ('ZFap_C', 'ZFap_A'), ('RZFant_C', 'RZFant_A'), ('RZFap_C', 'RZFap_A'),
              ('ZFant_A', 'MRT_C'), ('ZFap_A', 'MRT_C'), ('RZFant_A', 'MRT_C'), ('RZFap_A', 'MRT_C'),
              ('ZFant_C', 'MRT_C'), ('RZFant_C', 'MRT_C')]


def scale_links(g, beta):
    """服务链路（按名义映射 AP_UE）beta = 1，其余链路乘 sqrt(beta)。"""
    h = g.copy()
    for n, v in enumerate(AP_UE):
        for u in range(U):
            if u != v:
                h[u, n] *= np.sqrt(beta)
    return h


def draw_err(sample):
    rng = np.random.default_rng([SEED, sample, 1])
    re = rng.standard_normal((U, N, M_MAX))
    im = rng.standard_normal((U, N, M_MAX))
    return (re + 1j * im) / np.sqrt(2)


# ---------------------------------------------------------------- (a) 最强 UE 映射

def work_mapping(item):
    M, sample = item
    g = draw_g(sample)[:, :, :M]
    se = np.full((3, len(BETAS), len(SNRS_DB)), -np.inf)
    kk = np.full((3, len(BETAS), len(SNRS_DB), N), -1, dtype=np.int8)
    maps = np.zeros((len(BETAS), N), dtype=np.int8)
    for bi, beta in enumerate(BETAS):
        h = scale_links(g, beta)
        mp = antsel.strongest_mapping(h)
        maps[bi] = mp
        if len(set(mp)) < U:                      # 有 UE 没有任何 AP 服务
            continue
        v, k = antsel.solve(h, mp, 2 * M, P_TOTAL, P_MAX, [1.0], NOISE)
        se[:, bi], kk[:, bi] = v[:, 0], k[:, 0]
    return sample, se, kk, maps


def rows_mapping(M, se, kk, maps):
    rows = []
    for bi, beta in enumerate(BETAS):
        mp = maps[:, bi]
        changed = np.any(mp != np.array(AP_UE), axis=1)
        for si, snr in enumerate(SNRS_DB):
            a, b, c = (se[:, j, bi, si] for j in range(3))
            ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
            a, b, c = a[ok], b[ok], c[ok]
            dist = Counter(''.join(str(int(x) + 1) for x in row) for row in mp)
            row = {'M': M, 'beta': beta, 'snr_db': snr, 'n_samples': int(len(ok)),
                   'n_infeasible': int((~ok).sum()), 'frac_infeasible': float((~ok).mean()),
                   'frac_mapping_changed': float(changed.mean()),
                   'frac_mapping_changed_among_feasible': float(changed[ok].mean()) if ok.any() else None,
                   'mapping_dist (UE index per AP, 1-based)': {k: v / len(mp) for k, v in dist.most_common()}}
            if ok.sum() > 1:
                row.update({'mean_A': float(a.mean()), 'mean_B': float(b.mean()), 'mean_C': float(c.mean()),
                            'C_minus_A': paired(c - a), 'C_minus_B': paired(c - b), 'B_minus_A': paired(b - a),
                            'rel_gain_CA_pct': float(100 * (c.mean() - a.mean()) / a.mean()),
                            'rel_gain_CB_pct': float(100 * (c.mean() - b.mean()) / b.mean()),
                            'check': {'n_viol_C_ge_B': int(np.sum(c < b - TOL)),
                                      'n_viol_B_ge_A': int(np.sum(b < a - TOL))},
                            'counts_B': count_diag(kk[ok, 1, bi, si].astype(int), M)['dist'],
                            'counts_C': count_diag(kk[ok, 2, bi, si].astype(int), M)['dist']})
            rows.append(row)
    return rows


# ---------------------------------------------------------------- (b)+(c) 预编码与 CSI 误差

def work_precoder(item):
    M, sample, eps_list = item
    g = draw_g(sample)[:, :, :M]
    e = draw_err(sample)[:, :, :M]
    ne, ns, nb = len(eps_list), len(SNRS_DB), len(BETAS)
    se = np.zeros((ne, len(PREC_SCHEMES), nb, ns))
    kk = np.zeros((ne, len(PREC_SCHEMES), nb, ns, N), dtype=np.int8)
    for ei, eps in enumerate(eps_list):
        g_est = None if eps == 0 else g + np.sqrt(eps) * e
        v, k = antsel.solve(g, AP_UE, 2 * M, P_TOTAL, P_MAX, BETAS, NOISE, h_est=g_est)
        se[ei, :3], kk[ei, :3] = v, k
        for bi, beta in enumerate(BETAS):
            h = scale_links(g, beta)
            he = None if g_est is None else scale_links(g_est, beta)
            for kind, base in (('zf', 3), ('rzf', 7)):
                vz, kz = antsel_zf.solve_zf(h, 2 * M, P_TOTAL, P_MAX, NOISE, kind, h_est=he)
                # vz [variant, scheme, nS]：顺序 ant_A, ant_C, ap_A, ap_C
                for vi in range(2):
                    for ci in range(2):
                        se[ei, base + 2 * vi + ci, bi] = vz[vi, ci]
                        kk[ei, base + 2 * vi + ci, bi] = kz[vi, ci]
    return sample, se, kk


def rows_precoder(M, se, kk, eps_list):
    rows = []
    ix = {s: i for i, s in enumerate(PREC_SCHEMES)}
    for ei, eps in enumerate(eps_list):
        for bi, beta in enumerate(BETAS):
            for si, snr in enumerate(SNRS_DB):
                x = se[:, ei, :, bi, si]
                row = {'M': M, 'eps': eps, 'nmse_db': None if eps == 0 else float(10 * np.log10(eps)),
                       'beta': beta, 'snr_db': snr, 'n_samples': int(x.shape[0]),
                       'mean': {s: float(x[:, ix[s]].mean()) for s in PREC_SCHEMES}, 'diff': {}}
                for p, q in PREC_DIFFS:
                    d = x[:, ix[p]] - x[:, ix[q]]
                    row['diff'][f'{p}-{q}'] = dict(paired(d), rel_pct=float(
                        100 * (x[:, ix[p]].mean() - x[:, ix[q]].mean()) / x[:, ix[q]].mean()))
                if eps == 0:                                 # 理想 CSI 时穷举必然不差于子集
                    row['check'] = {
                        'MRT_C>=MRT_B>=MRT_A': int(np.sum(x[:, 2] < x[:, 1] - TOL) + np.sum(x[:, 1] < x[:, 0] - TOL)),
                        'X_C>=X_A (ZF/RZF)': int(sum(np.sum(x[:, ix[f'{k}_C']] < x[:, ix[f'{k}_A']] - TOL)
                                                     for k in ('ZFant', 'ZFap', 'RZFant', 'RZFap')))}
                row['counts_MRT_C'] = dict(list(count_diag(kk[:, ei, ix['MRT_C'], bi, si].astype(int), M)['dist'].items())[:5])
                row['counts_ZFant_C'] = dict(list(count_diag(kk[:, ei, ix['ZFant_C'], bi, si].astype(int), M)['dist'].items())[:5])
                rows.append(row)
    return rows


# ---------------------------------------------------------------- 汇总表

def ci(d):
    return f"{d['mean']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"


def summary_mapping(rows, timings):
    L = ['# (a) 最强 UE 映射：A / B / C 的平均 sum SE（bit/s/Hz，只统计可行样本）', '',
         '不可行 = 映射后有 UE 没有任何 AP 服务。差值为逐样本配对均值 [95% t 区间]。', '']
    for M in sorted({r['M'] for r in rows}):
        L += [f'## M = {M}（耗时 {timings[str(M)]:.1f} s）', '',
              '| β | SNR | 不可行比例 | 映射改变比例 | A | B | C | C−A | C−B | C−A % | C−B % |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
        for r in rows:
            if r['M'] != M:
                continue
            if 'mean_A' not in r:
                L.append(f"| {r['beta']} | {r['snr_db']} | {r['frac_infeasible']:.3f} | {r['frac_mapping_changed']:.3f} | — | — | — | — | — | — | — |")
                continue
            L.append(f"| {r['beta']} | {r['snr_db']} | {r['frac_infeasible']:.3f} | {r['frac_mapping_changed']:.3f} | "
                     f"{r['mean_A']:.3f} | {r['mean_B']:.3f} | {r['mean_C']:.3f} | {ci(r['C_minus_A'])} | "
                     f"{ci(r['C_minus_B'])} | {r['rel_gain_CA_pct']:.1f} | {r['rel_gain_CB_pct']:.1f} |")
        L.append('')
    return '\n'.join(L)


def summary_precoder(rows, timings):
    L = ['# (b)+(c) MRT 与 ZF/RZF，理想与非理想 CSI（bit/s/Hz）', '',
         'ant = 逐天线功率上限 P_max/M（M = 1 时即 [10] 式 (18)）；ap = 逐 AP 上限 P_max；两者都受总功率 P_total 约束。',
         'eps = 估计误差 NMSE（相对每条链路的 beta）。差值为逐样本配对均值 [95% t 区间]。', '']
    for M in sorted({r['M'] for r in rows}):
        L.append(f'## M = {M}（耗时 {timings[str(M)]:.1f} s）')
        for eps in sorted({r['eps'] for r in rows}):
            sub = [r for r in rows if r['M'] == M and r['eps'] == eps]
            tag = '理想 CSI' if eps == 0 else f"NMSE = {sub[0]['nmse_db']:.0f} dB"
            L += ['', f'### {tag}：平均 sum SE', '',
                  '| β | SNR | ' + ' | '.join(PREC_SCHEMES) + ' |', '|---|---|' + '---|' * len(PREC_SCHEMES)]
            for r in sub:
                L.append(f"| {r['beta']} | {r['snr_db']} | " + ' | '.join(f"{r['mean'][s]:.2f}" for s in PREC_SCHEMES) + ' |')
            keys = ['MRT_C-MRT_A', 'MRT_C-MRT_B', 'ZFant_C-ZFant_A', 'ZFap_C-ZFap_A', 'RZFant_C-RZFant_A',
                    'ZFant_A-MRT_C', 'RZFant_A-MRT_C']
            L += ['', f'### {tag}：关键差值', '', '| β | SNR | ' + ' | '.join(keys) + ' |', '|---|---|' + '---|' * len(keys)]
            for r in sub:
                L.append(f"| {r['beta']} | {r['snr_db']} | " + ' | '.join(ci(r['diff'][k]) for k in keys) + ' |')
        L.append('')
    return '\n'.join(L)


# ---------------------------------------------------------------- 主程序

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', choices=('mapping', 'precoder'), required=True)
    ap.add_argument('--M', type=int, nargs='+', default=[1, 2, 4, 6, 8])
    ap.add_argument('--samples', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--eps', type=float, nargs='+', default=EPS)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    name = {'mapping': 'round2a_mapping_20261119', 'precoder': 'round2bc_precoder_csi_20261119'}[args.exp]
    out = args.out or os.path.join(HERE, 'results', name)
    os.makedirs(out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(os.path.join(out, 'run.log')), logging.StreamHandler(sys.stdout)])
    log = logging.getLogger()
    os.makedirs(os.path.join(out, 'src'), exist_ok=True)
    for f in ('antsel.py', 'antsel_zf.py', 'run_round1.py', 'run_round2.py', 'test_antsel.py', 'test_round2.py'):
        shutil.copy2(os.path.join(HERE, f), os.path.join(out, 'src'))

    meta = {
        'experiment': args.exp,
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'command': ' '.join([os.path.basename(sys.executable)] + sys.argv),
        'git': git_info(),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        'host_cpus': os.cpu_count(), 'workers': args.workers,
        'params': {'N': N, 'U': U, 'nominal_ap_ue (0-based)': AP_UE, 'M': args.M, 'K': '2*M',
                   'P_total': P_TOTAL, 'P_max': P_MAX, 'betas': BETAS, 'snr_db': SNRS_DB, 'samples': args.samples},
        'seed_rule': (f'channel: same as round 1 (default_rng([{SEED}, s]), M_MAX = {M_MAX}, first M antennas); '
                      f'CSI error: default_rng([{SEED}, s, 1]) with the same layout'),
        'timings_s': {},
    }
    if args.exp == 'mapping':
        meta['design'] = ('beta matrix from the nominal mapping (AP1, AP2 near UE1; AP3 near UE2); per sample and beta, '
                          'AP n serves argmax_u ||h[u,n,:]||^2; MRT A/B/C as in round 1 with this mapping; '
                          'samples where some UE gets no AP are infeasible for all three schemes')
    else:
        meta['params']['eps'] = args.eps
        meta['design'] = {
            'MRT': 'round-1 A/B/C (fixed nominal mapping), weights and selection from h_est, evaluated on true h',
            'ZF/RZF': ('all active antennas jointly serve both UEs; W = Hs^H (Hs Hs^H + alpha I)^-1 from h_est; '
                       'alpha = 0 (ZF) or U*noise_var/P_total (RZF); common scaling rho = largest value meeting '
                       "'ant': per-antenna P_max/M and total P_total, or 'ap': per-AP P_max and total P_total; "
                       'A = any K/M full APs (incl. AP1+AP2), C = any K antennas; selection by estimated sum SE'),
            'csi_error': 'h_est = h + e, e ~ CN(0, eps*beta[u,n]) i.i.d.',
        }
        meta['schemes'] = PREC_SCHEMES

    rows = []
    for M in args.M:
        t0 = time.time()
        if args.exp == 'mapping':
            se = np.zeros((args.samples, 3, len(BETAS), len(SNRS_DB)))
            kk = np.zeros(se.shape + (N,), dtype=np.int8)
            maps = np.zeros((args.samples, len(BETAS), N), dtype=np.int8)
            with ProcessPoolExecutor(args.workers) as ex:
                for s, v, k, mp in ex.map(work_mapping, [(M, s) for s in range(args.samples)], chunksize=4):
                    se[s], kk[s], maps[s] = v, k, mp
            np.savez_compressed(os.path.join(out, f'samples_M{M}.npz'), se=se, k=kk, mapping=maps,
                                schemes=np.array(['A', 'B', 'C']), betas=np.array(BETAS), snr_db=np.array(SNRS_DB))
            new = rows_mapping(M, se, kk, maps)
        else:
            se = np.zeros((args.samples, len(args.eps), len(PREC_SCHEMES), len(BETAS), len(SNRS_DB)))
            kk = np.zeros(se.shape + (N,), dtype=np.int8)
            with ProcessPoolExecutor(args.workers) as ex:
                for s, v, k in ex.map(work_precoder, [(M, s, args.eps) for s in range(args.samples)], chunksize=2):
                    se[s], kk[s] = v, k
            np.savez_compressed(os.path.join(out, f'samples_M{M}.npz'), se=se, k=kk, schemes=np.array(PREC_SCHEMES),
                                eps=np.array(args.eps), betas=np.array(BETAS), snr_db=np.array(SNRS_DB))
            new = rows_precoder(M, se, kk, args.eps)
            for r in new:
                if r.get('check') and any(r['check'].values()):
                    log.warning(f"order violation M={M} beta={r['beta']} SNR={r['snr_db']}: {r['check']}")
        dt = time.time() - t0
        meta['timings_s'][str(M)] = dt
        log.info(f'{args.exp}: M = {M} done in {dt:.1f} s')
        rows += new
        with open(os.path.join(out, 'rows.jsonl'), 'w') as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + '\n')
        with open(os.path.join(out, 'summary.md'), 'w') as fh:
            fh.write(summary_mapping(rows, meta['timings_s']) if args.exp == 'mapping'
                     else summary_precoder(rows, meta['timings_s']))
        with open(os.path.join(out, 'metadata.json'), 'w') as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
    meta['status'] = 'done'
    with open(os.path.join(out, 'metadata.json'), 'w') as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    log.info('all done')


if __name__ == '__main__':
    main()
