"""第一轮 MRT 仿真：N = 3 AP、U = 2 UE、AP 1/2 → UE 1、AP 3 → UE 2，K = 2M，P_total = 2，P_max = 1。

用法：python run_round1.py [--M 1 2 4 6] [--samples 1000] [--workers 48] [--out results/round1_mrt_20261119]

随机数：样本 s 的底层信道 g = CN(0, 1)，形状 [U, N, 8]，由 default_rng([20261119, s]) 生成（先实部、后虚部，
各为 standard_normal，除以 sqrt(2)），再截取前 M 根天线。A/B/C、3 个 beta、4 个 SNR 共用同一个 g。
干扰链路乘 sqrt(beta)，服务链路 beta = 1。
"""

import argparse
import json
import logging
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')        # 多进程并行时每个进程单线程 BLAS，避免线程争用
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy
from scipy import stats

import antsel

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261119
M_MAX = 8
N, U = 3, 2
AP_UE = (0, 0, 1)
P_TOTAL, P_MAX = 2.0, 1.0
BETAS = [0.05, 0.25, 0.8]
SNRS_DB = [0, 10, 20, 30]
NOISE = [10 ** (-s / 10) for s in SNRS_DB]
SCHEMES = ('A', 'B', 'C')
TOL = 1e-9


def draw_g(sample):
    rng = np.random.default_rng([SEED, sample])
    re = rng.standard_normal((U, N, M_MAX))
    im = rng.standard_normal((U, N, M_MAX))
    return (re + 1j * im) / np.sqrt(2)


def work(item):
    M, sample = item
    g = draw_g(sample)[:, :, :M]
    v, k = antsel.solve(g, AP_UE, 2 * M, P_TOTAL, P_MAX, BETAS, NOISE)
    return sample, v, k.astype(np.int8)


def paired(x):
    n = len(x)
    m = float(np.mean(x))
    half = float(stats.t.ppf(0.975, n - 1) * np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float('nan')
    return {'mean': m, 'median': float(np.median(x)), 'p90': float(np.quantile(x, 0.9)),
            'ci95': [m - half, m + half]}


def count_diag(k, M):
    """k [S, N]：激活数分布，以及同时用 AP 1、AP 2 的样本中 k3 = M 与 k3 < M 的比例。"""
    both = (k[:, 0] > 0) & (k[:, 1] > 0)
    nb = int(both.sum())
    dist = Counter(','.join(str(int(x)) for x in row) for row in k)
    return {
        'dist': {key: c / len(k) for key, c in dist.most_common()},
        'frac_both_ap12': nb / len(k),
        'frac_k3_eq_M_given_both': float(np.mean(k[both, 2] == M)) if nb else None,
        'frac_k3_lt_M_given_both': float(np.mean(k[both, 2] < M)) if nb else None,
        'frac_k3_lt_M': float(np.mean(k[:, 2] < M)),
    }


def rows_for_M(M, se, kk, n_cand):
    """se [S, 3, nB, nS]，kk [S, 3, nB, nS, N] → 每组 (beta, SNR) 一行。"""
    rows = []
    for bi, beta in enumerate(BETAS):
        for si, snr in enumerate(SNRS_DB):
            a, b, c = (se[:, j, bi, si] for j in range(3))
            ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
            a, b, c = a[ok], b[ok], c[ok]
            row = {
                'M': M, 'beta': beta, 'snr_db': snr, 'noise_var': NOISE[si],
                'n_samples': int(se.shape[0]), 'n_infeasible': int((~ok).sum()),
                'frac_infeasible': float((~ok).mean()),
                'n_candidates': n_cand,
                'mean_A': float(a.mean()), 'mean_B': float(b.mean()), 'mean_C': float(c.mean()),
                'C_minus_A': paired(c - a), 'C_minus_B': paired(c - b), 'B_minus_A': paired(b - a),
                'rel_gain_CA_pct': float(100 * (c.mean() - a.mean()) / a.mean()),
                'rel_gain_CB_pct': float(100 * (c.mean() - b.mean()) / b.mean()),
                'check': {
                    'n_viol_C_ge_B': int(np.sum(c < b - TOL)), 'n_viol_B_ge_A': int(np.sum(b < a - TOL)),
                    'max_viol_C_ge_B': float(np.max(b - c)), 'max_viol_B_ge_A': float(np.max(a - b)),
                    'max_abs_C_minus_A': float(np.max(np.abs(c - a))),
                },
                'diag_B': count_diag(kk[ok, 1, bi, si], M),
                'diag_C': count_diag(kk[ok, 2, bi, si], M),
            }
            rows.append(row)
    return rows


def git_info():
    def run(*cmd):
        try:
            return subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None
    return {'commit': run('git', 'rev-parse', 'HEAD'),
            'validation_dir_dirty': bool(run('git', 'status', '--porcelain', '--', '.'))}


def ci_str(d):
    return f"{d['mean']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"


def write_summary(path, rows, timings):
    lines = ['# 第一轮 MRT：A / B / C 的平均 sum SE（bit/s/Hz）', '',
             '差值为逐样本配对均值 [95% t 区间]；相对增益 = 100·(mean C − mean X)/mean X。', '']
    for M in sorted({r['M'] for r in rows}):
        lines += [f'## M = {M}（C 可行候选 {rows_by(rows, M)[0]["n_candidates"]["C"]}，耗时 {timings[str(M)]:.1f} s）', '',
                  '| β | SNR (dB) | A | B | C | C−A | C−B | C−A % | C−B % | C−A 中位 / p90 | C−B 中位 / p90 |',
                  '|---|---|---|---|---|---|---|---|---|---|---|']
        for r in rows_by(rows, M):
            ca, cb = r['C_minus_A'], r['C_minus_B']
            lines.append(f"| {r['beta']} | {r['snr_db']} | {r['mean_A']:.3f} | {r['mean_B']:.3f} | {r['mean_C']:.3f} | "
                         f"{ci_str(ca)} | {ci_str(cb)} | {r['rel_gain_CA_pct']:.1f} | {r['rel_gain_CB_pct']:.1f} | "
                         f"{ca['median']:.3f} / {ca['p90']:.3f} | {cb['median']:.3f} / {cb['p90']:.3f} |")
        lines += ['', '诊断（比例）：P(k1>0 且 k2>0)；其中 k3 = M、k3 < M 的比例；最常见的 (k1,k2,k3)。', '',
                  '| β | SNR | B: 双 AP | B: k3=M / k3<M | C: 双 AP | C: k3=M / k3<M | C 最常见激活数 |',
                  '|---|---|---|---|---|---|---|']
        for r in rows_by(rows, M):
            def f(d):
                eq, lt = d['frac_k3_eq_M_given_both'], d['frac_k3_lt_M_given_both']
                return (f"{d['frac_both_ap12']:.3f}",
                        '—' if eq is None else f'{eq:.3f} / {lt:.3f}')
            b, c = f(r['diag_B']), f(r['diag_C'])
            top = ', '.join(f'({k}) {v:.2f}' for k, v in list(r['diag_C']['dist'].items())[:3])
            lines.append(f"| {r['beta']} | {r['snr_db']} | {b[0]} | {b[1]} | {c[0]} | {c[1]} | {top} |")
        lines.append('')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines))


def rows_by(rows, M):
    return [r for r in rows if r['M'] == M]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, nargs='+', default=[1, 2, 4, 6])
    ap.add_argument('--samples', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'round1_mrt_20261119'))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(os.path.join(args.out, 'run.log')),
                                  logging.StreamHandler(sys.stdout)])
    log = logging.getLogger()
    src = os.path.join(args.out, 'src')
    os.makedirs(src, exist_ok=True)
    for f in ('antsel.py', 'run_round1.py', 'test_antsel.py'):
        shutil.copy2(os.path.join(HERE, f), src)

    meta = {
        'purpose': 'Round-1 MRT: per-PRB AP-level (A) vs strongest-serving-antenna (B) vs antenna-level exhaustive (C) selection',
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'command': ' '.join([os.path.basename(sys.executable)] + sys.argv),
        'git': git_info(),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        'host_cpus': os.cpu_count(), 'workers': args.workers,
        'params': {'N': N, 'U': U, 'ap_ue (0-based)': AP_UE, 'M': args.M, 'K': '2*M',
                   'P_total': P_TOTAL, 'P_max': P_MAX, 'power_rule': 'p[n] = P_total*|S[n]|/K',
                   'betas_interference': BETAS, 'beta_serving': 1.0, 'snr_db': SNRS_DB,
                   'noise_var': '10^(-SNR_dB/10)', 'samples': args.samples, 'precoder': 'per-AP unit-norm MRT on active subset',
                   'combining': 'complex amplitudes summed over APs of the same stream, then |.|^2'},
        'seed_rule': (f'sample s: rng = numpy.random.default_rng([{SEED}, s]); g = (rng.standard_normal((U,N,{M_MAX})) '
                      f'+ 1j*rng.standard_normal((U,N,{M_MAX})))/sqrt(2); use g[:, :, :M]; '
                      'interference links scaled by sqrt(beta); shared by A/B/C and all (beta, SNR)'),
        'schemes': {'A': 'AP-level exhaustive (exactly K/M full APs, both UEs served)',
                    'B': 'exhaustive per-AP counts, strongest |h|^2 serving antennas per AP',
                    'C': 'exhaustive antenna subsets with |S| = K, both UEs served'},
        'stats': 'paired t 95% CI on per-sample C-A, C-B (and B-A); median, p90; relative gain of means',
        'timings_s': {}, 'candidate_counts': {},
    }

    rows = []
    for M in args.M:
        idx, counts, a_pos, kvecs = antsel.candidates(N, M, 2 * M, AP_UE, U, P_TOTAL, P_MAX)
        n_cand = {'A': int(len(a_pos)), 'B_count_vectors': int(len(kvecs)), 'C': int(len(idx))}
        meta['candidate_counts'][str(M)] = n_cand
        log.info(f'M = {M}: candidates {n_cand}, {args.samples} samples')
        t0 = time.time()
        se = np.zeros((args.samples, 3, len(BETAS), len(SNRS_DB)))
        kk = np.zeros((args.samples, 3, len(BETAS), len(SNRS_DB), N), dtype=np.int8)
        with ProcessPoolExecutor(args.workers) as ex:
            for s, v, k in ex.map(work, [(M, s) for s in range(args.samples)], chunksize=8):
                se[s], kk[s] = v, k
        dt = time.time() - t0
        meta['timings_s'][str(M)] = dt
        log.info(f'M = {M}: done in {dt:.1f} s')
        np.savez_compressed(os.path.join(args.out, f'samples_M{M}.npz'), se=se, k=kk,
                            schemes=np.array(SCHEMES), betas=np.array(BETAS), snr_db=np.array(SNRS_DB))
        new = rows_for_M(M, se, kk, n_cand)
        for r in new:
            ch = r['check']
            if ch['n_viol_C_ge_B'] or ch['n_viol_B_ge_A']:
                log.warning(f"order violation at M={M}, beta={r['beta']}, SNR={r['snr_db']}: {ch}")
            if M == 1 and ch['max_abs_C_minus_A'] != 0.0:
                log.warning(f"M = 1 but C != A: {ch['max_abs_C_minus_A']}")
            log.info(f"  beta={r['beta']:<5} SNR={r['snr_db']:>2} dB  A={r['mean_A']:.3f} B={r['mean_B']:.3f} "
                     f"C={r['mean_C']:.3f}  C-A={ci_str(r['C_minus_A'])}  C-B={ci_str(r['C_minus_B'])}")
        rows += new
        with open(os.path.join(args.out, 'rows.jsonl'), 'w') as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + '\n')
        with open(os.path.join(args.out, 'metadata.json'), 'w') as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        write_summary(os.path.join(args.out, 'summary.md'), rows, meta['timings_s'])
    meta['status'] = 'done'
    with open(os.path.join(args.out, 'metadata.json'), 'w') as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    log.info('all done')


if __name__ == '__main__':
    main()
