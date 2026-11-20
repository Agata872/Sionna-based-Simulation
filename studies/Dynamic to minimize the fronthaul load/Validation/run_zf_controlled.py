"""第二轮受控 ZF 实验（cell_free_round2_zf_simulation.txt）。

用法：python run_zf_controlled.py [--M 1 2 4 6] [--samples 1000] [--workers 48] [--out DIR]

信道：与第一轮相同的真信道样本（run_round1.draw_g，default_rng([20261119, s])，M_MAX = 8，取前 M 根），
     大尺度布局 UE 1 → [1, 1, beta_cross]，UE 2 → [beta_cross, beta_cross, 1]。
CSI：g_hat = (1 - eps) g + sqrt(eps (1 - eps)) e，e 来自独立流 default_rng([20261119, s, 1])（与 g 同布局），
     H_hat = sqrt(beta) g_hat；eps = 0（完美）、10^-2（NMSE -20 dB）、10^-1（-10 dB）。
     选择、B 的排序、预编码与功率归一化只用 H_hat，真实 SE 用 H。
对照：同一样本、同一 CSI 下的 MRT A/B/C（第一轮规则：固定 AP–UE 映射，AP 3 必须服务 UE 2，实际功率恒为 P_total）。
"""

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')        # 多进程并行时每个进程单线程 BLAS

import argparse
import json
import logging
import platform
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy

import antsel
import zfsel
from run_round1 import AP_UE, BETAS, N, NOISE, P_MAX, P_TOTAL, SEED, SNRS_DB, U, draw_g, git_info, paired
from run_round2 import draw_err, scale_links

HERE = os.path.dirname(os.path.abspath(__file__))
CSI = [('perfect', 0.0), ('nmse_-20dB', 1e-2), ('nmse_-10dB', 1e-1)]
SCH = zfsel.SCHEMES
TOL = 1e-9
DIFFS = [('C', 'A'), ('B', 'A'), ('C', 'B'), ('Ceq', 'A')]
CTRL_DIFFS = [('A_AP3', 'MRT_C'), ('C_AP3', 'MRT_C'), ('C_AP3', 'A_AP3'), ('C', 'C_AP3')]


def g_hat(g, e, eps):
    return g if eps == 0 else (1 - eps) * g + np.sqrt(eps * (1 - eps)) * e


def work(item):
    M, sample = item
    K = 2 * M
    g = draw_g(sample)[:, :, :M]
    e = draw_err(sample)[:, :, :M]
    nb, ne, ns, nk = len(BETAS), len(CSI), len(NOISE), len(SCH)
    r = {
        'se_true': np.zeros((nb, ne, nk, ns)), 'se_est': np.zeros((nb, ne, nk, ns)),
        'sinr_true': np.zeros((nb, ne, nk, ns, U)), 'intf_true': np.zeros((nb, ne, nk, U)),
        'p_used': np.zeros((nb, ne, nk)), 'p_ap': np.zeros((nb, ne, nk, N)), 'c2': np.zeros((nb, ne, nk)),
        'Q': np.zeros((nb, ne, nk)), 'maxq': np.zeros((nb, ne, nk)),
        'binding': np.zeros((nb, ne, nk), dtype=np.int8),      # 0 总功率预算，1 单 AP 上限，2 固定 P_ref（Ceq）
        'counts': np.zeros((nb, ne, nk, N), dtype=np.int8), 'key': np.zeros((nb, ne, nk), dtype=np.int64),
        'p_ref': np.zeros((nb, ne)), 'n_invalid': np.zeros((nb, ne), dtype=np.int64),
        'n_ceq_feasible': np.zeros((nb, ne), dtype=np.int64),
        'mrt_se': np.zeros((ne, 3, nb, ns)), 'mrt_k': np.zeros((ne, 3, nb, ns, N), dtype=np.int8),
        'nmse_num': np.zeros(ne), 'nmse_den': np.zeros(ne),
        'chk': {'resid_est': 0.0, 'q_rel': 0.0, 'p_over': 0, 'n_active': 0, 'ceq_p_rel': 0.0, 'est_formula': 0.0},
    }
    c = zfsel.cand(N, M, K)
    weights = np.int64(1) << np.arange(N * M, dtype=np.int64)
    for ei, (_, eps) in enumerate(CSI):
        gh = g_hat(g, e, eps)
        r['nmse_num'][ei] = np.sum(np.abs(gh - g) ** 2)
        r['nmse_den'][ei] = np.sum(np.abs(g) ** 2)
        v, k = antsel.solve(g, AP_UE, K, P_TOTAL, P_MAX, BETAS, NOISE, h_est=None if eps == 0 else gh)
        r['mrt_se'][ei], r['mrt_k'][ei] = v, k
        for bi, beta in enumerate(BETAS):
            H = zfsel.assemble(scale_links(g, beta))
            He = zfsel.assemble(scale_links(gh, beta))
            win, dg = zfsel.select(H, He, N, M, K, P_TOTAL, P_MAX)
            r['p_ref'][bi, ei] = dg['p_ref']
            r['n_invalid'][bi, ei] = dg['n_invalid']
            r['n_ceq_feasible'][bi, ei] = dg['n_ceq_feasible']
            for si, name in enumerate(SCH):
                j, c2 = win[name]
                mask = c['bits'][j]
                ev = zfsel.evaluate(H, He, mask, c2, N, M, NOISE)
                r['se_true'][bi, ei, si] = np.log2(1 + ev['sinr_true']).sum(1)
                r['se_est'][bi, ei, si] = np.log2(1 + ev['sinr_est']).sum(1)
                r['sinr_true'][bi, ei, si] = ev['sinr_true']
                r['intf_true'][bi, ei, si] = ev['intf_true']
                r['p_used'][bi, ei, si] = ev['p_used']
                r['p_ap'][bi, ei, si] = ev['p_ap']
                r['c2'][bi, ei, si] = c2
                r['Q'][bi, ei, si] = dg['Q'][j]
                r['maxq'][bi, ei, si] = dg['qap'][j].max()
                r['binding'][bi, ei, si] = 2 if name == 'Ceq' else int(P_MAX / dg['qap'][j].max() < P_TOTAL / dg['Q'][j])
                r['counts'][bi, ei, si] = ev['counts']
                r['key'][bi, ei, si] = int(mask.astype(np.int64) @ weights)
                ch = r['chk']
                ch['resid_est'] = max(ch['resid_est'], ev['resid_est'])
                ch['q_rel'] = max(ch['q_rel'], float(np.max(np.abs(ev['q_ap'] - dg['qap'][j])) / dg['Q'][j]))
                ch['p_over'] += int(ev['p_used'] > P_TOTAL * (1 + TOL)) + int(np.any(ev['p_ap'] > P_MAX * (1 + TOL)))
                ch['n_active'] += int(ev['n_active'] != K)
                ch['est_formula'] = max(ch['est_formula'], float(np.max(np.abs(
                    r['se_est'][bi, ei, si] - U * np.log2(1 + c2 / np.asarray(NOISE))))))
                if name == 'Ceq':
                    ch['ceq_p_rel'] = max(ch['ceq_p_rel'], abs(ev['p_used'] - dg['p_ref']) / dg['p_ref'])
    return M, sample, r


def stats_diff(x, y):
    d = x - y
    out = paired(d)
    out['p10'] = float(np.quantile(d, 0.1))
    out['frac_pos'] = float(np.mean(d > TOL))
    out['frac_neg'] = float(np.mean(d < -TOL))
    out['rel_pct'] = float(100 * (x.mean() - y.mean()) / y.mean())
    return out


def rows_for_M(M, R):
    rows = []
    ix = {s: i for i, s in enumerate(SCH)}
    for bi, beta in enumerate(BETAS):
        for ei, (tag, eps) in enumerate(CSI):
            nmse = float(R['nmse_num'][:, ei].sum() / R['nmse_den'][:, ei].sum())
            for si, snr in enumerate(SNRS_DB):
                t = {s: R['se_true'][:, bi, ei, ix[s], si] for s in SCH}
                es = {s: R['se_est'][:, bi, ei, ix[s], si] for s in SCH}
                for j, s in enumerate(('MRT_A', 'MRT_B', 'MRT_C')):
                    t[s] = R['mrt_se'][:, ei, j, bi, si]
                row = {'M': M, 'beta_cross': beta, 'csi': tag, 'eps': eps, 'nmse_measured': nmse, 'snr_db': snr,
                       'n_samples': int(R['se_true'].shape[0]),
                       'mean_true': {s: float(v.mean()) for s, v in t.items()},
                       'mean_est': {s: float(v.mean()) for s, v in es.items()},
                       'diff_true': {f'{a}-{b}': stats_diff(t[a], t[b]) for a, b in DIFFS + CTRL_DIFFS},
                       'diff_est_mean': {f'{a}-{b}': float((es[a] - es[b]).mean()) for a, b in DIFFS},
                       'per_ue_se_true': {s: [float(np.log2(1 + R['sinr_true'][:, bi, ei, ix[s], si, u]).mean())
                                              for u in range(U)] for s in SCH},
                       'power': {s: {'P_used_mean': float(R['p_used'][:, bi, ei, ix[s]].mean()),
                                     'P_used_p10_p90': [float(np.quantile(R['p_used'][:, bi, ei, ix[s]], q)) for q in (0.1, 0.9)],
                                     'p_ap_mean': [float(x) for x in R['p_ap'][:, bi, ei, ix[s]].mean(0)],
                                     'frac_ap_limited': float(np.mean(R['binding'][:, bi, ei, ix[s]] == 1)),
                                     'Q_mean': float(R['Q'][:, bi, ei, ix[s]].mean()),
                                     'maxq_mean': float(R['maxq'][:, bi, ei, ix[s]].mean()),
                                     'intf_true_mean': float(R['intf_true'][:, bi, ei, ix[s]].sum(-1).mean())}
                                 for s in SCH},
                       'mrt_P_used': P_TOTAL,
                       'counts_top': {s: dict(Counter(','.join(map(str, k)) for k in R['counts'][:, bi, ei, ix[s]]).most_common(3))
                                      for s in ('B', 'C', 'Ceq')},
                       'n_invalid_candidates_total': int(R['n_invalid'][:, bi, ei].sum()),
                       'ceq_feasible_mean': float(R['n_ceq_feasible'][:, bi, ei].mean())}
                v = {}
                for a, b in (('C', 'B'), ('B', 'A'), ('C', 'Ceq'), ('Ceq', 'A')):
                    v[f'est {a}>={b}'] = int(np.sum(es[a] < es[b] - TOL))
                    if eps == 0:
                        v[f'true {a}>={b}'] = int(np.sum(t[a] < t[b] - TOL))
                row['order_violations'] = v
                rows.append(row)
    return rows


def ci(d):
    return f"{d['mean']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"


def summary(rows, meta):
    L = ['# 第二轮受控 ZF：A / B / C / Ceq（真实信道上的 sum SE，bit/s/Hz）', '',
         '差值为逐样本配对均值 [95% t 区间]；neg = 负增益样本比例；P_used 为实际总发射功率（预算 2）；'
         'AP 限 = 由单 AP 上限决定缩放的样本比例。', '']
    for M in sorted({r['M'] for r in rows}):
        L += [f"## M = {M}（耗时 {meta['timings_s'][str(M)]:.1f} s）", '',
              '| β_cross | CSI | SNR | A | B | C | Ceq | C−A | B−A | C−B | Ceq−A | C−A neg | Ceq−A neg | P_used A / C / Ceq | AP 限 A / C |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for r in rows:
            if r['M'] != M:
                continue
            m, d, p = r['mean_true'], r['diff_true'], r['power']
            L.append(f"| {r['beta_cross']} | {r['csi']} | {r['snr_db']} | {m['A']:.2f} | {m['B']:.2f} | {m['C']:.2f} | {m['Ceq']:.2f} | "
                     f"{ci(d['C-A'])} | {ci(d['B-A'])} | {ci(d['C-B'])} | {ci(d['Ceq-A'])} | "
                     f"{d['C-A']['frac_neg']:.3f} | {d['Ceq-A']['frac_neg']:.3f} | "
                     f"{p['A']['P_used_mean']:.2f} / {p['C']['P_used_mean']:.2f} / {p['Ceq']['P_used_mean']:.2f} | "
                     f"{p['A']['frac_ap_limited']:.2f} / {p['C']['frac_ap_limited']:.2f} |")
        L += ['', '对照（候选集合与 MRT 匹配：AP 3 至少一根天线；MRT 实际功率恒为 2）', '',
              '| β_cross | CSI | SNR | MRT_A | MRT_C | ZF A_AP3 | ZF C_AP3 | A_AP3−MRT_C | C_AP3−MRT_C | C−C_AP3 | P_used A_AP3 / C_AP3 |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
        for r in rows:
            if r['M'] != M:
                continue
            m, d, p = r['mean_true'], r['diff_true'], r['power']
            L.append(f"| {r['beta_cross']} | {r['csi']} | {r['snr_db']} | {m['MRT_A']:.2f} | {m['MRT_C']:.2f} | {m['A_AP3']:.2f} | "
                     f"{m['C_AP3']:.2f} | {ci(d['A_AP3-MRT_C'])} | {ci(d['C_AP3-MRT_C'])} | {ci(d['C-C_AP3'])} | "
                     f"{p['A_AP3']['P_used_mean']:.2f} / {p['C_AP3']['P_used_mean']:.2f} |")
        L.append('')
    return '\n'.join(L)


def reuse_checks(M, R):
    """与已有结果逐样本核对：完美 CSI 的 A/C 对试跑 ZFap_A/ZFap_C，MRT 对第一轮。"""
    out = {}
    trial = os.path.join(HERE, 'results', 'round2bc_precoder_csi_20261119', f'samples_M{M}.npz')
    if os.path.exists(trial):
        z = np.load(trial)
        sch = list(z['schemes'])
        e0 = list(z['eps']).index(0.0)
        for new, old in (('A', 'ZFap_A'), ('C', 'ZFap_C')):
            a = R['se_true'][:, :, 0, SCH.index(new), :]                    # [S, nB, nS]
            b = z['se'][:len(a), e0, sch.index(old)]                        # [S, nB, nS]
            out[f'{new} vs trial {old}'] = float(np.max(np.abs(a - b)))
    r1 = os.path.join(HERE, 'results', 'round1_mrt_M8_20261119' if M == 8 else 'round1_mrt_20261119', f'samples_M{M}.npz')
    if os.path.exists(r1):
        z = np.load(r1)
        out['MRT perfect vs round 1'] = float(np.max(np.abs(R['mrt_se'][:, 0] - z['se'][:len(R['mrt_se'])])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, nargs='+', default=[1, 2, 4, 6])
    ap.add_argument('--samples', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'round2_zf_controlled_20261120'))
    args = ap.parse_args()
    out = args.out
    os.makedirs(os.path.join(out, 'src'), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(os.path.join(out, 'run.log')), logging.StreamHandler(sys.stdout)])
    log = logging.getLogger()
    for f in ('zfsel.py', 'antsel.py', 'antsel_zf.py', 'run_round1.py', 'run_round2.py', 'run_zf_controlled.py',
              'test_zfsel.py'):
        shutil.copy2(os.path.join(HERE, f), os.path.join(out, 'src'))
    meta = {
        'experiment': 'round-2 controlled ZF (cell_free_round2_zf_simulation.txt)',
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'command': ' '.join([os.path.basename(sys.executable)] + sys.argv),
        'git': git_info(),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        'host_cpus': os.cpu_count(), 'workers': args.workers,
        'params': {'N': N, 'U': U, 'M': args.M, 'K': '2*M', 'P_total': P_TOTAL, 'P_max': P_MAX,
                   'beta_layout': 'UE1 -> [1, 1, beta_cross], UE2 -> [beta_cross, beta_cross, 1]',
                   'beta_cross': BETAS, 'snr_db': SNRS_DB, 'noise_var': '10^(-SNR_dB/10)', 'samples': args.samples,
                   'csi': {t: e for t, e in CSI}},
        'conventions': 'H[u, n*M+m] = conj(h[u,n,m]) (round-1 column channel, receive term h^H w); received = H @ W',
        'seed_rule': (f'true channel g: default_rng([{SEED}, s]) as round 1 (M_MAX = 8, first M antennas); '
                      f'error source e: default_rng([{SEED}, s, 1]), same layout, independent stream; '
                      'g_hat = (1-eps) g + sqrt(eps(1-eps)) e; shared by all schemes, beta_cross, SNR'),
        'design': {
            'service_rule': 'centralised ZF, every active antenna serves both UEs; A includes AP1+AP2 (AP3 off)',
            'zf': 'W0 = H_hat_S^H G^-1 (G = H_hat_S H_hat_S^H); q[n] = per-AP power of W0; '
                  'c2 = min(P_total/Q, P_max/max q); W = sqrt(c2) W0; P_used = c2 Q',
            'selection': 'all schemes maximise estimated sum SE = U log2(1 + c2/noise), i.e. c2; SNR-independent; '
                         'ties -> smallest lexicographic candidate index',
            'B': 'exhaustive per-AP counts (sum = K), strongest estimated sum_u |H_hat[u,m]|^2 per AP, ties by antenna index',
            'Ceq': 'P_ref = P_used(A); c2 = P_ref/Q; keep candidates with c2 q[n] <= P_max for all n; maximise c2',
            'control': 'A_AP3 / C_AP3: A / C restricted to AP 3 having >= 1 active antenna (matches MRT candidate set); '
                       'MRT A/B/C under the same CSI (round-1 rules, fixed mapping, P_used = P_total)',
            'rank_check': f'candidate invalid if det(G) <= {zfsel.RANK_TOL} (tr G / 2)^2',
        },
        'reuse': {}, 'timings_s': {}, 'checks': {},
    }
    rows = []
    for M in args.M:
        t0 = time.time()
        res = [None] * args.samples
        with ProcessPoolExecutor(args.workers) as ex:
            for _, s, r in ex.map(work, [(M, s) for s in range(args.samples)], chunksize=4):
                res[s] = r
        R = {k: np.stack([r[k] for r in res]) for k in res[0] if k != 'chk'}
        chk = {k: (max(r['chk'][k] for r in res) if isinstance(res[0]['chk'][k], float) else sum(r['chk'][k] for r in res))
               for k in res[0]['chk']}
        dt = time.time() - t0
        meta['timings_s'][str(M)] = dt
        meta['checks'][str(M)] = chk
        meta['reuse'][str(M)] = reuse_checks(M, R)
        np.savez_compressed(os.path.join(out, f'samples_M{M}.npz'), schemes=np.array(SCH), csi=np.array([c[0] for c in CSI]),
                            eps=np.array([c[1] for c in CSI]), beta_cross=np.array(BETAS), snr_db=np.array(SNRS_DB),
                            mrt_schemes=np.array(['MRT_A', 'MRT_B', 'MRT_C']), **R)
        new = rows_for_M(M, R)
        bad = {k: v for r in new for k, v in r['order_violations'].items() if v}
        log.info(f'M = {M}: done in {dt:.1f} s; checks {chk}; reuse {meta["reuse"][str(M)]}; order violations {bad or "none"}')
        rows += new
        with open(os.path.join(out, 'rows.jsonl'), 'w') as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + '\n')
        with open(os.path.join(out, 'summary.md'), 'w') as fh:
            fh.write(summary(rows, meta))
        with open(os.path.join(out, 'metadata.json'), 'w') as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
    meta['status'] = 'done'
    with open(os.path.join(out, 'metadata.json'), 'w') as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    log.info('all done')


if __name__ == '__main__':
    main()
