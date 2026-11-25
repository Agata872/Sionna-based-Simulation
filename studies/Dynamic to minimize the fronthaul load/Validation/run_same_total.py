"""E8：同总天线数下单天线分布式 RU 与共址多天线 RU 的逐 PRB 天线激活（ZF，穷举；模型见 same_total.py）。

用法：python run_same_total.py [--drops 500] [--prbs 4] [--workers 48] [--out DIR]

每个撒点 d：UE 位置、小尺度随机数在各 K 之间共用，各 K 的 LOS 与阴影独立抽取（随机数规则见 same_total.drop）。
每个 (K, PRB)：对 |S| ≥ U 的全部子集算两种功率规则的 c²，逐 (规则, 粒度, m) 取最大；胜出子集用伪逆显式验算。
分析：每个噪声值、功率规则逐 PRB 求 M̂（SE ≥ 0.98 SE_max，SE_max 取 RU 级最大值），区间按撒点配对（先在撒点内平均 PRB）。
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
from scipy import stats

import same_total as st
from run_round1 import git_info

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261124
KS = (1, 2, 4, 8)
U = 4
NOISE_DBM = (-90, -70, -50)
NOISE_MW = tuple(10 ** (x / 10) for x in NOISE_DBM)
M_FIX = 8                  # 固定激活数的对照（对每个 K 都是 RU 级可取的值）
TOL = 1e-9


def work(item):
    d, prbs = item
    nk, nr, ng, nm = len(KS), len(st.RULES), len(st.GRANS), st.M_TOT + 1
    r = {'val': np.full((nk, prbs, nr, ng, nm), -np.inf),
         'key': np.full((nk, prbs, nr, ng, nm), -1, dtype=np.int64),
         'pw_ant_max_ru_rule': np.full((nk, prbs, ng, nm), np.nan),     # 'ru' 规则下胜出子集的单天线最大功率 / P_max
         'beta_db': np.full((nk, U, st.M_TOT), np.nan), 'los': np.zeros((nk, U, st.M_TOT), dtype=bool),
         'ue': st.ue_positions(SEED, d, U), 'n_invalid': np.zeros(nk, dtype=np.int64),
         'chk': {'c2_rel': 0.0, 'resid': 0.0, 'p_over': 0, 'n_checked': 0, 'order_viol': 0}}
    ch = r['chk']
    for ki, K in enumerate(KS):
        H, beta, los, _ = st.drop(SEED, d, K, U, prbs)
        n = st.M_TOT // K
        r['beta_db'][ki, :, :n] = 10 * np.log10(beta)
        r['los'][ki, :, :n] = los
        keys = st.candidates(K, U)[0]
        for b in range(prbs):
            c2, _, ninv = st.score(H[b], K)
            r['n_invalid'][ki] += ninv
            val, idx = st.best_per_m(c2, K, U)
            r['val'][ki, b] = val
            r['key'][ki, b] = np.where(idx >= 0, keys[np.maximum(idx, 0)], -1)
            fin = np.isfinite(val[:, 0])
            ch['order_viol'] += int(np.sum(val[:, 1][fin] < val[:, 0][fin] * (1 - TOL)))
            for ri in range(nr):
                for gi in range(ng):
                    for m in range(U, nm):
                        if idx[ri, gi, m] < 0:
                            continue
                        q, c2x, resid = st.explicit(H[b], keys[idx[ri, gi, m]], K)
                        ch['c2_rel'] = max(ch['c2_rel'], abs(c2x[ri] / val[ri, gi, m] - 1))
                        ch['resid'] = max(ch['resid'], resid)
                        ch['n_checked'] += 1
                        p = c2x[ri] * q
                        over = p.sum() > m * st.P_MAX_MW * (1 + TOL)
                        if ri == 0:
                            over |= p.max() > st.P_MAX_MW * (1 + TOL)
                        else:
                            over |= p.reshape(-1, K).sum(1).max() > K * st.P_MAX_MW * (1 + TOL)
                            r['pw_ant_max_ru_rule'][ki, b, gi, m] = p.max() / st.P_MAX_MW
                        ch['p_over'] += int(over)
    return d, r


def dstats(x):
    """按撒点的配对统计：x [D]。"""
    n = len(x)
    m = float(np.mean(x))
    half = float(stats.t.ppf(0.975, n - 1) * np.std(x, ddof=1) / np.sqrt(n))
    return {'mean': m, 'ci95': [m - half, m + half], 'median': float(np.median(x)), 'std': float(np.std(x, ddof=1))}


def rows_from(R):
    val, key, pw = R['val'], R['key'], R['pw_ant_max_ru_rule']          # val [D, nK, B, nR, nG, 17]
    D, _, B = val.shape[:3]
    rows = []
    for ri, rule in enumerate(st.RULES):
        for ndbm, noise in zip(NOISE_DBM, NOISE_MW):
            mh, c2max = st.m_hat(val[:, :, :, ri], noise)                  # [D, nK, B, nG]、[D, nK, B]
            se_max = st.sum_se(c2max, noise, U)
            se_allon = st.sum_se(val[:, :, :, ri, 0, st.M_TOT], noise, U)
            allon_is_max = np.argmax(val[:, :, :, ri, 0, :], axis=-1) == st.M_TOT
            for ki, K in enumerate(KS):
                mru, mant = mh[:, ki, :, 0].astype(float), mh[:, ki, :, 1].astype(float)
                delta = mru - mant
                part, act = np.zeros((D, B)), np.zeros((D, B))
                for d in range(D):
                    for b in range(B):
                        cnt = st.ru_counts(key[d, ki, b, ri, 1, int(mant[d, b])], K)
                        part[d, b] = np.sum((cnt > 0) & (cnt < K))
                        act[d, b] = np.sum(cnt > 0)
                se8 = {g: st.sum_se(val[:, ki, :, ri, gi, M_FIX], noise, U) for gi, g in enumerate(st.GRANS)}
                curve = {}
                for gi, g in enumerate(st.GRANS):
                    v = val[:, ki, :, ri, gi, :]
                    curve[g] = [float(st.sum_se(v[..., m], noise, U).mean()) if np.all(np.isfinite(v[..., m])) else None
                                for m in range(st.M_TOT + 1)]
                row = {
                    'K': K, 'N': st.M_TOT // K, 'rule': rule, 'noise_dbm': ndbm, 'n_drops': int(D), 'n_prb': int(B),
                    'se_max': {**dstats(se_max[:, ki].mean(1)),
                               'ratio_to_K1': float(se_max[:, ki].mean() / se_max[:, 0].mean()),
                               'diff_to_K1': dstats((se_max[:, ki] - se_max[:, 0]).mean(1))},
                    'se_allon': {'mean': float(se_allon[:, ki].mean()),
                                 'frac_allon_is_max': float(allon_is_max[:, ki].mean())},
                    'mhat': {'RU': dstats(mru.mean(1)), 'ant': dstats(mant.mean(1))},
                    'delta_mhat': {**dstats(delta.mean(1)),
                                   'rel_pct_of_RU': float(100 * delta.mean() / mru.mean()),
                                   'frac_pos_prb': float(np.mean(delta > 0)),
                                   'dist_prb': {str(int(k)): v / delta.size for k, v in sorted(Counter(delta.ravel()).items())}},
                    'saving_vs_allon': {'RU': float(st.M_TOT - mru.mean()), 'ant': float(st.M_TOT - mant.mean())},
                    'partial_ru_at_mhat_ant': {'mean_n_partial': float(part.mean()), 'frac_any_partial': float(np.mean(part > 0)),
                                               'mean_n_active_ru': float(act.mean())},
                    'mhat_dist_prb': {g: {str(int(k)): v / mh[:, ki, :, gi].size
                                          for k, v in sorted(Counter(mh[:, ki, :, gi].ravel()).items())}
                                      for gi, g in enumerate(st.GRANS)},
                    'se_m8': {'RU': float(se8['RU'].mean()), 'ant': float(se8['ant'].mean()),
                              'diff': dstats((se8['ant'] - se8['RU']).mean(1)),
                              'rel_pct': float(100 * (se8['ant'].mean() - se8['RU'].mean()) / se8['RU'].mean())},
                    'curve_mean_se': curve,
                }
                if rule == 'ru':
                    pa = np.take_along_axis(pw[:, ki, :, 1, :], mant.astype(int)[..., None], -1)[..., 0]
                    pr = np.take_along_axis(pw[:, ki, :, 0, :], mru.astype(int)[..., None], -1)[..., 0]
                    row['per_antenna_power_over_pmax'] = {
                        g: {'mean': float(x.mean()), 'p90': float(np.quantile(x, 0.9)), 'max': float(x.max()),
                            'frac_above_1': float(np.mean(x > 1 + 1e-9))} for g, x in (('RU', pr), ('ant', pa))}
                rows.append(row)
    return rows


def ci(d, f='.2f'):
    return f"{d['mean']:{f}} [{d['ci95'][0]:{f}}, {d['ci95'][1]:{f}}]"


def summary(rows, meta):
    L = ['# E8 同总天线数：单天线分布式 RU 与共址多天线 RU（ZF，理想 CSI，16 根天线，U = 4）', '',
         'M̂：满足 SE ≥ 0.98·SE_max 的最少激活天线数（SE_max 取 RU 级激活的最大值），逐 PRB 求解后平均。'
         f"ΔM̂ = M̂_RU − M̂_ant。区间为按撒点配对的 95% t 区间（{meta['params']['drops']} 个撒点，"
         f"先在撒点内平均 {meta['params']['prbs_per_drop']} 个 PRB）。"
         'SE 单位 bit/s/Hz（4 个 UE 之和）。"相对 K=1" = SE_max(K) / SE_max(K = 1)。'
         '"部分激活 RU" = M̂_ant 解中只开了部分天线的 RU 个数（平均）。', '']
    names = {'ant': "功率规则 'ant'（逐天线 P_max；判断潜力以此为准）",
             'ru': "功率规则 'ru'（论文式 (norm)：每 RU K·P_max）"}
    for rule in st.RULES:
        L += [f'## {names[rule]}', '',
              '| 噪声 (dBm) | K | N | SE_max | 相对 K=1 | 全开 SE | M̂_RU | M̂_ant | ΔM̂ | ΔM̂ > 0 | 部分激活 RU | m = 8：SE_ant − SE_RU |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for r in rows:
            if r['rule'] != rule:
                continue
            L.append(f"| {r['noise_dbm']} | {r['K']} | {r['N']} | {r['se_max']['mean']:.2f} | {r['se_max']['ratio_to_K1']:.3f} | "
                     f"{r['se_allon']['mean']:.2f} | {r['mhat']['RU']['mean']:.2f} | {r['mhat']['ant']['mean']:.2f} | "
                     f"{ci(r['delta_mhat'])} | {r['delta_mhat']['frac_pos_prb']:.2f} | "
                     f"{r['partial_ru_at_mhat_ant']['mean_n_partial']:.2f} | {ci(r['se_m8']['diff'], '.3f')} |")
        L += ['', '潜力对照：单天线分布式分组相对全开节省的激活天线数（16 − M̂_D1），与各 K 下逐天线激活比整 RU 多省的 ΔM̂。', '',
              '| 噪声 (dBm) | 16 − M̂_D1 | ΔM̂ (K = 2) | ΔM̂ (K = 4) | ΔM̂ (K = 8) | M̂_ant (K = 2 / 4 / 8) | M̂_RU (K = 2 / 4 / 8) |',
              '|---|---|---|---|---|---|---|']
        for ndbm in NOISE_DBM:
            sub = {r['K']: r for r in rows if r['rule'] == rule and r['noise_dbm'] == ndbm}
            L.append(f"| {ndbm} | {sub[1]['saving_vs_allon']['ant']:.2f} | "
                     + ' | '.join(f"{sub[K]['delta_mhat']['mean']:.2f}" for K in (2, 4, 8)) + ' | '
                     + ' / '.join(f"{sub[K]['mhat']['ant']['mean']:.2f}" for K in (2, 4, 8)) + ' | '
                     + ' / '.join(f"{sub[K]['mhat']['RU']['mean']:.2f}" for K in (2, 4, 8)) + ' |')
        if rule == 'ru':
            L += ['', "'ru' 规则下胜出子集的单天线最大功率 / P_max（M̂ 处；> 1 表示超过单个功放的上限）", '',
                  '| 噪声 (dBm) | K | RU 级 均值 / p90 / 最大 | 天线级 均值 / p90 / 最大 | 天线级 > 1 的比例 |', '|---|---|---|---|---|']
            for r in rows:
                if r['rule'] != 'ru' or r['K'] == 1:
                    continue
                p = r['per_antenna_power_over_pmax']
                L.append(f"| {r['noise_dbm']} | {r['K']} | {p['RU']['mean']:.2f} / {p['RU']['p90']:.2f} / {p['RU']['max']:.2f} | "
                         f"{p['ant']['mean']:.2f} / {p['ant']['p90']:.2f} / {p['ant']['max']:.2f} | {p['ant']['frac_above_1']:.2f} |")
        L.append('')
    c = meta['checks']
    L += [f"检查：显式验算 {c['n_checked']} 个胜出子集，c² 最大相对差 {c['c2_rel']:.1e}，ZF 残差 {c['resid']:.1e}，"
          f"功率越界 {c['p_over']}，天线级 < RU 级 {c['order_viol']}，秩不足候选 {c['n_invalid']}。"
          f"耗时 {meta['timing_s']:.1f} s。", '']
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drops', type=int, default=500)
    ap.add_argument('--prbs', type=int, default=4)
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'same_total_20261124'))
    args = ap.parse_args()
    out = args.out
    os.makedirs(os.path.join(out, 'src'), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(os.path.join(out, 'run.log')), logging.StreamHandler(sys.stdout)])
    log = logging.getLogger()
    for f in ('same_total.py', 'run_same_total.py', 'test_same_total.py', 'run_round1.py'):
        shutil.copy2(os.path.join(HERE, f), os.path.join(out, 'src'))
    meta = {
        'experiment': 'E8 same total antennas: single-antenna distributed RUs vs co-located K-antenna RUs (ZF, exhaustive)',
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'command': ' '.join([os.path.basename(sys.executable)] + sys.argv),
        'git': git_info(),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        'host_cpus': os.cpu_count(), 'workers': args.workers,
        'params': {'M_total': st.M_TOT, 'U': U, 'K': KS, 'N': [st.M_TOT // k for k in KS], 'drops': args.drops,
                   'prbs_per_drop': args.prbs, 'noise_dbm_per_prb': NOISE_DBM, 'P_max_mw_per_antenna_per_prb': st.P_MAX_MW,
                   'kappa': st.KAPPA, 'fc_ghz': st.FC_GHZ, 'h_bs_m': st.H_BS, 'h_ut_m': st.H_UT,
                   'antenna_gain_dbi': st.ANT_GAIN_DBI, 'area_m': st.AREA, 'm_fixed_compare': M_FIX},
        'layout': ('4x4 hexagonal lattice, 40 m spacing (lower-left block of the cfsim/EuCNC 8x8 lattice); RU r = antennas '
                   'r*K..r*K+K-1 (EuCNC grouping), co-located at the centroid of their lattice points; K = 1: '
                   'single-antenna distributed RUs'),
        'channel': ('beta[u, site]: ITU-R M.2135-1 UMi path loss, UMi LOS probability, log-normal shadowing 3/4 dB (LOS/NLOS), '
                    '2.15 dBi, 2D distance >= 10 m; shared by the K antennas of an RU; small scale i.i.d. CN(0,1) per '
                    'antenna and PRB; perfect CSI; ideal fronthaul; equal noise for all UEs'),
        'seed_rule': (f'UE positions default_rng([{SEED}, d, 0]); LOS + shadowing of deployment K default_rng([{SEED}, d, 1, K]) '
                      f'(LOS uniforms first, then shadowing normals, as cfsim.drop); small scale default_rng([{SEED}, d, 2]) '
                      '[prbs, U, 16] real then imaginary, shared by all K'),
        'design': {
            'precoder': 'ZF on the active set: W0 = H_S^H (H_S H_S^H)^-1, q_j = ||row j of W0||^2',
            'power_rules': {'ant': 'c2 = P_max / max_j q_j (per-antenna PA; K = 1: [10] eq. (18))',
                            'ru': 'c2 = min(m P_max / sum q, K P_max / max_n q_n) (paper eq. (norm))'},
            'granularity': {'RU': 'every RU all-on or all-off (S_RU)', 'ant': 'any subset (S_ant)'},
            'search': 'exhaustive over all subsets with |S| >= U (64,839); best c2 per (rule, granularity, m); ties -> smallest bitmask',
            'se': 'sum SE = U log2(1 + c2 / noise) (perfect CSI ZF, equal noise)',
            'metric': ('per PRB: SE_max = max over RU-level activations; M_hat_g = min m with SE >= kappa SE_max; '
                       'averaged over PRBs; paired CI over drops (PRBs averaged within a drop)'),
            'rank_check': f'candidate invalid if min eig(G) <= {st.RANK_TOL} max eig(G)',
        },
        'checks': {}, 'timing_s': None,
    }
    t0 = time.time()
    res = [None] * args.drops
    with ProcessPoolExecutor(args.workers) as ex:
        for d, r in ex.map(work, [(d, args.prbs) for d in range(args.drops)], chunksize=2):
            res[d] = r
    R = {k: np.stack([r[k] for r in res]) for k in res[0] if k != 'chk'}
    chk = {k: (max(r['chk'][k] for r in res) if isinstance(res[0]['chk'][k], float) else sum(r['chk'][k] for r in res))
           for k in res[0]['chk']}
    chk['n_invalid'] = int(R['n_invalid'].sum())
    meta['checks'] = chk
    meta['timing_s'] = time.time() - t0
    log.info(f"simulation done in {meta['timing_s']:.1f} s; checks {chk}")
    np.savez_compressed(os.path.join(out, 'samples.npz'), K=np.array(KS), rules=np.array(st.RULES),
                        grans=np.array(st.GRANS), noise_dbm=np.array(NOISE_DBM), **R)
    rows = rows_from(R)
    with open(os.path.join(out, 'rows.jsonl'), 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(os.path.join(out, 'summary.md'), 'w') as fh:
        fh.write(summary(rows, meta))
    meta['status'] = 'done'
    with open(os.path.join(out, 'metadata.json'), 'w') as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    for r in rows:
        log.info(f"rule={r['rule']:<3} noise={r['noise_dbm']} K={r['K']}  SE_max={r['se_max']['mean']:.2f} "
                 f"M̂_RU={r['mhat']['RU']['mean']:.2f} M̂_ant={r['mhat']['ant']['mean']:.2f} ΔM̂={ci(r['delta_mhat'])}")
    log.info('all done')


if __name__ == '__main__':
    main()
