"""跨 PRB 逐用户 ZF 速率交换：第一轮复现（委托：同目录 claude_brief.md）。

运行（cwd = 项目根 /home/user/Sionna-based Simulation）：
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -B \
      "studies/Dynamic to minimize the fronthaul load/Validation/cross_prb_zf/claude_replay.py"
输出：同目录 claude_replay.json。输入按脚本自身位置查找，与 cwd 无关；cwd 与线程环境记入 meta。

数据：../results/same_total_20261124/samples.npz 的 val、key，维度 [drop, K_index, PRB, power_rule, granularity, m]，
  取 K_index = 0（K = 1，16 个单天线分布式 RU）、power_rule = 0（'ant'，逐天线 10 mW）、granularity = 0（'RU'；
  K = 1 时与 'ant' 相同，脚本核对）、m = 4..16。真信道 H[b, u, a] = same_total.drop(20261124, d, 1, 4, 4)[0]。
  500 个布局 × 4 PRB，4 个 UE 全缓冲。

基线（逐 PRB b，与 E8 的 M̂ 定义一致）：
  gmax_b = max_{m=4..16} val[b, m]；r_b = 0.98 log2(1 + gmax_b / noise) 为每个 UE 的目标；t_b = noise (2^r_b − 1)。
  m*_b = 用归档掩码 key[b, m]、W = pinv(H_S) diag(sqrt(t_b)) 时通过 feasible(cap = ∞) 的最小 m；
  cap_b = 该 W 的实发总功率（原波束降到恰好达标）。
候选：四个 PRB 的 3 种两两配对 {01|23, 02|13, 03|12}；每对块 (i, j) 选一对 UE (u < v) 与 d：
  块 i 上 (u, v) 的速率加 (+d, −d)，块 j 上加 (−d, +d)；d ∈ {0} ∪ {±min(r_i, r_j)·f : f ∈ {1/32, 1/16, 1/8, 1/4}}。
  每个 UE 跨 4 块的总速率、每块 4 个 UE 的速率和都不变。每块每个 m 用归档掩码重算
  W = pinv(H_S) diag(sqrt(noise (2^rate − 1)))，取通过 feasible(cap_b) 的最小 m；目标为四块 m 之和
  （天线流 = RU–PRB 数）最小。d = 0 必须回到 m*（断言）；并列时依次优先 d = 0、|d| 小、先列的配对。
门槛 feasible()（基线与候选共用同一函数）：max_j p_j ≤ 10 mW·(1 + 1e-9) 且 Σ_j p_j ≤ cap_b·(1 + 1e-9)。
负载：7-1 固定宽度天线 IQ，前传负载 ∝ RU–PRB 数（K = 1 时即各 PRB 激活天线数之和），不含压缩。
核查：每个布局都对所选方案与基线显式构造 16 × U 复数 W，算真实 H @ W 与 SINR，检查分配速率、两个方向的速率守恒
  （每个 UE 跨 PRB、每个 PRB 跨 UE）、逐 PRB 总功率与逐天线上限；前 N_BRUTE 个布局与前 N_BRUTE 个胜出布局另做
  独立穷举（逐点 pinv，不用预存的 |pinv|² 矩阵，3 种配对 × 两对选择的笛卡尔积）。
"""

import os

_THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')
_ENV_GIVEN = {v: os.environ.get(v) for v in _THREAD_VARS}
for _v in _THREAD_VARS:
    os.environ[_v] = '1'                       # OpenBLAS 单线程（在 import numpy 之前）

import hashlib                                 # noqa: E402
import itertools                               # noqa: E402
import json                                    # noqa: E402
import platform                                # noqa: E402
import subprocess                              # noqa: E402
import sys                                     # noqa: E402
import time                                    # noqa: E402
from collections import Counter                # noqa: E402

import numpy as np                             # noqa: E402
import scipy                                   # noqa: E402
from scipy import stats                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
VAL_DIR = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, VAL_DIR)
import same_total as st                        # noqa: E402

E8_DIR = os.path.join(VAL_DIR, 'results', 'same_total_20261124')
SAMPLES = os.path.join(E8_DIR, 'samples.npz')
E8_ROWS = os.path.join(E8_DIR, 'rows.jsonl')
OUT = os.path.join(HERE, 'claude_replay.json')
SCRIPT_REL = os.path.relpath(os.path.abspath(__file__), ROOT)
COMMAND = (f'(cwd = project root) OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '
           f'.venv/bin/python -B "{SCRIPT_REL}"')

SEED = 20261124
K, KI, RI, GI = 1, 0, 0, 0
U, B = 4, 4
M_RANGE = tuple(range(U, st.M_TOT + 1))        # 4..16
NO_M = 99                                      # 不可行标记（两块之和 > 任何可行总数）
NOISE_DBM = (-90, -70, -50)
KAPPA = 0.98
P_ANT = 10.0                                   # mW，每根天线每 PRB
TOL = 1e-9                                     # 容量上限检查的相对浮点容差
FRACS = (1 / 32, 1 / 16, 1 / 8, 1 / 4)
MATCHINGS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
UE_PAIRS = tuple(itertools.combinations(range(U), 2))
N_BRUTE = 50
N_EXAMPLES = 5
CHECK_ATOL = 1e-6                              # 显式核查的速率 / 守恒误差上限（bit/s/Hz）


def feasible(p, cap):
    """唯一门槛，基线与候选共用。p [..., m]：各激活天线实发功率 (mW)；cap：该 PRB 的总功率上限 (mW)。"""
    return (p.max(-1) <= P_ANT * (1 + TOL)) & (p.sum(-1) <= cap * (1 + TOL))


def need_c2(rate, noise):
    """速率 rate (bit/s/Hz) 所需的 ZF 有效增益 |[H W]_uu|² = noise (2^rate − 1)，mW。"""
    return noise * (2.0 ** np.asarray(rate, dtype=float) - 1.0)


def active(key):
    """位掩码 → 激活天线下标（升序）。"""
    return np.flatnonzero((int(key) >> np.arange(st.M_TOT)) & 1)


def min_feasible_m(A2, T, cap):
    """A2 {m: |pinv(H_S)|² [m, U]}（归档掩码），T [n, U] 各 UE 所需 c² → 每行可行的最小 m [n]（无则 NO_M）。
    W = pinv(H_S) diag(sqrt(T)) 时天线 j 的功率 p_j = Σ_u |pinv_ju|² T_u，即 T @ A2[m].T。"""
    out = np.full(len(T), NO_M, dtype=np.int64)
    left = np.ones(len(T), dtype=bool)
    for m in M_RANGE:
        ok = left & feasible(T @ A2[m].T, cap)
        out[ok] = m
        left &= ~ok
        if not left.any():
            break
    return out


def swap_set(ri, rj):
    """块对 (i, j) 的全部选择：描述 [None | (u, v, 有符号 f, d)]（第 0 个为 d = 0）与块 i 的速率增量 Δ [n, U]
    （块 j 的增量为 −Δ）。顺序：d = 0，然后 f 升序、UE 对字典序、+ 先于 −。"""
    base = min(ri, rj)
    desc, delta = [None], [np.zeros(U)]
    for f in FRACS:
        for u, v in UE_PAIRS:
            for s in (1, -1):
                d = s * f * base
                row = np.zeros(U)
                row[u], row[v] = d, -d
                desc.append((u, v, s * f, float(d)))
                delta.append(row)
    return desc, np.array(delta)


def baseline(val_b, A2, noise):
    """val_b [B, 17]、A2[b] → r、t、m*、cap，以及直接按 val 判定的 m（相对容差 1e-12 即 E8 的写法，与 1e-9）。"""
    gmax = val_b[:, U:].max(1)
    r = KAPPA * np.log2(1 + gmax / noise)
    t = need_c2(r, noise)
    m_star = np.array([min_feasible_m(A2[b], np.full((1, U), t[b]), np.inf)[0] for b in range(B)], dtype=np.int64)
    assert np.all(m_star <= st.M_TOT), 'baseline target unreachable'
    cap = np.array([float(np.sum(A2[b][m_star[b]] @ np.full(U, t[b]))) for b in range(B)])
    v = val_b[:, U:]
    m_val12 = np.array(M_RANGE)[np.argmax(v >= t[:, None] * (1 - 1e-12), axis=1)]
    m_val9 = np.array(M_RANGE)[np.argmax(v >= t[:, None] * (1 - TOL), axis=1)]
    return r, t, m_star, cap, m_val12, m_val9


def optimise(r, cap, m_star, A2, noise):
    """三种配对 × 每对 49 种选择 × 每块最小可行 m。各对互不相交，所以每对单独取最优再比配对即为全局最优。"""
    best = {}
    for mt in MATCHINGS:
        for i, j in mt:
            desc, delta = swap_set(r[i], r[j])
            Ri, Rj = r[i] + delta, r[j] - delta
            mi = min_feasible_m(A2[i], need_c2(Ri, noise), cap[i])
            mj = min_feasible_m(A2[j], need_c2(Rj, noise), cap[j])
            assert mi[0] == m_star[i] and mj[0] == m_star[j], 'd = 0 must fall back to the baseline'
            cost = mi + mj
            k = int(np.argmin(cost))                                   # 并列取第一个（d = 0 优先）
            best[(i, j)] = {'cost': int(cost[k]), 'desc': desc[k], 'm': (int(mi[k]), int(mj[k])),
                            'rates': (Ri[k].copy(), Rj[k].copy())}
    totals = [sum(best[p]['cost'] for p in mt) for mt in MATCHINGS]
    w = int(np.argmin(totals))                                         # 并列取先列的配对
    m_sel = m_star.copy()
    rates = np.repeat(r[:, None], U, axis=1)
    swaps = []
    for i, j in MATCHINGS[w]:
        e = best[(i, j)]
        m_sel[i], m_sel[j] = e['m']
        rates[i], rates[j] = e['rates']
        if e['desc'] is not None:
            u, v, sf, d = e['desc']
            swaps.append({'prbs': [i, j], 'ues': [u, v], 'signed_frac': sf, 'd': d})
    return {'total': int(totals[w]), 'matching': w, 'm': m_sel, 'rates': rates, 'swaps': swaps}


def brute_force(H, key_b, r, cap, noise):
    """独立核对：逐 (块, 速率向量) 直接 pinv 求最小可行 m（不用 A2），3 种配对 × 两对选择的笛卡尔积取最小总数。"""
    memo = {}

    def m_of(b, rates):
        k = (b, rates)
        if k not in memo:
            tt = need_c2(np.array(rates), noise)
            memo[k] = NO_M
            for m in M_RANGE:
                W = np.linalg.pinv(H[b][:, active(key_b[b, m])]) @ np.diag(np.sqrt(tt))
                if feasible(np.sum(np.abs(W) ** 2, axis=1), cap[b]):
                    memo[k] = m
                    break
        return memo[k]

    best = None
    for mt in MATCHINGS:
        options = []
        for i, j in mt:
            _, delta = swap_set(r[i], r[j])
            options.append([m_of(i, tuple(r[i] + dl)) + m_of(j, tuple(r[j] - dl)) for dl in delta])
        for combo in itertools.product(*options):
            tot = sum(combo)
            best = tot if best is None else min(best, tot)
    return int(best)


def direct_check(H, key_b, m_sel, rates, r, cap, noise, detail=False):
    """真实复数信道上逐块构造 W [16, U]（未激活天线为 0），算 Y = H @ W（第 u 行 = UE u 收到的各流）、SINR 与功率。"""
    meas = np.zeros((B, U))
    out = {'ant_power_over_10mW_max': 0.0, 'prb_power_over_cap_max': 0.0, 'interference_over_noise_max': 0.0,
           'n_power_violations': 0}
    off = 1.0 - np.eye(U)
    blocks = []
    for b in range(B):
        S = active(key_b[b, m_sel[b]])
        assert len(S) == m_sel[b]
        tt = need_c2(rates[b], noise)
        W = np.zeros((st.M_TOT, U), dtype=complex)
        W[S] = np.linalg.pinv(H[b][:, S]) @ np.diag(np.sqrt(tt))
        g = np.abs(H[b] @ W) ** 2
        sig = np.diag(g).copy()
        intf = (g * off).sum(1)
        sinr = sig / (intf + noise)
        meas[b] = np.log2(1 + sinr)
        p = np.sum(np.abs(W) ** 2, axis=1)                             # [16]，未激活为 0
        out['ant_power_over_10mW_max'] = max(out['ant_power_over_10mW_max'], float(p.max() / P_ANT))
        out['prb_power_over_cap_max'] = max(out['prb_power_over_cap_max'], float(p.sum() / cap[b]))
        out['interference_over_noise_max'] = max(out['interference_over_noise_max'], float(intf.max() / noise))
        out['n_power_violations'] += int(not feasible(p, cap[b]))
        if detail:
            blocks.append({'prb': b, 'm': int(m_sel[b]), 'active_antennas': S.tolist(),
                           'rates_assigned': rates[b].tolist(), 'rates_measured': meas[b].tolist(),
                           'sinr_db_measured': (10 * np.log10(sinr)).tolist(),
                           'interference_over_noise': (intf / noise).tolist(),
                           'p_ant_mw': p[S].tolist(), 'p_total_mw': float(p.sum()), 'cap_mw': float(cap[b]),
                           'p_ant_max_over_10mW': float(p.max() / P_ANT), 'p_total_over_cap': float(p.sum() / cap[b])})
    out['rate_err_max'] = float(np.max(np.abs(meas - rates)))
    out['ue_conservation_err_max'] = float(np.max(np.abs(meas.sum(0) - r.sum())))     # 每个 UE 跨 4 个 PRB
    out['prb_conservation_err_max'] = float(np.max(np.abs(meas.sum(1) - U * r)))      # 每个 PRB 跨 4 个 UE
    return out, meas, blocks


def agg(acc, x):
    acc['n'] = acc.get('n', 0) + 1
    for k, v in x.items():
        acc[k] = acc.get(k, 0) + v if k == 'n_power_violations' else max(acc.get(k, 0.0), v)


def check_ok(acc):
    return acc.get('n', 0) == 0 or (acc['n_power_violations'] == 0 and acc['rate_err_max'] < CHECK_ATOL
                                    and acc['ue_conservation_err_max'] < CHECK_ATOL
                                    and acc['prb_conservation_err_max'] < CHECK_ATOL)


def dstats(x):
    """按布局的配对统计（每个布局一个数），95% t 区间。"""
    x = np.asarray(x, dtype=float)
    n, mu = len(x), float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return {'mean': mu, 'ci95': [mu - half, mu + half], 'std': sd, 'median': float(np.median(x)), 'n': n}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git_head():
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=HERE, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:                                                  # noqa: BLE001
        return None


def blas_info():
    try:
        from threadpoolctl import threadpool_info
        return [{k: x.get(k) for k in ('internal_api', 'num_threads', 'version')} for x in threadpool_info()]
    except Exception:                                                  # noqa: BLE001
        return None


def main():
    t0 = time.time()
    assert P_ANT == st.P_MAX_MW and KAPPA == st.KAPPA
    z = np.load(SAMPLES)
    assert int(z['K'][KI]) == K and str(z['rules'][RI]) == 'ant' and str(z['grans'][GI]) == 'RU'
    assert tuple(int(x) for x in z['noise_dbm']) == NOISE_DBM
    val_all, key_all = z['val'], z['key']
    D = val_all.shape[0]
    assert val_all.shape[1:] == (len(z['K']), B, len(z['rules']), len(z['grans']), st.M_TOT + 1)
    val = val_all[:, KI, :, RI, GI, :]                                 # [D, B, 17]
    key = key_all[:, KI, :, RI, GI, :]
    gran_same = bool(np.array_equal(key, key_all[:, KI, :, RI, 1, :]))
    assert np.all(np.isfinite(val[:, :, U:])) and np.all(key[:, :, U:] > 0)

    # 1) 重建真信道与归档掩码的 |pinv(H_S)|²，核对归档 val（'ant' 规则 c² = 10 / max_j q_j）
    Hs, A2s = [], []
    rec = {'n_masks': 0, 'mask_size_mismatch': 0, 'c2_rel_err_max': 0.0, 'zf_resid_max': 0.0,
           'key_RU_equals_key_ant_for_K1': gran_same}
    for d in range(D):
        H = st.drop(SEED, d, K, U, B)[0]
        assert H.shape == (B, U, st.M_TOT)
        A2 = []
        for b in range(B):
            Ab = {}
            for m in M_RANGE:
                S = active(key[d, b, m])
                rec['mask_size_mismatch'] += int(len(S) != m)
                P = np.linalg.pinv(H[b][:, S])
                Ab[m] = np.abs(P) ** 2
                c2 = P_ANT / Ab[m].sum(1).max()
                rec['c2_rel_err_max'] = max(rec['c2_rel_err_max'], float(abs(c2 / val[d, b, m] - 1)))
                rec['zf_resid_max'] = max(rec['zf_resid_max'], float(np.abs(H[b][:, S] @ P - np.eye(U)).max()))
                rec['n_masks'] += 1
            A2.append(Ab)
        Hs.append(H)
        A2s.append(A2)
    t_rec = time.time() - t0
    rec_ok = rec['mask_size_mismatch'] == 0 and rec['c2_rel_err_max'] < 1e-6 and gran_same

    e8 = {}
    if os.path.exists(E8_ROWS):
        with open(E8_ROWS) as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get('K') == 1 and row.get('rule') == 'ant':
                    e8[int(row['noise_dbm'])] = float(row['mhat']['ant']['mean'])

    results, checks, per_drop, examples = {}, {}, {}, {}
    all_ok = rec_ok
    for ndbm in NOISE_DBM:
        noise = 10 ** (ndbm / 10)
        tot_b, tot_c = np.zeros(D, dtype=np.int64), np.zeros(D, dtype=np.int64)
        mb, mc = np.zeros((D, B), dtype=np.int64), np.zeros((D, B), dtype=np.int64)
        ck = {'m_star_vs_val_tol1e-12_mismatch_prb': 0, 'm_star_vs_val_tol1e-9_mismatch_prb': 0,
              'brute_force': {'n_drops': 0, 'n_winner_drops': 0, 'mismatch': 0},
              'direct_selected_all': {}, 'direct_selected_winners': {}, 'direct_baseline_all': {},
              'winners_using_tolerance_slack': 0}
        recs, ex = [], []
        n_win, n_bf_win = 0, 0
        frac_used, pair_used, n_swaps = Counter(), Counter(), Counter()
        for d in range(D):
            H, A2, kd = Hs[d], A2s[d], key[d]
            r, t, m_star, cap, m_v12, m_v9 = baseline(val[d], A2, noise)
            ck['m_star_vs_val_tol1e-12_mismatch_prb'] += int(np.sum(m_v12 != m_star))
            ck['m_star_vs_val_tol1e-9_mismatch_prb'] += int(np.sum(m_v9 != m_star))
            sel = optimise(r, cap, m_star, A2, noise)
            tot_b[d], tot_c[d] = int(m_star.sum()), sel['total']
            mb[d], mc[d] = m_star, sel['m']
            assert tot_c[d] == int(sel['m'].sum()) and tot_c[d] <= tot_b[d]
            win = bool(tot_c[d] < tot_b[d])
            n_win += int(win)
            if d < N_BRUTE or (win and n_bf_win < N_BRUTE):
                ck['brute_force']['n_drops'] += 1
                if win:
                    n_bf_win += 1
                    ck['brute_force']['n_winner_drops'] += 1
                ck['brute_force']['mismatch'] += int(brute_force(H, kd, r, cap, noise) != tot_c[d])
            want_detail = win and len(ex) < N_EXAMPLES
            dc, meas, blocks = direct_check(H, kd, sel['m'], sel['rates'], r, cap, noise, detail=want_detail)
            dcb, _, _ = direct_check(H, kd, m_star, np.repeat(r[:, None], U, axis=1), r, cap, noise)
            agg(ck['direct_selected_all'], dc)
            agg(ck['direct_baseline_all'], dcb)
            item = {'drop': d, 'r': r.tolist(), 'cap_mw': cap.tolist(), 'm_base': m_star.tolist(),
                    'm_cand': sel['m'].tolist(), 'base_total': int(tot_b[d]), 'cand_total': int(tot_c[d])}
            if win:
                agg(ck['direct_selected_winners'], dc)
                if dc['ant_power_over_10mW_max'] > 1 + 1e-12 or dc['prb_power_over_cap_max'] > 1 + 1e-12:
                    ck['winners_using_tolerance_slack'] += 1
                n_swaps[len(sel['swaps'])] += 1
                for sw in sel['swaps']:
                    frac_used[f"{sw['signed_frac']:+.5f}"] += 1
                    pair_used[f"{sw['prbs'][0]}{sw['prbs'][1]}"] += 1
                item.update({'matching': [list(p) for p in MATCHINGS[sel['matching']]], 'swaps': sel['swaps'],
                             'rates': sel['rates'].tolist(), 'rates_measured': meas.tolist()})
                if want_detail:
                    for blk in blocks:
                        b = blk['prb']
                        if blk['m'] < m_star[b]:                       # 同一 m 下基线（等速率 r_b）为何不可行
                            p_uni = A2[b][blk['m']] @ np.full(U, t[b])
                            blk['uniform_rate_same_m'] = {'p_ant_max_mw': float(p_uni.max()),
                                                          'p_total_mw': float(p_uni.sum()),
                                                          'feasible': bool(feasible(p_uni, np.inf))}
                    ex.append({'drop': d, 'base_total': int(tot_b[d]), 'cand_total': int(tot_c[d]),
                               'm_base': m_star.tolist(), 'r': r.tolist(),
                               'matching': [list(p) for p in MATCHINGS[sel['matching']]], 'swaps': sel['swaps'],
                               'blocks': blocks, 'checks': dc})
            recs.append(item)

        saving = tot_b - tot_c
        dm = mc - mb
        ok = (ck['brute_force']['mismatch'] == 0 and check_ok(ck['direct_selected_all'])
              and check_ok(ck['direct_baseline_all']))
        ck['pass'] = bool(ok)
        all_ok &= ok
        e8m = e8.get(ndbm)
        results[str(ndbm)] = {
            'noise_mw': noise,
            'baseline_ru_prb_per_drop': dstats(tot_b),
            'candidate_ru_prb_per_drop': dstats(tot_c),
            'saving_ru_prb_per_drop': dstats(saving),
            'saving_rel_pct_of_mean_baseline': float(100 * saving.mean() / tot_b.mean()),
            'saving_rel_pct_per_drop': dstats(100 * saving / tot_b),
            'drops_with_saving': {'n': int(np.sum(saving > 0)), 'frac': float(np.mean(saving > 0))},
            'saving_dist': {str(k): v / D for k, v in sorted(Counter(saving.tolist()).items())},
            'mean_m_per_prb': {'baseline': float(mb.mean()), 'candidate': float(mc.mean())},
            'e8_cross_check': {'e8_mhat_ant_K1': e8m, 'e8_4_mhat_per_drop': None if e8m is None else 4 * e8m,
                               'baseline_mean_per_drop': float(tot_b.mean()),
                               'abs_diff': None if e8m is None else float(abs(tot_b.mean() - 4 * e8m))},
            'prb_m_change': {'down': float(np.mean(dm < 0)), 'up': float(np.mean(dm > 0)),
                             'same': float(np.mean(dm == 0)),
                             'dist': {str(k): v / dm.size for k, v in sorted(Counter(dm.ravel().tolist()).items())}},
            'winners': {'n': n_win, 'n_swaps': {str(k): v for k, v in sorted(n_swaps.items())},
                        'signed_frac_used': dict(sorted(frac_used.items())),
                        'prb_pair_used': dict(sorted(pair_used.items()))},
        }
        checks[str(ndbm)] = ck
        per_drop[str(ndbm)] = recs
        examples[str(ndbm)] = ex
        s = results[str(ndbm)]['saving_ru_prb_per_drop']
        e8s = f'{4 * e8m:.3f}' if e8m is not None else 'n/a'
        print(f"noise {ndbm} dBm: baseline {tot_b.mean():.3f} RU-PRB/drop (E8 4*M̂ = {e8s}), candidate {tot_c.mean():.3f}, "
              f"saving {s['mean']:.4f} [{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}] "
              f"({results[str(ndbm)]['saving_rel_pct_of_mean_baseline']:.3f} %), drops with saving {n_win}/{D}")
        print(f"   checks pass={ok}: m* vs val (1e-12 / 1e-9) {ck['m_star_vs_val_tol1e-12_mismatch_prb']} / "
              f"{ck['m_star_vs_val_tol1e-9_mismatch_prb']} PRB, brute force mismatch {ck['brute_force']['mismatch']}/"
              f"{ck['brute_force']['n_drops']} (winners {ck['brute_force']['n_winner_drops']}), "
              f"tolerance-slack winners {ck['winners_using_tolerance_slack']}")
        print(f"   direct (selected, all drops): {ck['direct_selected_all']}")

    meta = {
        'task': 'cross-PRB per-UE ZF rate swap, first replay (claude_brief.md)',
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'command': COMMAND,
        'argv': [sys.executable] + sys.argv,
        'cwd': os.getcwd(),
        'env_threads_given': _ENV_GIVEN,
        'env_threads_used': {v: os.environ.get(v) for v in _THREAD_VARS},
        'blas': blas_info(),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        'git_head': git_head(),
        'inputs': {'samples_npz': os.path.relpath(SAMPLES, ROOT), 'samples_sha256': sha256(SAMPLES),
                   'e8_rows': os.path.relpath(E8_ROWS, ROOT) if e8 else None,
                   'e8_rows_sha256': sha256(E8_ROWS) if e8 else None,
                   'same_total_py_sha256': sha256(os.path.join(VAL_DIR, 'same_total.py')),
                   'claude_replay_py_sha256': sha256(os.path.abspath(__file__))},
        'params': {'seed': SEED, 'K': K, 'drops': D, 'U': U, 'prbs': B,
                   'slice': 'val/key[:, K_index=0, :, power_rule=0 (ant), granularity=0 (RU), m=4..16]',
                   'channel': f'same_total.drop({SEED}, d, 1, {U}, {B})[0] -> H[b, u, a]',
                   'noise_dbm': NOISE_DBM, 'noise_mw': [10 ** (x / 10) for x in NOISE_DBM], 'kappa': KAPPA,
                   'p_ant_mw': P_ANT, 'tol_rel': TOL, 'd_fracs': FRACS, 'matchings': MATCHINGS, 'ue_pairs': UE_PAIRS,
                   'n_choices_per_pair': 1 + 2 * len(FRACS) * len(UE_PAIRS), 'brute_force_drops': N_BRUTE,
                   'check_atol_bits': CHECK_ATOL},
        'threshold': ('ONE function feasible(p, cap) for baseline and candidates: p_j = sum_u |pinv(H_S)_ju|^2 * '
                      'noise*(2**rate_u - 1) on the archived mask key[b, m]; feasible <=> max_j p_j <= 10 mW * (1 + 1e-9) '
                      'and sum_j p_j <= cap_b * (1 + 1e-9). Baseline m*_b = min m feasible with rate_u = r_b for all u and '
                      'cap = inf; cap_b = sum_j p_j at m*_b; rerun with cap_b and d = 0 asserted to return m*_b.'),
        'swap': ('per matched PRB pair (i, j): one UE pair (u < v) and d; rate[i, u] += d, rate[i, v] -= d, '
                 'rate[j, u] -= d, rate[j, v] += d; d in {0} U {+-min(r_i, r_j) f, f in 1/32, 1/16, 1/8, 1/4}; '
                 'objective: min sum over 4 PRBs of active antennas (RU-PRB); ties -> d = 0, smaller |d|, first matching'),
        'load_metric': '7-1 fixed-width antenna IQ: load proportional to RU-PRB count (K = 1: active antennas summed over PRBs)',
        'statistics': 'per-drop paired values (4 PRBs summed within a drop), mean and 95% Student-t interval over drops',
        'limitations': ('masks restricted to the archived equal-rate winners key[b, m] (as specified), so the candidate '
                        'search is conservative; perfect CSI, flat i.i.d. per-PRB channel, no MCS quantisation'),
        'not_done': ('multipath frequency-correlated channel, estimated CSI, true-channel SINR under estimated-CSI W, '
                     'same-MCS transport block CRC (next public check in the brief)'),
        'reconstruction_check': rec,
        'all_checks_pass': bool(all_ok),
        'timing_s': {'reconstruct': t_rec, 'total': time.time() - t0},
    }
    with open(OUT, 'w') as fh:
        json.dump({'meta': meta, 'results': results, 'checks': checks, 'examples': examples, 'per_drop': per_drop},
                  fh, ensure_ascii=False)
    print(f"reconstruction: {rec}; all checks pass: {all_ok}; wrote {os.path.relpath(OUT, ROOT)} "
          f"in {meta['timing_s']['total']:.1f} s")
    if not all_ok:
        raise SystemExit('some checks failed (see checks / meta.reconstruction_check in the JSON)')


if __name__ == '__main__':
    main()
