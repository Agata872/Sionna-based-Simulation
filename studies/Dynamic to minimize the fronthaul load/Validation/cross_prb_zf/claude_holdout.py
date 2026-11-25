"""跨 PRB 逐用户 ZF 速率交换：第二轮独立验证（holdout；委托 = 用户 2026-11-25 消息）。

运行（在本目录）：../../../../.venv/bin/python -B claude_holdout.py
输出：claude_holdout.json（同目录）。复用 ../same_total.py（E8，import）与第一轮 claude_replay.py 的门槛 / 交换集合 /
  统计函数（原样拷贝到本文件，因 claude_replay.py 在本轮期间被改过接口；运行时记录其 sha256）。

沿用第一轮：16 个单天线 RU（K = 1）、U = 4、B = 4 PRB、噪声 −70 dBm、逐天线 10 mW、逐 PRB 98 % 普通 ZF 参考、
  三种配对 × 每块对一对 UE 的速率交换 d ∈ {0} ∪ {±min(r_i, r_j) f, f ∈ {1/32, 1/16, 1/8, 1/4}}、逐 PRB TX 实发总功率匹配、
  同一门槛 feasible()：max_j p_j ≤ 10 mW (1 + 1e-9) 且 Σ_j p_j ≤ cap_b (1 + 1e-9)。全新布局 d = 0..95。

信道（布局 d）：beta = same_total.drop(20261125, d, 1, 4, 4)[1]，[U, 16]。四径 PDP [0.5, 0.3, 0.15, 0.05]，
  时延 [0, 0.1, 0.3, 0.8] µs。rng = default_rng([20261125, d, 78]) 依次生成两组抽头 g1、g2：每组先取 (4, 4, 16) 的全部实部，
  再取同形全部虚部，/ sqrt(2)。轴序取 (径 l, UE u, 天线 a)（委托未写明轴序，此为本实现的假设）。
  H[b, u, a] = sqrt(beta[u, a]) Σ_l sqrt(PDP_l) g1[l, u, a] exp(−2jπ · b · 360000 · τ_l)；E 同式用 g2（与 H 独立）。块内平坦。
估计：Hhat = (1 − eps) H + sqrt(eps (1 − eps)) E，eps ∈ {0, 0.01}；全部决策（赢家、r、m*、掩码、cap、候选）只用 Hhat。

基线（逐 PRB b，均在 Hhat[b] 上）：
  E8 score/best_per_m（规则 'ant'，粒度 'RU'；K = 1 时与 'ant' 粒度相同，已断言）→ 各 m 的共同增益 c² 赢家掩码与 val[m]；
  gmax = max_m val[m]；r_b = 0.98 log2(1 + gmax / σ²)（每个 UE）；t_b = σ² (2^r_b − 1)；
  m*_b = 赢家掩码在 t_b 下满足逐天线 ≤ 10 mW（cap = ∞）的最小 m（pinv 计算，同第一轮）。
  winner   ：掩码 = 赢家(m*_b)，cap_b = 其 TX 总功率 Σ_j p_j。
  lowpower ：m*_b 下全部满秩、在精确门槛 t_b 下逐天线可行的掩码里 TX 总功率最小者（并列取位掩码最小）；cap_b = 其 TX 总功率。
             选择用 E8 score 的 q_j = ||G^{-1} h_j||²，再以 pinv 重算该掩码；另用 itertools + 批量 pinv 独立核对。
候选：每块的掩码表 = 各 m 的赢家（m 升序）；lowpower 在 m*_b 处、赢家之后追加其基线掩码（回退）。给定速率向量，取表中
  第一个同时满足逐天线 10 mW 与 cap_b 的掩码（= 最小 m；同 m 时原赢家优先）。每块对 (i, j) 取 m_i + m_j 最小的行
  （并列取先：d = 0、|d| 升序、UE 对字典序、+ 先于 −）；若无严格节省，该块对用基线（基线掩码与速率 r）。
  三种配对取总数最小者（并列取先：01|23、02|13、03|12）。
评价：W_b = pinv(Hhat_S) diag(sqrt(σ² (2^R_bu − 1)))（未激活天线行为 0）。TX 功率由 W 决定；预测速率用 Hhat W，
  真速率用 H W；干扰 = 显式非对角 |·|² 之和；SE = log2(1 + SINR)，即 Shannon SE（不含 MCS / TBS / CRC）。
  每用户四块平均 SE 的变化 ΔSE_u = mean_b SE_true,cand[b, u] − mean_b SE_true,base[b, u]。
"""

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'                       # BLAS 单线程

import hashlib
import itertools
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
VAL_DIR = os.path.dirname(HERE)
sys.path.insert(0, VAL_DIR)
import same_total as st                        # noqa: E402  E8：drop / candidates / score / best_per_m

OUT = os.path.join(HERE, 'claude_holdout.json')
COMMAND = '../../../../.venv/bin/python -B claude_holdout.py  (cwd = Validation/cross_prb_zf)'

SEED = 20261125
DROPS = tuple(range(96))
TAP_STREAM = 78
PDP = np.array([0.5, 0.3, 0.15, 0.05])
DELAY_US = np.array([0.0, 0.1, 0.3, 0.8])
DELAY_S = DELAY_US * 1e-6
PRB_HZ = 360000.0
EPS = (0.0, 0.01)
NOISE_DBM = -70
NOISE = 10 ** (NOISE_DBM / 10)                 # mW
VARIANTS = ('winner', 'lowpower')
K, U, B, M = 1, 4, 4, st.M_TOT
M_RANGE = tuple(range(U, M + 1))               # 4..16
NO_M = 99                                      # 不可行标记
KAPPA = 0.98
P_ANT = 10.0                                   # mW，每根天线每 PRB
TOL = 1e-9
FRACS = (1 / 32, 1 / 16, 1 / 8, 1 / 4)
MATCHINGS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
BLOCK_PAIRS = tuple(p for mt in MATCHINGS for p in mt)
USER_PAIRS = tuple(itertools.combinations(range(U), 2))
OFFDIAG = ~np.eye(U, dtype=bool)
MAX_WORKERS = 8
assert M == 16 and P_ANT == st.P_MAX_MW and KAPPA == st.KAPPA


# ------------------------------------------------- 第一轮 claude_replay.py 的门槛与交换集合（原样拷贝，便于冻结复现）
def feasible(p, cap):
    """唯一门槛（基线与候选共用）：p [..., m] 各激活天线实发功率 (mW)。"""
    return (p.max(-1) <= P_ANT * (1 + TOL)) & (p.sum(-1) <= cap * (1 + TOL))


def target_c2(rate, noise):
    """速率 rate (bit/s/Hz) 所需的 ZF 有效增益 |[HW]_uu|² = noise (2^rate − 1)，mW。"""
    return noise * (2.0 ** np.asarray(rate, dtype=float) - 1)


def mask(key):
    return np.flatnonzero((int(key) >> np.arange(M)) & 1)


def pair_candidates(ri, rj):
    """块对 (i, j) 的候选：说明 [(u, v, sign, frac, d)]（第 0 个为 d = 0）与块 i 的速率增量 Δ [n, U]（块 j 为 −Δ）。
    顺序：d = 0，然后 |d| 升序（f 升序，UE 对字典序，+ 先于 −）。"""
    base = min(ri, rj)
    rows = [(-1, -1, 0, 0.0, 0.0)]
    for f in FRACS:
        for u, v in USER_PAIRS:
            for s in (1, -1):
                rows.append((u, v, s, f, s * f * base))
    delta = np.zeros((len(rows), U))
    for k, (u, v, _, _, dv) in enumerate(rows[1:], 1):
        delta[k, u], delta[k, v] = dv, -dv
    return rows, delta


def dstats(x):
    x = np.asarray(x, dtype=float)
    n, m, s = len(x), float(np.mean(x)), float(np.std(x, ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * s / np.sqrt(n))
    return {'mean': m, 'ci95': [m - half, m + half], 'std': s, 'median': float(np.median(x)), 'n': n}


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
    except Exception:                                                     # noqa: BLE001
        return None


def blas_info():
    try:
        from threadpoolctl import threadpool_info
        return [{k: x.get(k) for k in ('internal_api', 'num_threads', 'version')} for x in threadpool_info()]
    except Exception:                                                     # noqa: BLE001
        return None


def ek(eps):
    return f'{eps:g}'


# ---------------------------------------------------------------------------------------------- 信道
def channels(d):
    """→ 真 H [B, U, 16]、独立 E [B, U, 16]、beta [U, 16]。"""
    beta = st.drop(SEED, d, K, U, B)[1]
    assert beta.shape == (U, M)
    rng = np.random.default_rng([SEED, d, TAP_STREAM])
    taps = []
    for _ in range(2):                                                    # 先 g1（H），后 g2（E）
        re = rng.standard_normal((len(PDP), U, M))
        im = rng.standard_normal((len(PDP), U, M))
        taps.append((re + 1j * im) / np.sqrt(2))
    phase = np.exp(-2j * np.pi * np.arange(B)[:, None] * PRB_HZ * DELAY_S[None])       # [B, L]
    out = []
    for g in taps:
        gw = np.sqrt(PDP)[:, None, None] * g                              # [L, U, 16]
        f = (gw[None] * phase[:, :, None, None]).sum(1)                   # [B, U, 16]
        out.append(np.sqrt(beta)[None] * f)
    return out[0], out[1], beta


# ---------------------------------------------------------------------------------------------- 基线
def setup_block(Hb):
    """Hhat 的一个 PRB → 赢家掩码、r、t、m*、两种基线的掩码与 cap，以及核对量。"""
    keys, _, _, _, groups = st.candidates(K, U)
    c2, q, n_bad = st.score(Hb, K)
    val, idx = st.best_per_m(c2, K, U)
    assert np.array_equal(val[0, 0], val[0, 1]) and np.array_equal(idx[0, 0], idx[0, 1])     # K = 1：两粒度相同
    v, ix = val[0, 0], idx[0, 0]
    assert np.all(np.isfinite(v[U:])) and np.all(ix[U:] >= 0)
    win = {m: int(keys[ix[m]]) for m in M_RANGE}
    P2, c2_rel, zf_res = {}, 0.0, 0.0
    for m in M_RANGE:
        S = mask(win[m])
        assert len(S) == m
        Wp = np.linalg.pinv(Hb[:, S])
        P2[m] = np.abs(Wp) ** 2                                           # [m, U]
        c2_rel = max(c2_rel, abs(P_ANT / P2[m].sum(1).max() / v[m] - 1))
        zf_res = max(zf_res, float(np.max(np.abs(Hb[:, S] @ Wp - np.eye(U)))))
    gmax = float(v[U:].max())
    r = KAPPA * np.log2(1 + gmax / NOISE)
    t = float(target_c2(r, NOISE))
    tv = np.full(U, t)
    m_star = next((m for m in M_RANGE if feasible(P2[m] @ tv, np.inf)), NO_M)
    assert m_star < NO_M
    m_val = M_RANGE[int(np.argmax(v[U:] >= t * (1 - TOL)))]
    cap_w = float(np.sum(P2[m_star] @ tv))

    # lowpower：m* 下全部满秩且逐天线可行的掩码，精确门槛 t 下 TX 总功率最小（score 的 q，按位掩码升序，并列取先）
    sub = groups[('RU', m_star)]
    p = t * q[sub]                                                        # [n, 16]，未激活为 0
    feas = np.isfinite(c2[0, sub]) & feasible(p, np.inf)
    tot = np.where(feas, p.sum(1), np.inf)
    k = int(np.argmin(tot))
    assert feas[k]
    lp_key = int(keys[sub[k]])
    srt = np.sort(tot[feas])
    lp_gap = float(srt[1] / srt[0] - 1) if len(srt) > 1 else None
    Slp = mask(lp_key)
    P2lp = np.abs(np.linalg.pinv(Hb[:, Slp])) ** 2
    plp = P2lp @ tv
    assert feasible(plp, np.inf), 'lowpower mask must stay feasible with pinv'
    cap_lp = float(np.sum(plp))
    assert cap_lp <= cap_w * (1 + TOL)
    n_lower = 0                                                           # m* − 1 下按 score 可行的掩码数（应为 0）
    if m_star > U:
        sub1 = groups[('RU', m_star - 1)]
        n_lower = int(np.sum(np.isfinite(c2[0, sub1]) & feasible(t * q[sub1], np.inf)))
    return {'win': win, 'P2': P2, 'gmax': gmax, 'r': float(r), 't': t, 'm_star': int(m_star), 'm_val': int(m_val),
            'cap': {'winner': cap_w, 'lowpower': cap_lp}, 'base_key': {'winner': win[m_star], 'lowpower': lp_key},
            'P2lp': P2lp, 'lp_key': lp_key, 'lp_n_feasible': int(feas.sum()), 'lp_gap_rel': lp_gap,
            'lp_score_vs_pinv_rel': float(abs(tot[k] / cap_lp - 1)), 'n_lower_feasible': n_lower,
            'c2_rel': float(c2_rel), 'zf_res': zf_res, 'n_rank_def': int(n_bad)}


def brute_lowpower(Hb, m, t):
    """独立核对：itertools 枚举 m 子集，批量 pinv，逐天线可行者中 TX 总功率最小（并列取位掩码最小）。"""
    combos = np.array(list(itertools.combinations(range(M), m)))          # [n, m]
    A = np.transpose(Hb[:, combos], (1, 0, 2))                            # [n, U, m]
    p = t * np.sum(np.abs(np.linalg.pinv(A)) ** 2, axis=2)                # [n, m]
    ok = p.max(1) <= P_ANT * (1 + TOL)
    tot = np.where(ok, p.sum(1), np.inf)
    kmin = np.flatnonzero(tot == tot.min())
    keys = [int(np.sum(1 << combos[i])) for i in kmin]
    return min(keys), float(tot.min()), int(ok.sum())


# ---------------------------------------------------------------------------------------------- 候选
def mask_list(bk, variant):
    """块 bk 的候选掩码表 [(m, key, |pinv|² [m, U])]：m 升序，原赢家在先；lowpower 在 m* 处追加基线掩码。"""
    out = []
    for m in M_RANGE:
        out.append((m, bk['win'][m], bk['P2'][m]))
        if variant == 'lowpower' and m == bk['m_star']:
            out.append((m, bk['lp_key'], bk['P2lp']))
    return out


def first_feasible(mlist, T, cap):
    """T [n, U] 各 UE 目标 c² → 第一个可行掩码的 m [n] 与在表中的位置 [n]（无则 NO_M / −1）。"""
    n = T.shape[0]
    m_out, s_out = np.full(n, NO_M, dtype=np.int64), np.full(n, -1, dtype=np.int64)
    todo = np.ones(n, dtype=bool)
    for s, (m, _, P2) in enumerate(mlist):
        ok = feasible(T @ P2.T, cap) & todo
        m_out[ok], s_out[ok] = m, s
        todo &= ~ok
    return m_out, s_out


def optimise(blocks, variant):
    r = np.array([bk['r'] for bk in blocks])
    m_star = np.array([bk['m_star'] for bk in blocks])
    cap = np.array([bk['cap'][variant] for bk in blocks])
    base_key = [bk['base_key'][variant] for bk in blocks]
    mlists = [mask_list(bk, variant) for bk in blocks]
    best = {}
    for i, j in BLOCK_PAIRS:
        rows, delta = pair_candidates(r[i], r[j])
        Ri, Rj = r[i] + delta, r[j] - delta
        mi, si = first_feasible(mlists[i], target_c2(Ri, NOISE), cap[i])
        mj, sj = first_feasible(mlists[j], target_c2(Rj, NOISE), cap[j])
        assert mi[0] == m_star[i] and mj[0] == m_star[j], 'd = 0 must fall back to the baseline m*'
        cost = mi + mj
        k = int(np.argmin(cost))                                          # 并列取先（d = 0 优先）
        base_cost = int(m_star[i] + m_star[j])
        if cost[k] < base_cost:
            e = {'cost': int(cost[k]), 'row': rows[k], 'm': (int(mi[k]), int(mj[k])),
                 'keys': (int(mlists[i][si[k]][1]), int(mlists[j][sj[k]][1])), 'rates': (Ri[k], Rj[k])}
        else:                                                             # 无严格节省 → 基线
            e = {'cost': base_cost, 'row': rows[0], 'm': (int(m_star[i]), int(m_star[j])),
                 'keys': (base_key[i], base_key[j]), 'rates': (np.full(U, r[i]), np.full(U, r[j]))}
        e['base_cost'] = base_cost
        e['n_cand_feasible_rows'] = int(np.sum(cost < NO_M))
        best[(i, j)] = e
    totals = [best[p1]['cost'] + best[p2]['cost'] for p1, p2 in MATCHINGS]
    nswaps = [sum(int(best[p]['row'][2] != 0) for p in mt) for mt in MATCHINGS]
    mt = int(np.argmin(totals))                                           # 并列取先
    m_c, keys_c, rates = np.zeros(B, dtype=np.int64), [0] * B, np.zeros((B, U))
    swaps = []
    for i, j in MATCHINGS[mt]:
        e = best[(i, j)]
        m_c[i], m_c[j] = e['m']
        keys_c[i], keys_c[j] = e['keys']
        rates[i], rates[j] = e['rates']
        u, v, s, f, dv = e['row']
        if s != 0:
            swaps.append({'prbs': [i, j], 'ues': [u, v], 'frac': s * f, 'd': dv})
    assert int(m_c.sum()) == totals[mt] <= int(m_star.sum())
    fewer = any(totals[x] == totals[mt] and nswaps[x] < nswaps[mt] for x in range(len(MATCHINGS)))
    pairs = {f'{i}{j}': {'cost': e['cost'], 'base_cost': e['base_cost'], 'row': list(e['row']), 'm': list(e['m']),
                         'keys': list(e['keys']), 'n_rows_feasible': e['n_cand_feasible_rows']}
             for (i, j), e in best.items()}
    return {'total': int(totals[mt]), 'matching': mt, 'm': m_c, 'keys': keys_c, 'rates': rates, 'swaps': swaps,
            'matching_totals': totals, 'matching_nswaps': nswaps, 'tie_with_fewer_swaps': bool(fewer), 'pairs': pairs,
            'cap': cap, 'base_key': base_key, 'r': r, 'm_star': m_star, 'mlists': mlists}


def brute_total(Hh, mlists, r, cap):
    """独立核对：逐掩码标量 pinv（缓存），穷举 3 种配对 × 每对全部行，返回最小 RU–PRB 总数。"""
    cache = {}

    def block_m(b, rates):
        k = (b, rates)
        if k not in cache:
            t = target_c2(np.array(rates), NOISE)
            cache[k] = NO_M
            for m, key, _ in mlists[b]:
                W = np.linalg.pinv(Hh[b][:, mask(key)]) * np.sqrt(t)[None]
                if feasible(np.sum(np.abs(W) ** 2, axis=1), cap[b]):
                    cache[k] = m
                    break
        return cache[k]

    best = NO_M * B
    for mt in MATCHINGS:
        per_pair = []
        for i, j in mt:
            _, delta = pair_candidates(r[i], r[j])
            per_pair.append([(block_m(i, tuple(r[i] + dl)), block_m(j, tuple(r[j] - dl))) for dl in delta])
        for (a1, b1), (a2, b2) in itertools.product(*per_pair):
            best = min(best, a1 + b1 + a2 + b2)
    return best


# ---------------------------------------------------------------------------------------------- 评价
def evaluate(Hh, H, keys, rates, cap):
    """按 Hhat 设计 W；TX 功率、预测速率（Hhat W）、真速率（H W），干扰为显式非对角和。"""
    out = {'pmax': np.zeros(B), 'ptot': np.zeros(B), 'pred': np.zeros((B, U)), 'true': np.zeros((B, U)),
           'leak_pred': 0.0, 'leak_true': 0.0}
    for b in range(B):
        S = mask(keys[b])
        W = np.zeros((M, U), dtype=complex)
        W[S] = np.linalg.pinv(Hh[b][:, S]) * np.sqrt(target_c2(rates[b], NOISE))[None]
        p = np.sum(np.abs(W) ** 2, axis=1)
        out['pmax'][b], out['ptot'][b] = p.max(), p.sum()
        for name, Hx in (('pred', Hh[b]), ('true', H[b])):
            g = np.abs(Hx @ W) ** 2                                       # [u, v]：UE u 收到流 v
            sig = np.diag(g)
            intf = np.where(OFFDIAG, g, 0.0).sum(1)                       # 显式非对角和
            out[name][b] = np.log2(1 + sig / (intf + NOISE))
            out['leak_' + name] = max(out['leak_' + name], float(intf.max() / NOISE))
    out['pmax_ratio'] = out['pmax'] / P_ANT
    out['ptot_ratio'] = out['ptot'] / np.asarray(cap)
    return out


# ---------------------------------------------------------------------------------------------- 每个布局
def run_drop(d):
    t0 = time.time()
    H, E, beta = channels(d)
    rec = {'drop': d, 'eps': {}}
    chk = Counter()
    mx = {}

    def upd(k, v):
        mx[k] = max(mx.get(k, 0.0), float(v))

    for eps in EPS:
        Hh = (1 - eps) * H + np.sqrt(eps * (1 - eps)) * E
        if eps == 0:
            assert np.array_equal(Hh, H)
        blocks = [setup_block(Hh[b]) for b in range(B)]
        for b, bk in enumerate(blocks):
            chk['m_val_mismatch'] += int(bk['m_val'] != bk['m_star'])
            chk['n_lower_feasible'] += bk['n_lower_feasible']
            chk['n_rank_def'] += bk['n_rank_def']
            chk['lp_equals_winner'] += int(bk['lp_key'] == bk['win'][bk['m_star']])
            chk['blocks'] += 1
            upd('c2_rel', bk['c2_rel'])
            upd('zf_res', bk['zf_res'])
            upd('lp_score_vs_pinv_rel', bk['lp_score_vs_pinv_rel'])
            if bk['lp_gap_rel'] is not None:
                mx['lp_gap_rel_min'] = min(mx.get('lp_gap_rel_min', np.inf), bk['lp_gap_rel'])
            bkey, btot, bn = brute_lowpower(Hh[b], bk['m_star'], bk['t'])
            chk['lp_brute_key_mismatch'] += int(bkey != bk['lp_key'])
            chk['lp_brute_nfeas_mismatch'] += int(bn != bk['lp_n_feasible'])
            upd('lp_brute_tot_rel', abs(btot / bk['cap']['lowpower'] - 1))
        er = {'blocks': {'gmax_mw': [bk['gmax'] for bk in blocks], 'r': [bk['r'] for bk in blocks],
                         't_mw': [bk['t'] for bk in blocks], 'm_star': [bk['m_star'] for bk in blocks],
                         'winner_keys_m4_16': [[bk['win'][m] for m in M_RANGE] for bk in blocks],
                         'lowpower_key': [bk['lp_key'] for bk in blocks],
                         'cap_mw': {v: [bk['cap'][v] for bk in blocks] for v in VARIANTS},
                         'lowpower_n_feasible': [bk['lp_n_feasible'] for bk in blocks],
                         'lowpower_gap_rel': [bk['lp_gap_rel'] for bk in blocks]}}
        for variant in VARIANTS:
            sel = optimise(blocks, variant)
            r, cap = sel['r'], sel['cap']
            bt = brute_total(Hh, sel['mlists'], r, cap)
            chk[f'brute_mismatch_{variant}'] += int(bt != sel['total'])
            chk[f'tie_with_fewer_swaps_{variant}'] += int(sel['tie_with_fewer_swaps'])
            base_rates = np.repeat(r[:, None], U, axis=1)
            evb = evaluate(Hh, H, sel['base_key'], base_rates, cap)
            evc = evaluate(Hh, H, sel['keys'], sel['rates'], cap)
            se_b, se_c = evb['true'].mean(0), evc['true'].mean(0)
            pre = f'{ek(eps)}_{variant}_'
            for tag, ev, R in (('base', evb, base_rates), ('cand', evc, sel['rates'])):
                upd(pre + tag + '_pmax_ratio', ev['pmax_ratio'].max())
                upd(pre + tag + '_ptot_ratio', ev['ptot_ratio'].max())
                chk[pre + tag + '_pant_violations'] += int(np.sum(ev['pmax_ratio'] > 1 + TOL))
                chk[pre + tag + '_ptot_violations'] += int(np.sum(ev['ptot_ratio'] > 1 + TOL))
                chk[pre + tag + '_pant_above_1'] += int(np.sum(ev['pmax_ratio'] > 1))
                chk[pre + tag + '_ptot_above_1'] += int(np.sum(ev['ptot_ratio'] > 1))
                upd(pre + tag + '_pred_vs_assigned', np.max(np.abs(ev['pred'] - R)))
                upd(pre + tag + '_pred_ue_conservation', np.max(np.abs(ev['pred'].sum(0) - r.sum())))
                upd(pre + tag + '_pred_prb_conservation', np.max(np.abs(ev['pred'].sum(1) - U * r)))
                upd(pre + tag + '_assigned_ue_conservation', np.max(np.abs(R.sum(0) - r.sum())))
                upd(pre + tag + '_assigned_prb_conservation', np.max(np.abs(R.sum(1) - U * r)))
                upd(pre + tag + '_true_vs_pred', np.max(np.abs(ev['true'] - ev['pred'])))
                upd(pre + tag + '_leak_pred_over_noise', ev['leak_pred'])
                upd(pre + tag + '_leak_true_over_noise', ev['leak_true'])
            er[variant] = {
                'ru_prb_base': int(sel['m_star'].sum()), 'ru_prb_cand': sel['total'],
                'saving': int(sel['m_star'].sum() - sel['total']),
                'dse_user': (se_c - se_b).tolist(), 'se_user_base': se_b.tolist(), 'se_user_cand': se_c.tolist(),
                'base': {'keys': sel['base_key'], 'm': sel['m_star'].tolist(), 'cap_mw': cap.tolist(),
                         'pmax_mw': evb['pmax'].tolist(), 'ptot_mw': evb['ptot'].tolist(),
                         'se_true': evb['true'].tolist(), 'se_pred': evb['pred'].tolist()},
                'cand': {'keys': sel['keys'], 'm': sel['m'].tolist(), 'rates': sel['rates'].tolist(),
                         'matching': [list(p) for p in MATCHINGS[sel['matching']]], 'swaps': sel['swaps'],
                         'pmax_mw': evc['pmax'].tolist(), 'ptot_mw': evc['ptot'].tolist(),
                         'se_true': evc['true'].tolist(), 'se_pred': evc['pred'].tolist()},
                'matching_totals': sel['matching_totals'], 'matching_nswaps': sel['matching_nswaps'],
                'pairs': sel['pairs']}
        rec['eps'][ek(eps)] = er
    rec['time_s'] = time.time() - t0
    return rec, dict(chk), mx


# ---------------------------------------------------------------------------------------------- 汇总
def summarise(recs, e, variant):
    x = [rc['eps'][e][variant] for rc in recs]
    base = np.array([y['ru_prb_base'] for y in x])
    cand = np.array([y['ru_prb_cand'] for y in x])
    saving = base - cand
    dse = np.array([y['dse_user'] for y in x])                              # [D, U]
    seb = np.array([y['se_user_base'] for y in x])
    win = saving > 0
    nsw = Counter(len(y['cand']['swaps']) for y in x if y['saving'] > 0)
    fr = Counter(f"{sw['frac']:+.5f}" for y in x for sw in y['cand']['swaps'])
    out = {
        'ru_prb_base_per_drop': dstats(base), 'ru_prb_cand_per_drop': dstats(cand),
        'saving_ru_prb_per_drop': dstats(saving),
        'saving_rel_pct': float(100 * saving.mean() / base.mean()),
        'drops_with_saving': int(win.sum()), 'saving_dist': {str(k): v for k, v in sorted(Counter(saving.tolist()).items())},
        'n_swaps_in_saving_drops': {str(k): v for k, v in sorted(nsw.items())}, 'signed_frac_used': dict(sorted(fr.items())),
        'se_true_user_base_mean': float(seb.mean()),
        'dse_user_mean_per_drop': dstats(dse.mean(1)),
        'dse_worst_user_per_drop': dstats(dse.min(1)),
        'dse_sum_users_per_drop': dstats(dse.sum(1)),
        'dse_users_pooled': {'n': int(dse.size), 'mean': float(dse.mean()), 'min': float(dse.min()), 'max': float(dse.max()),
                             'frac_neg': float(np.mean(dse < 0)), 'frac_pos': float(np.mean(dse > 0)),
                             'frac_exact_zero': float(np.mean(dse == 0))},
        'max_abs_dse_all': float(np.abs(dse).max()),
        'max_abs_dse_no_saving_drops': float(np.abs(dse[~win]).max()) if (~win).any() else None,
        'max_abs_dse_saving_drops': float(np.abs(dse[win]).max()) if win.any() else None,
    }
    if win.sum() >= 2:
        out['saving_drops_only'] = {'dse_user_mean_per_drop': dstats(dse[win].mean(1)),
                                    'dse_worst_user_per_drop': dstats(dse[win].min(1)),
                                    'frac_users_neg': float(np.mean(dse[win] < 0))}
    compact = {'ru_prb_base': base.tolist(), 'ru_prb_cand': cand.tolist(), 'dse_user': dse.tolist()}
    return out, compact


def jdefault(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def main():
    t0 = time.time()
    try:
        n_cpu = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cpu = os.cpu_count() or 1
    workers = max(1, min(MAX_WORKERS, n_cpu))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        outs = list(ex.map(run_drop, DROPS))
    recs = [o[0] for o in outs]
    assert [rc['drop'] for rc in recs] == list(DROPS)
    chk, mx = Counter(), {}
    for _, c, m in outs:
        chk.update(c)
        for k, v in m.items():
            mx[k] = min(mx.get(k, np.inf), v) if k == 'lp_gap_rel_min' else max(mx.get(k, 0.0), v)
    results, compact, log = {}, {}, []
    for eps in EPS:
        e = ek(eps)
        results[e], compact[e] = {}, {}
        for variant in VARIANTS:
            s, c = summarise(recs, e, variant)
            pre = f'{e}_{variant}_'
            s['power'] = {tag: {'max_pant_over_10mW': mx[pre + tag + '_pmax_ratio'],
                                'max_ptot_over_cap': mx[pre + tag + '_ptot_ratio'],
                                'violations_pant_gt_1+1e-9': chk[pre + tag + '_pant_violations'],
                                'violations_ptot_gt_1+1e-9': chk[pre + tag + '_ptot_violations'],
                                'blocks_pant_gt_1': chk[pre + tag + '_pant_above_1'],
                                'blocks_ptot_gt_1': chk[pre + tag + '_ptot_above_1']} for tag in ('base', 'cand')}
            s['rate_errors_max'] = {tag: {k: mx[pre + tag + '_' + k] for k in
                                          ('pred_vs_assigned', 'pred_ue_conservation', 'pred_prb_conservation',
                                           'assigned_ue_conservation', 'assigned_prb_conservation', 'true_vs_pred',
                                           'leak_pred_over_noise', 'leak_true_over_noise')} for tag in ('base', 'cand')}
            s['checks'] = {'brute_force_total_mismatch': chk[f'brute_mismatch_{variant}'],
                           'drops_tie_matching_with_fewer_swaps': chk[f'tie_with_fewer_swaps_{variant}']}
            results[e][variant], compact[e][variant] = s, c
            sv, du, dw = s['saving_ru_prb_per_drop'], s['dse_user_mean_per_drop'], s['dse_worst_user_per_drop']
            line = (f"eps {e:>4} {variant:8s}: RU-PRB base {s['ru_prb_base_per_drop']['mean']:.4f} cand "
                    f"{s['ru_prb_cand_per_drop']['mean']:.4f} saving {sv['mean']:.4f} [{sv['ci95'][0]:.4f}, "
                    f"{sv['ci95'][1]:.4f}] ({s['saving_rel_pct']:.3f} %, {s['drops_with_saving']}/{len(DROPS)} drops) | "
                    f"dSE user-mean {du['mean']:+.3e} [{du['ci95'][0]:+.3e}, {du['ci95'][1]:+.3e}] worst-user "
                    f"{dw['mean']:+.3e} [{dw['ci95'][0]:+.3e}, {dw['ci95'][1]:+.3e}] max|dSE| {s['max_abs_dse_all']:.3e} | "
                    f"P viol ant/tot {s['power']['cand']['violations_pant_gt_1+1e-9']}/"
                    f"{s['power']['cand']['violations_ptot_gt_1+1e-9']} max ratio "
                    f"{s['power']['cand']['max_pant_over_10mW']:.12f}/{s['power']['cand']['max_ptot_over_cap']:.12f} | "
                    f"pred cons UE/PRB {s['rate_errors_max']['cand']['pred_ue_conservation']:.2e}/"
                    f"{s['rate_errors_max']['cand']['pred_prb_conservation']:.2e} | brute {s['checks']['brute_force_total_mismatch']}")
            print(line, flush=True)
            log.append(line)
    global_checks = {k: v for k, v in sorted(chk.items()) if not k.startswith(('0_', '0.01_', 'brute', 'tie'))}
    global_checks.update({k: v for k, v in sorted(mx.items()) if not k.startswith(('0_', '0.01_'))})
    line = f'checks: {global_checks}'
    print(line, flush=True)
    log.append(line)

    meta = {
        'task': 'cross-PRB per-UE ZF rate swap, round-2 holdout (multipath, estimated CSI, true-channel Shannon SE)',
        'metric': 'Shannon SE log2(1 + SINR) per UE per PRB (Gaussian signalling); no MCS / TBS / CRC',
        'date': time.strftime('%Y-%m-%d %H:%M:%S'), 'command': COMMAND, 'argv': [sys.executable] + sys.argv,
        'cwd': os.getcwd(), 'workers': workers, 'cpu_affinity': n_cpu, 'os_cpu_count': os.cpu_count(),
        'env_threads': {v: os.environ.get(v) for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
        'blas': blas_info(),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        'git_head': git_head(),
        'sources_sha256': {'claude_holdout.py': sha256(os.path.abspath(__file__)),
                           'claude_replay.py': sha256(os.path.join(HERE, 'claude_replay.py')),
                           'same_total.py': sha256(os.path.join(VAL_DIR, 'same_total.py'))},
        'params': {'seed': SEED, 'drops': [DROPS[0], DROPS[-1]], 'n_drops': len(DROPS), 'K': K, 'U': U, 'prbs': B,
                   'noise_dbm': NOISE_DBM, 'noise_mw': NOISE, 'kappa': KAPPA, 'p_ant_mw': P_ANT, 'tol_rel': TOL,
                   'eps': EPS, 'pdp': PDP.tolist(), 'delay_us': DELAY_US.tolist(), 'prb_hz': PRB_HZ,
                   'tap_stream': TAP_STREAM, 'd_fracs': FRACS, 'matchings': MATCHINGS, 'ue_pairs': USER_PAIRS},
        'channel': ('beta = same_total.drop(20261125, d, 1, 4, 4)[1]; rng = default_rng([20261125, d, 78]); '
                    'g1 = (N(4,4,16) + 1j N(4,4,16))/sqrt(2), then g2 likewise; axis order (path, UE, antenna) [assumption]; '
                    'H[b] = sqrt(beta) * sum_l sqrt(PDP_l) g1[l] exp(-2j pi b 360000 tau_l); E from g2; '
                    'Hhat = (1-eps) H + sqrt(eps(1-eps)) E'),
        'decisions': ('all decisions from Hhat: E8 score/best_per_m winners (rule ant, granularity RU), r, m*, masks, '
                      'cap = TX total power of the baseline W (Hhat, same r); true H only in the final SINR'),
        'baselines': {'winner': 'mask = E8 winner at m*; cap = its TX total power',
                      'lowpower': ('mask = argmin TX total power at exact t over all full-rank masks of size m* that are '
                                   'per-antenna feasible (E8 score q; ties -> smallest bitmask); re-evaluated with pinv; '
                                   'cap = its TX total power; mask appended to the candidate list after the m* winner')},
        'tie_breaking': ('per block pair: lowest m_i + m_j, ties -> d = 0, then |d| ascending, UE pair lexicographic, '
                         '+ before -; no strict saving -> baseline masks and rates; matchings: lowest total, ties -> '
                         '01|23, 02|13, 03|12; within a block: first feasible mask in (m ascending, E8 winner before '
                         'lowpower mask)'),
        'interference': 'sum of explicit off-diagonal |[H W]_uv|^2 (no subtraction)',
        'log': log, 'timing_s': {'total': time.time() - t0, 'per_drop_sum': float(sum(rc['time_s'] for rc in recs))},
    }
    with open(OUT, 'w') as fh:
        json.dump({'meta': meta, 'results': results, 'global_checks': global_checks, 'per_drop_compact': compact,
                   'per_drop': recs}, fh, ensure_ascii=False, default=jdefault)
    print(f'wrote {os.path.relpath(OUT, HERE)} in {meta["timing_s"]["total"]:.1f} s with {workers} workers', flush=True)


if __name__ == '__main__':
    main()
