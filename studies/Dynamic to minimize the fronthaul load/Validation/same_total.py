"""E8：同总天线数下，单天线分布式 RU（K = 1）与共址 K 天线 RU 的逐 PRB 天线激活（ZF，穷举）。
设定由用户 2026-11-24 确认。

场景
  16 根天线在 4 × 4 六角格上（间距 40 m，区域 160 m × 138.56 m），即 EuCNC 8 × 8 布局在同密度下缩小。
  编号沿用 dynamic_split/cfsim.py：j = 8·block + 2·col + sub，row = 2·block + sub，
  格点 (10 + 40·col + 20·(row mod 2), 17.32 + 34.64·row)。
  部署 K ∈ {1, 2, 4, 8}，N = 16/K：RU r 由编号连续的 K 根天线组成（EuCNC 的分组），这 K 根天线共址于
  它们格点的质心。K = 1 时站点就是格点，即 16 个单天线分布式 RU（D1）。
  大尺度：ITU-R M.2135-1 UMi 路损（与 cfsim._pathloss_m2135_db 相同）、UMi LOS 概率、对数正态阴影
  （LOS 3 dB、NLOS 4 dB）、天线增益 2.15 dBi。β[u, r] 只取决于 UE u 与站点 r，同一 RU 的 K 根天线共享
  （论文式 (channel)）。小尺度：i.i.d. CN(0, 1)，各 PRB 独立；天线 j 在各 K 下用同一组随机数。
  理想 CSI、理想前传，各 UE 噪声相同。

记号（论文）：H [U, 16] 的第 u 行是 UE u 的信道，接收 = H W；h_j = H[:, j] 是天线 j 到各 UE 的信道。

ZF（论文式 (zf)）：激活集合 S（m = |S| ≥ U），G = H_S H_S^H，W0 = H_S^H G^{-1}，
  天线 j 的功率 q_j = ||W0 第 j 行||² = ||G^{-1} h_j||²。两种功率规则：
    'ant'  逐天线上限：c² = P_max / max_j q_j（K = 1 时即 [10] 式 (18)）；
    'ru'   论文式 (norm)：c² = min(m P_max / Σ_j q_j, K P_max / max_n q_n)，q_n 为 RU n 内 q_j 之和。
  理想 CSI 下 H_S (c W0) = c I，各 UE 的 SINR = c² / σ²，sum SE = U log2(1 + c² / σ²) 随 c² 单调增，
  所以固定 m 时最优子集与噪声无关：逐 m 取 c² 最大的子集（并列取位掩码最小者）。
粒度：'RU' = 每个 RU 全开或全关（论文的 S_RU），'ant' = 任意子集（S_ant）。
指标（论文内层问题，逐 PRB 达标）：SE_max = RU 级全部激活中的最大 sum SE；
  M̂_g = 满足 SE ≥ κ SE_max 的最小 m（κ = 0.98）。
"""

from functools import lru_cache

import numpy as np

M_TOT = 16
N_COL = 4
SPACING_M = 40.0
ROW_PITCH_M = 34.64
AREA = (N_COL * SPACING_M, (M_TOT // N_COL) * ROW_PITCH_M)      # (160, 138.56)
FC_GHZ = 4.0
H_BS, H_UT = 10.0, 1.5
ANT_GAIN_DBI = 2.15
P_MAX_MW = 10.0                  # 每根天线每 PRB 10 dBm（EuCNC 表 II）
KAPPA = 0.98
RANK_TOL = 1e-12                 # 最小 / 最大特征值低于此值视为秩不足
RULES = ('ant', 'ru')
GRANS = ('RU', 'ant')


def antenna_positions():
    """16 根天线的格点 (x, y)，单位 m，编号规则同 cfsim.antenna_positions。"""
    pos = np.zeros((M_TOT, 2))
    for j in range(M_TOT):
        block, i = divmod(j, 2 * N_COL)
        col, sub = divmod(i, 2)
        row = 2 * block + sub
        pos[j] = (10.0 + SPACING_M * col + 20.0 * (row % 2), 17.32 + ROW_PITCH_M * row)
    return pos


def ru_sites(K):
    """部署 K 的 N = 16/K 个 RU 站点：编号连续的 K 个格点的质心 [N, 2]。"""
    return antenna_positions().reshape(M_TOT // K, K, 2).mean(1)


def pathloss_m2135_db(d2d, los):
    """ITU-R M.2135-1 UMi，与 dynamic_split/cfsim._pathloss_m2135_db 相同（test_same_total 逐点核对）。"""
    fc = FC_GHZ
    d_bp = 4 * (H_BS - 1) * (H_UT - 1) * fc * 1e9 / 3e8
    pl_los = np.where(d2d <= d_bp, 22.0 * np.log10(d2d) + 28.0 + 20 * np.log10(fc),
                      40 * np.log10(d2d) + 7.8 - 18 * np.log10(H_BS - 1) - 18 * np.log10(H_UT - 1)
                      + 2 * np.log10(fc))
    pl_nlos = 36.7 * np.log10(d2d) + 22.7 + 26 * np.log10(fc)
    return np.where(los, pl_los, pl_nlos)


def los_probability(d2d):
    """UMi LOS 概率（与 cfsim.drop 相同）。"""
    return np.where(d2d <= 18, 1.0, 18 / d2d + np.exp(-d2d / 36) * (1 - 18 / d2d))


def large_scale(rng, ue, sites):
    """β [U, N]（线性增益）与 LOS 标志 [U, N]：公式与抽样顺序同 cfsim.drop 的 'm2135' 分支（先 LOS，后阴影）。"""
    d2d = np.maximum(np.linalg.norm(ue[:, None] - sites[None], axis=-1), 10.0)
    los = rng.random(d2d.shape) < los_probability(d2d)
    sf = rng.standard_normal(d2d.shape) * np.where(los, 3.0, 4.0)
    return 10 ** ((ANT_GAIN_DBI - pathloss_m2135_db(d2d, los) - sf) / 10), los


def ue_positions(seed, d, num_ue):
    return np.random.default_rng([seed, d, 0]).uniform((0, 0), AREA, size=(num_ue, 2))


def small_scale(seed, d, num_ue, num_prb):
    """f [B, U, 16]，CN(0, 1)，各 K 共用（先实部、后虚部）。"""
    rng = np.random.default_rng([seed, d, 2])
    re = rng.standard_normal((num_prb, num_ue, M_TOT))
    im = rng.standard_normal((num_prb, num_ue, M_TOT))
    return (re + 1j * im) / np.sqrt(2)


def drop(seed, d, K, num_ue, num_prb):
    """撒点 d、部署 K → H [B, U, 16]、β [U, N]、LOS [U, N]、UE 位置 [U, 2]。"""
    ue = ue_positions(seed, d, num_ue)
    beta, los = large_scale(np.random.default_rng([seed, d, 1, K]), ue, ru_sites(K))
    f = small_scale(seed, d, num_ue, num_prb)
    H = np.sqrt(np.repeat(beta, K, axis=1))[None] * f
    return H, beta, los, ue


@lru_cache(maxsize=None)
def candidates(K, num_ue):
    """|S| ≥ U 的全部子集，按位掩码升序：keys [Nc]、bits [Nc, 16]、size [Nc]、is_ru [Nc]；
    groups[(粒度, m)] 为该粒度下激活数为 m 的候选下标（升序）。"""
    keys = np.arange(1, 2 ** M_TOT, dtype=np.int64)
    bits = ((keys[:, None] >> np.arange(M_TOT)) & 1).astype(bool)
    size = bits.sum(1)
    keep = size >= num_ue
    keys, bits, size = keys[keep], bits[keep], size[keep]
    per_ru = bits.reshape(len(keys), M_TOT // K, K).sum(2)
    is_ru = np.all((per_ru == 0) | (per_ru == K), axis=1)
    groups = {}
    for m in range(num_ue, M_TOT + 1):
        at_m = size == m
        groups[('ant', m)] = np.flatnonzero(at_m)
        groups[('RU', m)] = np.flatnonzero(at_m & is_ru)
    return keys, bits, size, is_ru, groups


def score(H, K):
    """一个 PRB 的 H [U, 16] → 全部候选的 c² [2 规则, Nc]（秩不足为 -inf）、q [Nc, 16]、秩不足候选数。"""
    U = H.shape[0]
    _, bits, size, _, _ = candidates(K, U)
    b = bits.astype(float)
    outer = np.einsum('uj,vj->juv', H, np.conj(H)).reshape(M_TOT, U * U)    # 第 j 个为 h_j h_j^H
    G = (b @ outer).reshape(-1, U, U)
    ev = np.linalg.eigvalsh(G)
    valid = ev[:, 0] > RANK_TOL * ev[:, -1]
    A = np.linalg.inv(np.where(valid[:, None, None], G, np.eye(U)))
    q = np.sum(np.abs(A @ H) ** 2, axis=1) * b                                 # q_j = ||G^{-1} h_j||²，仅激活天线
    q_ru = q.reshape(len(q), M_TOT // K, K).sum(2)
    with np.errstate(divide='ignore', invalid='ignore'):
        c2_ant = P_MAX_MW / q.max(1)
        c2_ru = np.minimum(size * P_MAX_MW / q.sum(1), K * P_MAX_MW / q_ru.max(1))
    c2 = np.where(valid[None], np.stack([c2_ant, c2_ru]), -np.inf)
    return c2, q, int((~valid).sum())


def best_per_m(c2, K, num_ue):
    """c2 [2 规则, Nc] → val [2 规则, 2 粒度, 17]（激活数 m 下的最大 c²；无候选为 -inf）与同形的候选下标 idx
    （无为 -1）。粒度顺序 GRANS = ('RU', 'ant')；并列取下标最小者（即位掩码最小）。"""
    groups = candidates(K, num_ue)[4]
    nr = c2.shape[0]
    val = np.full((nr, len(GRANS), M_TOT + 1), -np.inf)
    idx = np.full(val.shape, -1, dtype=np.int64)
    for gi, g in enumerate(GRANS):
        for m in range(num_ue, M_TOT + 1):
            sub = groups[(g, m)]
            if len(sub) == 0:
                continue
            j = sub[np.argmax(c2[:, sub], axis=1)]
            v = c2[np.arange(nr), j]
            ok = np.isfinite(v)
            val[ok, gi, m] = v[ok]
            idx[ok, gi, m] = j[ok]
    return val, idx


def c2_threshold(c2_max, noise, kappa=KAPPA):
    """SE ≥ κ SE_max（SE = U log2(1 + c²/σ²)）等价于 c² ≥ σ² ((1 + c2_max/σ²)^κ − 1)。"""
    return noise * np.expm1(kappa * np.log1p(c2_max / noise))


def m_hat(val, noise, kappa=KAPPA):
    """val [..., 2 粒度, 17] → M̂ [..., 2 粒度] 与 c2_max [...]。SE_max 取 RU 级的最大值（论文定义）。"""
    c2_max = val[..., 0, :].max(-1)
    thr = c2_threshold(c2_max, noise, kappa)
    ok = val >= thr[..., None, None] * (1 - 1e-12)
    assert np.all(ok.any(-1)), 'target unreachable'
    return np.argmax(ok, axis=-1), c2_max


def sum_se(c2, noise, num_ue):
    return num_ue * np.log2(1 + c2 / noise)


def explicit(H, key, K):
    """位掩码 key 的子集直接用伪逆求 W0：q [16]（未激活为 0）、两种规则的 c² [2]、ZF 残差 max|H_S W0 − I|。"""
    S = np.flatnonzero((int(key) >> np.arange(M_TOT)) & 1)
    W0 = np.linalg.pinv(H[:, S])                                                # [m, U] = H_S^H (H_S H_S^H)^{-1}
    q = np.zeros(M_TOT)
    q[S] = np.sum(np.abs(W0) ** 2, axis=1)
    q_ru = q.reshape(-1, K).sum(1)
    c2 = np.array([P_MAX_MW / q.max(), min(len(S) * P_MAX_MW / q.sum(), K * P_MAX_MW / q_ru.max())])
    resid = float(np.max(np.abs(H[:, S] @ W0 - np.eye(H.shape[0]))))
    return q, c2, resid


def ru_counts(key, K):
    """位掩码 → 各 RU 的激活天线数 [N]。"""
    return ((int(key) >> np.arange(M_TOT)) & 1).reshape(-1, K).sum(1)
