"""Cell-free mMIMO 下行模型，复现 Girycki 等（EuCNC/6G Summit 2024）第 II、IV 节和表 II 的设置。

论文没有给出的细节在这里补上，都写进 ASSUMPTIONS，运行时存进 metadata.json：
  - 64 根天线按六角格排成 8 列 × 8 行（最近邻 40 m，区域 320 m × 277 m，与图 2 一致），
    RU r 由编号连续的 K 根天线组成（K = 4 时 r = 1 含 n = 1..4，与图 2 一致）；
  - 大尺度：默认 ITU-R M.2135-1 UMi 路损（论文引用 [15]），LOS 概率按 UMi，对数正态阴影链路间独立。
    与论文图 3、图 5 的 ZF 和速率对照，M.2135 吻合（10 UE、-90 dBm：128.5 vs 127），
    TR 38.901 / M.2412 UMi street canyon（'m2412'）偏低约 7%，保留为选项；
  - 小尺度：每条链路一个指数功率时延谱的抽头模型（时延扩展取 UMi 中位数，LOS 加 K = 9 dB 的直射分量），
    在每个 PRB 中心取一个频响，PRB 内平坦；
  - ZF：每 PRB 对服务天线集合做 ZF，列归一化（各 UE 等功率），再按"最大单天线功率 = Pmax"统一缩放；
  - MRT：每根服务天线只服务一个 UE（论文第 IV 节），满功率相位共轭；
  - 天线顺序：按"该天线到最强 UE 的大尺度增益"降序；逐根考察，能提高总 SE 才加入（论文第 III-C 节）。
    MRT 时对每根天线试遍所有 UE，取总 SE 最大的那个 UE。
"""

import numpy as np

FC_GHZ = 4.0
H_BS, H_UT = 10.0, 1.5
ANT_GAIN_DBI = 2.15
NUM_ANT = 64
NUM_PRB = 50                 # 18 MHz，30 kHz
SCS_HZ = 30e3
P_MAX_PRB_MW = 10.0          # 表 II：0.833 mW/子载波 ≈ 10 dBm/PRB
AREA = (320.0, 277.0)

# TS 38.214 V19.3.0 表 5.1.3.1-2（PDSCH MCS 表 2）：(Qm, 频谱效率)
MCS_TABLE2 = np.array([
    (2, 0.2344), (2, 0.3770), (2, 0.6016), (2, 0.8770), (2, 1.1758),
    (4, 1.4766), (4, 1.6953), (4, 1.9141), (4, 2.1602), (4, 2.4063), (4, 2.5703),
    (6, 2.7305), (6, 3.0293), (6, 3.3223), (6, 3.6094), (6, 3.9023), (6, 4.2129),
    (6, 4.5234), (6, 4.8164), (6, 5.1152),
    (8, 5.3320), (8, 5.5547), (8, 5.8906), (8, 6.2266), (8, 6.5703), (8, 6.9141),
    (8, 7.1602), (8, 7.4063)])

# TS 38.214 V19.3.0 表 5.1.3.1-4（PDSCH MCS 表 4，含 1024QAM）
MCS_TABLE4 = np.array([
    (2, 0.2344), (2, 0.3770), (2, 0.8770), (4, 1.4766), (4, 1.9141), (4, 2.4063),
    (6, 2.7305), (6, 3.0293), (6, 3.3223), (6, 3.6094), (6, 3.9023), (6, 4.2129), (6, 4.5234), (6, 4.8164),
    (6, 5.1152), (8, 5.3320), (8, 5.5547), (8, 5.8906), (8, 6.2266), (8, 6.5703), (8, 6.9141), (8, 7.1602),
    (8, 7.4063), (10, 7.8662), (10, 8.3301), (10, 8.7939), (10, 9.2578)])

# ZF 功率归一化：'equal'（列归一化后统一缩放，阶段 0 的口径）或 'inversion'（[10] 式 (18)，不做列归一化）
ZF_NORM = 'equal'


def set_zf_norm(mode):
    global ZF_NORM
    if mode not in ('equal', 'inversion'):
        raise ValueError(mode)
    ZF_NORM = mode


ASSUMPTIONS = [
    "64 antennas on a hexagonal 8x8 lattice (40 m spacing, 320 m x 277 m); RU r = antennas with consecutive indices.",
    "Large scale (default 'm2135'): ITU-R M.2135-1 UMi path loss (hexagonal NLOS formula, 2D distance), UMi LOS "
    "probability, log-normal shadowing 3 / 4 dB independent across links; 'm2412' = TR 38.901 UMi street canyon "
    "(4 / 7.82 dB). UE height 1.5 m, 2D distance >= 10 m.",
    "Small scale: per-link tapped delay line (exponential PDP, DS 49 ns LOS / 100 ns NLOS, 25 ns tap grid, "
    "Rician K = 9 dB in LOS), one response per PRB centre, flat within a PRB.",
    "ZF per PRB on the serving set, unit-norm columns, common scaling to the per-antenna limit Pmax per PRB.",
    "MRT: each serving antenna serves one UE with full power and phase conjugation (paper Sec. IV).",
    "Antenna order: descending large-scale gain to the strongest UE; an antenna is kept only if it raises the "
    "sum SE (paper Sec. III-C); for MRT every UE is tried and the best one kept.",
    "Full CSI, single TTI, full buffer (every served UE uses all PRBs).",
]


def antenna_positions():
    """64 根天线的 (x, y)，单位 m。编号 n-1 = 16·块 + 2·列 + 行内序号。"""
    pos = np.zeros((NUM_ANT, 2))
    for n in range(NUM_ANT):
        block, j = divmod(n, 16)
        col, sub = divmod(j, 2)
        row = 2 * block + sub
        pos[n] = (10.0 + 40.0 * col + 20.0 * (row % 2), 17.32 + 34.64 * row)
    return pos


def ru_of_antenna(k):
    """每根天线所属的 RU 编号（K 根一组）。"""
    return np.arange(NUM_ANT) // k


def _pathloss_db(d2d, d3d, los):
    fc = FC_GHZ
    d_bp = 4 * (H_BS - 1) * (H_UT - 1) * fc * 1e9 / 3e8
    pl1 = 32.4 + 21 * np.log10(d3d) + 20 * np.log10(fc)
    pl2 = 32.4 + 40 * np.log10(d3d) + 20 * np.log10(fc) - 9.5 * np.log10(d_bp ** 2 + (H_BS - H_UT) ** 2)
    pl_los = np.where(d2d <= d_bp, pl1, pl2)
    pl_nlos = 35.3 * np.log10(d3d) + 22.4 + 21.3 * np.log10(fc) - 0.3 * (H_UT - 1.5)
    return np.where(los, pl_los, np.maximum(pl_los, pl_nlos))


def _pathloss_m2135_db(d2d, los):
    """ITU-R M.2135-1 UMi（六角格布局的 NLOS 公式），d 取二维距离。"""
    fc = FC_GHZ
    d_bp = 4 * (H_BS - 1) * (H_UT - 1) * fc * 1e9 / 3e8
    pl_los = np.where(d2d <= d_bp, 22.0 * np.log10(d2d) + 28.0 + 20 * np.log10(fc),
                      40 * np.log10(d2d) + 7.8 - 18 * np.log10(H_BS - 1) - 18 * np.log10(H_UT - 1)
                      + 2 * np.log10(fc))
    pl_nlos = 36.7 * np.log10(d2d) + 22.7 + 26 * np.log10(fc)
    return np.where(los, pl_los, pl_nlos)


def drop(rng, num_ue, pathloss='m2135'):
    """一次撒点：返回信道 h [B, U, M]（单位 sqrt(mW 增益)）、大尺度增益 beta [U, M]、UE 位置。

    pathloss：'m2412'（TR 38.901 UMi street canyon，阴影 4/7.82 dB）或 'm2135'（M.2135-1 UMi，阴影 3/4 dB）。
    """
    ue = rng.uniform((0, 0), AREA, size=(num_ue, 2))
    ant = antenna_positions()
    d2d = np.maximum(np.linalg.norm(ue[:, None] - ant[None], axis=-1), 10.0)
    d3d = np.sqrt(d2d ** 2 + (H_BS - H_UT) ** 2)
    p_los = np.where(d2d <= 18, 1.0, 18 / d2d + np.exp(-d2d / 36) * (1 - 18 / d2d))
    los = rng.random(d2d.shape) < p_los
    if pathloss == 'm2412':
        pl, sigma = _pathloss_db(d2d, d3d, los), np.where(los, 4.0, 7.82)
    else:
        pl, sigma = _pathloss_m2135_db(d2d, los), np.where(los, 3.0, 4.0)
    sf = rng.standard_normal(d2d.shape) * sigma
    beta = 10 ** ((ANT_GAIN_DBI - pl - sf) / 10)
    # 抽头模型
    tau = np.arange(16) * 25e-9
    ds = np.where(los, 49e-9, 100e-9)[..., None]
    pdp = np.exp(-tau / ds)
    pdp /= pdp.sum(-1, keepdims=True)
    k_lin = np.where(los, 10 ** 0.9, 0.0)[..., None]
    taps = (rng.standard_normal(pdp.shape) + 1j * rng.standard_normal(pdp.shape)) * np.sqrt(pdp / 2)
    taps = taps * np.sqrt(1 / (1 + k_lin))
    taps[..., 0] += np.sqrt(k_lin[..., 0] / (1 + k_lin[..., 0])) * np.exp(2j * np.pi * rng.random(d2d.shape))
    f = (np.arange(NUM_PRB) * 12 + 6) * SCS_HZ
    g = np.einsum('umt,bt->bum', taps, np.exp(-2j * np.pi * f[:, None] * tau[None]))
    return np.sqrt(beta)[None] * g, beta, ue


def se_from_sinr(sinr, model):
    """频谱效率：'shannon' 为 log2(1+SINR)（论文）；'mcs' / 'mcs4' 取 MCS 表 2 / 表 4 中不超过它的最高档。"""
    c = np.log2(1 + sinr)
    if model == 'shannon':
        return c
    tab = MCS_TABLE4 if model == 'mcs4' else MCS_TABLE2
    idx = np.searchsorted(tab[:, 1], c, side='right') - 1
    return np.where(idx >= 0, tab[np.maximum(idx, 0), 1], 0.0)


def qm_from_se(se_per_ue, model='mcs'):
    """按每 UE 的平均频谱效率取 MCS 表的调制阶数（调制压缩每 RE 的比特数 = Qm）；'mcs4' 用表 4（含 1024QAM）。"""
    tab = MCS_TABLE4 if model == 'mcs4' else MCS_TABLE2
    idx = np.clip(np.searchsorted(tab[:, 1], se_per_ue, side='right') - 1, 0, None)
    return tab[idx, 0].astype(int)


def noise_per_ue(rng, mean_mw, num_ue, sigma_db=4.8):
    """[10] 的噪声模型：每个 UE 一个对数正态值（标准差 sigma_db），线性均值等于 mean_mw。"""
    s = sigma_db * np.log(10) / 10
    return mean_mw * np.exp(s * rng.standard_normal(num_ue) - s ** 2 / 2)


def zf_sinr(h_s, noise_mw):
    """h_s [B, U, S] → 每 UE 每 PRB 的 SINR [B, U]（S >= U）。noise_mw 可为标量或每 UE 的数组。"""
    g = h_s @ np.conj(np.swapaxes(h_s, 1, 2))
    w = np.conj(np.swapaxes(h_s, 1, 2)) @ np.linalg.inv(g)          # [B, S, U]，H W = I
    if ZF_NORM == 'inversion':                                       # [10] 式 (18)：各 UE 接收幅度相同
        p = P_MAX_PRB_MW / np.max(np.sum(np.abs(w) ** 2, axis=2), axis=1)
        return np.repeat(p[:, None], h_s.shape[1], axis=1) / noise_mw
    norm = np.linalg.norm(w, axis=1)                                 # [B, U]
    w = w / norm[:, None, :]
    p = P_MAX_PRB_MW / np.max(np.sum(np.abs(w) ** 2, axis=2), axis=1)  # [B]
    return p[:, None] / norm ** 2 / noise_mw


def mrt_sinr(a, noise_mw):
    """a [B, U, U']：UE u 收到发给 u' 的幅度和。返回 SINR [B, U]。"""
    pw = np.abs(a) ** 2
    sig = np.einsum('buu->bu', pw)
    return sig / (pw.sum(-1) - sig + noise_mw)


def antenna_order(beta):
    return np.argsort(-beta.max(0), kind='stable')


def zf_sequence(h, order, noise_mw, se_model):
    """按顺序逐根考察天线（ZF-H 的近似）。返回每步的总 SE、每 UE SE、是否加入。"""
    num_prb, num_ue, _ = h.shape
    chosen, cur, cur_ue = [], 0.0, np.zeros(num_ue)
    curve, per_ue, incl = [], [], []
    for m in order:
        trial = chosen + [m]
        if len(trial) < num_ue:
            chosen, keep = trial, True
        else:
            se_ue = se_from_sinr(zf_sinr(h[:, :, trial], noise_mw), se_model).mean(0)
            keep = len(chosen) < num_ue or se_ue.sum() > cur
            if keep:
                chosen, cur, cur_ue = trial, se_ue.sum(), se_ue
        curve.append(cur)
        per_ue.append(cur_ue.copy())
        incl.append(keep)
    return np.array(curve), np.array(per_ue), np.array(incl), np.full(len(order), -1)


def mrt_sequence(h, order, noise_mw, se_model):
    """MRT-H2 的近似：每根天线试遍所有 UE，取总 SE 最大者，能提高才加入。"""
    num_prb, num_ue, _ = h.shape
    a = np.zeros((num_prb, num_ue, num_ue), complex)
    cur, cur_ue = 0.0, np.zeros(num_ue)
    curve, per_ue, incl, assign = [], [], [], []
    for m in order:
        hm = h[:, :, m]                                             # [B, U]
        w = np.sqrt(P_MAX_PRB_MW) * np.conj(hm) / np.abs(hm)        # 发给各 UE 时的权值 [B, U']
        best = (cur, None, None)
        for u in range(num_ue):
            a_try = a.copy()
            a_try[:, :, u] += hm * w[:, u][:, None]
            se_ue = se_from_sinr(mrt_sinr(a_try, noise_mw), se_model).mean(0)
            if se_ue.sum() > best[0]:
                best = (se_ue.sum(), u, se_ue)
        if best[1] is not None:
            u = best[1]
            a[:, :, u] += hm * w[:, u][:, None]
            cur, cur_ue = best[0], best[2]
        curve.append(cur)
        per_ue.append(cur_ue.copy())
        incl.append(best[1] is not None)
        assign.append(-1 if best[1] is None else best[1])
    return np.array(curve), np.array(per_ue), np.array(incl), np.array(assign)
