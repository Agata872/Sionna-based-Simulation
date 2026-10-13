"""阶段 2a 场景：两个 32 阵元 O-RU 的 cell-free，简化的 3GPP UMi 簇信道（不用 Sionna RT）。

参数与 paper 1 对齐：3.5 GHz，30 kHz，52 PRB，每组 2 PRB（F = 26 组），O-RU 30 dBm，杆高 12 m，UE 噪声系数 9 dB。
  - O-RU：(0, 0, 12) 和 (120, 0, 12)，都朝 +y；UPA 4 行 × 8 列，λ/2，单极化，3GPP TR 38.901 单元方向图（8 dBi）；
  - UE：在 x ∈ [0, 120]、y ∈ [10, 80] m 内均匀撒点，高 1.5 m，全向；
  - 大尺度：TR 38.901 UMi street canyon 路损、LOS 概率、阴影 4 / 7.82 dB；
  - 小尺度：每条链路 6 个簇 × 20 条射线（TR 38.901 的射线偏移），簇时延和功率按 UMi 的时延扩展与 r_τ，
    簇方位按 UMi 的 ASD 围绕 LOS 方向散开，簇内 c_ASD = 3°、c_ZSD = 2°；LOS 时加 K = 9 dB 的直射径；
    每条射线的初相独立均匀；静止 UE。
  - "孪生"（URTS-CODE 的代理）知道每条射线的角度、时延和功率，不知道初相；
    twin_err_deg > 0 时，孪生的簇中心方位和天顶角各加 N(0, σ²) 的误差，时延加 N(0, (5 ns)²)。
信道单位：sqrt(mW 增益)，即 |h|² 是含单元增益的功率增益。
"""

import numpy as np

FC = 3.5e9
LAM = 3e8 / FC
SCS = 30e3
NUM_PRB = 52
GRP_PRB = 2
F = NUM_PRB // GRP_PRB
ROWS, COLS = 4, 8
M = ROWS * COLS
RU_POS = np.array([[0.0, 0.0, 12.0], [120.0, 0.0, 12.0]])
L = len(RU_POS)
AREA_LO, AREA_HI = np.array([0.0, 10.0]), np.array([120.0, 80.0])
H_UT = 1.5
P_RU_DBM = 30.0
P_RU_PRB_MW = 10 ** ((P_RU_DBM - 10 * np.log10(NUM_PRB)) / 10)
UE_NF_DB = 9.0
N_TH_PRB_DBM = -174 + 10 * np.log10(12 * SCS) + UE_NF_DB          # 约 -109.4 dBm
P_UE_DBM = 23.0
# SRS 信道估计噪声（每个元素，每个 2-PRB 组）：UE 23 dBm 均分到 52 PRB，按组内 2 PRB 平均
CI_NOISE_MW = 10 ** ((N_TH_PRB_DBM - (P_UE_DBM - 10 * np.log10(NUM_PRB)) - 10 * np.log10(GRP_PRB)) / 10)

RAY_OFFSETS = np.array([0.0447, 0.1413, 0.2492, 0.3715, 0.5129, 0.6797, 0.8844, 1.1481, 1.5195, 2.1551])
RAY_OFFSETS = np.concatenate([RAY_OFFSETS, -RAY_OFFSETS])
N_CLUSTER = 6

ASSUMPTIONS = [
    'Two O-RUs at (0,0,12) and (120,0,12) m facing +y; UPA 4x8 lambda/2 single-pol, TR 38.901 element pattern 8 dBi; '
    'UEs uniform in x 0..120, y 10..80 m, 1.5 m, omni. 3.5 GHz, 30 kHz, 52 PRB, groups of 2 PRB (F = 26).',
    'TR 38.901 UMi street canyon path loss, LOS probability, shadowing 4/7.82 dB; 6 clusters x 20 rays per link '
    '(UMi DS 50/103 ns, r_tau 3/2.1, ASD 15/24 deg, ZSD 6/8 deg, c_ASD 3 deg, c_ZSD 2 deg), LOS ray K = 9 dB; '
    'independent ray phases; static UEs; one channel per 2-PRB group centre.',
    'O-RU 30 dBm over 52 PRB; UE noise figure 9 dB (thermal -109.4 dBm/PRB); SRS: UE 23 dBm, per-element CI '
    'estimation noise = thermal / (UE power per PRB x 2 PRB).',
    'Twin (URTS-CODE proxy) knows ray angles, delays and powers but not phases; twin_err adds N(0, s^2) to cluster '
    'azimuth/zenith and N(0, 5 ns^2) to delays.',
]


def element_gain_db(theta, phi):
    """TR 38.901 表 7.3-1 单元方向图；theta 天顶角，phi 相对视轴的方位角（度）。"""
    a_v = -np.minimum(12 * ((theta - 90) / 65) ** 2, 30)
    a_h = -np.minimum(12 * (phi / 65) ** 2, 30)
    return 8.0 - np.minimum(-(a_v + a_h), 30)


def steering(theta, phi):
    """UPA 阵列响应 [..., M]，theta/phi 为度。阵元位于 (列·λ/2, 0, 行·λ/2)，视轴 +y。"""
    t, p = np.deg2rad(theta)[..., None], np.deg2rad(phi)[..., None]
    col = np.tile(np.arange(COLS), ROWS)
    row = np.repeat(np.arange(ROWS), COLS)
    phase = np.pi * (col * np.sin(t) * np.sin(p) + row * np.cos(t))
    return np.exp(1j * phase)


def group_freqs():
    return (np.arange(F) * GRP_PRB * 12 + GRP_PRB * 6) * SCS


def _pathloss(d2d, d3d, los, h_bs):
    fc = FC / 1e9
    d_bp = 4 * (h_bs - 1) * (H_UT - 1) * FC / 3e8
    pl1 = 32.4 + 21 * np.log10(d3d) + 20 * np.log10(fc)
    pl2 = 32.4 + 40 * np.log10(d3d) + 20 * np.log10(fc) - 9.5 * np.log10(d_bp ** 2 + (h_bs - H_UT) ** 2)
    pl_los = np.where(d2d <= d_bp, pl1, pl2)
    pl_nlos = 35.3 * np.log10(d3d) + 22.4 + 21.3 * np.log10(fc) - 0.3 * (H_UT - 1.5)
    return np.where(los, pl_los, np.maximum(pl_los, pl_nlos))


def link_rays(rng, ru, ue):
    """一条链路的射线：返回 dict(theta, phi, tau, power, beta, los)，power 归一化（和为 1，不含单元增益）。"""
    v = ue - ru
    d2d = max(np.hypot(v[0], v[1]), 10.0)
    d3d = np.hypot(d2d, v[2])
    p_los = 1.0 if d2d <= 18 else 18 / d2d + np.exp(-d2d / 36) * (1 - 18 / d2d)
    los = rng.random() < p_los
    sf = rng.standard_normal() * (4.0 if los else 7.82)
    beta = 10 ** (-(_pathloss(d2d, d3d, los, ru[2]) + sf) / 10)
    phi_los = np.rad2deg(np.arctan2(v[0], v[1]))
    theta_los = np.rad2deg(np.arccos(v[2] / np.linalg.norm(v)))
    ds, r_tau, asd, zsd = (50e-9, 3.0, 15.0, 6.0) if los else (103e-9, 2.1, 24.0, 8.0)
    tau = np.sort(-r_tau * ds * np.log(rng.random(N_CLUSTER)))
    tau -= tau[0]
    pw = np.exp(-tau * (r_tau - 1) / (r_tau * ds)) * 10 ** (-rng.standard_normal(N_CLUSTER) * 3 / 10)
    pw /= pw.sum()
    phi_c = phi_los + rng.standard_normal(N_CLUSTER) * asd
    theta_c = theta_los + rng.standard_normal(N_CLUSTER) * zsd
    if los:
        phi_c[0], theta_c[0] = phi_los, theta_los
    n_r = len(RAY_OFFSETS)
    rays = dict(
        # 天顶方向的射线偏移与方位随机配对（TR 38.901 的随机耦合）
        theta=np.repeat(theta_c, n_r) + 2.0 * np.concatenate(
            [RAY_OFFSETS[rng.permutation(n_r)] for _ in range(N_CLUSTER)]),
        phi=np.repeat(phi_c, n_r) + 3.0 * np.tile(RAY_OFFSETS, N_CLUSTER),
        tau=np.repeat(tau, n_r),
        power=np.repeat(pw, n_r) / n_r,
        cluster=np.repeat(np.arange(N_CLUSTER), n_r))
    if los:
        k_r = 10 ** 0.9
        rays['power'] = rays['power'] / (1 + k_r)
        for key, val in (('theta', theta_los), ('phi', phi_los), ('tau', 0.0), ('power', k_r / (1 + k_r)),
                         ('cluster', -1)):
            rays[key] = np.append(rays[key], val)
    rays['phi'] = np.clip(rays['phi'], -179.9, 179.9)
    rays.update(beta=beta, los=bool(los), d2d=d2d)
    return rays


def ray_vectors(rays, beta=None):
    """每条射线的空间-频率向量（含幅度，不含初相）[F·M, R]，顺序为 (组, 阵元)。"""
    beta = rays['beta'] if beta is None else beta
    amp = np.sqrt(beta * rays['power'] * 10 ** (element_gain_db(rays['theta'], rays['phi']) / 10))
    a = steering(rays['theta'], rays['phi'])                                   # [R, M]
    fr = np.exp(-2j * np.pi * group_freqs()[:, None] * rays['tau'][None])       # [F, R]
    v = fr[:, :, None] * a[None]                                                # [F, R, M]
    return (np.swapaxes(v, 0, 1).reshape(len(amp), F * M) * amp[:, None]).T     # [F·M, R]


def perturb(rng, rays, err_deg):
    """孪生的射线几何：簇中心角度加误差，时延加 5 ns 误差（同一簇的射线一起移动）。"""
    if err_deg <= 0:
        return rays
    tw = dict(rays)
    clusters = np.unique(rays['cluster'])
    d_phi = dict(zip(clusters, rng.standard_normal(len(clusters)) * err_deg))
    d_th = dict(zip(clusters, rng.standard_normal(len(clusters)) * err_deg))
    d_tau = dict(zip(clusters, rng.standard_normal(len(clusters)) * 5e-9))
    tw['phi'] = rays['phi'] + np.array([d_phi[c] for c in rays['cluster']])
    tw['theta'] = rays['theta'] + np.array([d_th[c] for c in rays['cluster']])
    tw['tau'] = np.maximum(rays['tau'] + np.array([d_tau[c] for c in rays['cluster']]), 0)
    return tw


def drop(rng, num_ue):
    """一次撒点。返回 H [F, K, L·M]（真实信道）、每条链路的射线 rays[k][l]、UE 位置。"""
    ue = np.column_stack([rng.uniform(AREA_LO, AREA_HI, size=(num_ue, 2)), np.full(num_ue, H_UT)])
    h = np.zeros((F, num_ue, L * M), complex)
    rays = [[None] * L for _ in range(num_ue)]
    for k in range(num_ue):
        for l_ in range(L):
            r = link_rays(rng, RU_POS[l_], ue[k])
            v = ray_vectors(r)                                             # [F·M, R]
            ph = np.exp(2j * np.pi * rng.random(v.shape[1]))
            h[:, k, l_ * M:(l_ + 1) * M] = (v @ ph).reshape(F, M)
            rays[k][l_] = r
    return h, rays, ue
