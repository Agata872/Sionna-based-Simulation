"""阶段 2a：双 O-RU 场景下，一个撒点上各配置的和速率与前传负载。

配置（网络统一，两个 O-RU 相同）：
  ('7-1',  U-plane BFP 宽度, None, 'raw')            DU 集中式 RZF；前传传 32 路天线级 IQ；CI 为天线级 SRS IQ（7-1 上行固有）
  ('7-2c', 'modcomp', 权值宽度, CI 方案)             DU 集中式 RZF；前传传 K 路层符号（调制压缩）+ SE-11 权值（每时隙）；
                                                       CI 由 O-RU 上送：'raw' 天线级 SRS IQ、'st6_bfpW' 每 UE 每组每 TRX 的
                                                       SRS-CE 信道估计（ST 6，BFP W）、'klt_oracle' / 'klt_twin2' URTS-CODE 代理
  ('7-2l', 'modcomp', None, 'nu0' / 'nu0.5')         O-RU 本地 RZF（SRS-BF，只用本 O-RU 的 SRS），无权值、无 CI；
                                                       每 O-RU 按本地大尺度增益的 ν 次方给各 UE 分功率（ν = 0 为等功率）
理想参考：DU 集中式 RZF，用含估计噪声但未压缩的 CI，前传无失真。集中式的功率规则（UE 功率 ∝ 网络平均增益^ν，
ν ∈ {0, 0.5}）由 DU 用自己的 CI 预测和速率选出，同一撒点、同一噪声下所有集中式配置共用。
URTS-CODE 代理：两端由孪生射线几何得到同一个逐链路 KLT 基与特征值 λ；O-RU 只发 λ > σ²（CI 噪声）的维度，
按噪声匹配的反注水分配比特（每复系数 log2((λ+σ²)/σ²) + 0.508 bit，后者为标量量化相对率失真界的 1.53 dB/实维），
量化误差按方差 σ² 的高斯噪声模拟；每链路另计 10 bit 的维数。
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import cfsim  # noqa: E402
import distortion as dz  # noqa: E402
import fhload  # noqa: E402
import geo2ru as g  # noqa: E402
from oran.fronthaul import CPLANE_HEADERS, ECPRI_HEADER, ETH_OVERHEAD  # noqa: E402

UP71 = [f'bfp{w}' for w in range(4, 10)] + ['none']
BFW = list(range(4, 10)) + [12]
CI_SCHEMES = ['raw', 'st6_bfp4', 'st6_bfp6', 'st6_bfp9', 'klt_oracle', 'klt_twin2']
CONFIGS = ([('7-1', up, None, 'raw') for up in UP71]
           + [('7-2c', 'modcomp', b, ci) for ci in CI_SCHEMES for b in BFW]
           + [('7-2l', 'modcomp', None, 'nu0'), ('7-2l', 'modcomp', None, 'nu0.5')])
N_RE = dz.N_RE
UE_PER_SRS_SYMBOL = 16
SLOTS_PER_S = 2000


def rzf_central(h_est, noise_mw, nu=0.0):
    """h_est [F, K, LM] → W [F, LM, K]。UE k 的功率 ∝ (网络平均增益)^ν，再统一缩放到每 O-RU 功率 ≤ P_RU。"""
    f_, k_, _ = h_est.shape
    alpha = k_ * noise_mw / (g.L * g.P_RU_PRB_MW)
    hh = np.conj(np.swapaxes(h_est, 1, 2))
    w = hh @ np.linalg.inv(h_est @ hh + alpha * np.eye(k_))
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    gain = np.mean(np.abs(h_est) ** 2, axis=(0, 2)) ** nu
    w = w * np.sqrt(gain / gain.mean())[None, None, :]
    per_ru = np.sum(np.abs(w.reshape(f_, g.L, g.M, k_)) ** 2, axis=(2, 3))     # [F, L]
    return w * np.sqrt(g.P_RU_PRB_MW / per_ru.max(1))[:, None, None]


def rzf_local(h_est, noise_mw, nu=0.0):
    """每个 O-RU 只用自己的信道做 RZF；UE k 分到 P_RU · g_kl^ν / Σ_j g_jl^ν（g 为本 O-RU 看到的平均增益）。"""
    f_, k_, _ = h_est.shape
    alpha = k_ * noise_mw / g.P_RU_PRB_MW
    w = np.zeros((f_, g.L * g.M, k_), complex)
    for l_ in range(g.L):
        hl = h_est[:, :, l_ * g.M:(l_ + 1) * g.M]
        hh = np.conj(np.swapaxes(hl, 1, 2))
        wl = hh @ np.linalg.inv(hl @ hh + alpha * np.eye(k_))
        wl /= np.linalg.norm(wl, axis=1, keepdims=True)
        gain = np.mean(np.abs(hl) ** 2, axis=(0, 2)) ** nu
        w[:, l_ * g.M:(l_ + 1) * g.M] = wl * np.sqrt(g.P_RU_PRB_MW * gain / gain.sum())[None, None, :]
    return w


def sinr_exact(h, w, noise_mw):
    gm = h @ w                                                     # [F, K, K]
    p = np.abs(gm) ** 2
    sig = np.einsum('fkk->fk', p)
    return sig / (p.sum(-1) - sig + noise_mw)


def ci_reconstructions(rng, h, rays, noise_ci):
    """各 CI 方案下 DU 得到的信道 [F, K, LM]，以及每 O-RU 每次 SRS 的 CI 净荷比特（raw 在负载里另算）。"""
    f_, k_, lm = h.shape
    h_noisy = h + np.sqrt(noise_ci / 2) * (rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape))
    out = {'raw': (h_noisy, None)}
    # ST 6（SRS-CE）：每条链路按自身平均功率定点化（相当于逐 UE 的增益归一），每 16 个 TRX 一块 BFP
    links = h_noisy.reshape(f_, k_, g.L, g.M)
    scale = np.sqrt(np.mean(np.abs(links) ** 2, axis=(0, 3), keepdims=True))
    for w in (4, 6, 9):
        rec = dz.apply(f'bfp{w}', links / scale, 1.0, 16) * scale
        bits = np.full(g.L, k_ * f_ * 2 * (16 * 2 * w + 8), float)
        out[f'st6_bfp{w}'] = (rec.reshape(f_, k_, lm), bits)
    for name, err in (('klt_oracle', 0.0), ('klt_twin2', 2.0)):
        rec = np.empty_like(h_noisy)
        bits = np.zeros(g.L)
        for k in range(k_):
            for l_ in range(g.L):
                v = g.ray_vectors(g.perturb(rng, rays[k][l_], err))          # [F·M, R]
                lam, e = np.linalg.eigh(v.conj().T @ v)                       # 孪生特征值 = Gram 特征值
                keep = lam > noise_ci
                uk = (v @ e[:, keep]) / np.sqrt(lam[keep])
                x = h_noisy[:, k, l_ * g.M:(l_ + 1) * g.M].reshape(-1)
                c = uk.conj().T @ x
                c = c + np.sqrt(noise_ci / 2) * (rng.standard_normal(c.shape) + 1j * rng.standard_normal(c.shape))
                rec[:, k, l_ * g.M:(l_ + 1) * g.M] = (uk @ c).reshape(f_, g.M)
                bits[l_] += np.sum(np.log2((lam[keep] + noise_ci) / noise_ci) + 0.508) + 10
        out[name] = (rec, bits)
    return out


def quantize_weights(w, width):
    """SE-11：每 O-RU、每 UE、每组的 32 个权值为一块做 BFP（12 为 BFP12）。"""
    f_, lm, k_ = w.shape
    w_hat = np.empty_like(w)
    nominal = g.P_RU_PRB_MW / (g.M * k_)
    for l_ in range(g.L):
        blk = np.swapaxes(w[:, l_ * g.M:(l_ + 1) * g.M, :], 1, 2)          # [F, K, M]
        w_hat[:, l_ * g.M:(l_ + 1) * g.M, :] = np.swapaxes(dz.apply(f'bfp{width}', blk, nominal, g.M), 1, 2)
    return w_hat


class Drop2:
    """ci 为 ci_reconstructions 的结果（与噪声水平无关，一个撒点算一次）。"""

    def __init__(self, h, ci, rng, noise_mw, se_model):
        self.h, self.noise, self.model = h, noise_mw, se_model
        self.ci = ci
        # DU 按自己的 CI 预测和速率，在 ν ∈ {0, 0.5} 中选集中式功率规则（不用真实信道）
        h_du = self.ci['raw'][0]
        pred = {nu: cfsim.se_from_sinr(sinr_exact(h_du, rzf_central(h_du, noise_mw, nu), noise_mw), se_model).sum()
                for nu in (0.0, 0.5)}
        self.nu_central = max(pred, key=pred.get)
        self.w_ideal = rzf_central(h_du, noise_mw, self.nu_central)
        se_ideal_ue = cfsim.se_from_sinr(sinr_exact(h, self.w_ideal, noise_mw), se_model).mean(0)
        f_, k_, _ = h.shape
        qm = cfsim.qm_from_se(se_ideal_ue)
        self.s = np.stack([dz.qam(rng, qm[k], (f_, N_RE)) for k in range(k_)], axis=1)   # [F, K, N_RE]
        self.ss = np.sum(np.abs(self.s) ** 2, axis=2)
        self.se_ideal = self._se(h @ (self.w_ideal @ self.s))

    def _se(self, y):
        ys = np.sum(y * np.conj(self.s), axis=2)
        yy = np.sum(np.abs(y) ** 2, axis=2)
        return dz.se_per_ue(dz.sinr_from_stats(ys, yy, self.ss, self.noise), self.model)

    def evaluate(self):
        """返回 {config: 每 UE 频谱效率}。"""
        out = {}
        w_by_ci = {ci: (self.w_ideal if ci == 'raw' else rzf_central(rec, self.noise, self.nu_central))
                   for ci, (rec, _) in self.ci.items()}
        x_raw = w_by_ci['raw'] @ self.s                                    # 7-1 用 raw CI 的 RZF
        for up in UP71:
            x_hat = dz.apply(up, x_raw, g.P_RU_PRB_MW / g.M, 12)
            out[('7-1', up, None, 'raw')] = self._se(self.h @ x_hat)
        for ci in CI_SCHEMES:
            for b in BFW:
                w_hat = quantize_weights(w_by_ci[ci], b)
                out[('7-2c', 'modcomp', b, ci)] = self._se(self.h @ (w_hat @ self.s))
        for nu in ('nu0', 'nu0.5'):
            w_loc = rzf_local(self.ci['raw'][0], self.noise, float(nu[2:]))
            out[('7-2l', 'modcomp', None, nu)] = self._se(self.h @ (w_loc @ self.s))
        return out


def _rate_up(n_eaxc, bits_per_prb, symbols=14, slots=SLOTS_PER_S):
    return fhload.uplane_rate(int(n_eaxc), g.NUM_PRB, int(bits_per_prb), symbols, slots)[1]


def _cp_bits(payload_bits):
    """非定时 C-plane 消息（ST 6）的总比特：净荷 + 每 1450 字节一条消息的 eCPRI/以太网/C-plane 头。"""
    n_msg = math.ceil(payload_bits / 8 / 1450)
    return payload_bits + n_msg * (ETH_OVERHEAD + ECPRI_HEADER + CPLANE_HEADERS) * 8


def loads(cfg, num_ue, qm, ci_bits):
    """每 O-RU 的下行负载（bit/s）和每次 SRS 的上行 CI 比特。"""
    fam, up, b, ci = cfg
    srs_sym = -(-num_ue // UE_PER_SRS_SYMBOL)
    raw_bits = _rate_up(g.M, fhload.bits_per_prb('bfp9'), srs_sym, 1)
    if fam == '7-1':
        dl = _rate_up(g.M, fhload.bits_per_prb(up)) + fhload.cplane_rate(g.M, SLOTS_PER_S)
        return np.full(g.L, dl), np.full(g.L, raw_bits)
    data = sum(_rate_up(1, 12 * qm[k]) for k in range(num_ue))
    if fam == '7-2l':
        return np.full(g.L, data + fhload.cplane_rate(num_ue, SLOTS_PER_S)), np.zeros(g.L)
    dl = data + fhload.cplane_rate(num_ue, SLOTS_PER_S, g.F, g.M, fhload.bfw_block_bits(g.M, b))
    ul = np.full(g.L, raw_bits) if ci == 'raw' else np.array([_cp_bits(x) for x in ci_bits[ci]])
    return np.full(g.L, dl), ul
