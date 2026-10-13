"""阶段 1b 的核心：一个撒点上，前传压缩失真经过真实编解码器后对每 UE SINR 的影响。

发送端（每 PRB 一组权值 W，单位与 cfsim 相同：每天线每 PRB 功率 ≤ Pmax）：
  - 7-1（≈ Cat A）：DU 算 x = W s，逐天线、逐 PRB、逐符号 12 个 RE 为一块做 U-plane 压缩，RU 直接发送 x̂；
  - 7-2（≈ Cat B）：DU 压缩层符号 ŝ（BFP，或无损的调制压缩）；ZF 时 DU 用 SE-11 下发权值，
    每个 RU 每个 UE 每个 PRB 的 K_r 个权值为一块做 BFP（bundle = 1 PRB），RU 发 x̂ = Ŵ ŝ；
    MRT 时 RU 用本地 SRS 自算权值，不经前传、不量化。
编解码器全部来自 oran/compression（经 oran.fronthaul.IQCodec：先按标称功率 + 15 dB 回退定点化到 16 bit）。

接收端：UE 按每个 PRB 的 DMRS 估计有效增益，等价于对 168 个 RE 做 LMMSE：
    g = Σ y s* / Σ|s|²，SINR = (|Σ y s*|² / Σ|s|²) / (Σ|y|² − |Σ y s*|² / Σ|s|² + n·N)，
其中 y 不含热噪声，n = 168。干扰、压缩失真、权值误差、与信号相关的失真分量都由此自动计入。
7-2 用相关矩阵避免逐配置重算：C = Σ ŝ ŝᴴ、D = Σ ŝ sᴴ（每 PRB），则 Σ y_u s_u* = (G D)_uu，Σ|y_u|² = (G C Gᴴ)_uu。
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
import cfsim  # noqa: E402
from oran.config import CompressionConfig  # noqa: E402
from oran.fronthaul import IQCodec  # noqa: E402

N_RE = 14 * 12          # 一个时隙内每 PRB 的 RE 数
UP71 = [f'bfp{w}' for w in range(4, 13)] + ['none']
UP72 = UP71 + ['modcomp']
BFW = list(range(4, 13)) + [16]


def qam(rng, qm, shape):
    m = int(round(2 ** (qm / 2)))
    lv = np.arange(m) * 2 - (m - 1)
    norm = np.sqrt(2 * (m * m - 1) / 3)
    return (lv[rng.integers(0, m, shape)] + 1j * lv[rng.integers(0, m, shape)]) / norm


def codec(option, nominal, block_len):
    """option: 'bfpW' / 'none' / 'modcomp'（无损，返回 None）。"""
    if option == 'modcomp':
        return None
    if option == 'none':
        return IQCodec(CompressionConfig('none', 16), nominal, block_len)
    return IQCodec(CompressionConfig('bfp', int(option[3:])), nominal, block_len)


def apply(option, x, nominal, block_len):
    c = codec(option, nominal, block_len)
    if c is None:
        return x
    t = torch.from_numpy(np.ascontiguousarray(x)).to(torch.complex128)
    return c(t).numpy()


def ideal_weights(h_s, bf, assign_s):
    """h_s [B, U, S]。返回 W [B, S, U]（每天线每 PRB 功率 ≤ Pmax）。"""
    b_, u_, s_ = h_s.shape
    if bf == 'zf':
        g = h_s @ np.conj(np.swapaxes(h_s, 1, 2))
        w = np.conj(np.swapaxes(h_s, 1, 2)) @ np.linalg.inv(g)
        if cfsim.ZF_NORM != 'inversion':                         # [10] 式 (18) 不做列归一化
            w = w / np.linalg.norm(w, axis=1, keepdims=True)
        p = cfsim.P_MAX_PRB_MW / np.max(np.sum(np.abs(w) ** 2, axis=2), axis=1)
        return w * np.sqrt(p)[:, None, None]
    w = np.zeros((b_, s_, u_), complex)
    for j, u in enumerate(assign_s):
        hm = h_s[:, u, j]
        w[:, j, u] = np.sqrt(cfsim.P_MAX_PRB_MW) * np.conj(hm) / np.abs(hm)
    return w


def sinr_from_stats(ys, yy, ss, noise_mw):
    """ys = Σ y s*，yy = Σ|y|²，ss = Σ|s|²，形状 [B, U]。"""
    sig = np.abs(ys) ** 2 / ss
    return sig / (np.maximum(yy - sig, 0) + N_RE * noise_mw)


def se_per_ue(sinr, se_model):
    return cfsim.se_from_sinr(sinr, se_model).mean(0)


class DropEval:
    """一个撒点、一个状态：保存理想权值与符号，按配置返回每 UE 频谱效率。"""

    def __init__(self, h, serving, assign_s, bf, noise_mw, se_model, qm, rng):
        self.h_s = h[:, :, serving]                         # [B, U, S]
        self.serving = np.asarray(serving)
        self.bf, self.noise, self.model = bf, noise_mw, se_model
        b_, u_, _ = self.h_s.shape
        self.w = ideal_weights(self.h_s, bf, assign_s)       # [B, S, U]
        self.s = np.stack([qam(rng, qm[u], (b_, N_RE)) for u in range(u_)], axis=1)   # [B, U, N_RE]
        self.ss = np.sum(np.abs(self.s) ** 2, axis=2)        # [B, U]
        self.x = self.w @ self.s                             # [B, S, N_RE]
        self._c72 = {}

    def se_ideal(self):
        y = self.h_s @ self.x
        return self._se(y)

    def _se(self, y):
        ys = np.sum(y * np.conj(self.s), axis=2)
        yy = np.sum(np.abs(y) ** 2, axis=2)
        return se_per_ue(sinr_from_stats(ys, yy, self.ss, self.noise), self.model)

    def se_71(self, option):
        x_hat = apply(option, self.x, cfsim.P_MAX_PRB_MW, 12)
        return self._se(self.h_s @ x_hat)

    def _stats72(self, option):
        if option not in self._c72:
            s_hat = apply(option, self.s, 1.0, 12)
            c = s_hat @ np.conj(np.swapaxes(s_hat, 1, 2))          # [B, U, U]
            d = s_hat @ np.conj(np.swapaxes(self.s, 1, 2))         # [B, U, U]，D[u', u] = Σ ŝ_u' s_u*
            self._c72[option] = (c, d)
        return self._c72[option]

    def weights_72(self, k, bfw_width):
        """按 RU 分块量化的权值 Ŵ [B, S, U]。MRT 不量化。"""
        if self.bf != 'zf' or bfw_width is None:
            return self.w
        w_hat = np.empty_like(self.w)
        ru = self.serving // k
        nominal = cfsim.P_MAX_PRB_MW / self.w.shape[2]
        opt = 'none' if bfw_width >= 16 else f'bfp{bfw_width}'
        for r in np.unique(ru):
            idx = np.where(ru == r)[0]
            blk = np.swapaxes(self.w[:, idx, :], 1, 2)                # [B, U, n_r]：一块 = 一个 UE 的 n_r 个权值
            w_hat[:, idx, :] = np.swapaxes(apply(opt, blk, nominal, len(idx)), 1, 2)
        return w_hat

    def se_72(self, option, w_hat):
        c, d = self._stats72(option)
        g = self.h_s @ w_hat                                          # [B, U, U]
        ys = np.einsum('bij,bji->bi', g, d)
        yy = np.real(np.einsum('bij,bjk,bik->bi', g, c, np.conj(g)))
        return se_per_ue(sinr_from_stats(ys, yy, self.ss, self.noise), self.model)
