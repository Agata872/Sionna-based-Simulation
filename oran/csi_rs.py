"""CSI-RS / CSI-IM：测量时隙中的发送（O-DU）与 UE 侧的 CSI 估计。

资源配置（TS 38.211 V18.6.0 §7.4.1.5；TS 38.214 V18.2.0 §5.2.1.4、§5.2.2.3.1、§5.2.2.4）：
  - 每个 UE 一个 ν 端口 NZP CSI-RS 作为信道测量资源（CMR），用该 UE 的预编码权值发送（波束赋形 CSI-RS），
    端口 i 对应第 i 层（reportQuantity = 'cri-RI-CQI'，未配置 non-PMI-PortIndication）。
    ν = 1、2、4 分别对应表 7.4.1.5.3-1 的第 2、3、4 行，密度 1；ν ≥ 2 时为 fd-CDM2。
  - MU：其他同调度 UE 的 CMR 同时作为本 UE 的干扰测量资源（NZP-IMR），每个端口是一个干扰层。
  - CSI-IM：pattern1（1 个符号 × 4 个子载波），测噪声与小区间干扰。
  - 全部位于 OFDM 符号 l0 = cfg.csi_rs_symbol（默认 12，避开 DMRS 与控制区）。每个 PRB 内，UE k 的 CMR
    从子载波 k·span 开始（span = 1、2、4），CSI-IM 位于其后第一个 4 的倍数处（subcarrierLocation-p1 ∈ {0, 4, 8}），
    该符号其余子载波仍放 PDSCH。默认 2 UE × 2 层：子载波 0–1、2–3 为 UE0、UE1 的 CMR，4–7 为 CSI-IM。
  - 序列：r(m) = [(1 − 2c(2m)) + j(1 − 2c(2m+1))] / √2，c_init = (2^10·(14·n_s + l + 1)·(2·n_ID + 1) + n_ID) mod 2^31；
    映射 a = β·w_f(k')·w_t(l')·r(m')，m' = ⌊n·α⌋ + k' + ⌊k̄·ρ/12⌋，α = ρ（X = 1）或 2ρ（X > 1），
    w_f 见表 7.4.1.5.3-3（fd-CDM2：端口 0 为 [+1, +1]，端口 1 为 [+1, −1]）。
  - powerControlOffset = 0 dB：每端口 EPRE 与 PDSCH 每层 EPRE 相同（β = 1）。
  - 测量时隙的 PDSCH 在这些 RE 上打孔（规范中本 UE 的 NZP CSI-RS 不用于 PDSCH，其他 UE 的 CSI-RS 与 CSI-IM
    配置为 ZP CSI-RS 做速率匹配）。测量时隙的 PDSCH 只用来让前传压缩块的内容真实，不译码。

UE 侧估计（:class:`CSIRSReceiver`）：
  1. 定时：用测量时隙中 PDSCH DMRS 估计的相位斜率（与 DMRS 信道估计的定时偏差补偿相同）去旋转接收网格；
  2. 每个 PRB、每个 CDM 组：y / r 后按 w_f 解扩，得到各端口的等效信道 Ĝ（N_r × ν）；
  3. CSI-IM 上接收功率的平均值作为噪声（及小区间干扰）方差 σ²；
  4. 其他 UE 的端口（NZP-IMR）给出干扰协方差 Σ_j Ĝ_j Ĝ_j^H，在 PRG 内平均，扣除估计噪声 ν_j·σ²/L
    （L 为 CDM 长度）后投影为半正定；
  5. 每个 PRB、每层的 LMMSE SINR：1 / [(I + Ĝ_k^H R^{-1} Ĝ_k)^{-1}]_ss − 1，R = 干扰协方差 + σ²·I。
     本 UE 信道估计中的噪声不做修正（低 SNR 时 SINR 略偏高，由 OLLA 吸收）。

注意：三种压缩算法都把值为 0 的 RE 还原为严格的 0，因此 CSI-IM 上看不到前传压缩失真；U-plane 失真只以
CMR 信道估计误差的形式进入 CQI。Cat B 权值压缩造成的 BD 零陷误差会出现在配对 UE 的 CSI-RS 上，被 NZP-IMR 测到。
"""

from typing import Dict, Tuple

import numpy as np
import torch
from sionna.phy.nr.utils import generate_prng_seq

from .config import ORANConfig

# 表 7.4.1.5.3-1：端口数 -> (行号, CDM 长度, 各 CDM 组相对 k0 的子载波偏移)
CSI_RS_ROWS = {1: (2, 1, [0]), 2: (3, 2, [0]), 4: (4, 2, [0, 2])}
# 表 7.4.1.5.3-3（fd-CDM2）：w_f[s][k']
W_F = {1: np.array([[1.0]]), 2: np.array([[1.0, 1.0], [1.0, -1.0]])}


class CSIRS:
    """测量时隙中所有 UE 的 CMR 与 CSI-IM（同一个 OFDM 符号）。

    :param pdsch_config: 任一 UE 的 PDSCH 配置（用于检查与 DMRS 不重叠）
    """

    def __init__(self, cfg: ORANConfig, pdsch_config, device=None):
        nu = cfg.num_layers_per_ue
        if nu not in CSI_RS_ROWS:
            raise ValueError("CSI-RS 只支持每 UE 1、2、4 层（表 7.4.1.5.3-1 第 2、3、4 行）")
        self.row, self.cdm, group_offsets = CSI_RS_ROWS[nu]
        self.num_ue = cfg.num_ue
        self.num_ports = nu
        self.num_prb = cfg.num_prb
        self.symbol = int(cfg.csi_rs_symbol)
        if self.symbol in [int(l) for l in pdsch_config.dmrs_symbol_indices] or self.symbol < 2:
            raise ValueError(f"CSI-RS 符号 {self.symbol} 与 DMRS 或控制区重叠")
        span = max(nu, 1)
        self.im_offset = int(4 * np.ceil(cfg.num_ue * span / 4))     # subcarrierLocation-p1
        if self.im_offset + 4 > 12:
            raise ValueError("一个 PRB 内放不下所有 UE 的 CMR 和 CSI-IM，请减少 UE 数或层数")

        # 序列（密度 ρ = 1）：c_init 按 TS 38.211 7.4.1.5.2
        n_id = cfg.n_cell_id if cfg.csi_rs_scrambling_id is None else cfg.csi_rs_scrambling_id
        slot = cfg.slot_number
        c_init = (2 ** 10 * (14 * slot + self.symbol + 1) * (2 * n_id + 1) + n_id) % 2 ** 31
        alpha = 1 if nu == 1 else 2
        num_m = alpha * self.num_prb + self.cdm
        c = generate_prng_seq(2 * num_m, c_init)
        r = ((1 - 2 * c[0::2]) + 1j * (1 - 2 * c[1::2])) / np.sqrt(2)

        num_sc = 12 * self.num_prb
        prb = np.arange(self.num_prb)
        # 每个 (UE, CDM 组)：子载波下标与序列值，[num_prb, cdm]
        self.cmr_sc, self.cmr_seq = [], []
        layers = np.zeros([cfg.num_ue * nu, num_sc], complex)       # 测量符号上各层的 CSI-RS
        mask = np.zeros(num_sc, bool)                                # 该符号上 CSI-RS / CSI-IM 占用的 RE
        w_f = W_F[self.cdm]
        for k in range(cfg.num_ue):
            sc_k, seq_k = [], []
            for j, off in enumerate(group_offsets):
                k_bar = k * span + off
                kp = np.arange(self.cdm)
                sc = 12 * prb[:, None] + k_bar + kp[None, :]
                m = alpha * prb[:, None] + kp[None, :] + (k_bar * 1) // 12
                seq = r[m]
                for s in range(self.cdm):
                    port = s + j * self.cdm
                    layers[k * nu + port, sc] = w_f[s][None, :] * seq
                mask[sc] = True
                sc_k.append(sc)
                seq_k.append(seq)
            self.cmr_sc.append(np.stack(sc_k))                       # [num_groups, num_prb, cdm]
            self.cmr_seq.append(np.stack(seq_k))
        self.im_sc = 12 * prb[:, None] + self.im_offset + np.arange(4)[None, :]   # [num_prb, 4]
        mask[self.im_sc] = True

        self.device = device
        self._layers = torch.as_tensor(layers, dtype=torch.complex64, device=device)
        self._mask = torch.as_tensor(mask, device=device)

    @property
    def overhead_re_per_prb(self) -> int:
        """测量符号上每个 PRB 被 CSI-RS / CSI-IM 占用的 RE 数。"""
        return int(self._mask.sum()) // self.num_prb

    def insert(self, x_layers: torch.Tensor) -> torch.Tensor:
        """把 CSI-RS / CSI-IM 放进层资源网格（PDSCH 在这些 RE 上打孔）。

        :param x_layers: [batch, L, num_symbols, num_sc]
        """
        x = x_layers.clone()
        sym = x[:, :, self.symbol, :]
        sym = torch.where(self._mask, torch.zeros_like(sym), sym) + self._layers.to(sym.dtype)
        x[:, :, self.symbol, :] = sym
        return x


class CSIRSReceiver:
    """UE 侧：由测量时隙的接收网格估计 CMR / NZP-IMR / CSI-IM，得到每 PRB 每层的 SINR。

    :param prg_num_prb: 预编码粒度（干扰协方差在 PRG 内平均）
    :param dmrs_estimator: 用于定时偏差估计的 DMRS 估计器（:class:`~oran.channel_estimation.DMRSChannelEstimator`），
        None 时不做定时补偿
    """

    def __init__(self, csirs: CSIRS, prg_num_prb: int, dmrs_estimator=None):
        self.csirs = csirs
        self.dmrs_estimator = dmrs_estimator
        dev = csirs.device
        self._cmr_sc = torch.as_tensor(np.stack(csirs.cmr_sc), device=dev)         # [K, G, P, cdm]
        self._cmr_seq = torch.as_tensor(np.stack(csirs.cmr_seq), dtype=torch.complex64, device=dev)
        self._im_sc = torch.as_tensor(csirs.im_sc.reshape(-1), device=dev)
        self._w_f = torch.as_tensor(W_F[csirs.cdm], dtype=torch.float32, device=dev)  # [s, k']
        self._prg_of_prb = torch.arange(csirs.num_prb, device=dev) // prg_num_prb
        self._num_prg = int(self._prg_of_prb.max()) + 1

    def estimate(self, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        :param y: 频域接收网格 [B, K, N_r, num_symbols, num_sc]
        :return: dict：
            g      各 UE 端口的信道估计 [B, K_rx, N_r, num_prb, K_tx, ν]
            noise  CSI-IM 测得的噪声方差 [B, K_rx]
        """
        if self.dmrs_estimator is not None and getattr(self.dmrs_estimator, "_delay_compensation", False):
            slope = self.dmrs_estimator.phase_slope(y)                         # [B, K]
            k = self.dmrs_estimator._sc_centered.to(slope.dtype)
            rot = torch.polar(torch.ones_like(slope)[..., None], -slope[..., None] * k)
            y = y * rot[:, :, None, None, :].to(y.dtype)
        ys = y[..., self.csirs.symbol, :]                                       # [B, K, N_r, F]
        noise = (ys[..., self._im_sc].abs() ** 2).mean(dim=(-1, -2))           # [B, K]
        num_ue_tx, num_groups, num_prb, cdm = self._cmr_sc.shape
        z = ys[..., self._cmr_sc.reshape(-1)].reshape(*ys.shape[:3], num_ue_tx, num_groups, num_prb, cdm)
        z = z / self._cmr_seq.to(z.dtype)                                       # [B, K, N_r, K_tx, G, P, cdm]
        # 解扩：端口 s + j·cdm 的估计 = mean_k' w_f[s][k'] · z
        h = torch.einsum("buatgpk,sk->buatpgs", z, self._w_f.to(z.dtype)) / cdm
        h = h.reshape(*h.shape[:5], num_groups * cdm)                           # [B, K, N_r, K_tx, P, ν]
        return {"g": h.permute(0, 1, 2, 4, 3, 5), "noise": noise}

    def sinr(self, y: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        :return: sinr [B, K, ν, num_prb]（线性），以及诊断量：noise [B, K]、inr [B, K]
                 （NZP-IMR 测得的干扰功率与噪声之比，每接收天线平均）
        """
        est = self.estimate(y)
        g, noise = est["g"], est["noise"]                                       # g: [B, K, N_r, P, K_tx, ν]
        num_ue, num_rx_ant = g.shape[1], g.shape[2]
        nu, cdm = g.shape[-1], self.csirs.cdm
        eye_r = torch.eye(num_rx_ant, dtype=g.dtype, device=g.device)
        eye_nu = torch.eye(nu, dtype=g.dtype, device=g.device)
        out, inr = [], []
        for k in range(num_ue):
            gk = g[:, k].permute(0, 2, 1, 3, 4)                                 # [B, P, N_r, K_tx, ν]
            desired = gk[:, :, :, k, :]                                         # [B, P, N_r, ν]
            sigma2 = noise[:, k].to(g.real.dtype)[:, None, None, None]
            r_i = torch.zeros(*desired.shape[:2], num_rx_ant, num_rx_ant, dtype=g.dtype, device=g.device)
            if num_ue > 1:
                interf = torch.cat([gk[:, :, :, j, :] for j in range(num_ue) if j != k], dim=-1)
                cov = interf @ interf.mH                                        # [B, P, N_r, N_r]
                # PRG 内平均，扣除估计噪声 ν_j·σ²/L，投影为半正定
                prg = self._prg_of_prb
                cov_prg = torch.zeros(cov.shape[0], self._num_prg, *cov.shape[2:], dtype=cov.dtype,
                                      device=cov.device).index_add_(1, prg, cov)
                count = torch.bincount(prg, minlength=self._num_prg).to(cov.real.dtype)
                cov_prg = cov_prg / count[None, :, None, None]
                cov_prg = cov_prg - (num_ue - 1) * nu * sigma2 / cdm * eye_r
                lam, v = torch.linalg.eigh(cov_prg)
                cov_prg = (v * lam.clamp_min(0)[..., None, :].to(v.dtype)) @ v.mH
                r_i = cov_prg[:, prg]
            inr.append(torch.diagonal(r_i, dim1=-2, dim2=-1).real.mean(dim=(-1, -2)) / noise[:, k])
            r = r_i + sigma2 * eye_r
            a = desired.mH @ torch.linalg.solve(r, desired)
            e = torch.linalg.inv(eye_nu + a)
            s = 1 / torch.diagonal(e, dim1=-2, dim2=-1).real - 1                # [B, P, ν]
            out.append(s.clamp_min(0).permute(0, 2, 1))
        return torch.stack(out, dim=1), {"noise": noise, "inr": torch.stack(inr, dim=1)}
