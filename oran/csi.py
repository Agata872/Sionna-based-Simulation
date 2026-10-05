"""UE 侧 CSI（L1）：CQI 表、CSI 参考资源、等效信道上的 LMMSE SINR、宽带 CQI。

对应 TS 38.214（Rel-18，V18.2.0）：
  - 5.2.2.1：CQI 定义——在 CSI 参考资源上按该 CQI 的调制和码率传一个 PDSCH TB，TB 误块率不超过 0.1 的
    最大 CQI 序号。表 5.2.2.1-2（cqi-Table 'table1'，最高 64QAM）、表 5.2.2.1-3（'table2'，最高 256QAM）。
    表 5.2.2.1-4（'table3'，目标 1e-5）超出 BLER 表的精度，未实现。
  - 5.2.2.5.1：CSI 参考资源的假设——前 2 个 OFDM 符号为控制区，PDSCH+DMRS 共 12 个符号，DMRS 个数与配置
    相同且 DMRS 符号不放数据，没有 CSI-RS / SSB，PRB bundling 为 2，RV0。CQI 对应的 TBS 按这样的 RE 数计算。
  - 5.2.1.4.2：reportQuantity = 'cri-RI-CQI'——gNB 发预编码（波束赋形）的 NZP CSI-RS，每个端口对应一层，
    UE 直接在等效信道 H_k·W_k 上算 CQI；同调度 UE 的干扰由 CSI-IM / 干扰测量用 NZP CSI-RS 测得
    （5.2.2.4）。这与 TDD 互易（O-DU 由 SRS 得到 CSI 并做 BD 预编码）相配。

简化（第一版）：UE 不估计 CSI-RS，而是直接用真实等效信道 H_k·W 和同调度 UE 的干扰 H_k·W_j，算出每个 RE、
每层的 LMMSE 输出 SINR（"精灵"测量），经 EESM 得到有效 SINR，再与各 CQI 的 AWGN 门限比较。
CSI 不经过前传，因此看不到前传压缩失真；上报（UCI）无误码、无时延；层数固定，不上报 RI。
"""

from typing import List, Sequence, Tuple

import numpy as np
import torch

from .bler_model import AWGNLinkModel, Entry
from .precoding import effective_channel

# TS 38.214 表 5.2.2.1-2 / 5.2.2.1-3：CQI 序号 1~15 的 (Qm, 码率 × 1024)，序号 0 为 "out of range"
CQI_TABLES = {
    1: [(2, 78), (2, 120), (2, 193), (2, 308), (2, 449), (2, 602), (4, 378), (4, 490), (4, 616),
        (6, 466), (6, 567), (6, 666), (6, 772), (6, 873), (6, 948)],
    2: [(2, 78), (2, 193), (2, 449), (4, 378), (4, 490), (4, 616), (6, 466), (6, 567), (6, 666),
        (6, 772), (6, 873), (8, 711), (8, 797), (8, 885), (8, 948)],
}

NUM_CONTROL_SYMBOLS = 2         # CSI 参考资源：前 2 个符号为控制区
NUM_PDSCH_SYMBOLS = 12          # CSI 参考资源：PDSCH + DMRS 共 12 个符号


def reference_data_symbols(pdsch_config) -> List[int]:
    """CSI 参考资源中承载 PDSCH 数据的 OFDM 符号（符号 2~13 中除去 DMRS 符号）。

    PDSCH 映射类型 A 的 DMRS 位置相对时隙起点定义，持续到时隙末尾，因此与实际 PDSCH 的 DMRS 位置相同。
    """
    dmrs = set(int(l) for l in pdsch_config.dmrs_symbol_indices)
    start = NUM_CONTROL_SYMBOLS
    return [l for l in range(start, start + NUM_PDSCH_SYMBOLS) if l not in dmrs]


def reference_num_re(pdsch_config, num_prb: int, num_layers: int) -> int:
    """CSI 参考资源中一个 TB 占用的 RE 数（含各层），用于 CQI 对应的 TBS。"""
    return 12 * num_prb * len(reference_data_symbols(pdsch_config)) * num_layers


def lmmse_sinr(h: torch.Tensor, w: torch.Tensor, no: float, prg_of_sc: torch.Tensor,
               num_layers_per_ue: int) -> torch.Tensor:
    """各 UE 在等效信道上的 LMMSE 输出 SINR（本 UE 的层为期望信号，其他 UE 的层为干扰并做白化）。

    对 UE k：G_k 为本 UE 各层的等效信道（N_r × ν），R = Σ_{j≠k} G_j G_j^H + N0·I，
    SINR_s = 1 / [(I + G_k^H R^{-1} G_k)^{-1}]_ss − 1，与 Sionna LMMSEEqualizer 的 1 / no_eff 相同。

    :param h: [B, K, N_r, N_t, num_symbols, num_sc]
    :param w: [B, num_prg, N_t, L]，||W||_F = 1
    :return: [B, K, ν, num_symbols, num_sc]（线性）
    """
    g = effective_channel(h, w, prg_of_sc)                  # [B, K, N_r, L, S, F]
    g = g.permute(0, 1, 4, 5, 2, 3)                         # [B, K, S, F, N_r, L]
    num_ue, num_rx_ant = g.shape[1], g.shape[-2]
    nu = num_layers_per_ue
    eye_r = torch.eye(num_rx_ant, dtype=g.dtype, device=g.device)
    eye_nu = torch.eye(nu, dtype=g.dtype, device=g.device)
    out = []
    for k in range(num_ue):
        gk = g[:, k]
        desired = gk[..., k * nu:(k + 1) * nu]
        interf = torch.cat([gk[..., :k * nu], gk[..., (k + 1) * nu:]], dim=-1)
        r = interf @ interf.mH + no * eye_r
        a = desired.mH @ torch.linalg.solve(r, desired)
        e = torch.linalg.inv(eye_nu + a)
        sinr = 1 / torch.diagonal(e, dim1=-2, dim2=-1).real - 1
        out.append(sinr.clamp_min(0).permute(0, 3, 1, 2))   # [B, ν, S, F]
    return torch.stack(out, dim=1)


class CQIReporter:
    """宽带 CQI（每个 UE 一个码字）：满足 TBLER ≤ 目标的最大 CQI 序号。

    :param cqi_table: 1 或 2（CSI-ReportConfig 的 cqi-Table）
    :param num_re_ref: CSI 参考资源中 TB 的 RE 数（含各层），见 :func:`reference_num_re`
    """

    def __init__(self, cqi_table: int, num_re_ref: int, model: AWGNLinkModel):
        if cqi_table not in CQI_TABLES:
            raise ValueError("只支持 CQI 表 1、2（cqi-Table 'table1' / 'table2'）")
        self.cqi_table = cqi_table
        self.entries: List[Entry] = [(q, float(r)) for q, r in CQI_TABLES[cqi_table]]
        self.num_re_ref = num_re_ref
        self.model = model
        # 各 CQI（序号 1~15）在参考资源上 TBLER = 目标值时的有效 SINR [dB]
        self.threshold_db = np.array([model.threshold_db(e, num_re_ref) for e in self.entries])
        self.betas = [model.beta(e) for e in self.entries]

    def __call__(self, sinr: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param sinr: [K, ν, ...] 各 UE 在参考资源数据 RE 上的线性 SINR
        :return: cqi [K]（0 表示 out of range），以及按所报 CQI 的 β 算出的有效 SINR [dB]
                 （CQI 为 0 时按 CQI 1 的 β）
        """
        cqi, eff = [], []
        for k in range(sinr.shape[0]):
            eff_db = self.model.effective_sinr_db(sinr[k], self.betas)      # 每个 CQI 一个值
            ok = np.flatnonzero(eff_db >= self.threshold_db)
            c = int(ok[-1]) + 1 if len(ok) else 0
            cqi.append(c)
            eff.append(eff_db[max(c, 1) - 1])
        return np.array(cqi), np.array(eff)
