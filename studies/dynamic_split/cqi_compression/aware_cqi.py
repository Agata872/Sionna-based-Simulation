"""考虑前传压缩失真的精灵 CQI。

精灵 CQI（:meth:`oran.closed_loop.ClosedLoopDownlink.measure_cqi_genie`）用真实等效信道 H·W 算每个 RE、
每层的 LMMSE SINR，不经前传，看不到压缩失真。这里在同一个式子里加入前传实际造成的失真：
  - 等效信道用 O-RU 实际使用的权值（Cat B 为解压后的权值，权值量化造成的残余多用户干扰随之进入 SINR；
    本 UE 自己的权值误差改变的是等效信道，UE 由 DMRS 估计吸收，不算失真）；
  - U-plane 失真 e = O-RU 天线口信号 − 用实际权值预编码的理想层信号，经信道到达 UE k：d_k = H_k·e。
UE 的 LMMSE 只知道热噪声（与 oran/ue.py 一致，见 oran/README 设计第 5 条），第 l 层的合并向量
f = R⁻¹·g_l，R = Σ_所有层 g·g^H + N0·I。记 a = g_l^H R⁻¹ g_l，无失真时 SINR = a / (1 − a)，
与 :func:`oran.csi.lmmse_sinr` 相同；加入失真后
    SINR = a² / (a − a² + f^H C_d f)，
C_d 为 d_k 在每个 PRB 上（CSI 参考资源的全部数据符号 × 12 个子载波）的样本协方差。

上式只计入数据 RE 上的失真。DMRS 与数据在同一个压缩块里，同样带有失真，使 UE 的信道估计变差，
这一部分只能实测：:func:`measure_cqi_oracle` 发一个经压缩前传的 PDSCH 时隙，UE 照常估计信道、均衡，
按 PRB、按层统计均衡后的 SINR（含噪声、干扰、信道估计误差和全部前传失真），再经 EESM 得到 CQI。
"""

from typing import Tuple

import numpy as np
import torch

from oran.channel import ChannelState
from oran.precoding import apply_precoder, effective_channel


def distortion_aware_sinr(h: torch.Tensor, w: torch.Tensor, no: float, prg_of_sc: torch.Tensor,
                          num_layers_per_ue: int, d: torch.Tensor) -> torch.Tensor:
    """各 UE 的 LMMSE 输出 SINR，计入接收端看到的前传失真 d。

    :param h: [B, K, N_r, N_t, S, F]
    :param w: [B, num_prg, N_t, L]，O-RU 实际使用的权值
    :param d: [B, K, N_r, S, F]，前传失真经信道到达各 UE 接收天线的分量
    :return: [B, K, ν, S, F]（线性）
    """
    g = effective_channel(h, w, prg_of_sc).permute(0, 1, 4, 5, 2, 3)     # [B, K, S, F, N_r, L]
    b, num_ue, s, f, num_rx_ant, _ = g.shape
    nu = num_layers_per_ue
    blocks = d.permute(0, 1, 3, 4, 2).reshape(b, num_ue, s, f // 12, 12, num_rx_ant)
    c = torch.einsum("bkspir,bkspiq->bkprq", blocks, blocks.conj()) / (s * 12)
    c = c.repeat_interleave(12, dim=2)                                    # [B, K, F, N_r, N_r]
    eye_r = torch.eye(num_rx_ant, dtype=g.dtype, device=g.device)
    out = []
    for k in range(num_ue):
        gk = g[:, k]
        desired = gk[..., k * nu:(k + 1) * nu]                            # [B, S, F, N_r, ν]
        comb = torch.linalg.solve(gk @ gk.mH + no * eye_r, desired)
        a = (desired.conj() * comb).sum(-2).real                          # [B, S, F, ν]
        dist = torch.einsum("bsfrv,bfrq,bsfqv->bsfv", comb.conj(), c[:, k], comb).real
        sinr = a ** 2 / (a - a ** 2 + dist).clamp_min(1e-30)
        out.append(sinr.clamp_min(0).permute(0, 3, 1, 2))                 # [B, ν, S, F]
    return torch.stack(out, dim=1)


@torch.no_grad()
def measure_cqi_aware(cl, state: ChannelState) -> Tuple[np.ndarray, np.ndarray]:
    """考虑压缩的精灵 CQI：用 state 第一个样本的信道发一个 PDSCH 时隙经过前传，取实际失真算 CQI。

    :param cl: :class:`~oran.closed_loop.ClosedLoopDownlink`（使用其当前压缩配置）
    :return: (cqi [K], 有效 SINR [K] dB)
    """
    sim, cfg = cl.sim, cl.cfg
    one = ChannelState(state.h_freq[:1])
    tx = sim.odu(1, sim.channel.csi(one, sim.csi_symbol))
    n = cfg.num_oru_ant
    ws, errs = [], []
    for o in range(cfg.num_oru):
        uplane_hat, w_hat = sim.fronthaul[o].transmit(tx.uplane[o], tx.weights[o])
        w_o = tx.w[:, :, o * n:(o + 1) * n] if w_hat is None else w_hat
        grid = sim.orus[o].frequency_domain(uplane_hat, w_hat)
        errs.append(grid - apply_precoder(tx.x_layers, w_o, sim.odu.prg_of_sc))
        ws.append(w_o)
    w, e = torch.cat(ws, dim=2), torch.cat(errs, dim=1)                  # [1, num_prg, N_t, L]、[1, N_t, S, F]
    h = one.stacked()
    d = torch.einsum("bkrtsf,btsf->bkrsf", h, e)
    ref = cl.ref_symbols
    sinr = distortion_aware_sinr(h.index_select(4, ref), w, 1.0, sim.odu.prg_of_sc,
                                 cfg.num_layers_per_ue, d.index_select(3, ref))
    return cl.reporter(sinr[0])


@torch.no_grad()
def measure_cqi_oracle(cl, state: ChannelState) -> Tuple[np.ndarray, np.ndarray]:
    """PDSCH 实测 CQI：用 state 第一个样本的信道发一个经压缩前传的 PDSCH 时隙，UE 照常接收，
    每个 PRB、每层的均衡后 SINR = 发送数据符号功率 / 均衡误差功率（Sionna LMMSE 输出无偏），再经 EESM。

    :return: (cqi [K], 有效 SINR [K] dB)
    """
    sim = cl.sim
    tx, y, _ = sim._transmit(1, 1.0, ChannelState(state.h_freq[:1]))   # 信道已含链路预算，噪声方差为 1
    rx = sim.ue(y, 1.0)
    mapper = sim.odu.rg_mapper                       # 映射回资源网格；两次映射相减，导频处为 0
    x = mapper(tx.x_data) - mapper(torch.zeros_like(tx.x_data))
    e = mapper(rx.x_hat.to(tx.x_data.dtype)) - mapper(tx.x_data)
    per_prb = lambda g: (g.abs() ** 2).unflatten(-1, (-1, 12)).sum((-3, -1))   # [1, K, ν, num_prb]
    sinr = per_prb(x) / per_prb(e).clamp_min(1e-30)
    return cl.reporter(sinr[0])
