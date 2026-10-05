"""UE 接收机（所有 UE 批量处理，Sionna 的 num_rx = UE 数）。

    （时域仿真时）去 CP + FFT，取出 PRB 所在子载波
    -> DMRS 信道估计：定时偏差补偿 + LS + PRG 内局部线性模型做 CDM 分离 + PRG 内线性插值
       或理想 CSI（真实信道乘实际使用的预编码）
    -> LMMSE 均衡（本 UE 的层为期望流，其他 UE 的层作为已估计信道的干扰做白化）
    -> 软解调 -> 解层映射 -> 解扰 + 速率解匹配 + LDPC 译码 + CRC 校验（TBDecoder）

UE 能估计所有 DMRS 端口（含同调度的其他 UE），因为各端口的 DMRS 序列由小区参数决定。
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from sionna.phy.mapping import Demapper
from sionna.phy.mimo import StreamManagement
from sionna.phy.nr import LayerDemapper, TBDecoder
from sionna.phy.ofdm import LMMSEEqualizer, OFDMDemodulator

from .channel_estimation import DMRSChannelEstimator
from .config import ORANConfig
from .odu import ODU
from .pdsch import PDSCHConfig


@dataclass
class UEOutput:
    bits: List[torch.Tensor]          # 每个 UE 的译码比特 [batch, 1, tb_size_k]
    crc_ok: torch.Tensor              # [batch, K]
    x_hat: torch.Tensor               # 均衡后的数据符号 [batch, K, ν, num_data_symbols]
    no_eff: torch.Tensor              # 均衡后的等效噪声方差，同上


class UEReceivers:
    def __init__(self, cfg: ORANConfig, pdsch_configs: List[PDSCHConfig], resource_grid, odu: ODU,
                 air=None, l_min: Optional[int] = None, num_bp_iter: int = 20,
                 precision=None, device=None):
        """
        :param air: 时域仿真时的空口参数 :class:`~oran.oru.AirNumerology`
        :param l_min: 时域信道的最小时延抽头（FFT 窗口提前量）
        """
        self.cfg = cfg
        self.rg = resource_grid
        self.air = air
        self.num_bp_iter = num_bp_iter
        self.precision = precision
        self.device = device
        self.demodulator = None
        if cfg.domain == "time":
            self.demodulator = OFDMDemodulator(air.fft_size, l_min, air.cp_length,
                                               precision=precision, device=device)
        pc = pdsch_configs[0]
        self.estimator = None
        if cfg.channel_estimation == "ls":
            self.estimator = DMRSChannelEstimator(
                resource_grid, pc.dmrs.length, pc.dmrs.additional_position,
                pc.dmrs.num_cdm_groups_without_data, 12 * cfg.prg_num_prb,
                window=cfg.ce_window, delay_compensation=cfg.ce_delay_compensation,
                precision=precision, device=device)
        # UE k（rx k）接收虚拟发射机 k（即第 k 个 UE 的层组）
        sm = StreamManagement(np.eye(cfg.num_ue, dtype=bool), cfg.num_layers_per_ue)
        self.equalizer = LMMSEEqualizer(resource_grid, sm, precision=precision, device=device)
        self._codewords = {}            # 与 O-DU 相同的键 -> (Demapper, LayerDemapper, TBDecoder)
        self.set_codewords(odu)

    def set_codewords(self, odu: ODU) -> None:
        """与 O-DU 当前各 UE 的 MCS 保持一致（链路自适应改变 MCS 后调用）。"""
        for k, key in enumerate(odu.codeword_keys):
            if key not in self._codewords:
                m = odu.mappers[k].constellation.num_bits_per_symbol
                self._codewords[key] = (
                    Demapper("maxlog", "qam", m, precision=self.precision, device=self.device),
                    LayerDemapper(odu.layer_mappers[k], num_bits_per_symbol=m,
                                  precision=self.precision, device=self.device),
                    TBDecoder(odu.tb_encoders[k], num_bp_iter=self.num_bp_iter,
                              precision=self.precision, device=self.device))
        codewords = [self._codewords[key] for key in odu.codeword_keys]
        self.demappers = [c[0] for c in codewords]
        self.layer_demappers = [c[1] for c in codewords]
        self.tb_decoders = [c[2] for c in codewords]

    def __call__(self, y: torch.Tensor, no, h_eff: Optional[torch.Tensor] = None) -> UEOutput:
        """
        :param y: 接收信号，频域 [B, K, N_r, num_symbols, num_sc] 或时域 [B, K, N_r, num_samples]
        :param no: 噪声方差
        :param h_eff: 理想 CSI 时的等效信道 [B, K, N_r, K, ν, num_symbols, num_sc]
        """
        if self.demodulator is not None:
            y = self.air.active(self.demodulator(y))
        if not isinstance(no, torch.Tensor):
            no = torch.tensor(no, dtype=y.real.dtype, device=y.device)
        if self.estimator is not None:
            h_hat, err_var = self.estimator(y, no)
        else:
            if h_eff is None:
                raise ValueError("理想 CSI 需要提供 h_eff")
            h_hat, err_var = h_eff, torch.zeros((), dtype=no.dtype, device=y.device)
        x_hat, no_eff = self.equalizer(y, h_hat, err_var, no)   # [B, K, ν, num_data_symbols]

        bits, crc = [], []
        for k in range(self.cfg.num_ue):
            llr = self.demappers[k](x_hat[:, k:k + 1], no_eff[:, k:k + 1])
            llr = self.layer_demappers[k](llr)
            b_hat, crc_k = self.tb_decoders[k](llr)
            bits.append(b_hat)
            crc.append(crc_k)
        return UEOutput(bits, torch.cat(crc, dim=1), x_hat, no_eff)
