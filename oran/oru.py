"""O-RU：7.2x 分割点以下的 Low-PHY。

    前传收到的 eAxC 流（已在 FronthaulLink 中解压）
    -> Cat B：按 C-plane 下发的权值逐 RE 预编码（层 -> 天线）
    -> [O-RU 数字波束赋形：预留，当前为直通]
    -> 时域仿真时：把 PRB 映射到 IFFT 的中间（两侧为保护带）-> IFFT + 加 CP
    -> [DAC、模拟波束赋形：预留，当前为直通]

前传只承载 PRB（12·num_prb 个子载波）。IFFT 点数按 NR 惯例取不小于 num_sc/0.85 的最小 2 的幂，
例如 30 kHz 下 51 PRB（20 MHz）为 1024 点、30.72 Msps，273 PRB（100 MHz）为 4096 点。

option 8 的 O-RU 只有 DAC 和射频。前传送来的是时域 IQ，FronthaulLink 把它换算成 PRB 上的频域等效网格
（见 :class:`oran.fronthaul.TimeDomainCodec`），这里与 Cat A 一样直通。
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sionna.phy.ofdm import OFDMModulator, ResourceGrid

from .config import ORANConfig
from .precoding import apply_precoder, subcarrier_to_prg


@dataclass
class AirNumerology:
    """空口（时域）参数：IFFT 点数、保护带、CP，以及供 Sionna 工具函数使用的资源网格。"""

    fft_size: int
    num_sc: int
    guard_left: int
    cp_length: int
    resource_grid: ResourceGrid

    @property
    def guard_right(self) -> int:
        return self.fft_size - self.num_sc - self.guard_left

    @property
    def bandwidth(self) -> float:
        return self.resource_grid.bandwidth

    def to_fft_bins(self, x: torch.Tensor) -> torch.Tensor:
        """[..., num_sc] -> [..., fft_size]，两侧补零。"""
        return F.pad(x, (self.guard_left, self.guard_right))

    def active(self, x: torch.Tensor) -> torch.Tensor:
        """[..., fft_size] -> [..., num_sc]，取出 PRB 所在子载波。"""
        return x[..., self.guard_left:self.guard_left + self.num_sc]


def make_air_numerology(cfg: ORANConfig, pdsch_config, precision=None, device=None) -> AirNumerology:
    num_sc = cfg.num_subcarriers
    fft_size = cfg.ifft_size()
    if fft_size < num_sc:
        raise ValueError("fft_size 不能小于子载波数")
    scs = cfg.subcarrier_spacing * 1e3
    cp_length = int(np.ceil(pdsch_config.carrier.cyclic_prefix_length * fft_size * scs))
    guard_left = (fft_size - num_sc) // 2
    rg = ResourceGrid(num_ofdm_symbols=pdsch_config.carrier.num_symbols_per_slot,
                      fft_size=fft_size, subcarrier_spacing=scs, cyclic_prefix_length=cp_length,
                      num_guard_carriers=(guard_left, fft_size - num_sc - guard_left),
                      precision=precision, device=device)
    return AirNumerology(fft_size, num_sc, guard_left, cp_length, rg)


class ORU:
    """一个 O-RU。

    :param air: 时域仿真时的空口参数（频域仿真时为 None）
    :param dac: 可选的 DAC 模型，作用于时域样点 [batch, N_TRX, num_time_samples]，默认直通
    """

    def __init__(self, cfg: ORANConfig, air: Optional[AirNumerology] = None, oru_index: int = 0,
                 dac: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 precision=None, device=None):
        self.cfg = cfg
        self.oru_index = oru_index
        self.prg_of_sc = subcarrier_to_prg(cfg.num_prb, cfg.prg_num_prb)
        self.air = air
        self.dac = dac
        self.modulator = None
        if cfg.domain == "time":
            if air is None:
                raise ValueError("时域仿真需要提供 air（空口参数）")
            self.modulator = OFDMModulator(air.cp_length, precision=precision, device=device)

    def frequency_domain(self, uplane: torch.Tensor, weights: Optional[torch.Tensor] = None
                         ) -> torch.Tensor:
        """得到各天线在 PRB 上的频域资源网格 [batch, N_TRX, num_symbols, num_sc]。"""
        if self.cfg.category == "B":
            if weights is None:
                raise ValueError("Cat B 的 O-RU 需要 C-plane 下发的预编码权值")
            return apply_precoder(uplane, weights, self.prg_of_sc)
        return uplane

    def to_air(self, x: torch.Tensor) -> torch.Tensor:
        """频域网格 -> 送入信道的信号：频域仿真时原样输出，时域仿真时做 IFFT + CP。"""
        if self.modulator is not None:
            x = self.modulator(self.air.to_fft_bins(x))    # [batch, N_TRX, num_time_samples]
            if self.dac is not None:
                x = self.dac(x)
        return x

    def __call__(self, uplane: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.to_air(self.frequency_domain(uplane, weights))
