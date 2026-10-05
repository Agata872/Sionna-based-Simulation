"""下行多链路信道：每个 (UE, O-RU) 对一条独立的 3GPP CDL 链路。

CDL 模型只描述一对收发机，且簇的离开角由表格固定。若两个 UE 直接用同一个 CDL，
它们在 BS 侧的角度分布相同，MU-MIMO 几乎无法分开。这里对每条链路、每个批次样本随机旋转
BS 阵列的方位角（bs_orientation 的 yaw），相当于把 UE 放到不同方向上。

输出张量的轴约定与 Sionna 一致：num_rx = UE 数，num_tx = O-RU 数。
信道按链路逐样本归一化，使每对收发天线的平均功率为 1，再乘以大尺度增益 link_gain_db。

channel_model="rt" 时改用 Sionna RT 射线追踪（:mod:`oran.rt_channel`），信道含链路预算，噪声方差为 1。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from sionna.phy import config as sionna_config
from sionna.phy.channel import (ApplyOFDMChannel, ApplyTimeChannel, cir_to_ofdm_channel,
                                cir_to_time_channel, subcarrier_frequencies,
                                time_lag_discrete_time_channel, time_to_ofdm_channel)
from sionna.phy.channel.tr38901 import CDL, PanelArray

from .config import ORANConfig
from .precoding import prg_center_subcarriers


def _panel_array(num_ant: int, carrier_frequency: float, pattern: str, precision, device):
    """天线数为偶数时用双极化（±45° 交叉）单行面板，否则用垂直极化 ULA。"""
    if num_ant % 2 == 0:
        return PanelArray(num_rows_per_panel=1, num_cols_per_panel=num_ant // 2,
                          polarization="dual", polarization_type="cross",
                          antenna_pattern=pattern, carrier_frequency=carrier_frequency,
                          precision=precision, device=device)
    return PanelArray(num_rows_per_panel=1, num_cols_per_panel=num_ant,
                      polarization="single", polarization_type="V",
                      antenna_pattern=pattern, carrier_frequency=carrier_frequency,
                      precision=precision, device=device)


class MultiLinkCDL:
    """所有 (UE, O-RU) 链路的 CDL 信道冲激响应。

    调用返回 a: [batch, K, N_r, num_oru, N_TRX, num_paths, num_time_steps]，
             tau: [batch, K, num_oru, num_paths]
    """

    def __init__(self, cfg: ORANConfig, precision=None, device=None):
        self.cfg = cfg
        self.device = device
        ut_array = _panel_array(cfg.num_ue_ant, cfg.carrier_frequency, "omni", precision, device)
        bs_array = _panel_array(cfg.num_oru_ant, cfg.carrier_frequency, "omni", precision, device)
        self.links = [[CDL(cfg.cdl_model, cfg.delay_spread, cfg.carrier_frequency,
                           ut_array=ut_array, bs_array=bs_array, direction="downlink",
                           min_speed=0.0, max_speed=cfg.ue_speed,
                           precision=precision, device=device)
                       for _ in range(cfg.num_oru)] for _ in range(cfg.num_ue)]
        gain_db = np.zeros([cfg.num_ue, cfg.num_oru]) if cfg.link_gain_db is None \
            else np.asarray(cfg.link_gain_db, dtype=float)
        self.link_amplitude = torch.tensor(10 ** (gain_db / 20), device=device)

    @property
    def max_delay(self) -> float:
        """所有链路的最大路径时延 [s]（CDL 的时延由表格和 delay_spread 决定，与随机实现无关）。"""
        return max(float(cdl._delays.max()) for row in self.links for cdl in row)

    def _random_bs_orientation(self, batch_size: int, dtype) -> torch.Tensor:
        rng = sionna_config.torch_rng(self.device)
        span = np.deg2rad(self.cfg.ue_azimuth_range)
        yaw = (torch.rand(batch_size, generator=rng, device=self.device, dtype=dtype) - 0.5) * span
        zeros = torch.zeros_like(yaw)
        return torch.stack([yaw, zeros, zeros], dim=-1)

    def __call__(self, batch_size: int, num_time_steps: int, sampling_frequency: float):
        a_rows, tau_rows = [], []
        for k in range(self.cfg.num_ue):
            a_row, tau_row = [], []
            for o in range(self.cfg.num_oru):
                cdl = self.links[k][o]
                cdl._bs_orientation = self._random_bs_orientation(batch_size, cdl.dtype)
                a, tau = cdl(batch_size, num_time_steps, sampling_frequency)
                # 逐样本归一化：每对收发天线平均功率为 1（对路径求和，对天线和时间平均）
                power = (a.abs() ** 2).sum(dim=5).mean(dim=(1, 2, 3, 4, 5), keepdim=True)
                a = a / torch.sqrt(power[..., None]) * self.link_amplitude[k, o]
                a_row.append(a)                             # [B, 1, N_r, 1, N_TRX, P, T]
                tau_row.append(tau)                         # [B, 1, 1, P]
            a_rows.append(torch.cat(a_row, dim=3))
            tau_rows.append(torch.cat(tau_row, dim=2))
        return torch.cat(a_rows, dim=1), torch.cat(tau_rows, dim=1)


@dataclass
class ChannelState:
    """一个批次的信道实现。"""

    h_freq: torch.Tensor                    # [B, K, N_r, num_oru, N_TRX, num_symbols, num_sc]
    h_time: Optional[torch.Tensor] = None   # 时域仿真时：[B, K, N_r, num_oru, N_TRX, T + l_tot - 1, l_tot]

    def stacked(self) -> torch.Tensor:
        """把各 O-RU 的天线拼成一维：[B, K, N_r, N_t, num_symbols, num_sc]（天线序号 = o·N_TRX + a）。"""
        h = self.h_freq
        return h.flatten(3, 4)


class DownlinkChannel:
    """生成信道、提供 CSI、把 O-RU 发射信号通过信道并加噪声。"""

    def __init__(self, cfg: ORANConfig, resource_grid, air=None, precision=None, device=None):
        """
        :param resource_grid: PRB 资源网格（频域仿真用）
        :param air: 时域仿真时的空口参数 :class:`~oran.oru.AirNumerology`
        """
        self.cfg = cfg
        self.rg = resource_grid
        self.air = air
        self.frequencies = subcarrier_frequencies(resource_grid.fft_size,
                                                  resource_grid.subcarrier_spacing,
                                                  precision=precision, device=device)
        if cfg.channel_model == "rt":
            from .rt_channel import RTChannel             # 只有用到时才加载 Mitsuba / Sionna RT
            self.model = RTChannel(cfg, self.frequencies, device=device)
        else:
            self.model = MultiLinkCDL(cfg, precision=precision, device=device)
        self.prg_centers = prg_center_subcarriers(cfg.num_prb, cfg.prg_num_prb)
        if cfg.domain == "freq":
            self.apply_ofdm = ApplyOFDMChannel(precision=precision, device=device)
        else:
            # 离散时域信道的抽头范围：覆盖最大路径时延（再加 sinc 拖尾，由 Sionna 函数处理）
            self.l_min, self.l_max = time_lag_discrete_time_channel(air.bandwidth,
                                                                    self.model.max_delay)
            self.l_tot = self.l_max - self.l_min + 1
            self.apply_time = ApplyTimeChannel(air.resource_grid.num_time_samples, self.l_tot,
                                               precision=precision, device=device)

    def set_ue_positions(self, positions) -> None:
        """射线追踪信道：放置 UE（[K, 3]，单位 m）并做射线追踪。"""
        if self.cfg.channel_model != "rt":
            raise ValueError("只有射线追踪信道（channel_model='rt'）才有 UE 位置")
        self.model.set_ue_positions(positions)

    def sample(self, batch_size: int) -> ChannelState:
        rg = self.rg
        if self.cfg.channel_model == "rt":
            # 静止 UE：同一位置的信道在各时隙（批次样本）、各符号上相同
            return ChannelState(self.model.ofdm_channel(rg.num_ofdm_symbols, batch_size))
        if self.cfg.domain == "freq":
            # 每个 OFDM 符号采样一次（符号内信道不变，无 ICI）
            a, tau = self.model(batch_size, rg.num_ofdm_symbols, 1 / rg.ofdm_symbol_duration)
            h_freq = cir_to_ofdm_channel(self.frequencies, a, tau)
            return ChannelState(h_freq)
        # 时域：CDL 若按采样率逐点生成，显存随样点数（一个时隙一万多点）线性增长，batch 稍大就溢出。
        # 这里每个 OFDM 符号生成一次 CIR，符号内保持不变（忽略符号内多普勒引起的 ICI，适用于低速），
        # 时延扩展、CP、符号间干扰等时域效应仍然完整保留。
        air = self.air.resource_grid
        a, tau = self.model(batch_size, air.num_ofdm_symbols, 1 / air.ofdm_symbol_duration)
        h_sym = cir_to_time_channel(air.bandwidth, a, tau, self.l_min, self.l_max)  # [..., S, l_tot]
        num_steps = air.num_time_samples + self.l_tot - 1
        symbol_len = air.fft_size + air.cyclic_prefix_length
        idx = torch.clamp(torch.arange(num_steps, device=h_sym.device) // symbol_len,
                          max=air.num_ofdm_symbols - 1)
        h_time = h_sym.index_select(-2, idx)
        h_freq = self.air.active(time_to_ofdm_channel(h_time, air, self.l_min))
        return ChannelState(h_freq, h_time)

    def csi(self, state: ChannelState, symbol: int) -> torch.Tensor:
        """O-DU 用于计算预编码的 CSI：各 PRG 中心子载波、第 symbol 个符号处的信道。

        返回 [B, num_prg, K, N_r, N_t]。这里假设 O-DU 已知理想 CSI（如由 SRS 测得且无时延）。
        """
        h = state.stacked()[..., symbol, :]                 # [B, K, N_r, N_t, num_sc]
        h = h[..., self.prg_centers.to(h.device)]           # [B, K, N_r, N_t, num_prg]
        return h.permute(0, 4, 1, 2, 3)

    def __call__(self, x: torch.Tensor, state: ChannelState, no) -> torch.Tensor:
        """
        :param x: 所有 O-RU 的发射信号，频域 [B, num_oru, N_TRX, num_symbols, num_sc]
                  或时域 [B, num_oru, N_TRX, num_time_samples]
        :return: 各 UE 的接收信号，频域 [B, K, N_r, num_symbols, num_sc] 或时域 [B, K, N_r, T + l_tot - 1]
        """
        if self.cfg.domain == "freq":
            return self.apply_ofdm(x, state.h_freq, no)
        return self.apply_time(x, state.h_time, no)
