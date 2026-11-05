"""TR 38.901 UMi-Street Canyon 随机信道下的 cell-free 撒点（不用射线追踪）。

用 Sionna 的系统级信道 ``sionna.phy.channel.tr38901.UMi``，按 TR 38.901 §7.5 的步骤由 O-RU 与 UE 的位置生成信道：
    步骤 1：布置（O-RU 方格布置、间距默认为 UMi 的 ISD 200 m、高 10 m（表 7.2-1）；UE 在区域内均匀撒点、室外、
            高 1.5 m、静止，朝向随机）；
    步骤 2：LoS / NLoS（表 7.4.2-1，按 d_2D 的概率随机）；
    步骤 3：路损（表 7.4.1-1，UMi-Street Canyon）；
    步骤 4~11：大尺度参数（含阴影衰落，表 7.5-6）、簇时延 / 功率 / 角度、射线耦合、交叉极化、信道系数。
每个 (UE, O-RU) 是一条独立链路，各链路的角度由几何决定（与 CDL 模式中随机旋转方位不同）。

- 天线：与 CDL / 射线追踪模式相同（O-RU 为 1×(N_TRX/2) 的 ±45° 交叉极化面板，UE 为 1×(N_r/2)），
  阵元方向图默认全向（cell-free 的接入点），可选 TR 38.901 表 7.3-1 的扇区方向图（朝区域中心）。
- 规范版本：TR 38.901 V16.1.0 到 V18.0.0 之间 UMi 没有改动（唯一的 CR 0025 只改 CDL），用 Sionna 的
  spec_version="16.1"。
- 时延：TR 38.901 的基础模型中每条链路的簇时延从 0 开始（步骤 5）。cell-free 中各 O-RU 与 UE 的距离不同，
  默认按 §7.6.9（绝对到达时间）给每条链路加上 d_3D / c（UMi 没有 Δτ 项，Δτ 只用于 InF），再以每个 UE 在所有
  O-RU 中最早到达的路径为时延零点（UE 下行定时同步到第一径，与射线追踪模式相同）。
- 链路预算：每个 O-RU 最大发射功率 P_oru，总功率 num_oru·P_oru；仿真沿用"每 RE 总发射功率为 1"的约定，
  信道乘以 sqrt(P_RE / N0)，P_RE = num_oru·P_oru / (12·N_PRB)，N0 = k·T·Δf·NF，噪声方差取 1（snr_db = 0）。
  每个 O-RU 不超过 P_oru 由 O-DU 的预编码缩放保证（cfg.per_oru_power）。
- UE 静止：每个撒点只生成一次小尺度衰落，信道在时隙内、时隙间都不变（与射线追踪模式相同）。
"""

from typing import Optional

import numpy as np
import torch
from sionna.phy import SPEED_OF_LIGHT
from sionna.phy import config as sionna_config
from sionna.phy.channel import cir_to_ofdm_channel
from sionna.phy.channel.tr38901 import UMi

from .channel import _panel_array
from .config import ORANConfig

BOLTZMANN = 1.380649e-23


def snr_scale(cfg: ORANConfig) -> float:
    """链路预算：P_RE / N0（线性），P_RE 为所有 O-RU 的总功率均分到每个 RE。"""
    uc = cfg.umi
    p_re = cfg.num_oru * 10 ** ((uc.tx_power_dbm - 30) / 10) / cfg.num_subcarriers
    n0 = BOLTZMANN * uc.temperature * cfg.subcarrier_spacing * 1e3 * 10 ** (uc.noise_figure_db / 10)
    return p_re / n0


class UMiChannel:
    """cell-free UMi 撒点信道源，供 :class:`~oran.channel.DownlinkChannel` 使用（静止 UE）。

    用法与射线追踪信道相同：``drop_ues`` 撒点，``set_ue_positions`` 生成该撒点的信道，之后每个批次复用。
    """

    def __init__(self, cfg: ORANConfig, frequencies: torch.Tensor, precision=None, device=None):
        self.cfg = cfg
        uc = cfg.umi
        self.device = device
        self.frequencies = frequencies
        self.oru_positions = uc.positions(cfg.num_oru)
        self.precision = precision
        self.ut_array = _panel_array(cfg.num_ue_ant, cfg.carrier_frequency, "omni", precision, device)
        self.bs_array = _panel_array(cfg.num_oru_ant, cfg.carrier_frequency, uc.oru_pattern, precision, device)
        self.model: Optional[UMi] = None
        self.amplitude = float(np.sqrt(snr_scale(cfg)))
        self.cdtype = torch.complex128 if frequencies.dtype == torch.float64 else torch.complex64
        self.ue_positions: Optional[np.ndarray] = None
        self.h: Optional[torch.Tensor] = None          # [1, K, N_r, num_oru, N_TRX, 1, num_sc]
        self.los: Optional[np.ndarray] = None          # [K, num_oru]
        self.num_paths: Optional[np.ndarray] = None    # [K, num_oru]
        self.delay_offset: Optional[np.ndarray] = None  # [K, num_oru]：各链路首径相对该 UE 定时的时延 [s]

    @property
    def max_delay(self) -> float:
        raise NotImplementedError("UMi 撒点信道只支持频域仿真")

    def drop_ues(self, num_ue: int, seed: int) -> np.ndarray:
        """在区域内均匀撒 num_ue 个 UE（与每个 O-RU 的水平距离不小于 min_dist_2d），返回 [num_ue, 3]。"""
        uc = self.cfg.umi
        x0, x1, y0, y1 = uc.area
        rng = np.random.default_rng(seed)
        chosen = []
        for _ in range(1000):
            xy = rng.uniform([x0, y0], [x1, y1], size=(64, 2))
            d2d = np.linalg.norm(xy[:, None, :] - self.oru_positions[None, :, :2], axis=-1).min(axis=1)
            chosen.extend(xy[d2d >= uc.min_dist_2d])
            if len(chosen) >= num_ue:
                xy = np.stack(chosen[:num_ue])
                return np.column_stack([xy, np.full(num_ue, uc.ue_height)])
        raise RuntimeError("区域内找不到满足最小距离的 UE 位置，请放宽 umi.area 或 umi.min_dist_2d")

    def set_ue_positions(self, positions: np.ndarray, seed: Optional[int] = None) -> None:
        """放置 UE 并生成该撒点的信道（LoS 状态、阴影衰落、小尺度衰落都随机生成一次）。

        位置与当前相同且未给 seed 时沿用已生成的信道（同一撒点上依次比较多种配置）。

        :param seed: 给定时先把 Sionna 的随机种子设为它（UE 朝向也由它决定），同一 seed 得到同一信道
        """
        positions = np.asarray(positions, dtype=float)
        if seed is None and self.h is not None and self.ue_positions is not None \
                and np.array_equal(positions, self.ue_positions):
            return
        cfg = self.cfg
        num_ue = len(positions)
        rng = np.random.default_rng(None if seed is None else seed + 7_919)
        if seed is not None:
            sionna_config.seed = seed
        dev, real = self.device, self.frequencies.dtype
        t = lambda v: torch.as_tensor(np.asarray(v), dtype=real, device=dev)
        # UE 朝向：方位均匀随机（TR 38.901 表 7.8-1 的 UT 朝向）；O-RU 面板朝区域中心
        ut_orient = np.zeros((1, num_ue, 3))
        ut_orient[0, :, 0] = rng.uniform(0, 2 * np.pi, num_ue)
        x0, x1, y0, y1 = cfg.umi.area
        center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        d = center[None] - self.oru_positions[:, :2]
        bs_orient = np.zeros((1, cfg.num_oru, 3))
        bs_orient[0, :, 0] = np.where(np.linalg.norm(d, axis=-1) > 1e-6, np.arctan2(d[:, 1], d[:, 0]), 0.0)
        # 每个撒点新建一次 Sionna 的 UMi：拓扑不变时 set_topology 会跳过部分随机量的生成，复用同一个对象时
        # 同一 seed 不能复现同一信道
        uc = cfg.umi
        self.model = UMi(cfg.carrier_frequency, uc.o2i_model, self.ut_array, self.bs_array, "downlink",
                         enable_pathloss=True, enable_shadow_fading=uc.shadow_fading,
                         precision=self.precision, device=dev, spec_version=uc.spec_version)
        self.model.set_topology(
            ut_loc=t(positions[None]), bs_loc=t(self.oru_positions[None]),
            ut_orientations=t(ut_orient), bs_orientations=t(bs_orient),
            ut_velocities=t(np.zeros((1, num_ue, 3))),
            in_state=torch.zeros(1, num_ue, dtype=torch.bool, device=dev), los="random")
        a, tau = self.model(1, 1.0)                    # a: [1, K, N_r, O, N_TRX, P, 1]，tau: [1, K, O, P]
        a = a.to(self.cdtype)
        tau = tau.to(real)
        power = (a.abs() ** 2).sum(dim=(2, 4, 6))      # [1, K, O, P]
        valid = power > 0
        if cfg.umi.absolute_delay:
            d3d = np.linalg.norm(positions[:, None, :] - self.oru_positions[None], axis=-1)   # [K, O]
            tau = tau + t(d3d / SPEED_OF_LIGHT)[None, :, :, None]
        # 每个 UE 以所有 O-RU 中最早到达的路径为时延零点
        first = torch.where(valid, tau, torch.full_like(tau, float("inf"))).amin(dim=-1)     # [1, K, O]
        t0 = first.amin(dim=-1, keepdim=True)
        t0 = torch.where(torch.isfinite(t0), t0, torch.zeros_like(t0))
        tau = torch.where(valid, tau - t0[..., None], torch.zeros_like(tau))
        h = cir_to_ofdm_channel(self.frequencies.to(dev), a, tau, normalize=False)          # [1, K, N_r, O, N_TRX, 1, F]
        self.h = (h * self.amplitude).to(self.cdtype)
        self.ue_positions = positions
        self.los = self.model._scenario.los[0].T.cpu().numpy()                              # [K, O]
        self.num_paths = valid[0].sum(-1).cpu().numpy()
        self.delay_offset = (first - t0)[0].cpu().numpy()

    def ofdm_channel(self, num_symbols: int, batch_size: int) -> torch.Tensor:
        """[batch, K, N_r, num_oru, N_TRX, num_symbols, num_sc]（各样本、各符号相同）。"""
        if self.h is None:
            raise ValueError("UMi 撒点信道需要先调用 set_ue_positions 放置 UE")
        return self.h.expand(batch_size, -1, -1, -1, -1, num_symbols, -1)

    def link_gain_db(self) -> np.ndarray:
        """每条 (UE, O-RU) 链路的平均功率增益 [dB]（路损 + 阴影 + 天线，对天线对和子载波平均），[K, num_oru]。"""
        p = (self.h.abs() ** 2).mean(dim=(0, 2, 4, 5, 6))
        return 10 * np.log10(p.cpu().numpy() + 1e-30) - 10 * np.log10(snr_scale(self.cfg))

    def wideband_snr_db(self) -> np.ndarray:
        """每个 UE 的宽带 SNR [dB]：每根接收天线、各发射天线等功率、不含波束赋形增益（与射线追踪模式相同）。"""
        p = (self.h.abs() ** 2).mean(dim=(0, 2, 3, 4, 5, 6))
        return 10 * np.log10(p.cpu().numpy())

    def path_gain_db(self) -> np.ndarray:
        """每个 UE 的平均路径增益 [dB]（对所有 O-RU 天线和子载波平均）。"""
        return self.wideband_snr_db() - 10 * np.log10(snr_scale(self.cfg))
