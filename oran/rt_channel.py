"""Sionna RT 射线追踪信道：物理环境（场景几何 + 材料）中的 O-RU 与 UE。

射线追踪即 TR 38.901 第 8 节的 map-based hybrid channel model（可选信道模型）。与 CDL 不同，
信道由几何决定：同一位置只有一个信道实现，路径增益是绝对值，SNR 由链路预算给出而不再扫描。

- 场景：Sionna 内置场景（如 munich）或 Mitsuba XML 文件，载频取 cfg.carrier_frequency。
- 天线：与 CDL 模式相同。O-RU 为 1×(N_TRX/2) 的 ±45° 交叉极化面板，阵元方向图为 TR 38.901 7.3 节；
  UE 为 1×(N_r/2) 的交叉极化全向阵元（天线数为奇数时用垂直极化 ULA）。
- 撒点：先算一张单阵元方向图、不含折射（避免把 UE 撒进建筑）的覆盖图，在其范围内均匀抽点，保留距离
  和路径增益满足门限的点，再向上发一条射线，头顶有遮挡（室内或骑楼下）的点丢弃。
- 链路预算：每个 RE 的发射功率 P_RE = P_tx / (12·N_PRB)，噪声 N0 = k·T·Δf·NF。仿真中仍沿用
  "每 RE 总发射功率为 1"的约定，把信道乘以 sqrt(P_RE / N0)，噪声方差取 1（即 run_batch 的 snr_db = 0）。
- 定时：每个 UE 以所有 O-RU 中最早到达的路径为时延零点（UE 下行定时同步到第一径）。
- UE 静止：信道在时隙内、时隙间都不变，一个批次的各样本是同一位置的连续时隙。
"""

from typing import Optional

import numpy as np
import torch

import mitsuba as mi
import sionna.rt
from sionna.rt import PathSolver, PlanarArray, RadioMapSolver, Receiver, Transmitter, load_scene

from .config import ORANConfig

BOLTZMANN = 1.380649e-23


def _scene_file(name: str) -> str:
    """内置场景名（sionna.rt.scene.munich 等）或场景文件路径。"""
    return getattr(sionna.rt.scene, name, name)


def _rt_array(num_ant: int, pattern: str) -> PlanarArray:
    """与 channel._panel_array 相同的阵列：偶数天线为 1×(n/2) 交叉极化，奇数为垂直极化 ULA。"""
    if num_ant % 2 == 0:
        return PlanarArray(num_rows=1, num_cols=num_ant // 2, pattern=pattern, polarization="cross")
    return PlanarArray(num_rows=1, num_cols=num_ant, pattern=pattern, polarization="V")


def snr_scale(cfg: ORANConfig) -> float:
    """链路预算：P_RE / N0（线性）。信道幅度乘以它的平方根后，噪声方差为 1。"""
    rc = cfg.rt
    p_re = 10 ** ((rc.tx_power_dbm - 30) / 10) / cfg.num_subcarriers
    n0 = BOLTZMANN * rc.temperature * cfg.subcarrier_spacing * 1e3 * 10 ** (rc.noise_figure_db / 10)
    return p_re / n0


class RTScene:
    """射线追踪场景：O-RU（发射机）固定，UE（接收机）位置可变。"""

    def __init__(self, cfg: ORANConfig):
        self.cfg = cfg
        rc = cfg.rt
        self.scene = load_scene(_scene_file(rc.scene), merge_shapes=True)
        self.scene.frequency = cfg.carrier_frequency
        self.tx_array = _rt_array(cfg.num_oru_ant, "tr38901")
        self.rx_array = _rt_array(cfg.num_ue_ant, "iso")
        self.scene.tx_array = self.tx_array
        self.scene.rx_array = self.rx_array
        for o, (pos, look_at) in enumerate(zip(rc.oru_positions, rc.oru_look_at)):
            self.scene.add(Transmitter(f"oru{o}", position=list(map(float, pos)),
                                       look_at=list(map(float, look_at))))
        self.solver = PathSolver(deterministic=rc.deterministic)
        self.ue_positions: Optional[np.ndarray] = None
        self._coverage = None

    # ------------------------------------------------------------------
    def coverage_map(self):
        """撒点用的覆盖图（单阵元、无折射，在 UE 高度的水平面上），只算一次。"""
        if self._coverage is None:
            rc = self.cfg.rt
            self.scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="tr38901",
                                              polarization="V")
            self.scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso",
                                              polarization="V")
            try:
                center = np.mean(np.asarray(rc.oru_positions, dtype=float), axis=0)
                size = 2 * (rc.drop_max_dist + rc.coverage_cell_size)
                self._coverage = RadioMapSolver()(
                    self.scene, center=[float(center[0]), float(center[1]), rc.ue_height],
                    orientation=[0.0, 0.0, 0.0], size=[size, size],
                    cell_size=[rc.coverage_cell_size, rc.coverage_cell_size],
                    samples_per_tx=rc.coverage_samples, max_depth=rc.max_depth,
                    refraction=False, diffraction=rc.diffraction)
            finally:
                self.scene.tx_array = self.tx_array
                self.scene.rx_array = self.rx_array
        return self._coverage

    def is_outdoor(self, positions: np.ndarray) -> np.ndarray:
        """从每个位置竖直向上发射线，没有打到任何物体即为室外。"""
        p = np.asarray(positions, dtype=np.float32)
        origin = mi.Point3f(mi.Float(p[:, 0]), mi.Float(p[:, 1]), mi.Float(p[:, 2]))
        si = self.scene.mi_scene.ray_intersect(mi.Ray3f(origin, mi.Vector3f(0, 0, 1)))
        return ~si.is_valid().numpy()

    def drop_ues(self, num_ue: int, seed: int) -> np.ndarray:
        """在覆盖区内均匀随机撒 num_ue 个室外 UE，返回 [num_ue, 3]。

        在覆盖图范围内均匀抽候选点，保留与最近 O-RU 的距离在 [drop_min_dist, drop_max_dist] 内、
        所在格子的路径增益（各 O-RU 中最强）不低于门限、且头顶无遮挡的点（拒绝采样）。
        覆盖图由蒙特卡洛射线追踪得到，每次计算略有差异；拒绝采样只让门限附近的少数候选受影响，
        同一 seed 的撒点基本可复现。
        """
        rc = self.cfg.rt
        rm = self.coverage_map()
        cc = rm.cell_centers.numpy()                    # [ny, nx, 3]，x 沿第 2 维，y 沿第 1 维
        pg_db = 10 * np.log10(rm.path_gain.numpy().max(axis=0) + 1e-30)
        cell = float(rc.coverage_cell_size)
        oru = np.asarray(rc.oru_positions, dtype=float)
        lo, hi = cc[0, 0, :2] - cell / 2, cc[-1, -1, :2] + cell / 2
        rng = np.random.default_rng(seed)
        chosen = []
        for _ in range(500):
            xy = rng.uniform(lo, hi, size=(64, 2))
            p = np.column_stack([xy, np.full(len(xy), rc.ue_height)])
            dist = np.linalg.norm(p[:, None, :] - oru[None], axis=-1).min(axis=1)
            ix = np.clip(np.round((xy[:, 0] - cc[0, 0, 0]) / cell).astype(int), 0, cc.shape[1] - 1)
            iy = np.clip(np.round((xy[:, 1] - cc[0, 0, 1]) / cell).astype(int), 0, cc.shape[0] - 1)
            ok = ((dist >= rc.drop_min_dist) & (dist <= rc.drop_max_dist)
                  & (pg_db[iy, ix] >= rc.drop_min_path_gain_db))
            p = p[ok]
            if len(p):
                chosen.extend(p[self.is_outdoor(p)])
            if len(chosen) >= num_ue:
                return np.stack(chosen[:num_ue])
        raise RuntimeError("覆盖区内找不到足够的室外 UE 位置，请放宽 rt.drop_* 参数")

    def set_ue_positions(self, positions: np.ndarray) -> None:
        for name in list(self.scene.receivers):
            self.scene.remove(name)
        for k, p in enumerate(np.asarray(positions, dtype=float)):
            self.scene.add(Receiver(f"ue{k}", position=[float(v) for v in p]))
        self.ue_positions = np.asarray(positions, dtype=float)

    def cir(self):
        """当前 UE 位置的信道冲激响应。

        返回 a: [K, N_r, num_oru, N_TRX, num_paths]（复数，绝对增益），
             tau: [K, num_oru, num_paths]（绝对时延 [s]，无效路径为负）
        """
        if self.ue_positions is None:
            raise ValueError("请先调用 set_ue_positions 放置 UE")
        rc = self.cfg.rt
        paths = self.solver(self.scene, max_depth=rc.max_depth, samples_per_src=rc.samples_per_src,
                            diffraction=rc.diffraction, synthetic_array=True)
        a, tau = paths.cir(normalize_delays=False, out_type="torch")
        return a[..., 0], tau


class RTChannel:
    """射线追踪信道源，供 :class:`~oran.channel.DownlinkChannel` 使用（静止 UE）。

    set_ue_positions 时做一次射线追踪并算出频域响应，之后每个批次直接复用（批次和符号维只做 expand）。
    """

    def __init__(self, cfg: ORANConfig, frequencies: torch.Tensor, device=None):
        self.cfg = cfg
        self.device = device
        self.scene = RTScene(cfg)
        self.frequencies = frequencies
        self.amplitude = float(np.sqrt(snr_scale(cfg)))
        self.cdtype = torch.complex128 if frequencies.dtype == torch.float64 else torch.complex64
        self.h: Optional[torch.Tensor] = None          # [1, K, N_r, num_oru, N_TRX, 1, num_sc]
        self.num_paths: Optional[np.ndarray] = None    # [K, num_oru]

    def set_ue_positions(self, positions: np.ndarray) -> None:
        positions = np.asarray(positions, dtype=float)
        if self.h is not None and self.scene.ue_positions is not None \
                and np.array_equal(positions, self.scene.ue_positions):
            return                                      # 位置未变，沿用已算出的信道
        self.scene.set_ue_positions(positions)
        a, tau = self.scene.cir()
        a = a.to(self.device, self.cdtype) * self.amplitude
        tau = tau.to(self.device, self.frequencies.dtype)
        valid = tau >= 0
        # 每个 UE 以最早到达的路径为时延零点（对所有 O-RU 统一）
        t0 = torch.where(valid, tau, torch.full_like(tau, float("inf"))).amin(dim=(1, 2), keepdim=True)
        t0 = torch.where(torch.isfinite(t0), t0, torch.zeros_like(t0))
        tau = torch.where(valid, tau - t0, torch.zeros_like(tau))
        phase = -2 * np.pi * tau[..., None] * self.frequencies.to(self.device)   # [K, O, P, F]
        e = torch.polar(torch.ones_like(phase), phase).to(self.cdtype)
        h = torch.einsum("krotp,kopf->krotf", a, e)   # 无效路径的 a 为 0
        self.h = h[None, :, :, :, :, None, :]
        self.num_paths = valid.sum(-1).cpu().numpy()

    def ofdm_channel(self, num_symbols: int, batch_size: int) -> torch.Tensor:
        """[batch, K, N_r, num_oru, N_TRX, num_symbols, num_sc]（各样本、各符号相同）。"""
        if self.h is None:
            raise ValueError("射线追踪信道需要先调用 set_ue_positions 放置 UE")
        return self.h.expand(batch_size, -1, -1, -1, -1, num_symbols, -1)

    def wideband_snr_db(self) -> np.ndarray:
        """每个 UE 的宽带 SNR [dB]：每根接收天线、不含波束赋形增益（各发射天线等功率时的平均接收 SNR）。"""
        p = (self.h.abs() ** 2).mean(dim=(0, 2, 3, 4, 5, 6))
        return 10 * np.log10(p.cpu().numpy())

    def path_gain_db(self) -> np.ndarray:
        """每个 UE 的平均路径增益 [dB]（含天线方向图，对所有 O-RU 天线和子载波平均）。"""
        return self.wideband_snr_db() - 10 * np.log10(snr_scale(self.cfg))
