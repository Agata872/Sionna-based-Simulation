"""仿真配置。

所有可调参数集中在 :class:`ORANConfig` 中；O-DU、前传、O-RU、信道、UE 各模块只读取它。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import numpy as np


@dataclass
class CompressionConfig:
    """一条前传数据流的压缩配置。

    :param method: 压缩方法，见 :func:`oran.compression.available_compressors`
    :param iq_width: 位宽（BFP 为尾数位宽 udIqWidth）
    :param backoff_db: 定点化时 I/Q 分量 RMS 相对 16 bit 满量程的回退
    """

    method: str = "bfp"
    iq_width: int = 9
    backoff_db: float = 15.0
    kwargs: dict = field(default_factory=dict)


@dataclass
class RTConfig:
    """Sionna RT 射线追踪场景（channel_model="rt" 时使用），见 :mod:`oran.rt_channel`。

    默认：munich 场景，1 个 O-RU 架在 26 m 高的楼顶（抱杆 4 m），扇区朝北、下倾约 11°，
    覆盖北侧广场和几条街道；UE 在覆盖区内随机撒点（室外，高 1.5 m）。
    """

    scene: str = "munich"               # Sionna 内置场景名，或 Mitsuba XML 场景文件路径
    oru_positions: List[List[float]] = field(default_factory=lambda: [[8.5, 21.0, 30.0]])
    oru_look_at: List[List[float]] = field(default_factory=lambda: [[8.5, 171.0, 1.5]])

    # ---- 链路预算（默认值参考 TR 38.901 表 7.8-1 的 UMi：20 MHz 35 dBm，UE 噪声系数 9 dB）----
    tx_power_dbm: float = 35.0          # 每个 O-RU 的总发射功率（所有天线、整个载波带宽）
    noise_figure_db: float = 9.0        # UE 噪声系数
    temperature: float = 290.0          # K

    # ---- 射线追踪 ----
    max_depth: int = 3                  # 最大反射/绕射次数
    diffraction: bool = True
    samples_per_src: int = 1_000_000
    deterministic: bool = True          # PathSolver 结果可复现（不同进程间配对比较需要），开销很小

    # ---- 撒点 ----
    ue_height: float = 1.5
    drop_min_dist: float = 10.0         # 与 O-RU 的距离范围 [m]
    drop_max_dist: float = 250.0
    drop_min_path_gain_db: float = -135.0   # 覆盖门限：单阵元方向图下的路径增益
    coverage_cell_size: float = 2.0     # 撒点用覆盖图的栅格 [m]
    coverage_samples: int = 10_000_000


@dataclass
class ORANConfig:
    """O-RAN 7.2x 下行链路级仿真配置（默认：单小区、1 个 O-RU、2 个 UE 的 MU-MIMO）。"""

    # ---- 功能划分 ----
    # "A"：7.2x Cat A，预编码在 O-DU；"B"：7.2x Cat B，预编码在 O-RU；
    # "8"：3GPP split option 8（PHY-RF），O-DU 做到 IFFT + CP，前传传各天线的时域 IQ，O-RU 只剩射频
    category: str = "A"

    # ---- 载波 ----
    subcarrier_spacing: int = 30        # kHz
    num_prb: int = 51                   # 30 kHz 下 20 MHz 为 51 个 PRB（612 个子载波）
    carrier_frequency: float = 3.5e9    # Hz
    n_cell_id: int = 1
    slot_number: int = 0
    frame_number: int = 0

    # ---- 节点与天线 ----
    num_ue: int = 2
    num_ue_ant: int = 4                 # 每个 UE 的接收天线数
    num_oru: int = 1                    # O-RU 个数（cell-free 时 > 1）
    num_oru_ant: int = 8                # 每个 O-RU 的发射通道数（N_TRX）

    # ---- PDSCH ----
    num_layers_per_ue: int = 2
    mcs_index: Union[int, Sequence[int]] = 14   # 可按 UE 给出列表
    mcs_table: int = 1                  # 1：最高 64QAM；2：最高 256QAM
    rnti: Optional[Sequence[int]] = None        # 默认 UE k 的 RNTI 为 100 + k
    dmrs_config_type: int = 1
    dmrs_length: int = 1
    dmrs_additional_position: int = 1
    dmrs_type_a_position: int = 2

    # ---- 预编码（由 O-DU 根据 CSI 计算；Cat A 在 O-DU 执行，Cat B 经 C-plane 下发给 O-RU 执行）----
    precoder: str = "bd"                # "bd"：块对角化；"ezf"：特征 ZF；"rzf"：特征 RZF
    rzf_alpha: float = 0.0              # RZF 正则化系数（"rzf" 时使用；0 退化为 ZF）
    prg_size: int = 2                   # 预编码粒度（PRB），0 表示宽带（全带宽一个预编码）

    # ---- 前传 ----
    # option 8 时为时域样点的量化："fixed" 为 CPRI 式均匀定点（每个 I/Q 分量 iq_width 位，按位宽取最优削波电平），
    # 也可用 bfp / block_scaling / mu_law 对时域样点分块压缩（块长 TIME_BLOCK_LEN，非标准，作对照）
    uplane: CompressionConfig = field(default_factory=CompressionConfig)
    # Cat B 的 C-plane 波束赋形权值压缩（bfwCompMeth / bfwIqWidth）
    bfw: CompressionConfig = field(default_factory=lambda: CompressionConfig("bfp", 9))
    mtu_bytes: int = 1500               # 以太网 MTU（eCPRI 消息最大长度）

    # ---- 仿真域 ----
    domain: str = "freq"                # "freq"：频域（不做 IFFT/FFT）；"time"：时域（O-RU 做 IFFT+CP）
    fft_size: Optional[int] = None      # 时域 IFFT 点数，None 时取不小于 num_sc/0.85 的最小 2 的幂

    # ---- 信道 ----
    # "cdl"：每个 (UE, O-RU) 一条 3GPP CDL 链路，链路增益归一化，扫描 SNR；
    # "rt"：Sionna RT 射线追踪（rt 中的场景与链路预算），SNR 由 UE 位置决定，只支持频域仿真
    channel_model: str = "cdl"
    rt: RTConfig = field(default_factory=RTConfig)
    cdl_model: str = "C"
    delay_spread: float = 100e-9        # s
    ue_speed: float = 0.0               # m/s（最大速度，在 [0, ue_speed] 内均匀取值）
    ue_azimuth_range: float = 120.0     # 各链路 BS 侧方位随机旋转范围（度），使不同 UE 处于不同方向
    link_gain_db: Optional[List[List[float]]] = None   # [num_ue][num_oru] 大尺度增益，None 为全 0 dB

    # ---- UE 接收机 ----
    channel_estimation: str = "ls"      # "ls"：DMRS LS + PRG 内线性插值；"perfect"：理想信道
    ce_window: int = 6                  # LS 估计时每个导频使用的邻近导频数（PRG 内局部线性拟合）
    ce_delay_compensation: bool = True  # LS 估计前先补偿定时偏差（平均时延）带来的线性相位

    # ---- CSI 测量（闭环）----
    # "csi-rs"：测量时隙中经前传发送的预编码 CSI-RS 与 CSI-IM（见 csi_rs.py）；"genie"：直接用真实等效信道
    cqi_source: str = "csi-rs"
    csi_rs_symbol: int = 12             # CSI-RS / CSI-IM 所在 OFDM 符号（firstOFDMSymbolInTimeDomain）
    csi_rs_scrambling_id: Optional[int] = None   # NZP-CSI-RS 的 scramblingID，None 时取 n_cell_id

    # ---- 其他 ----
    device: Optional[str] = None        # None 时使用 sionna.phy.config.device

    # ------------------------------------------------------------------
    @property
    def num_layers(self) -> int:
        """总层数 L。"""
        return self.num_ue * self.num_layers_per_ue

    @property
    def num_tx_ant(self) -> int:
        """所有 O-RU 的发射天线总数。"""
        return self.num_oru * self.num_oru_ant

    @property
    def num_subcarriers(self) -> int:
        return 12 * self.num_prb

    @property
    def mu(self) -> int:
        return {15: 0, 30: 1, 60: 2, 120: 3}[self.subcarrier_spacing]

    @property
    def slots_per_second(self) -> int:
        return 1000 * 2 ** self.mu

    @property
    def prg_num_prb(self) -> int:
        return self.num_prb if self.prg_size == 0 else self.prg_size

    @property
    def num_prg(self) -> int:
        return -(-self.num_prb // self.prg_num_prb)

    def ifft_size(self, num_prb: Optional[int] = None) -> int:
        """IFFT 点数：未给出 fft_size 时取不小于子载波数 / 0.85 的最小 2 的幂（30 kHz 下 51 PRB 为 1024）。

        :param num_prb: 另行指定的 PRB 数（如估算 100 MHz 的前传比特率时取 273），此时按默认规则计算
        """
        if self.fft_size and (num_prb is None or num_prb == self.num_prb):
            return self.fft_size
        num_sc = 12 * (self.num_prb if num_prb is None else num_prb)
        return int(2 ** np.ceil(np.log2(num_sc / 0.85)))

    def mcs_of(self, ue: int) -> int:
        if isinstance(self.mcs_index, int):
            return self.mcs_index
        return list(self.mcs_index)[ue]

    def rnti_of(self, ue: int) -> int:
        return 100 + ue if self.rnti is None else list(self.rnti)[ue]

    def validate(self) -> None:
        if self.category not in ("A", "B", "8"):
            raise ValueError("category 必须为 'A'、'B' 或 '8'")
        if self.uplane.method == "fixed" and self.category != "8":
            raise ValueError("uplane.method='fixed'（时域均匀定点）只用于 option 8")
        if self.domain not in ("freq", "time"):
            raise ValueError("domain 必须为 'freq' 或 'time'")
        if self.channel_model not in ("cdl", "rt"):
            raise ValueError("channel_model 必须为 'cdl' 或 'rt'")
        if self.channel_model == "rt":
            if self.domain != "freq":
                raise ValueError("射线追踪信道暂只支持频域仿真（domain='freq'）")
            if len(self.rt.oru_positions) != self.num_oru or len(self.rt.oru_look_at) != self.num_oru:
                raise ValueError("rt.oru_positions / rt.oru_look_at 的个数必须等于 num_oru")
            if self.link_gain_db is not None:
                raise ValueError("射线追踪信道的大尺度增益由场景决定，link_gain_db 必须为 None")
        if self.precoder not in ("bd", "ezf", "rzf"):
            raise ValueError("precoder 必须为 'bd'、'ezf' 或 'rzf'")
        if self.cqi_source not in ("csi-rs", "genie"):
            raise ValueError("cqi_source 必须为 'csi-rs' 或 'genie'")
        if self.channel_estimation not in ("ls", "perfect"):
            raise ValueError("channel_estimation 必须为 'ls' 或 'perfect'")
        if self.num_layers > self.num_tx_ant:
            raise ValueError("总层数不能超过发射天线总数")
        if self.num_layers_per_ue > self.num_ue_ant:
            raise ValueError("每个 UE 的层数不能超过其接收天线数")
        if self.precoder == "bd":
            dim = self.num_tx_ant - (self.num_ue - 1) * self.num_ue_ant
            if dim < self.num_layers_per_ue:
                raise ValueError(
                    f"块对角化要求 N_t - (K-1)·N_r >= 每 UE 层数（当前 {dim}），"
                    "请增加发射天线或改用 precoder='ezf'/'rzf'")
        if self.link_gain_db is not None:
            if (len(self.link_gain_db) != self.num_ue or
                    any(len(r) != self.num_oru for r in self.link_gain_db)):
                raise ValueError("link_gain_db 形状必须为 [num_ue][num_oru]")
