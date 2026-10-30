"""O-RAN 7.2x 与 option 8 前传：U-plane IQ 与 C-plane 波束赋形权值的压缩传输，以及前传比特率估算。

一条 FronthaulLink 对应 O-DU 与一个 O-RU 之间的链路：
  - Cat A：U-plane 承载 N_TRX 路天线端口流（已预编码），C-plane 不带权值；
  - Cat B：U-plane 承载 L 路层流，C-plane 用 Section Extension 11（按 PRB bundle 的权值）
           下发 N_TRX×L 预编码权值，权值同样可压缩（bfwCompMeth / bfwIqWidth）；
  - option 8（3GPP TR 38.801 §11.1.2.8，PHY-RF 切分）：O-DU 做到 IFFT + CP，前传承载 N_TRX 路时域 IQ，
           没有 C-plane。3GPP 不规定 option 8 的接口（实际由 CPRI 实现，TR 38.801 §11.1.3.1），
           也没有压缩，位宽即每个 I/Q 分量的定点位宽（TR 38.801 附录 A 表 A-2 取 2×16 bit）。
cell-free 场景下每个 O-RU 一条链路：Cat A / option 8 各 O-RU 收到自己天线的信号，
Cat B 各 O-RU 都收到全部 L 路层流和属于自己天线的那部分权值。
"""

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sionna.phy.ofdm import OFDMDemodulator, OFDMModulator

from .compression import (FixedPoint, IdealIQ, from_blocks, make_compressor,
                          to_blocks)
from .config import CompressionConfig, ORANConfig

# option 8 用 BFP / 块缩放 / μ 律压缩时域样点时的块长（非标准，作对照；须整除每根天线一个时隙的样点数）
TIME_BLOCK_LEN = 16


@dataclass
class CodecStats:
    """压缩误差统计（跨批次累加）。"""

    signal_energy: float = 0.0
    error_energy: float = 0.0
    num_clipped: int = 0
    num_blocks: int = 0
    num_bits: int = 0

    def update(self, x: torch.Tensor, x_hat: torch.Tensor, num_clipped: int, num_blocks: int,
               num_bits: int) -> None:
        self.signal_energy += float((x.abs() ** 2).sum())
        self.error_energy += float(((x - x_hat).abs() ** 2).sum())
        self.num_clipped += num_clipped
        self.num_blocks += num_blocks
        self.num_bits += num_bits

    @property
    def sqnr_db(self) -> float:
        if self.error_energy == 0:
            return float("inf")
        return 10 * np.log10(self.signal_energy / self.error_energy)

    @property
    def evm_percent(self) -> float:
        if self.signal_energy == 0:
            return 0.0
        return 100 * np.sqrt(self.error_energy / self.signal_energy)


class IQCodec:
    """一类前传数据的编解码：定点化 -> 分块压缩 ->（传输）-> 解压 -> 反定点化。

    :param comp: 压缩配置
    :param nominal_power: 该数据流每个复样点的标称功率（决定定点缩放）
    :param block_len: 块长（U-plane 为 12，权值为 N_TRX）
    """

    def __init__(self, comp: CompressionConfig, nominal_power: float, block_len: int):
        self.config = comp
        self.compressor = make_compressor(comp.method, iq_width=comp.iq_width, **comp.kwargs)
        self.ideal = isinstance(self.compressor, IdealIQ)
        self.fixed_point = FixedPoint(nominal_power, comp.backoff_db)
        self.block_len = block_len
        self.stats = CodecStats()

    @property
    def bits_per_block(self) -> int:
        return self.compressor.bits_per_block(self.block_len)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., n]，沿最后一维每 block_len 个样点为一块。返回同形状的重建信号。"""
        num_blocks = x.numel() // self.block_len
        num_bits = num_blocks * self.bits_per_block
        if self.ideal:
            self.stats.update(x, x, 0, num_blocks, num_bits)
            return x
        x_int, num_clipped = self.fixed_point.quantize(x)
        x_rec, _ = self.compressor.roundtrip(to_blocks(x_int, self.block_len))
        x_hat = self.fixed_point.dequantize(from_blocks(x_rec)).to(x.dtype)
        self.stats.update(x, x_hat, num_clipped, num_blocks, num_bits)
        return x_hat

    def reset_stats(self) -> None:
        self.stats = CodecStats()


@lru_cache(maxsize=None)
def gaussian_loading_db(width: int) -> float:
    """width 位均匀定点量化高斯分量时，使均方误差最小的回退（分量 RMS 相对满量程，dB）。

    CPRI 不规定量化方法，O-DU 按位宽设定数字增益：增益大则削波噪声大，增益小则粒状噪声大。OFDM 时域样点的
    I/Q 分量近似高斯，均方误差 = 步长²/12 + 2·∫_k^∞ (x − k)² φ(x) dx（k 为削波电平 / RMS），在 0.001 的
    网格上取最小。例如 16 bit 时 k ≈ 5.9（回退 15.5 dB），8 bit 时 k ≈ 3.9（11.9 dB），4 bit 时 k ≈ 2.6（8.2 dB）。
    """
    def mse(k):
        step = k / 2 ** (width - 1)
        tail = (1 + k * k) * 0.5 * math.erfc(k / math.sqrt(2)) - k * math.exp(-k * k / 2) / math.sqrt(2 * math.pi)
        return step * step / 12 + 2 * tail

    k = min(np.arange(1.0, 8.0, 0.001), key=mse)
    return float(20 * np.log10(k))


def time_bits_per_sample(comp: CompressionConfig) -> float:
    """option 8 每个时域复样点的前传比特数（"fixed" 为 2·位宽；分块压缩含每块的压缩参数）。"""
    if comp.method == "fixed":
        return 2 * comp.iq_width
    c = make_compressor(comp.method, iq_width=comp.iq_width, **comp.kwargs)
    return c.bits_per_block(TIME_BLOCK_LEN) / TIME_BLOCK_LEN


class TimeDomainCodec:
    """option 8 的 U-plane：O-DU 做 IFFT + CP，时域 IQ 经量化送到 O-RU（O-RU 只做 DAC 和射频）。

    输入、输出都是 PRB 上的频域网格 [..., num_symbols, num_sc]，O-RU 之后的处理与 7.2x 共用：
        O-DU：补保护带 -> IFFT + CP                    -> 时域样点 [..., T]
        前传："fixed" 均匀定点，或 16 bit 定点后按 TIME_BLOCK_LEN 个样点分块压缩
        仿真等效：去 CP + FFT，取 PRB 上的子载波
    最后一步不是 O-RU 的功能，而是把 O-RU 发出的时域信号换算成频域信道模型的输入（射线追踪信道只支持频域）。
    信道时延不超过 CP、且量化后的 CP 仍是符号尾部的拷贝时，"先过信道、UE 再去 CP + FFT"与"先去 CP + FFT、
    再逐子载波乘信道"完全相同。"fixed" 逐样点量化，CP 与尾部样点的量化结果相同，等效是精确的；分块压缩时
    CP 与尾部落在不同的块里，等效忽略了 CP 上的量化误差经多径漏进 FFT 窗口的部分（与量化误差同阶的二阶小量）。
    保护带上的量化噪声在 UE 的 FFT 之后落在 PRB 以外，不影响接收。

    stats 在 PRB 上（频域）统计，与 7.2x 的 U-plane SQNR 口径相同；时域量化噪声均匀分布在全部 IFFT 点上，
    所以 PRB 上的 SQNR 比时域 SQNR（time_stats）高约 10·log10(IFFT 点数 / 子载波数)，51 PRB 时为 2.2 dB。

    :param nominal_power: 频域每个 RE 的标称功率（与 Cat A 相同，为 1 / 发射天线总数）
    """

    def __init__(self, comp: CompressionConfig, air, nominal_power: float, precision=None, device=None):
        self.config = comp
        self.air = air
        self.ideal = comp.method == "ideal"
        self.samples_per_symbol = air.fft_size + air.cp_length
        self.bits_per_sample = time_bits_per_sample(comp)
        self.modulator = OFDMModulator(air.cp_length, precision=precision, device=device)
        self.demodulator = OFDMDemodulator(air.fft_size, 0, air.cp_length, precision=precision, device=device)
        # Sionna 的 IFFT 是归一化的：时域样点平均功率 = 频域每 RE 功率 × 占用子载波比例
        time_power = nominal_power * air.num_sc / air.fft_size
        self.compressor = None
        if comp.method == "fixed":
            self.fixed_point = FixedPoint(time_power, gaussian_loading_db(comp.iq_width), width=comp.iq_width)
        elif not self.ideal:
            self.compressor = make_compressor(comp.method, iq_width=comp.iq_width, **comp.kwargs)
            self.fixed_point = FixedPoint(time_power, comp.backoff_db)
        self.stats = CodecStats()
        self.time_stats = CodecStats()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        num_samples = x.numel() // x.shape[-1] * self.samples_per_symbol
        num_bits = int(num_samples * self.bits_per_sample)
        if self.ideal:
            # 理想前传不经 IFFT/FFT，与 Cat A 理想前传逐比特相同
            self.stats.update(x, x, 0, num_samples, num_bits)
            self.time_stats.update(x, x, 0, num_samples, num_bits)
            return x
        t = self.modulator(self.air.to_fft_bins(x))                   # [..., T]
        t_int, num_clipped = self.fixed_point.quantize(t)
        num_blocks = num_samples
        if self.compressor is not None:
            t_int = from_blocks(self.compressor.roundtrip(to_blocks(t_int, TIME_BLOCK_LEN))[0])
            num_blocks = num_samples // TIME_BLOCK_LEN
        t_hat = self.fixed_point.dequantize(t_int).to(t.dtype)
        x_hat = self.air.active(self.demodulator(t_hat)).to(x.dtype)
        self.time_stats.update(t, t_hat, num_clipped, num_blocks, num_bits)
        self.stats.update(x, x_hat, num_clipped, num_blocks, num_bits)
        return x_hat

    def reset_stats(self) -> None:
        self.stats = CodecStats()
        self.time_stats = CodecStats()


class FronthaulLink:
    """O-DU 与一个 O-RU 之间的前传链路。

    :param air: 空口参数 :class:`~oran.oru.AirNumerology`，option 8 需要（O-DU 做 IFFT + CP）
    """

    def __init__(self, cfg: ORANConfig, oru_index: int = 0, air=None, precision=None, device=None):
        self.cfg = cfg
        self.oru_index = oru_index
        self.category = cfg.category
        if cfg.category in ("A", "8"):
            # 天线端口流：每 RE 总功率 1 均分到所有发射天线
            uplane_power = 1.0 / cfg.num_tx_ant
        else:
            # 层流：调制符号功率 1
            uplane_power = 1.0
        if cfg.category == "8":
            if air is None:
                raise ValueError("option 8 的前传需要空口参数 air（O-DU 做 IFFT + CP）")
            self.uplane = TimeDomainCodec(cfg.uplane, air, uplane_power, precision, device)
        else:
            self.uplane = IQCodec(cfg.uplane, uplane_power, block_len=12)
        self.bfw = None
        if cfg.category == "B":
            # 权值 ||W||_F = 1，每个元素平均功率 1/(N_t·L)；一块 = 一层在一个 bundle 上的 N_TRX 个权值
            self.bfw = IQCodec(cfg.bfw, 1.0 / (cfg.num_tx_ant * cfg.num_layers),
                               block_len=cfg.num_oru_ant)

    def transmit(self, uplane: torch.Tensor, weights: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        :param uplane: [batch, num_eaxc, num_symbols, num_sc]（option 8 与 Cat A 相同，IFFT + CP 在前传内部做）
        :param weights: Cat B 时为 [batch, num_prg, N_TRX, L]
        :return: O-RU 收到的 (uplane, weights)
        """
        uplane_hat = self.uplane(uplane)
        weights_hat = None
        if weights is not None:
            # 块沿 N_TRX 维：[batch, num_prg, L, N_TRX]
            w = weights.transpose(-1, -2)
            weights_hat = self.bfw(w).transpose(-1, -2)
        return uplane_hat, weights_hat

    def reset_stats(self) -> None:
        self.uplane.reset_stats()
        if self.bfw is not None:
            self.bfw.reset_stats()


# ---------------------------------------------------------------------------
# 前传比特率估算（eCPRI over Ethernet）
# 头部长度参照 O-RAN WG4 CUS：
#   以太网：前导码+SFD 8 + MAC 头 14 + VLAN 4 + FCS 4 + 帧间隔 12 = 42 字节
#   eCPRI 公共头 4 + ecpriPcid/RtcId 2 + ecpriSeqid 2 = 8 字节
#   U-plane：应用层头 4（dataDirection ... symbolId）+ 段头 4（sectionId, rb, symInc,
#            startPrbu, numPrbu）+ udCompHdr 1 + reserved 1
#   C-plane（段类型 1）：公共头 8 + 段头 8；SE-11 头 6（extType/ef 1, extLen 2,
#            disableBFWs/RAD 1, numBundPrb 1, bfwCompHdr 1）；每个 bundle：beamId 2 + bfwCompParam + 权值
# ---------------------------------------------------------------------------

ETH_OVERHEAD = 42
ECPRI_HEADER = 8
UPLANE_HEADERS = 4 + 4 + 2
CPLANE_HEADERS = 8 + 8
SE11_HEADER = 6
MAX_NUM_PRBU = 255


@dataclass
class FronthaulRate:
    """一个 O-RU 链路的前传比特率（bit/s）。"""

    num_eaxc: int
    uplane_payload: float
    uplane_total: float
    cplane_total: float
    detail: Dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return self.uplane_total + self.cplane_total


def uplane_rate(num_eaxc: int, num_prb: int, bits_per_prb: int, symbols_per_slot: int,
                slots_per_second: int, mtu_bytes: int = 1500) -> Tuple[float, float, int]:
    """U-plane 比特率。返回 (净荷比特率, 含协议头比特率, 每 eAxC 每符号的消息数)。"""
    bytes_per_prb = bits_per_prb / 8
    prb_per_msg = int(min(MAX_NUM_PRBU, (mtu_bytes - ECPRI_HEADER - UPLANE_HEADERS) // bytes_per_prb))
    num_msgs = int(np.ceil(num_prb / prb_per_msg))
    payload_bytes = num_prb * bytes_per_prb
    overhead_bytes = num_msgs * (ETH_OVERHEAD + ECPRI_HEADER + UPLANE_HEADERS)
    per_second = num_eaxc * symbols_per_slot * slots_per_second * 8
    return per_second * payload_bytes, per_second * (payload_bytes + overhead_bytes), num_msgs


def cplane_rate(num_eaxc: int, slots_per_second: int, num_bundles: int = 0, num_trx: int = 0,
                bfw_bits_per_block: int = 0, mtu_bytes: int = 1500) -> float:
    """C-plane 比特率（每 eAxC 每时隙一个段类型 1 的段；Cat B 时带 SE-11 权值）。"""
    per_bundle = 0 if num_bundles == 0 else 2 + bfw_bits_per_block / 8
    body = CPLANE_HEADERS + (SE11_HEADER + num_bundles * per_bundle if num_bundles else 0)
    body = 4 * np.ceil(body / 4)                       # 扩展补齐到 4 字节
    num_msgs = int(np.ceil((body + ECPRI_HEADER) / mtu_bytes))
    bytes_per_slot = body + num_msgs * (ETH_OVERHEAD + ECPRI_HEADER)
    return num_eaxc * slots_per_second * bytes_per_slot * 8


def option8_rate(cfg: ORANConfig, num_prb: Optional[int] = None) -> FronthaulRate:
    """option 8 单个 O-RU 链路的前传比特率。

    净荷 = 天线数 × 采样率 × 每样点比特数，与 TR 38.801 §11.1.4.2 的算法相同（"天线数 × 采样率 × 位宽 + 开销"）。
    采样率 = IFFT 点数 × 子载波间隔：30 kHz 下 51 PRB 为 30.72 Msps，273 PRB 为 122.88 Msps。TR 38.801 附录 A
    按 LTE 的 30.72 Msps / 20 MHz 线性放大到 100 MHz（153.6 Msps，32 天线 2×16 bit 得 157.3 Gbps），
    比 NR 100 MHz 的实际采样率高 25%。
    含协议头的比特率按 eCPRI IQ 数据消息（消息类型 0）走以太网估算，与 7.2x 的 U-plane 口径一致：
    每根天线每个 OFDM 符号的样点按 MTU 分成若干消息，每条加 eCPRI 头 8 字节和以太网开销 42 字节。
    option 8 没有 C-plane。走 CPRI 时另有控制字（16/15）和线路编码（8B/10B 或 64B/66B）开销，这里不计入。
    """
    fs = cfg.ifft_size(num_prb) * cfg.subcarrier_spacing * 1e3
    bits_per_sample = time_bits_per_sample(cfg.uplane)
    symbols_per_second = 14 * cfg.slots_per_second
    bytes_per_symbol = fs / symbols_per_second * bits_per_sample / 8
    num_msgs = int(np.ceil(bytes_per_symbol / (cfg.mtu_bytes - ECPRI_HEADER)))
    num_ant = cfg.num_oru_ant
    payload = num_ant * fs * bits_per_sample
    overhead = num_ant * symbols_per_second * num_msgs * (ETH_OVERHEAD + ECPRI_HEADER) * 8
    return FronthaulRate(num_ant, payload, payload + overhead, 0.0,
                         {"sample_rate": fs, "bits_per_sample": bits_per_sample,
                          "msgs_per_symbol": num_msgs})


def fronthaul_rate(cfg: ORANConfig, num_prb: Optional[int] = None) -> FronthaulRate:
    """按配置估算单个 O-RU 链路的前传比特率；num_prb 可另行指定（如 273 对应 100 MHz）。"""
    if cfg.category == "8":
        return option8_rate(cfg, num_prb)
    num_prb = cfg.num_prb if num_prb is None else num_prb
    symbols = 14
    uplane_comp = make_compressor(cfg.uplane.method, iq_width=cfg.uplane.iq_width, **cfg.uplane.kwargs)
    bits_per_prb = uplane_comp.bits_per_block(12)
    if cfg.category == "A":
        num_eaxc = cfg.num_oru_ant
        c_rate = cplane_rate(num_eaxc, cfg.slots_per_second, mtu_bytes=cfg.mtu_bytes)
    else:
        num_eaxc = cfg.num_layers
        prg = cfg.prg_num_prb
        num_bundles = -(-num_prb // prg)
        bfw_comp = make_compressor(cfg.bfw.method, iq_width=cfg.bfw.iq_width, **cfg.bfw.kwargs)
        c_rate = cplane_rate(num_eaxc, cfg.slots_per_second, num_bundles, cfg.num_oru_ant,
                             bfw_comp.bits_per_block(cfg.num_oru_ant), cfg.mtu_bytes)
    payload, total, num_msgs = uplane_rate(num_eaxc, num_prb, bits_per_prb, symbols,
                                           cfg.slots_per_second, cfg.mtu_bytes)
    return FronthaulRate(num_eaxc, payload, total, c_rate,
                         {"bits_per_prb": bits_per_prb, "msgs_per_symbol": num_msgs,
                          "num_prb": num_prb})
