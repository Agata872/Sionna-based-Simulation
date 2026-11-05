"""O-RAN 7.2x、option 7-1 与 option 8 前传：U-plane IQ 与 C-plane 波束赋形权值的压缩传输，以及前传比特率估算。

一条 FronthaulLink 对应 O-DU 与一个 O-RU 之间的链路：
  - Cat A：U-plane 承载 N_TRX 路天线端口流（已预编码），C-plane 不带权值；
  - Cat B：U-plane 承载 L 路层流，C-plane 用 Section Extension 11（按 PRB bundle 的权值）
           下发 N_TRX×L 预编码权值，权值同样可压缩（bfwCompMeth / bfwIqWidth）；U-plane 还可用调制压缩
           （附录 A.5，:class:`ModulationCodec`），层符号按星座无损表示，每 RE 只要 Qm bit；
  - option 7-1（3GPP TR 38.801 §11.1.2.7，TR 38.816 §4.2）：O-DU 做到预编码和 RE 映射，前传承载 N_TRX 路
           整个载波（全部子载波、每个符号，与调度无关）的频域 IQ，O-RU 只做 IFFT + CP。3GPP 不规定接口与压缩
           （TR 38.816 附录 A 取 2×16 bit），这里用每个 I/Q 分量 w bit 的均匀定点（"fixed"，按位宽取最优削波电平）。
           空口信号与 Cat A 相同；与 Cat A 的区别在接口：不分段、只传整个载波、没有 O-RAN 压缩和 C-plane；
  - option 8（3GPP TR 38.801 §11.1.2.8，PHY-RF 切分）：O-DU 做到 IFFT + CP，前传承载 N_TRX 路时域 IQ，
           没有 C-plane。3GPP 不规定 option 8 的接口（实际由 CPRI 实现，TR 38.801 §11.1.3.1），
           也没有压缩，位宽即每个 I/Q 分量的定点位宽（TR 38.801 附录 A 表 A-2 取 2×16 bit）。
cell-free 场景下每个 O-RU 一条链路：Cat A / 7-1 / option 8 各 O-RU 收到自己天线的信号，
Cat B 各 O-RU 都收到全部 L 路层流和属于自己天线的那部分权值。
"""

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sionna.phy.ofdm import OFDMDemodulator, OFDMModulator

from .compression import (FixedPoint, IdealIQ, ModulationCompressor, decode_scaler,
                          encode_scaler, from_blocks, make_compressor, qam_width, to_blocks)
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

    method = "fixed" 时只做均匀定点（每个 I/Q 分量 iq_width 位，削波电平按位宽取高斯信号的最优值，
    见 :func:`gaussian_loading_db`），用于 option 7-1 的频域 IQ。

    :param comp: 压缩配置
    :param nominal_power: 该数据流每个复样点的标称功率（决定定点缩放）
    :param block_len: 块长（U-plane 为 12，权值为 N_TRX）
    """

    def __init__(self, comp: CompressionConfig, nominal_power: float, block_len: int):
        self.config = comp
        self.uniform = comp.method == "fixed"
        self.compressor = None if self.uniform else \
            make_compressor(comp.method, iq_width=comp.iq_width, **comp.kwargs)
        self.ideal = isinstance(self.compressor, IdealIQ)
        self.block_len = block_len
        self.set_nominal_power(nominal_power)
        self.stats = CodecStats()

    def set_nominal_power(self, nominal_power) -> None:
        """重设定点化的标称功率（数字增益）。可为张量，按广播规则对应输入的各数据流（如 [num_eaxc, 1, 1]）。"""
        if self.uniform:
            w = self.config.iq_width
            self.fixed_point = FixedPoint(nominal_power, gaussian_loading_db(w), width=w)
        else:
            self.fixed_point = FixedPoint(nominal_power, self.config.backoff_db)

    @property
    def bits_per_block(self) -> int:
        if self.uniform:
            return 2 * self.config.iq_width * self.block_len
        return self.compressor.bits_per_block(self.block_len)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., n]，沿最后一维每 block_len 个样点为一块。返回同形状的重建信号。"""
        num_blocks = x.numel() // self.block_len
        num_bits = num_blocks * self.bits_per_block
        if self.ideal:
            self.stats.update(x, x, 0, num_blocks, num_bits)
            return x
        x_int, num_clipped = self.fixed_point.quantize(x)
        if not self.uniform:
            x_int = from_blocks(self.compressor.roundtrip(to_blocks(x_int, self.block_len))[0])
        x_hat = self.fixed_point.dequantize(x_int).to(x.dtype)
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
        self.compressor = None
        if comp.method not in ("fixed", "ideal"):
            self.compressor = make_compressor(comp.method, iq_width=comp.iq_width, **comp.kwargs)
        self.set_nominal_power(nominal_power)
        self.stats = CodecStats()
        self.time_stats = CodecStats()

    def set_nominal_power(self, nominal_power) -> None:
        """重设定点化的标称功率（频域每 RE，可为 [num_eaxc, 1] 的张量，对应每根天线的时域样点）。"""
        if self.ideal:
            return
        # Sionna 的 IFFT 是归一化的：时域样点平均功率 = 频域每 RE 功率 × 占用子载波比例
        time_power = nominal_power * self.air.num_sc / self.air.fft_size
        if self.config.method == "fixed":
            w = self.config.iq_width
            self.fixed_point = FixedPoint(time_power, gaussian_loading_db(w), width=w)
        else:
            self.fixed_point = FixedPoint(time_power, self.config.backoff_db)

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


@dataclass
class REMap:
    """调制压缩所需的逐 RE 星座信息（O-DU 已知，见 :meth:`oran.odu.ODU.re_map`）。

    width：每层每个 RE 的星座位宽 w_c（数据为 Qm/2，DMRS、CSI-RS 为 1，不承载的 RE 为 0），[L, num_symbols, num_sc]；
    unit：星座的幅度单位 u（RE 的值为 u·(a + jb)，a、b 为奇数），同形状。
    """

    width: torch.Tensor
    unit: torch.Tensor


class ModulationCodec:
    """Cat B U-plane 的调制压缩（O-RAN CUS 附录 A.5，udCompMeth = 4），算法见 :mod:`oran.compression.modulation`。

    段的划分：每层（eAxC）每个 OFDM 符号一个段，覆盖全部 PRB，udIqWidth 取该层该符号上最大的星座位宽：
    数据符号为 Qm/2，DMRS 符号为 1（QPSK，另一 CDM 组的 RE 为 0，不被任何掩码指到）；测量时隙的 CSI-RS 符号上
    数据与 CSI-RS 同段，CSI-RS（QPSK）不平移（csf = 0），CSI-IM 与配对 UE 的 CMR 位置不承载。
    参数下发：数据段用 SE 5（一组掩码），CSI-RS 符号用两组；每个 PRB 的 U-plane 为 24·udIqWidth bit，没有 udCompParam。
    O-RU 的满量程与 16 bit 定点相同：标称功率信号的 I/Q 分量 RMS 比满量程低 backoff_db。
    唯一的失真来自 15 bit scaler 的量化（相对误差不超过 2^-11，约 −66 dB 以下）。
    """

    def __init__(self, comp: CompressionConfig, num_prb: int, nominal_power: float = 1.0):
        self.config = comp
        self.num_prb = num_prb
        self.ideal = False
        self.full_scale = float(np.sqrt(nominal_power / 2) * 10 ** (comp.backoff_db / 20))
        self.stats = CodecStats()

    @staticmethod
    def section_width(re_map: REMap) -> torch.Tensor:
        """每层每个符号一个段的 udIqWidth：[L, num_symbols, 1]。"""
        return re_map.width.amax(dim=-1, keepdim=True).clamp(min=1)

    def __call__(self, x: torch.Tensor, re_map: REMap) -> torch.Tensor:
        """x: [batch, L, num_symbols, num_sc] 层资源网格。返回 O-RU 解压后的层网格。"""
        if re_map is None:
            raise ValueError("调制压缩需要 O-DU 给出逐 RE 的星座信息（re_map）")
        width = re_map.width.to(x.device)
        unit = re_map.unit.to(x.device)
        w_sec = self.section_width(REMap(width, unit))
        scaler = decode_scaler(encode_scaler(ModulationCompressor.scaler(unit, width, self.full_scale)))
        i, q, csf, clipped = ModulationCompressor.compress(x, unit, width, w_sec)
        y = ModulationCompressor.decompress(i, q, csf, scaler, w_sec, width > 0)
        x_hat = (self.full_scale * y).to(x.dtype)
        batch, num_layers, num_symbols = x.shape[:3]
        num_blocks = batch * num_layers * num_symbols * self.num_prb
        num_bits = batch * self.num_prb * ModulationCompressor.bits_per_prb(int(w_sec.sum()))
        self.stats.update(x, x_hat, clipped, num_blocks, num_bits)
        return x_hat

    def reset_stats(self) -> None:
        self.stats = CodecStats()


class FronthaulLink:
    """O-DU 与一个 O-RU 之间的前传链路。

    :param air: 空口参数 :class:`~oran.oru.AirNumerology`，option 8 需要（O-DU 做 IFFT + CP）
    """

    def __init__(self, cfg: ORANConfig, oru_index: int = 0, air=None, precision=None, device=None):
        self.cfg = cfg
        self.oru_index = oru_index
        self.category = cfg.category
        if cfg.category in ("A", "7-1", "8"):
            # 天线端口流：每 RE 总功率 1 均分到所有发射天线
            uplane_power = 1.0 / cfg.num_tx_ant
        else:
            # 层流：调制符号功率 1
            uplane_power = 1.0
        if cfg.category == "8":
            if air is None:
                raise ValueError("option 8 的前传需要空口参数 air（O-DU 做 IFFT + CP）")
            self.uplane = TimeDomainCodec(cfg.uplane, air, uplane_power, precision, device)
        elif cfg.uplane.method == "modulation":
            if cfg.category != "B":
                raise ValueError("调制压缩只用于 Cat B")
            self.uplane = ModulationCodec(cfg.uplane, cfg.num_prb, uplane_power)
        else:
            self.uplane = IQCodec(cfg.uplane, uplane_power, block_len=12)
        self.bfw = None
        if cfg.category == "B":
            # 权值 ||W||_F = 1，每个元素平均功率 1/(N_t·L)；一块 = 一层在一个 bundle 上的 N_TRX 个权值
            self.bfw = IQCodec(cfg.bfw, 1.0 / (cfg.num_tx_ant * cfg.num_layers),
                               block_len=cfg.num_oru_ant)

    def set_stream_power(self, power: torch.Tensor) -> None:
        """按每路天线流的实际平均功率（每 RE）重设 U-plane 定点化的数字增益（Cat A / 7-1 / option 8）。

        :param power: [N_TRX]，该 O-RU 各天线的平均功率（层符号功率为 1 时由预编码权值算出）
        """
        if self.category == "B":
            raise ValueError("Cat B 的 U-plane 是层流（功率为 1），不需要按天线设定增益")
        shape = (-1, 1) if self.category == "8" else (-1, 1, 1)
        self.uplane.set_nominal_power(power.reshape(shape))

    def transmit(self, uplane: torch.Tensor, weights: Optional[torch.Tensor] = None,
                 re_map: Optional[REMap] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        :param uplane: [batch, num_eaxc, num_symbols, num_sc]（option 8 与 Cat A 相同，IFFT + CP 在前传内部做）
        :param weights: Cat B 时为 [batch, num_prg, N_TRX, L]
        :param re_map: 调制压缩时 O-DU 给出的逐 RE 星座信息
        :return: O-RU 收到的 (uplane, weights)
        """
        if isinstance(self.uplane, ModulationCodec):
            uplane_hat = self.uplane(uplane, re_map)
        else:
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
# 调制压缩另需的 C-plane 扩展：SE 5 一组掩码 8 字节（表 7.7.5.1-1）；SE 6 的符号掩码 8 字节
# （extType/ef 1、extLen 1、repetition/rbgSize/rbgMask 4、priority/symbolMask 2），用于把不连续的数据符号放进一个段
SE5_BYTES = 8
SE6_BYTES = 8


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
                bfw_bits_per_block: int = 0, mtu_bytes: int = 1500, extra_bytes: int = 0) -> float:
    """C-plane 比特率（每 eAxC 每时隙一个段类型 1 的段；Cat B 时带 SE-11 权值）。

    :param extra_bytes: 该段另带的扩展字节数（如调制压缩的 SE 5、SE 6）
    """
    per_bundle = 0 if num_bundles == 0 else 2 + bfw_bits_per_block / 8
    body = CPLANE_HEADERS + (SE11_HEADER + num_bundles * per_bundle if num_bundles else 0) + extra_bytes
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


def _bits_per_re(comp: CompressionConfig) -> float:
    """频域每个 RE 的前传比特数（"fixed" 为 2·位宽；分块压缩含每 PRB 的 udCompParam）。"""
    if comp.method == "fixed":
        return 2.0 * comp.iq_width
    return make_compressor(comp.method, iq_width=comp.iq_width, **comp.kwargs).bits_per_block(12) / 12


def split71_rate(cfg: ORANConfig, num_prb: Optional[int] = None) -> FronthaulRate:
    """option 7-1 单个 O-RU 链路的前传比特率。

    净荷 = 天线数 × 子载波数 × 每秒 OFDM 符号数 × 每 RE 比特数，即 TR 38.816 附录 A 中 7-1 公式的
    B1·C1·D1·E1·F1 项（"子载波数 × 符号率 × 天线端口数 × IQ 位宽"）。num_prb 为载波的 PRB 数：7-1 不论调度
    都传整个载波，所以负载与 PRB 占用率无关。
    含协议头的比特率与 option 8 同口径（eCPRI IQ 数据消息 + 以太网，每根天线每个符号按 MTU 分成若干消息）。
    7-1 没有 O-RAN 的 C-plane；TR 38.816 另计的"调度/控制信令"开销（100 MHz、32 天线时 121 Mbit/s）见
    :func:`tr38816_rate`。
    """
    num_prb = cfg.num_prb if num_prb is None else num_prb
    num_sc = 12 * num_prb
    bits_per_re = _bits_per_re(cfg.uplane)
    symbols_per_second = 14 * cfg.slots_per_second
    bytes_per_symbol = num_sc * bits_per_re / 8
    num_msgs = int(np.ceil(bytes_per_symbol / (cfg.mtu_bytes - ECPRI_HEADER)))
    num_ant = cfg.num_oru_ant
    payload = num_ant * num_sc * symbols_per_second * bits_per_re
    overhead = num_ant * symbols_per_second * num_msgs * (ETH_OVERHEAD + ECPRI_HEADER) * 8
    return FronthaulRate(num_ant, payload, payload + overhead, 0.0,
                         {"bits_per_re": bits_per_re, "msgs_per_symbol": num_msgs, "num_prb": num_prb})


def modulation_rate(cfg: ORANConfig, qm_per_ue, num_prb: Optional[int] = None) -> FronthaulRate:
    """Cat B + 调制压缩时单个 O-RU 链路的前传比特率（负载由各 UE 的 MCS 决定）。

    U-plane：每层每个 PRB 24·udIqWidth bit，没有 udCompParam；数据符号 udIqWidth = Qm/2，DMRS 符号为 1
    （另一 CDM 组的空 RE 也占位，没有用选择性 RE 发送）。测量时隙中 CSI-RS 与数据同段，位宽不变。
    C-plane：在 Cat B 的段（SE 11 权值）上另加 SE 5 与 SE 6（数据符号不连续，用符号掩码放进一个段）；
    DMRS 符号的 udIqWidth 不同，每 eAxC 每时隙另发一条 C-plane 消息（段头 + SE 6 + SE 5 + SE 11 只带 beamId，
    即 disableBFWs = 1 复用数据段的权值）。测量时隙 CSI-RS 符号上第二组 SE 5 掩码（4 字节）忽略不计。

    :param qm_per_ue: 每个 UE 当前 MCS 的调制阶数 Qm（每符号比特数）
    """
    from .pdsch import make_pdsch_configs
    num_prb = cfg.num_prb if num_prb is None else num_prb
    pc = make_pdsch_configs(cfg)[0]
    num_dmrs = len(pc.dmrs_symbol_indices)
    num_data = pc.carrier.num_symbols_per_slot - num_dmrs
    nu = cfg.num_layers_per_ue
    payload = total = 0.0
    for qm in qm_per_ue:
        for num_sym, width in ((num_data, qam_width(int(qm))), (num_dmrs, 1)):
            p, t, _ = uplane_rate(nu, num_prb, ModulationCompressor.bits_per_prb(width), num_sym,
                                  cfg.slots_per_second, cfg.mtu_bytes)
            payload += p
            total += t
    num_bundles = -(-num_prb // cfg.prg_num_prb)
    bfw_comp = make_compressor(cfg.bfw.method, iq_width=cfg.bfw.iq_width, **cfg.bfw.kwargs)
    c_data = cplane_rate(cfg.num_layers, cfg.slots_per_second, num_bundles, cfg.num_oru_ant,
                         bfw_comp.bits_per_block(cfg.num_oru_ant), cfg.mtu_bytes,
                         extra_bytes=SE5_BYTES + SE6_BYTES)
    se11_beam_only = 4 * np.ceil((SE11_HEADER - 1 + 2 * num_bundles) / 4)   # 无 bfwCompHdr，每 bundle 一个 beamId
    dmrs_body = CPLANE_HEADERS + SE6_BYTES + SE5_BYTES + se11_beam_only
    dmrs_msgs = int(np.ceil((dmrs_body + ECPRI_HEADER) / cfg.mtu_bytes))
    c_dmrs = cfg.num_layers * cfg.slots_per_second * (dmrs_body + dmrs_msgs * (ETH_OVERHEAD + ECPRI_HEADER)) * 8
    return FronthaulRate(cfg.num_layers, payload, total, c_data + c_dmrs,
                         {"qm_per_ue": [int(q) for q in qm_per_ue], "num_prb": num_prb,
                          "num_data_symbols": num_data, "num_dmrs_symbols": num_dmrs})


def tr38816_rate(split: str, num_ports: int, bandwidth_mhz: float, iq_bits: int = 32,
                 with_margin: bool = True) -> float:
    """TR 38.816 V15.0.0 附录 A 的下行 7-1 / 7-2 前传带宽（bit/s），作为口径对照。

    A1 = 1200 子载波/20 MHz × (100/20) × 14 符号/ms × N × IQ 位宽 + G1（TR 38.801 的 100 MHz 参考值，
    7-1 的 N 为 32 个天线端口、G1 = 121 Mbit/s；7-2 的 N 为 8 层、G1 = 713.9 Mbit/s），再按带宽和端口数（层数）
    线性缩放；with_margin 时乘以频谱利用率 99%/90% 与 20% 余量（参数集 #1~#6 的口径，表 A-2）。
    例如 7-1、100 MHz、32 端口、2×16 bit 为 113.6 Gbit/s，7-2、8 层为 29.3 Gbit/s。

    :param num_ports: 7-1 为天线端口数，7-2 为层数
    """
    ref = {"7-1": (32, 121e6), "7-2": (8, 713.9e6)}
    if split not in ref:
        raise ValueError("split 必须为 '7-1' 或 '7-2'")
    n_ref, g1 = ref[split]
    a1 = 1200 * (100 / 20) * 14e3 * n_ref * iq_bits + g1
    rate = a1 * (bandwidth_mhz / 100) * (num_ports / n_ref)
    if with_margin:
        rate *= (0.99 / 0.90) * 1.2
    return rate


def fronthaul_rate(cfg: ORANConfig, num_prb: Optional[int] = None, qm_per_ue=None) -> FronthaulRate:
    """按配置估算单个 O-RU 链路的前传比特率；num_prb 可另行指定（如 273 对应 100 MHz）。

    :param qm_per_ue: Cat B + 调制压缩时各 UE 的调制阶数，None 时按 cfg.mcs_index 查表
    """
    if cfg.category == "8":
        return option8_rate(cfg, num_prb)
    if cfg.category == "7-1":
        return split71_rate(cfg, num_prb)
    if cfg.uplane.method == "modulation":
        if qm_per_ue is None:
            from .bler_model import mcs_entry
            qm_per_ue = [mcs_entry(cfg.mcs_table, cfg.mcs_of(k))[0] for k in range(cfg.num_ue)]
        return modulation_rate(cfg, qm_per_ue, num_prb)
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
