"""O-RAN 7.2x 前传：U-plane IQ 与 C-plane 波束赋形权值的压缩传输，以及前传比特率估算。

一条 FronthaulLink 对应 O-DU 与一个 O-RU 之间的链路：
  - Cat A：U-plane 承载 N_TRX 路天线端口流（已预编码），C-plane 不带权值；
  - Cat B：U-plane 承载 L 路层流，C-plane 用 Section Extension 11（按 PRB bundle 的权值）
           下发 N_TRX×L 预编码权值，权值同样可压缩（bfwCompMeth / bfwIqWidth）。
cell-free 场景下每个 O-RU 一条链路：Cat A 各 O-RU 收到自己天线的端口流，
Cat B 各 O-RU 都收到全部 L 路层流和属于自己天线的那部分权值。
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .compression import (FixedPoint, IdealIQ, from_blocks, make_compressor,
                          to_blocks)
from .config import CompressionConfig, ORANConfig


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


class FronthaulLink:
    """O-DU 与一个 O-RU 之间的前传链路。"""

    def __init__(self, cfg: ORANConfig, oru_index: int = 0):
        self.cfg = cfg
        self.oru_index = oru_index
        self.category = cfg.category
        if cfg.category == "A":
            # 天线端口流：每 RE 总功率 1 均分到所有发射天线
            uplane_power = 1.0 / cfg.num_tx_ant
        else:
            # 层流：调制符号功率 1
            uplane_power = 1.0
        self.uplane = IQCodec(cfg.uplane, uplane_power, block_len=12)
        self.bfw = None
        if cfg.category == "B":
            # 权值 ||W||_F = 1，每个元素平均功率 1/(N_t·L)；一块 = 一层在一个 bundle 上的 N_TRX 个权值
            self.bfw = IQCodec(cfg.bfw, 1.0 / (cfg.num_tx_ant * cfg.num_layers),
                               block_len=cfg.num_oru_ant)

    def transmit(self, uplane: torch.Tensor, weights: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        :param uplane: [batch, num_eaxc, num_symbols, num_sc]
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


def fronthaul_rate(cfg: ORANConfig, num_prb: Optional[int] = None) -> FronthaulRate:
    """按配置估算单个 O-RU 链路的前传比特率；num_prb 可另行指定（如 273 对应 100 MHz）。"""
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
