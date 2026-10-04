"""IQ 压缩的公共部分：压缩器基类、压缩结果、定点转换、比特打包。

O-RAN 前传（WG4 CUS 规范）中，IQ 样点按"块"压缩，每块共用一个压缩参数：
  - U-plane 数据：一块 = 一个 eAxC 在一个 OFDM 符号上的一个 PRB（12 个 RE），
    压缩参数即 udCompParam（BFP 时为指数）；
  - C-plane 波束赋形权值（Cat B）：一块 = 一个 eAxC 在一个 PRB 组（bundle）上的
    N_TRX 个权值，压缩参数即 bfwCompParam。
压缩器只处理"块"，不关心块里装的是数据还是权值，所以两处可以共用同一套算法。

数据流：
    浮点复数 --FixedPoint.quantize--> 16 bit 定点 --compress--> CompressedIQ（可 pack 成字节流）
            <--FixedPoint.dequantize-- 16 bit 定点 <--decompress--
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

# O-RAN WG4 CUS 中 udCompMeth / bfwCompMeth 的编码
COMP_METH_CODES = {
    "none": 0,                     # 不压缩（定点，位宽由 udIqWidth 给出，0 表示 16 bit）
    "bfp": 1,                      # 块浮点
    "block_scaling": 2,            # 块缩放
    "mu_law": 3,                   # μ 律
    "modulation": 4,               # 调制压缩
    "bfp_selective_re": 5,         # BFP + 选择性 RE 发送
    "modulation_selective_re": 6,  # 调制压缩 + 选择性 RE 发送
}


@dataclass
class CompressedIQ:
    """一批压缩后的 IQ 块。

    ``i``/``q`` 是压缩后的样点（整数取值，int32 存储），形状 [..., num_blocks, block_len]；
    ``param`` 是每块 udCompParam（或 bfwCompParam）字段的整数值，形状 [..., num_blocks]，
    没有该字段的方法为 None。各方法的含义：BFP 为指数；块缩放为 sblockScaler；
    μ 律为 compBitWidth << 4 | compShift。
    """

    method: str
    iq_width: int
    param_bits: int
    i: torch.Tensor
    q: torch.Tensor
    param: Optional[torch.Tensor] = None

    @property
    def block_len(self) -> int:
        return self.i.shape[-1]

    @property
    def num_blocks(self) -> int:
        return self.i.numel() // self.block_len

    @property
    def bits_per_block(self) -> int:
        return block_bits(self.block_len, self.iq_width, self.param_bits)

    @property
    def total_bits(self) -> int:
        return self.num_blocks * self.bits_per_block


def block_bits(block_len: int, iq_width: int, param_bits: int) -> int:
    """一块压缩数据的比特数：压缩参数 + block_len 对 I/Q，样点部分按字节对齐。

    U-plane 一个 PRB 为 24·iq_width 比特，总是整字节；权值块（N_TRX 对 I/Q）不一定，
    这里按 O-RAN 的做法补零到字节边界。
    """
    sample_bits = 2 * block_len * iq_width
    return param_bits + 8 * int(np.ceil(sample_bits / 8))


class IQCompressor(ABC):
    """IQ 压缩器基类。

    输入输出都是"定点取值"的复数张量（实部、虚部为 [-2^15, 2^15-1] 内的整数，
    以浮点类型存储），形状 [..., num_blocks, block_len]。

    子类需定义 ``method``（注册名），并实现 :meth:`compress` 和 :meth:`decompress`。
    """

    method: str = ""
    #: 每块压缩参数的比特数（如 BFP 的 udCompParam 为 8 bit：4 bit 保留 + 4 bit 指数）
    param_bits: int = 0

    def __init__(self, iq_width: int, input_width: int = 16):
        if not 1 <= iq_width <= input_width:
            raise ValueError(f"iq_width 必须在 [1, {input_width}] 内，当前为 {iq_width}")
        self.iq_width = int(iq_width)
        self.input_width = int(input_width)

    @property
    def comp_meth(self) -> int:
        """O-RAN udCompMeth / bfwCompMeth 编码。"""
        return COMP_METH_CODES[self.method]

    @property
    def is_lossless(self) -> bool:
        return False

    def bits_per_block(self, block_len: int) -> int:
        return block_bits(block_len, self.iq_width, self.param_bits)

    @abstractmethod
    def compress(self, x: torch.Tensor) -> CompressedIQ:
        """压缩。x: [..., num_blocks, block_len] 复数，实虚部为定点整数取值。"""

    @abstractmethod
    def decompress(self, c: CompressedIQ) -> torch.Tensor:
        """解压，返回与压缩前同形状的复数张量（定点整数取值）。"""

    def roundtrip(self, x: torch.Tensor) -> Tuple[torch.Tensor, CompressedIQ]:
        """压缩再解压，返回 (重建信号, 压缩结果)。"""
        c = self.compress(x)
        return self.decompress(c), c

    def __repr__(self) -> str:
        return f"{type(self).__name__}(iq_width={self.iq_width})"


def round_half_away(x: torch.Tensor) -> torch.Tensor:
    """四舍五入，0.5 远离零（MATLAB round 的语义，规范附录 A 的伪代码按此书写）。"""
    return torch.sign(x) * torch.floor(x.abs() + 0.5)


def saturate(x: torch.Tensor, width: int) -> torch.Tensor:
    """把整数取值饱和到 width 位有符号整数范围 [-2^(w-1), 2^(w-1)-1]。"""
    lim = 2 ** (width - 1)
    return x.clamp(-lim, lim - 1)


class FixedPoint:
    """浮点复数与定点（默认 16 bit）之间的转换，对应 O-DU/O-RU 内部的 IQ 定点格式。

    缩放系数按"标称功率 + 回退"确定：功率为 ``nominal_power`` 的信号，其 I、Q 分量的
    RMS 比满量程低 ``backoff_db`` dB。缩放是静态的（不随时隙变化）；BFP 的指数会在
    每个 PRB 内自适应动态范围，所以静态缩放只需保证不溢出、不过分浪费分辨率。

    :param nominal_power: 标称平均功率 E|x|^2（每个复样点）
    :param backoff_db: I/Q 分量 RMS 相对满量程的回退
    :param width: 定点位宽（O-RAN 不压缩时为 16）
    """

    def __init__(self, nominal_power: float, backoff_db: float = 15.0, width: int = 16):
        if nominal_power <= 0:
            raise ValueError("nominal_power 必须为正")
        self.width = int(width)
        self.backoff_db = float(backoff_db)
        self.nominal_power = float(nominal_power)
        full_scale = 2.0 ** (self.width - 1)
        rms_per_component = np.sqrt(self.nominal_power / 2)
        self.scale = full_scale * 10 ** (-self.backoff_db / 20) / rms_per_component

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """浮点 -> 定点（四舍五入 + 饱和）。返回 (定点取值的复数张量, 被削顶的分量个数)。"""
        lim = 2 ** (self.width - 1)
        re = torch.round(x.real * self.scale)
        im = torch.round(x.imag * self.scale)
        num_clipped = int(((re < -lim) | (re > lim - 1)).sum() + ((im < -lim) | (im > lim - 1)).sum())
        return torch.complex(saturate(re, self.width), saturate(im, self.width)), num_clipped

    def dequantize(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.scale


def to_blocks(x: torch.Tensor, block_len: int) -> torch.Tensor:
    """把最后一维（如子载波）切成长度为 block_len 的块：[..., n] -> [..., n/block_len, block_len]。"""
    n = x.shape[-1]
    if n % block_len:
        raise ValueError(f"最后一维长度 {n} 不是块长 {block_len} 的整数倍")
    return x.reshape(*x.shape[:-1], n // block_len, block_len)


def from_blocks(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(*x.shape[:-2], -1)


# ---------------------------------------------------------------------------
# 比特打包：按 O-RAN U-plane 的 PRB 格式把压缩结果序列化成字节流
#   每块：[udCompParam（param_bits，如有）][I0 Q0 I1 Q1 ...（各 iq_width 位，二进制补码，MSB 在前）]
#   样点部分补零到字节边界。打包只用于检查格式与比特数，仿真主流程不需要。
# ---------------------------------------------------------------------------

def _int_to_bits(v: np.ndarray, width: int) -> np.ndarray:
    """有符号整数 -> 补码比特（MSB 在前），v: [N] -> [N, width]。"""
    u = np.asarray(v, dtype=np.int64) & ((1 << width) - 1)
    shifts = np.arange(width - 1, -1, -1)
    return ((u[:, None] >> shifts) & 1).astype(np.uint8)


def _bits_to_int(b: np.ndarray, width: int, signed: bool = True) -> np.ndarray:
    """补码比特 -> 整数，b: [N, width] -> [N]。"""
    weights = (1 << np.arange(width - 1, -1, -1)).astype(np.int64)
    u = (b.astype(np.int64) * weights).sum(-1)
    if signed:
        u = np.where(u >= (1 << (width - 1)), u - (1 << width), u)
    return u


def pack(c: CompressedIQ) -> bytes:
    """把 CompressedIQ 打包成字节流（所有块依次排列）。"""
    i = c.i.reshape(-1, c.block_len).cpu().numpy()
    q = c.q.reshape(-1, c.block_len).cpu().numpy()
    nb = i.shape[0]
    iq = np.stack([i, q], axis=-1).reshape(nb, -1)  # I0 Q0 I1 Q1 ...
    sample_bits = _int_to_bits(iq.reshape(-1), c.iq_width).reshape(nb, -1)
    pad = c.bits_per_block - c.param_bits - sample_bits.shape[1]
    parts = []
    if c.param_bits:
        p = c.param.reshape(-1).cpu().numpy()
        parts.append(_int_to_bits(p, c.param_bits))
    parts.append(sample_bits)
    if pad:
        parts.append(np.zeros([nb, pad], np.uint8))
    return np.packbits(np.concatenate(parts, axis=1).reshape(-1)).tobytes()


def unpack(data: bytes, method: str, iq_width: int, param_bits: int,
           block_len: int, shape: Tuple[int, ...], device=None) -> CompressedIQ:
    """pack 的逆过程。shape 为块的前导形状 [..., num_blocks]。"""
    nb = int(np.prod(shape))
    bpb = block_bits(block_len, iq_width, param_bits)
    bits = np.unpackbits(np.frombuffer(data, np.uint8))[: nb * bpb].reshape(nb, bpb)
    param = None
    if param_bits:
        param = _bits_to_int(bits[:, :param_bits], param_bits, signed=False)
        param = torch.as_tensor(param.reshape(shape), device=device)
    samples = bits[:, param_bits: param_bits + 2 * block_len * iq_width]
    iq = _bits_to_int(samples.reshape(-1, iq_width), iq_width).reshape(nb, block_len, 2)
    i = torch.as_tensor(iq[..., 0].reshape(*shape, block_len), dtype=torch.int32, device=device)
    q = torch.as_tensor(iq[..., 1].reshape(*shape, block_len), dtype=torch.int32, device=device)
    return CompressedIQ(method, iq_width, param_bits, i, q, param)
