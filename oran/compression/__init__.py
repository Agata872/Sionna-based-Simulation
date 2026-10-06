"""O-RAN 前传 IQ 压缩算法。

用法::

    from oran.compression import make_compressor
    comp = make_compressor("bfp", iq_width=9)
    x_hat, c = comp.roundtrip(x_int)      # x_int: [..., num_blocks, block_len] 定点取值复数
    c.total_bits                          # 压缩后的比特数

已实现（算法按 O-RAN WG4 CUS v22.00 附录 A，与 v08.00 相同）：
    "ideal"          理想前传（不量化，仿真参考）
    "none"           定点不压缩（udCompMeth 0），iq_width 默认 16
    "bfp"            块浮点（udCompMeth 1，附录 A.1）
    "block_scaling"  块缩放（udCompMeth 2，附录 A.2）
    "mu_law"         μ 律（udCompMeth 3，附录 A.3），iq_width 取 3~15

扩展新算法：在本目录新建文件，继承 :class:`IQCompressor`，设置 ``method`` 和
``param_bits``，实现 compress/decompress，再用 :func:`register_compressor` 注册。
O-RAN 中其他方法（尚未实现）：modulation(4)、bfp_selective_re(5)、modulation_selective_re(6)、
bfp_selective_re_hdr(7)、modulation_selective_re_hdr(8)。
"""

from typing import Dict, Type

from .base import (COMP_METH_CODES, CompressedIQ, FixedPoint, IQCompressor,
                   block_bits, from_blocks, pack, round_half_away, to_blocks, unpack)
from .bfp import BFPCompressor
from .block_scaling import BlockScalingCompressor
from .mu_law import MuLawCompressor
from .none import IdealIQ, NoCompression

_REGISTRY: Dict[str, Type[IQCompressor]] = {}


def register_compressor(cls: Type[IQCompressor]) -> Type[IQCompressor]:
    """注册压缩器类（可作装饰器使用），注册名取 cls.method。"""
    if not cls.method:
        raise ValueError("压缩器类必须定义 method")
    _REGISTRY[cls.method] = cls
    return cls


def available_compressors():
    return sorted(_REGISTRY)


def make_compressor(method: str = "bfp", **kwargs) -> IQCompressor:
    """按名称创建压缩器，kwargs 传给对应类（如 iq_width=9）。"""
    if method not in _REGISTRY:
        raise ValueError(f"未知压缩方法 '{method}'，可选：{available_compressors()}")
    return _REGISTRY[method](**kwargs)


for _cls in (IdealIQ, NoCompression, BFPCompressor, BlockScalingCompressor, MuLawCompressor):
    register_compressor(_cls)

__all__ = [
    "COMP_METH_CODES", "CompressedIQ", "FixedPoint", "IQCompressor",
    "BFPCompressor", "BlockScalingCompressor", "MuLawCompressor", "NoCompression", "IdealIQ",
    "block_bits", "to_blocks", "from_blocks", "pack", "unpack", "round_half_away",
    "register_compressor", "available_compressors", "make_compressor",
]
