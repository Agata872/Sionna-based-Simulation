"""块缩放（Block Scaling）压缩，O-RAN WG4 CUS（v22.00）附录 A.2，udCompMeth = 2。

与 BFP 一样按块（U-plane 为一个 PRB 的 12 个 RE）共用一个参数，但参数不是 2 的幂次指数，
而是 8 bit 的乘性缩放因子 sblockScaler（无符号 Q1.7，128 表示 1.0），缩放粒度更细。
附录 A.2.2 / A.2.3 的参考算法（输入、输出均为 16 bit 定点）：

    maxV, minV   = 块内 I/Q 的最大值、最小值
    maxValue     = max(maxV, |minV| - 1)                              # 15 bit
    sblockScaler = ceil(maxValue / 2^8)，为 0 时取 1                   # 1~128
    c            = sat(round(2^7 / sblockScaler · x / 2^(16 - iqWidth)), iqWidth)
    x_hat        = round(sblockScaler · c / 2^(iqWidth - 9))          # 解压回 16 bit

round 按 MATLAB 语义（0.5 远离零）。udCompParam 即 sblockScaler（1 字节）。
规范伪代码先算倒数 2^7/sblockScaler 再相乘；这里按精确有理数计算（最后一步才做除法），
避免浮点误差在恰好为 .5 时改变取整方向。
"""

import torch

from .base import CompressedIQ, IQCompressor, round_half_away, saturate


class BlockScalingCompressor(IQCompressor):
    """块缩放压缩器。

    :param iq_width: 压缩后 I、Q 各自的位宽（udIqWidth）
    :param input_width: 输入定点位宽，规范的算法假定为 16
    """

    method = "block_scaling"
    param_bits = 8

    def __init__(self, iq_width: int = 9, input_width: int = 16):
        if input_width != 16:
            raise ValueError("规范的块缩放算法假定输入为 16 bit")
        super().__init__(iq_width, input_width)

    def scaler(self, x: torch.Tensor) -> torch.Tensor:
        """每块的 sblockScaler，x: [..., num_blocks, block_len] -> [..., num_blocks]（int32，1~128）。"""
        max_v = torch.maximum(x.real.amax(-1), x.imag.amax(-1))
        min_v = torch.minimum(x.real.amin(-1), x.imag.amin(-1))
        max_value = torch.maximum(max_v, min_v.abs() - 1).to(torch.float64)
        s = torch.ceil(max_value / 2 ** 8)
        return torch.where(s == 0, torch.ones_like(s), s).to(torch.int32)

    def compress(self, x: torch.Tensor) -> CompressedIQ:
        s = self.scaler(x)
        # c = x · 2^7 / (s · 2^(16 - w))：分子、分母均为整数，在 float64 中精确，只在除法处舍入一次
        den = (s.to(torch.float64) * 2.0 ** (16 - self.iq_width)).unsqueeze(-1)

        def one(v):
            c = round_half_away(v.to(torch.float64) * 2 ** 7 / den)
            return saturate(c, self.iq_width).to(torch.int32)

        return CompressedIQ(self.method, self.iq_width, self.param_bits,
                            one(x.real), one(x.imag), s)

    def decompress(self, c: CompressedIQ) -> torch.Tensor:
        s = c.param.to(torch.float64).unsqueeze(-1)

        def one(v):
            return round_half_away(s * v.to(torch.float64) / 2.0 ** (c.iq_width - 9)).float()

        return torch.complex(one(c.i), one(c.q))
