"""μ 律（μ-law）压缩，O-RAN WG4 CUS（v08.00）附录 A.3，udCompMeth = 3。

先对整个 PRB 做左移（compShift，0~7）以利用动态范围，再对每个样点的幅度做 μ = 8 的
三段分段线性压扩（小幅度量化步长更细），最后恢复符号。输入固定为 16 bit 定点。

压缩（附录 A.3.1，absBitWidth = 15，B = compBitWidth）：

    maxVal    = 块内 |I|、|Q| 的最大值
    compShift = 0（maxVal ≥ 2^14），1（< 2^14），2（< 2^13），…，7（< 2^8）
    a         = min(|x| · 2^compShift, 2^15 - 1)
    c         = a / 2^(15-B)                    a ≤ 2^13
                a / 2^(16-B) + 2^(B-3)          2^13 < a ≤ 2^14
                a / 2^(17-B) + 2^(B-2)          a > 2^14
    comp      = round(sign(x) · c)

解压（附录 A.3.2）：

    a     = min(|comp|, 2^(B-1) - 1)
    d     = a · 2^(15-B)                        a ≤ 2^(B-2)
            a · 2^(16-B) - 2^13                 a ≤ 2^(B-2) + 2^(B-3)
            a · 2^(17-B) - 2^15                 其他
    x_hat = sign(comp) · d / 2^compShift

udCompParam 为 1 字节：高 4 位 compBitWidth，低 4 位 compShift。

规范未说明的两处：
  - 幅度接近满量程时 c 取整后可达 2^(B-1)，超出 B 位补码范围，这里饱和到 2^(B-1) - 1；
  - 解压的最后一步 "/ 2^compShift" 按伪代码做实数除法，不取整。
"""

import torch

from .base import CompressedIQ, IQCompressor, round_half_away, saturate

ABS_BITS = 15        # absBitWidth
MAX_SHIFT = 7


class MuLawCompressor(IQCompressor):
    """μ 律压缩器。

    :param iq_width: compBitWidth，即压缩后 I、Q 各自的位宽。udCompParam 中只有 4 bit，
        且分段常数 2^(B-3) 要求 B ≥ 3，因此取值范围为 3~15
    :param input_width: 输入定点位宽，规范固定为 16
    """

    method = "mu_law"
    param_bits = 8

    def __init__(self, iq_width: int = 8, input_width: int = 16):
        if input_width != 16:
            raise ValueError("规范的 μ 律算法输入固定为 16 bit")
        if not 3 <= iq_width <= 15:
            raise ValueError("μ 律的 compBitWidth 须在 3~15 内")
        super().__init__(iq_width, input_width)

    def comp_shift(self, x: torch.Tensor) -> torch.Tensor:
        """每块的 compShift，x: [..., num_blocks, block_len] -> [..., num_blocks]（int32）。"""
        max_val = torch.maximum(x.real.abs().amax(-1), x.imag.abs().amax(-1))
        # maxVal 的位长 n = floor(log2(maxVal)) + 1（maxVal=0 时为 0），compShift = 15 - n，限制在 0~7
        _, nbits = torch.frexp(max_val)
        return (ABS_BITS - nbits).clamp(0, MAX_SHIFT).to(torch.int32)

    def _compand(self, a: torch.Tensor) -> torch.Tensor:
        b = self.iq_width
        return torch.where(
            a <= 2 ** (ABS_BITS - 2), a / 2.0 ** (ABS_BITS - b),
            torch.where(a <= 2 ** (ABS_BITS - 1),
                        a / 2.0 ** (ABS_BITS - b + 1) + 2.0 ** (b - 3),
                        a / 2.0 ** (ABS_BITS - b + 2) + 2.0 ** (b - 2)))

    def compress(self, x: torch.Tensor) -> CompressedIQ:
        shift = self.comp_shift(x)
        scale = torch.exp2(shift.to(torch.float64)).unsqueeze(-1)

        def one(v):
            v = v.to(torch.float64)
            a = torch.clamp(v.abs() * scale, max=2 ** ABS_BITS - 1)      # 左移并饱和
            c = round_half_away(torch.sign(v) * self._compand(a))
            return saturate(c, self.iq_width).to(torch.int32)

        param = (self.iq_width << 4) | shift
        return CompressedIQ(self.method, self.iq_width, self.param_bits,
                            one(x.real), one(x.imag), param)

    def decompress(self, c: CompressedIQ) -> torch.Tensor:
        b = c.iq_width                                  # 即 udCompParam 的高 4 位
        shift = (c.param & 0xF).to(torch.float64).unsqueeze(-1)

        def one(v):
            v = v.to(torch.float64)
            a = torch.clamp(v.abs(), max=2 ** (b - 1) - 1)
            d = torch.where(
                a <= 2 ** (b - 2), a * 2.0 ** (ABS_BITS - b),
                torch.where(a <= 2 ** (b - 2) + 2 ** (b - 3),
                            a * 2.0 ** (ABS_BITS - b + 1) - 2.0 ** 13,
                            a * 2.0 ** (ABS_BITS - b + 2) - 2.0 ** 15))
            return (torch.sign(v) * d / torch.exp2(shift)).float()

        return torch.complex(one(c.i), one(c.q))
