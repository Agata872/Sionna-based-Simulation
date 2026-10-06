"""块浮点（Block Floating Point, BFP）压缩，O-RAN WG4 CUS（v22.00）附录 A.1，udCompMeth = 1。

每块（U-plane 为一个 PRB 的 12 个 RE）确定一个公共指数 e，使块内样点右移 e 位后能放进
iq_width 位有符号整数（附录 A.1.2 的参考算法）：

    maxV     = max(Re, Im)，minV = min(Re, Im)            # 块内 I/Q 的最大值、最小值
    maxValue = max(maxV, |minV| - 1)                     # 补码负数可多表示一位，如 -256 用 9 bit 即可
    raw_exp  = floor(log2(maxValue)) + 1                 # 表示 maxValue 所需位数（不含符号位）
    e        = max(raw_exp - iq_width + 1, 0)            # +1 为符号位
    m        = sat(round(x / 2^e), iq_width)             # 尾数，iq_width 位补码
    x_hat    = m * 2^e                                   # 解压（附录 A.1.3）

规范的伪代码未说明取整进位溢出（如 511/2 -> 256 超出 9 bit）如何处理，这里做饱和。

udCompParam 为 1 字节：高 4 位保留，低 4 位为指数（0~15）。
每个 PRB 的比特数 = 8 + 24 * iq_width，例如 9 bit 时为 224 bit（28 字节），
相对 16 bit 不压缩（384 bit）压缩比约 1.71。
"""

import torch

from .base import CompressedIQ, IQCompressor, saturate


class BFPCompressor(IQCompressor):
    """块浮点压缩器。

    :param iq_width: 尾数位宽（udIqWidth），常用 9、12、14 等
    :param input_width: 输入定点位宽，O-RAN 为 16
    :param exponent_bits: 指数位数，O-RAN 为 4
    :param rounding: "nearest"（四舍五入，默认）或 "truncate"（直接算术右移，硬件最简实现）
    """

    method = "bfp"
    param_bits = 8

    def __init__(self, iq_width: int = 9, input_width: int = 16,
                 exponent_bits: int = 4, rounding: str = "nearest"):
        super().__init__(iq_width, input_width)
        if rounding not in ("nearest", "truncate"):
            raise ValueError("rounding 必须为 'nearest' 或 'truncate'")
        self.rounding = rounding
        self.max_exponent = 2 ** exponent_bits - 1
        if input_width - iq_width > self.max_exponent:
            raise ValueError("指数位数不足以覆盖输入与尾数的位宽差")

    @property
    def is_lossless(self) -> bool:
        return self.iq_width >= self.input_width

    def exponent(self, x: torch.Tensor) -> torch.Tensor:
        """每块的指数，x: [..., num_blocks, block_len] -> [..., num_blocks]（int32）。

        x 的实部、虚部须为定点整数取值。
        """
        max_v = torch.maximum(x.real.amax(-1), x.imag.amax(-1))
        min_v = torch.minimum(x.real.amin(-1), x.imag.amin(-1))
        max_value = torch.maximum(max_v, min_v.abs() - 1)   # 恒 >= 0
        # 对整数 v>0，frexp 给出 v = f * 2^k，f∈[0.5,1)，k = floor(log2(v)) + 1；v=0 时 k=0
        _, raw_exp = torch.frexp(max_value)
        return (raw_exp - self.iq_width + 1).clamp(0, self.max_exponent).to(torch.int32)

    def compress(self, x: torch.Tensor) -> CompressedIQ:
        e = self.exponent(x)
        step = torch.exp2(e.to(x.real.dtype)).unsqueeze(-1)
        if self.rounding == "nearest":
            # 四舍五入（0.5 向上），避免 torch.round 的银行家舍入
            i = torch.floor(x.real / step + 0.5)
            q = torch.floor(x.imag / step + 0.5)
        else:
            i = torch.floor(x.real / step)
            q = torch.floor(x.imag / step)
        # 四舍五入可能使最大值进位越界（如 511/2 -> 256），需要饱和
        i = saturate(i, self.iq_width).to(torch.int32)
        q = saturate(q, self.iq_width).to(torch.int32)
        return CompressedIQ(self.method, self.iq_width, self.param_bits, i, q, e)

    def decompress(self, c: CompressedIQ) -> torch.Tensor:
        dtype = torch.float32
        step = torch.exp2(c.param.to(dtype)).unsqueeze(-1)
        return torch.complex(c.i.to(dtype) * step, c.q.to(dtype) * step)
