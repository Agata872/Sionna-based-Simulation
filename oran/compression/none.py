"""不压缩的两种基准。

- ``NoCompression``：O-RAN udCompMeth = 0，定点 IQ。iq_width = 16 时就是 16 bit 原样传输；
  iq_width < 16 时等价于固定指数（16 - iq_width）的截位量化，可用来对比 BFP 自适应指数的好处。
- ``IdealIQ``：理想前传，不做定点化也不量化，仅作仿真参考（比特数按 16 bit 计）。
"""

import torch

from .base import CompressedIQ, IQCompressor, saturate


class NoCompression(IQCompressor):
    """定点 IQ，不压缩（udCompMeth = 0）。"""

    method = "none"
    param_bits = 0

    def __init__(self, iq_width: int = 16, input_width: int = 16):
        super().__init__(iq_width, input_width)
        self.shift = input_width - iq_width

    @property
    def is_lossless(self) -> bool:
        return self.shift == 0

    def compress(self, x: torch.Tensor) -> CompressedIQ:
        step = 2.0 ** self.shift
        i = saturate(torch.floor(x.real / step + 0.5), self.iq_width).to(torch.int32)
        q = saturate(torch.floor(x.imag / step + 0.5), self.iq_width).to(torch.int32)
        return CompressedIQ(self.method, self.iq_width, self.param_bits, i, q, None)

    def decompress(self, c: CompressedIQ) -> torch.Tensor:
        step = 2.0 ** self.shift
        return torch.complex(c.i.float() * step, c.q.float() * step)


class IdealIQ(IQCompressor):
    """理想前传：信号原样通过（不经定点化），比特数按 16 bit 不压缩计算。"""

    method = "ideal"
    param_bits = 0

    def __init__(self, iq_width: int = 16, input_width: int = 16):
        super().__init__(16, 16)

    @property
    def comp_meth(self) -> int:
        return 0

    @property
    def is_lossless(self) -> bool:
        return True

    def compress(self, x: torch.Tensor) -> CompressedIQ:
        # 浮点值直接存放在 i/q 中（不取整），仅为接口一致
        return CompressedIQ(self.method, self.iq_width, self.param_bits, x.real, x.imag, None)

    def decompress(self, c: CompressedIQ) -> torch.Tensor:
        return torch.complex(c.i, c.q)
