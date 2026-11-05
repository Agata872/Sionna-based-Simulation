"""调制压缩（O-RAN WG4 CUS v22.00 附录 A.5，udCompMeth = 4，只用于下行）。

原理：未经预编码的层符号是 QAM 星座点，每个 I、Q 分量只有 2^w 种取值（w = Qm/2），可以无损地用 w bit 表示。
所以它只适用于 Cat B（O-RU 预编码，前传传层符号）；Cat A、7-1、option 8 前传的是预编码后的叠加信号，不在星座网格上。

与 BFP 等按块压缩的方法不同，调制压缩要知道每个 RE 的星座类型，所以不是 :class:`IQCompressor` 的子类：
O-DU 按 RE 的用途（数据、DMRS、CSI-RS、空 RE）给出星座位宽与幅度单位，压缩参数经 C-plane 下发
（SE 4：一组 csf + modCompScaler；SE 5：多组 mcScaleReMask + csf + mcScaleOffset），U-plane 每个 PRB 只有
12 对 udIqWidth 位的 I/Q，没有 udCompParam（表 8.3.3.15-1）。

记号（每个 RE）：
    x = u·(a + j·b)，a、b ∈ {±1, ±3, …, ±(2^{w_c} − 1)}，u 为该星座的幅度单位（如 256QAM 为 1/√170），
    w_c 为该星座所需的位宽（QPSK 1、16QAM 2、64QAM 3、256QAM 4、1024QAM 5，即 Qm/2）；
    w_c = 0 表示该 RE 不承载数据：没有被任何 mcScaleReMask 指到（§7.7.5.2），O-RU 输出 0，但在 U-plane 中仍占位。
    一个段（section）的 udIqWidth 记为 W，取段内最大的 w_c（附录 A.5："the larger number needed to represent the
    largest constellation possible in the compression block"）。

压缩（规范只有文字描述）：先写成分数 a / 2^{w_c}；w_c = W 的星座平移（csf = 1，减去表 7.7.4.2-1 的 2^{-W}），
    w_c < W 的星座不平移（csf = 0，其点已落在 W 位网格上，如混合 MCS 时低阶星座、或 DMRS 与高阶数据同段）；
    再按 Q1.(W−1) 表示为 W 位二进制补码整数。
解压（附录 A.5 伪代码）：iqSampleFx = 整数 / 2^{W−1}；csf = 1 时加 2^{-W}；
    iqSampleScaled = scaler × iqSampleFx × √2，|iqSampleScaled| = 1 对应 0 dBFS。
    若 O-RU 的满量程为每个 I/Q 分量 FS，则 scaler = u·2^{w_c} / (√2·FS)。scaler 量化为 15 bit：4 bit 指数 +
    11 bit 尾数（§7.7.4.3），这是唯一的误差来源，相对误差不超过 2^-11（规范 A.5 的 NOTE）。
"""

import math
from typing import Tuple

import torch

SCALER_MANTISSA_BITS = 11          # modCompScaler / mcScaleOffset：11 bit 尾数
SCALER_EXPONENT_MAX = 15           # 4 bit 指数
SCALER_MAX = 1.0 - 2.0 ** -SCALER_MANTISSA_BITS
MAX_UD_IQ_WIDTH = 5                # 表 7.7.4.2-1 定义到 udIqWidth = 5（1024QAM）


def qam_width(qm: int) -> int:
    """QAM（每符号 qm bit）每个 I/Q 分量所需的位宽 w_c = qm / 2（表 7.7.4.2-1）。"""
    if qm % 2 or not 2 <= qm <= 2 * MAX_UD_IQ_WIDTH:
        raise ValueError(f"调制压缩只支持 QPSK 到 1024QAM（qm = 2, 4, ..., 10），当前 qm = {qm}")
    return qm // 2


def qam_unit(qm: int) -> float:
    """单位平均功率的方形 QAM（TS 38.211 §5.1）的幅度单位 u：星座点为 u·(a + jb)，a、b 为奇数。"""
    m = 2 ** qm
    return math.sqrt(3.0 / (2.0 * (m - 1)))


def encode_scaler(value: torch.Tensor) -> torch.Tensor:
    """modCompScaler / mcScaleOffset 的 15 bit 字段（§7.7.4.3）：value = mantissa × 2^-exponent。

    exponent 为字段高 4 bit（0~15），mantissa 为低 11 bit 表示的 [0, 1 − 2^-11] 内的纯小数。取使
    value·2^exponent ∈ [0.5, 1) 的指数（尾数精度最高），尾数四舍五入；舍入进位到 1 时改用小一档的指数
    （尾数 0.5），指数已为 0 时饱和到 1 − 2^-11。value = 0 编码为全 0 尾数。
    """
    v = value.double()
    if bool(((v < 0) | (v > SCALER_MAX)).any()):
        raise ValueError(f"scaler 必须在 [0, {SCALER_MAX}] 内")
    full = 2 ** SCALER_MANTISSA_BITS
    pos = v > 0
    safe = torch.where(pos, v, torch.ones_like(v))
    e = torch.clamp(torch.ceil(-torch.log2(safe)) - 1, 0, SCALER_EXPONENT_MAX)
    m = torch.round(safe * torch.pow(2.0, e + SCALER_MANTISSA_BITS))
    carry = m >= full
    m = torch.where(carry, torch.where(e > 0, torch.full_like(m, full // 2), torch.full_like(m, full - 1)), m)
    e = torch.where(carry & (e > 0), e - 1, e)
    m = torch.where(pos, m, torch.zeros_like(m))
    return (e.long() << SCALER_MANTISSA_BITS) | m.long()


def decode_scaler(field: torch.Tensor) -> torch.Tensor:
    """15 bit 字段 -> scaler（§7.7.4.3）：mantissa = Σ_{k=0}^{10} b_k·2^{k−11}，exponent = Σ_{k=11}^{14} b_k·2^{k−11}。"""
    field = field.long()
    mantissa = (field & (2 ** SCALER_MANTISSA_BITS - 1)).double() / 2 ** SCALER_MANTISSA_BITS
    exponent = (field >> SCALER_MANTISSA_BITS) & SCALER_EXPONENT_MAX
    return mantissa * torch.pow(2.0, -exponent.double())


def _nearest_odd(v: torch.Tensor) -> torch.Tensor:
    """最近的奇数（星座点本身是奇数，取整只为吸收浮点误差）。"""
    return 2 * torch.floor(v / 2) + 1


class ModulationCompressor:
    """调制压缩的逐 RE 编解码（无状态）。各参数按广播规则逐元素对应。

    - ``width``：每个 RE 的星座位宽 w_c（0 表示不承载）；
    - ``ud_iq_width``：所在段的 udIqWidth W（不小于段内所有 w_c）；
    - ``unit``：星座的幅度单位 u；
    - ``full_scale``：O-RU 每个 I/Q 分量的满量程 FS（与信号同单位）。
    """

    method = "modulation"
    comp_meth = 4               # udCompMeth（表 8.3.3.13-3）
    param_bits = 0              # U-plane 中没有 udCompParam（表 8.3.3.15-1）

    @staticmethod
    def bits_per_prb(ud_iq_width) -> int:
        """一个 PRB 的 U-plane 比特数：12 对 I/Q，各 udIqWidth 位（24·W 总是整字节）。"""
        return 24 * int(ud_iq_width)

    @staticmethod
    def csf(width: torch.Tensor, ud_iq_width: torch.Tensor) -> torch.Tensor:
        """星座平移标志（§7.7.4.2）：只有位宽等于段位宽的星座平移。"""
        return (width > 0) & (width == ud_iq_width)

    @staticmethod
    def scaler(unit: torch.Tensor, width: torch.Tensor, full_scale: float) -> torch.Tensor:
        """O-DU 下发的 scaler（量化前）：u·2^{w_c} / (√2·FS)；不承载的 RE 为 0。"""
        s = unit.double() * torch.pow(2.0, width.double()) / (math.sqrt(2.0) * full_scale)
        return torch.where(width > 0, s, torch.zeros_like(s))

    @classmethod
    def compress(cls, x: torch.Tensor, unit: torch.Tensor, width: torch.Tensor,
                 ud_iq_width: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """O-DU 侧压缩。返回 (I, Q, csf, 超出星座范围而被饱和的分量数)。

        I、Q 为 udIqWidth 位二进制补码的整数取值（int32），不承载的 RE 为 0。
        """
        width = width.to(torch.int64)
        w_sec = ud_iq_width.to(torch.int64)
        if bool((width > w_sec).any()) or bool((w_sec > MAX_UD_IQ_WIDTH).any()):
            raise ValueError("RE 的星座位宽不能超过段的 udIqWidth，udIqWidth 最大为 5")
        populated = width > 0
        u = torch.where(populated, unit.to(x.real.dtype), torch.ones_like(unit, dtype=x.real.dtype))
        lim = (torch.pow(2.0, width.to(x.real.dtype)) - 1).clamp(min=1)
        csf = cls.csf(width, w_sec)
        # 不平移时乘的 2^{W−1−w_c}；平移时为 (a − 1)/2
        up = torch.pow(2.0, (w_sec - 1 - width).clamp(min=0).to(x.real.dtype))
        clipped = 0
        out = []
        for comp in (x.real, x.imag):
            a = _nearest_odd(comp / u)
            clipped += int(((a.abs() > lim) & populated).sum())
            a = torch.maximum(torch.minimum(a, lim), -lim)
            v = torch.where(csf, (a - 1) / 2, a * up)
            out.append(torch.where(populated, v, torch.zeros_like(v)).to(torch.int32))
        return out[0], out[1], csf, clipped

    @staticmethod
    def decompress(i: torch.Tensor, q: torch.Tensor, csf: torch.Tensor, scaler: torch.Tensor,
                   ud_iq_width: torch.Tensor, populated: torch.Tensor) -> torch.Tensor:
        """O-RU 侧解压，逐行对应附录 A.5 的伪代码，返回 iqSampleScaled（满量程为 1）。

        1) 读 udIqWidth 位的 iqSample；2) 按 Q1.(udIqWidth−1) 映射到 [−1, 1)；
        3b) csf = 1 时加 2^{−udIqWidth}；3c) iqSampleScaled = scaler × iqSampleFx × √2。
        未被任何 mcScaleReMask 指到的 RE 视为不承载数据（§7.7.5.2），输出 0。
        """
        w = ud_iq_width.to(torch.float64)
        shift = torch.pow(2.0, -w)
        res = []
        for v in (i, q):
            fx = v.to(torch.float64) / torch.pow(2.0, w - 1)
            fx = torch.where(csf, fx + shift, fx)
            res.append(scaler.to(torch.float64) * fx * math.sqrt(2.0))
        y = torch.complex(res[0], res[1])
        return torch.where(populated, y, torch.zeros_like(y))
