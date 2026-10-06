"""IQ 压缩模块的单元测试（CPU 即可）。

运行：python -m unittest discover -s tests -v
"""

import math
import unittest
from fractions import Fraction

import torch

from oran.compression import (BFPCompressor, BlockScalingCompressor, FixedPoint, MuLawCompressor,
                              available_compressors, from_blocks, make_compressor, pack, to_blocks,
                              unpack)


def _ints(values):
    """一个 12 RE 的块，第 0 个 RE 的 I 取给定值，其余为 0。"""
    x = torch.zeros(len(values), 12, dtype=torch.complex64)
    x[:, 0] = torch.tensor(values, dtype=torch.float32).to(torch.complex64)
    return x


def _sqnr_db(x, x_hat):
    return float(10 * torch.log10((x.abs() ** 2).sum() / ((x - x_hat).abs() ** 2).sum()))


class TestBFP(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        x = torch.complex(torch.randn(64, 14, 612, generator=g), torch.randn(64, 14, 612, generator=g))
        self.x = x / 2 ** 0.5
        self.fp = FixedPoint(nominal_power=1.0, backoff_db=15.0)
        self.x_int, self.clipped = self.fp.quantize(self.x)
        self.blocks = to_blocks(self.x_int, 12)

    def test_exponent_edge_cases(self):
        comp = BFPCompressor(iq_width=9)
        c = comp.compress(_ints([0, 255, 256, -256, -257, 32767, -32768]))
        # 规范 A.1.1：maxValue = max(maxV, |minV| - 1)，-256、-32768 可分别用指数 0、7 表示
        self.assertEqual(c.param.tolist(), [0, 0, 1, 0, 1, 7, 7])
        x_hat = comp.decompress(c).real[:, 0].tolist()
        # 32767/2^7 四舍五入为 256，超出 9 bit 范围，被饱和到 255
        self.assertEqual(x_hat, [0.0, 255.0, 256.0, -256.0, -256.0, 255.0 * 128, -32768.0])
        self.assertTrue(c.i.max() <= 255 and c.i.min() >= -256)

    def test_negative_peak_keeps_precision(self):
        # 块内峰值为 -256 时指数为 0，其余样点（如 3）无损
        comp = BFPCompressor(iq_width=9)
        x = _ints([-256])
        x[0, 1] = 3
        x_hat = comp.decompress(comp.compress(x))
        self.assertTrue(torch.equal(x_hat, x))

    def test_lossless_at_16_bit(self):
        for method in ("bfp", "none"):
            x_hat, _ = make_compressor(method, iq_width=16).roundtrip(self.blocks)
            self.assertTrue(torch.equal(x_hat, self.blocks), method)

    def test_sqnr_about_6db_per_bit(self):
        prev = None
        for w in (12, 11, 10, 9, 8, 7, 6, 5, 4):
            x_hat, _ = make_compressor("bfp", iq_width=w).roundtrip(self.blocks)
            sqnr = _sqnr_db(self.x, self.fp.dequantize(from_blocks(x_hat)))
            if prev is not None:
                self.assertAlmostEqual(prev - sqnr, 6.02, delta=0.5)
            prev = sqnr
        # 9 bit BFP 对高斯信号约 48~49 dB
        x_hat, _ = make_compressor("bfp", iq_width=9).roundtrip(self.blocks)
        self.assertGreater(_sqnr_db(self.x, self.fp.dequantize(from_blocks(x_hat))), 47.0)

    def test_bfp_beats_fixed_shift(self):
        for w in (9, 6):
            bfp, _ = make_compressor("bfp", iq_width=w).roundtrip(self.blocks)
            fix, _ = make_compressor("none", iq_width=w).roundtrip(self.blocks)
            s_bfp = _sqnr_db(self.x, self.fp.dequantize(from_blocks(bfp)))
            s_fix = _sqnr_db(self.x, self.fp.dequantize(from_blocks(fix)))
            self.assertGreater(s_bfp, s_fix + 3.0)

    def test_bit_count_and_pack_roundtrip(self):
        for w in (9, 7, 14):
            comp = make_compressor("bfp", iq_width=w)
            c = comp.compress(self.blocks[:2, :3])
            self.assertEqual(c.bits_per_block, 8 + 24 * w)          # udCompParam + 12 对 I/Q
            data = pack(c)
            self.assertEqual(len(data), c.num_blocks * (1 + 3 * w))
            u = unpack(data, "bfp", w, 8, 12, tuple(c.i.shape[:-1]))
            self.assertTrue(torch.equal(u.i, c.i) and torch.equal(u.q, c.q))
            self.assertTrue(torch.equal(u.param.to(c.param.dtype), c.param))

    def test_registry(self):
        self.assertTrue({"bfp", "none", "ideal"} <= set(available_compressors()))
        with self.assertRaises(ValueError):
            make_compressor("no_such_method")
        self.assertEqual(make_compressor("bfp").comp_meth, 1)
        self.assertEqual(make_compressor("none").comp_meth, 0)

    def test_fixed_point_no_clipping_at_nominal_power(self):
        self.assertEqual(self.clipped, 0)


# ---------------------------------------------------------------------------
# 规范 v22.00 附录 A.2 / A.3 伪代码（与 v08.00 相同）的逐行标量翻译（精确有理数），作为参考实现
# ---------------------------------------------------------------------------

def _mround(v):
    """MATLAB round：0.5 远离零。"""
    v = Fraction(v)
    sign = 1 if v >= 0 else -1
    return sign * math.floor(abs(v) + Fraction(1, 2))


def ref_block_scaling(re, im, iq_width):
    max_v = max(max(re), max(im))
    min_v = min(min(re), min(im))
    max_value = max(max_v, abs(min_v) - 1)
    s = math.ceil(Fraction(max_value, 2 ** 8))
    if s == 0:
        s = 1
    inv = Fraction(2 ** 7, s)
    qs = 2 ** (iq_width - 1)
    comp = lambda v: min(max(_mround(inv * v / 2 ** (16 - iq_width)), -qs), qs - 1)
    ci, cq = [comp(v) for v in re], [comp(v) for v in im]
    decomp = lambda c: _mround(Fraction(s * c) / Fraction(2) ** (iq_width - 9))
    return s, ci, cq, [decomp(c) for c in ci], [decomp(c) for c in cq]


def ref_mu_law(re, im, b):
    max_val = max(max(abs(v) for v in re), max(abs(v) for v in im))
    shift = 0
    for k, limit in enumerate((2 ** 14, 2 ** 13, 2 ** 12, 2 ** 11, 2 ** 10, 2 ** 9, 2 ** 8), 1):
        if max_val < limit:
            shift = k

    def comp(v):
        a = min(abs(v) * 2 ** shift, 2 ** 15 - 1)
        if a <= 2 ** 13:
            c = Fraction(a, 2 ** (15 - b))
        elif a <= 2 ** 14:
            c = Fraction(a, 2 ** (16 - b)) + 2 ** (b - 3)
        else:
            c = Fraction(a, 2 ** (17 - b)) + 2 ** (b - 2)
        c = _mround((1 if v >= 0 else -1) * c)
        return min(c, 2 ** (b - 1) - 1)          # 规范未写：正向取整溢出时饱和

    def decomp(c):
        a = min(abs(c), 2 ** (b - 1) - 1)
        if a <= 2 ** (b - 2):
            d = a * 2 ** (15 - b)
        elif a <= 2 ** (b - 2) + 2 ** (b - 3):
            d = a * 2 ** (16 - b) - 2 ** 13
        else:
            d = a * 2 ** (17 - b) - 2 ** 15
        return Fraction((1 if c >= 0 else -1) * d, 2 ** shift)

    ci, cq = [comp(v) for v in re], [comp(v) for v in im]
    return shift, ci, cq, [decomp(c) for c in ci], [decomp(c) for c in cq]


def _test_blocks(num_random=400, seed=1):
    """随机块（覆盖各种幅度）+ 边界块。返回 [N, 12] 复数（定点取值）。"""
    g = torch.Generator().manual_seed(seed)
    k = torch.randint(0, 16, (num_random, 1), generator=g).float()
    amp = torch.exp2(k)
    re = torch.clamp(torch.round(torch.randn(num_random, 12, generator=g) * amp), -32768, 32767)
    im = torch.clamp(torch.round(torch.randn(num_random, 12, generator=g) * amp), -32768, 32767)
    edges = [[0] * 12, [32767] + [0] * 11, [-32768] + [0] * 11, [-32768, 32767] + [5] * 10,
             [255] + [1] * 11, [256] + [-1] * 11, [-256, 3] + [0] * 10, [2 ** 14] + [0] * 11,
             [2 ** 14 - 1] + [7] * 11, [2 ** 13, -2 ** 13] + [0] * 10, list(range(-6, 6))]
    edge = torch.tensor(edges, dtype=torch.float32)
    return torch.complex(torch.cat([re, edge]), torch.cat([im, edge.flip(-1)]))


class TestSpecReference(unittest.TestCase):
    """与规范伪代码逐比特一致。"""

    def test_block_scaling_matches_spec(self):
        x = _test_blocks()
        for w in (4, 6, 8, 9, 12, 16):
            comp = BlockScalingCompressor(iq_width=w)
            c = comp.compress(x)
            x_hat = comp.decompress(c)
            for n in range(x.shape[0]):
                re, im = x[n].real.int().tolist(), x[n].imag.int().tolist()
                s, ci, cq, di, dq = ref_block_scaling(re, im, w)
                self.assertEqual(int(c.param[n]), s)
                self.assertEqual(c.i[n].tolist(), ci)
                self.assertEqual(c.q[n].tolist(), cq)
                self.assertEqual(x_hat[n].real.tolist(), [float(v) for v in di])
                self.assertEqual(x_hat[n].imag.tolist(), [float(v) for v in dq])

    def test_mu_law_matches_spec(self):
        x = _test_blocks()
        for b in (4, 6, 8, 9, 12, 15):
            comp = MuLawCompressor(iq_width=b)
            c = comp.compress(x)
            x_hat = comp.decompress(c)
            for n in range(x.shape[0]):
                re, im = x[n].real.int().tolist(), x[n].imag.int().tolist()
                shift, ci, cq, di, dq = ref_mu_law(re, im, b)
                self.assertEqual(int(c.param[n]), (b << 4) | shift)
                self.assertEqual(c.i[n].tolist(), ci)
                self.assertEqual(c.q[n].tolist(), cq)
                self.assertEqual(x_hat[n].real.tolist(), [float(v) for v in di])
                self.assertEqual(x_hat[n].imag.tolist(), [float(v) for v in dq])


class TestBlockScalingAndMuLaw(TestBFP):
    """块缩放与 μ 律：位宽-SQNR 关系、比特数、打包、注册。"""

    def _sqnr(self, method, w):
        x_hat, _ = make_compressor(method, iq_width=w).roundtrip(self.blocks)
        return _sqnr_db(self.x, self.fp.dequantize(from_blocks(x_hat)))

    def test_sqnr_about_6db_per_bit(self):
        for method in ("block_scaling", "mu_law"):
            prev = None
            for w in (12, 11, 10, 9, 8, 7, 6, 5):
                sqnr = self._sqnr(method, w)
                if prev is not None:
                    self.assertAlmostEqual(prev - sqnr, 6.02, delta=0.6, msg=f"{method} {w}")
                prev = sqnr
            self.assertGreater(self._sqnr(method, 9), 45.0, method)

    def test_param_ranges(self):
        bs = BlockScalingCompressor(9).compress(self.blocks)
        self.assertTrue(int(bs.param.min()) >= 1 and int(bs.param.max()) <= 128)
        mu = MuLawCompressor(8).compress(self.blocks)
        self.assertTrue(bool(((mu.param >> 4) == 8).all()))
        self.assertTrue(int((mu.param & 0xF).max()) <= 7)

    def test_bit_count_and_pack_roundtrip(self):
        for method, w in (("block_scaling", 9), ("block_scaling", 7), ("mu_law", 8), ("mu_law", 9)):
            comp = make_compressor(method, iq_width=w)
            c = comp.compress(self.blocks[:2, :3])
            self.assertEqual(c.bits_per_block, 8 + 24 * w)
            data = pack(c)
            self.assertEqual(len(data), c.num_blocks * (1 + 3 * w))
            u = unpack(data, method, w, 8, 12, tuple(c.i.shape[:-1]))
            self.assertTrue(torch.equal(u.i, c.i) and torch.equal(u.q, c.q))
            self.assertTrue(torch.equal(u.param.to(c.param.dtype), c.param))
            self.assertTrue(torch.equal(comp.decompress(u), comp.decompress(c)))

    def test_registry(self):
        self.assertEqual(make_compressor("block_scaling").comp_meth, 2)
        self.assertEqual(make_compressor("mu_law").comp_meth, 3)
        with self.assertRaises(ValueError):
            MuLawCompressor(iq_width=16)

    # 父类中只针对 BFP 的测试不在这里重复
    test_exponent_edge_cases = None
    test_negative_peak_keeps_precision = None
    test_lossless_at_16_bit = None
    test_bfp_beats_fixed_shift = None


if __name__ == "__main__":
    unittest.main()
