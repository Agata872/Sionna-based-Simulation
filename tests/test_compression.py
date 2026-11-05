"""IQ 压缩模块的单元测试（CPU 即可）。

运行：python -m unittest discover -s tests -v
"""

import math
import unittest
from fractions import Fraction

import torch

from oran.compression import (BFPCompressor, BlockScalingCompressor, FixedPoint, ModulationCompressor,
                              MuLawCompressor, available_compressors, decode_scaler, encode_scaler,
                              from_blocks, make_compressor, pack, qam_unit, to_blocks, unpack)


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



def ref_mod_decompress(sample, w, csf, scaler):
    """附录 A.5 解压伪代码的逐行翻译（单个 I 或 Q 分量）。

    1) 读 udIqWidth 位的 iqSample；2) 按 Q1.(udIqWidth−1) 映射到 [−1, 1)；
    3b) csf == 1 时 iqSampleFx += 2^−udIqWidth；3c) iqSampleScaled = scaler × iqSampleFx × √2。
    """
    u = sample & ((1 << w) - 1)                         # udIqWidth 位二进制补码
    v = u - (1 << w) if u >= (1 << (w - 1)) else u
    fx = Fraction(v, 1 << (w - 1))
    if csf:
        fx += Fraction(1, 1 << w)
    return float(scaler) * float(fx) * math.sqrt(2)


class TestModulationCompression(unittest.TestCase):
    FS = 3.976                                          # 满量程（与 16 bit 定点、回退 15 dB 时相同的量级）

    def _qam(self, qm, n=4000, seed=0):
        g = torch.Generator().manual_seed(seed)
        w = qm // 2
        a = 2 * torch.randint(-(1 << (w - 1)), 1 << (w - 1), (2, n), generator=g) + 1
        u = qam_unit(qm)
        return torch.complex(a[0].double() * u, a[1].double() * u), a

    def test_qam_unit_matches_sionna(self):
        from sionna.phy.mapping import Constellation
        for qm in (2, 4, 6, 8, 10):
            pts = Constellation("qam", qm).points.cpu().to(torch.complex128)
            a = pts / qam_unit(qm)
            # 星座点为 u·(a + jb)，a、b 为奇数
            self.assertTrue(torch.allclose(a.real, torch.round((a.real - 1) / 2) * 2 + 1, atol=1e-4), qm)
            self.assertEqual(float(a.real.abs().max().round()), 2 ** (qm // 2) - 1)

    def test_scaler_field(self):
        # §7.7.4.3：value = mantissa × 2^-exponent，mantissa = 低 11 bit / 2^11，exponent = 高 4 bit
        self.assertEqual(int(encode_scaler(torch.tensor(0.5))), 1024)
        self.assertEqual(int(encode_scaler(torch.tensor(0.25))), (1 << 11) | 1024)
        self.assertEqual(int(encode_scaler(torch.tensor(0.0))), 0)
        v = torch.tensor([1 - 2 ** -11, 0.3, 0.1, 1e-3, 2 ** -15])
        rec = decode_scaler(encode_scaler(v))
        self.assertTrue(bool(((rec - v.double()).abs() / v.double() <= 2 ** -11).all()))
        for field in (0, 1, 2047, 1 << 11, (15 << 11) | 1500, 12345):
            bits = [(field >> k) & 1 for k in range(15)]
            mantissa = sum(bits[k] * 2.0 ** (k - 11) for k in range(11))
            exponent = sum(bits[k] * 2 ** (k - 11) for k in range(11, 15))
            self.assertAlmostEqual(float(decode_scaler(torch.tensor(field))), mantissa * 2.0 ** -exponent)
        with self.assertRaises(ValueError):
            encode_scaler(torch.tensor(1.0))

    def test_lossless_for_every_qam(self):
        for qm in (2, 4, 6, 8, 10):
            w = qm // 2
            x, _ = self._qam(qm)
            width = torch.full(x.shape, w)
            unit = torch.full(x.shape, qam_unit(qm), dtype=torch.float64)
            sec = torch.tensor(w)
            i, q, csf, clipped = ModulationCompressor.compress(x, unit, width, sec)
            self.assertEqual(clipped, 0)
            self.assertTrue(bool(csf.all()))
            lim = 1 << (w - 1)
            self.assertTrue(bool(((i >= -lim) & (i < lim) & (q >= -lim) & (q < lim)).all()))
            scaler = decode_scaler(encode_scaler(ModulationCompressor.scaler(unit, width, self.FS)))
            y = self.FS * ModulationCompressor.decompress(i, q, csf, scaler, sec, width > 0)
            err = float(((y - x).abs() / x.abs()).max())
            self.assertLess(err, 2 ** -11, qm)                  # 只剩 15 bit scaler 的量化误差
            self.assertEqual(ModulationCompressor.bits_per_prb(w), 12 * qm)

    def test_mixed_widths_not_shifted(self):
        # 图 A.5-3：16QAM 与 64QAM 同段（udIqWidth 3），只有 64QAM 平移；QPSK（如 DMRS、CSI-RS）放进 256QAM 段同理
        for qm_low, qm_high in ((4, 6), (2, 8), (2, 4)):
            w_sec = qm_high // 2
            x, a = self._qam(qm_low, n=200)
            width = torch.full(x.shape, qm_low // 2)
            unit = torch.full(x.shape, qam_unit(qm_low), dtype=torch.float64)
            i, q, csf, _ = ModulationCompressor.compress(x, unit, width, torch.tensor(w_sec))
            self.assertFalse(bool(csf.any()))
            # 不平移：分数表示 a / 2^{w_c} 直接落在 Q1.(W−1) 网格上
            self.assertTrue(torch.equal(i.long(), a[0] * 2 ** (w_sec - 1 - qm_low // 2)))
            scaler = decode_scaler(encode_scaler(ModulationCompressor.scaler(unit, width, self.FS)))
            y = self.FS * ModulationCompressor.decompress(i, q, csf, scaler, torch.tensor(w_sec), width > 0)
            self.assertLess(float(((y - x).abs() / x.abs()).max()), 2 ** -11)

    def test_decompress_matches_spec_pseudocode(self):
        g = torch.Generator().manual_seed(3)
        for w in (1, 2, 3, 4, 5):
            n = 300
            i = torch.randint(-(1 << (w - 1)), 1 << (w - 1), (n,), generator=g, dtype=torch.int32)
            q = torch.randint(-(1 << (w - 1)), 1 << (w - 1), (n,), generator=g, dtype=torch.int32)
            csf = torch.rand(n, generator=g) < 0.5
            field = torch.randint(0, 1 << 15, (n,), generator=g)
            scaler = decode_scaler(field)
            y = ModulationCompressor.decompress(i, q, csf, scaler, torch.tensor(w), torch.ones(n, dtype=torch.bool))
            for k in range(n):
                self.assertAlmostEqual(float(y[k].real),
                                       ref_mod_decompress(int(i[k]), w, bool(csf[k]), scaler[k]), places=12)
                self.assertAlmostEqual(float(y[k].imag),
                                       ref_mod_decompress(int(q[k]), w, bool(csf[k]), scaler[k]), places=12)

    def test_unpopulated_res_are_zero(self):
        x = torch.tensor([0j, (1 + 1j) / math.sqrt(2)], dtype=torch.complex128)
        width = torch.tensor([0, 1])
        unit = torch.tensor([0.0, 1 / math.sqrt(2)], dtype=torch.float64)
        i, q, csf, _ = ModulationCompressor.compress(x, unit, width, torch.tensor(1))
        scaler = decode_scaler(encode_scaler(ModulationCompressor.scaler(unit, width, self.FS)))
        y = ModulationCompressor.decompress(i, q, csf, scaler, torch.tensor(1), width > 0)
        self.assertEqual(complex(y[0]), 0j)
        self.assertAlmostEqual(abs(complex(y[1]) * self.FS - complex(x[1])), 0.0, places=3)

if __name__ == "__main__":
    unittest.main()
