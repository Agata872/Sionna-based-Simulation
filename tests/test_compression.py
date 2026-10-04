"""IQ 压缩模块的单元测试（CPU 即可）。

运行：python -m unittest discover -s tests -v
"""

import unittest

import torch

from oran.compression import (BFPCompressor, FixedPoint, available_compressors, from_blocks,
                              make_compressor, pack, to_blocks, unpack)


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
        c = comp.compress(_ints([0, 255, 256, -256, 32767, -32768]))
        self.assertEqual(c.param.tolist(), [0, 0, 1, 1, 7, 8])
        x_hat = comp.decompress(c).real[:, 0].tolist()
        # 32767/2^7 四舍五入为 256，超出 9 bit 范围，被饱和到 255
        self.assertEqual(x_hat, [0.0, 255.0, 256.0, -256.0, 255.0 * 128, -32768.0])
        self.assertTrue(c.i.max() <= 255 and c.i.min() >= -256)

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


if __name__ == "__main__":
    unittest.main()
