"""链路级功能测试（有 GPU 时在 cuda:0 上运行，小带宽，约 1 分钟）。

运行：python -m unittest discover -s tests -v
"""

import math
import unittest

import numpy as np
import sionna.phy
import torch

from oran import CompressionConfig, ORANConfig, ORANDownlink
from oran.channel_estimation import PRGLinearInterpolator
from oran.fronthaul import fronthaul_rate, gaussian_loading_db
from oran.precoding import compute_precoder

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
IDEAL = CompressionConfig("ideal", 16)


def small_config(**kwargs):
    base = dict(num_prb=12, device=DEVICE)
    base.update(kwargs)
    return ORANConfig(**base)


class TestLink(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sionna.phy.config.device = DEVICE

    def _antenna_signal(self, cfg, seed=7, batch_size=4):
        sionna.phy.config.seed = seed
        sim = ORANDownlink(cfg)
        sionna.phy.config.seed = seed
        state = sim.channel.sample(batch_size)
        tx = sim.odu(batch_size, sim.channel.csi(state, sim.csi_symbol))
        uplane, w = sim.fronthaul[0].transmit(tx.uplane[0], tx.weights[0])
        return sim.orus[0].frequency_domain(uplane, w)

    def test_cat_a_equals_cat_b_without_compression(self):
        xa = self._antenna_signal(small_config(category="A", uplane=IDEAL))
        xb = self._antenna_signal(small_config(category="B", uplane=IDEAL, bfw=IDEAL))
        self.assertLess(float((xa - xb).abs().max()), 1e-5)

    def test_option8_equals_cat_a_without_compression(self):
        xa = self._antenna_signal(small_config(category="A", uplane=IDEAL))
        x8 = self._antenna_signal(small_config(category="8", uplane=IDEAL))
        self.assertTrue(torch.equal(xa, x8))

    def test_option8_fixed_point_sqnr(self):
        # 时域 SQNR 不超过高斯分量最优均匀量化的 SQNR；定点增益按全链路平均功率设定，各天线功率相差约 ±1.5 dB，
        # 所以实际低 1~2 dB。量化噪声均匀落在全部 IFFT 点上，PRB 上的 SQNR 再高 10·log10(IFFT 点数 / 子载波数)
        sionna.phy.config.seed = 7
        sim = ORANDownlink(small_config(category="8", uplane=CompressionConfig("fixed", 8)))
        state = sim.channel.sample(4)
        tx = sim.odu(4, sim.channel.csi(state, sim.csi_symbol))
        codec = sim.fronthaul[0].uplane
        codec(tx.uplane[0])
        k = 10 ** (gaussian_loading_db(8) / 20)
        tail = (1 + k * k) * 0.5 * math.erfc(k / math.sqrt(2)) - k * math.exp(-k * k / 2) / math.sqrt(2 * math.pi)
        sqnr_time = -10 * np.log10((k / 128) ** 2 / 12 + 2 * tail)
        gain = 10 * np.log10(sim.air.fft_size / sim.air.num_sc)
        self.assertLess(codec.time_stats.sqnr_db, sqnr_time + 0.3)
        self.assertGreater(codec.time_stats.sqnr_db, sqnr_time - 3.0)
        self.assertAlmostEqual(codec.stats.sqnr_db - codec.time_stats.sqnr_db, gain, delta=0.3)

    def test_cat_a_and_b_differ_with_compression(self):
        comp = CompressionConfig("bfp", 5)
        xa = self._antenna_signal(small_config(category="A", uplane=comp))
        xb = self._antenna_signal(small_config(category="B", uplane=comp, bfw=comp))
        self.assertGreater(float((xa - xb).abs().max()), 1e-3)

    def test_block_diagonalization(self):
        g = torch.Generator().manual_seed(0)
        h = torch.complex(torch.randn(3, 4, 2, 4, 8, generator=g), torch.randn(3, 4, 2, 4, 8, generator=g))
        w = compute_precoder(h, 2, "bd")
        self.assertTrue(torch.allclose((w.abs() ** 2).sum((-1, -2)), torch.ones(3, 4), atol=1e-5))
        leak = (h[:, :, 1] @ w[..., :2]).abs().max() + (h[:, :, 0] @ w[..., 2:]).abs().max()
        self.assertLess(float(leak), 1e-4)

    def test_prg_interpolator_does_not_cross_prg(self):
        sim = ORANDownlink(small_config(prg_size=2))
        pp = sim.rg.pilot_pattern
        interp = PRGLinearInterpolator(pp, 24, device=DEVICE)
        # 每个 PRG 内信道为常数，各 PRG 取不同值：插值结果应逐 PRG 恢复该常数
        mask = pp.mask.cpu().numpy().astype(bool)
        pos = np.argwhere(mask[0, 0])
        prg_value = torch.arange(1, 7, dtype=torch.float32)
        h_pilot = prg_value[torch.as_tensor(pos[:, 1]) // 24].to(torch.complex64).to(DEVICE)
        h_pilot = h_pilot.expand(1, 1, 1, *mask.shape[:2], -1)
        h_full, _ = interp(h_pilot, torch.zeros_like(h_pilot.real))
        expect = prg_value[torch.arange(144) // 24].to(DEVICE)
        self.assertTrue(torch.allclose(h_full[0, 0, 0, 0, 0].real, expect.expand(14, -1)))

    def test_high_snr_decodes(self):
        for cat in ("A", "B"):
            sionna.phy.config.seed = 1
            sim = ORANDownlink(small_config(category=cat))
            stats = sim.run_batch(8, 30.0)
            self.assertEqual(stats.block_errors.sum(), 0, cat)
            self.assertLess(stats.evm_percent.max(), 10.0)

    def test_all_compression_methods_decode(self):
        for method, w in (("block_scaling", 9), ("mu_law", 8)):
            comp = CompressionConfig(method, w)
            for cat in ("A", "B"):
                sionna.phy.config.seed = 1
                sim = ORANDownlink(small_config(category=cat, uplane=comp, bfw=comp))
                stats = sim.run_batch(4, 30.0)
                self.assertEqual(stats.block_errors.sum(), 0, f"{method} Cat {cat}")
                self.assertGreater(sim.fronthaul_stats()["uplane"].sqnr_db, 35.0)

    def test_time_domain_matches_frequency_domain(self):
        evm = {}
        for domain in ("freq", "time"):
            sionna.phy.config.seed = 3
            sim = ORANDownlink(small_config(category="B", domain=domain, uplane=IDEAL, bfw=IDEAL))
            evm[domain] = sim.run_batch(4, 25.0).evm_percent.mean()
        self.assertLess(abs(evm["freq"] - evm["time"]), 0.25 * evm["freq"])

    def test_fronthaul_rate_matches_book(self):
        # 书中：100 MHz、30 kHz、每层 3276 子载波 × 14 符号 × 2000 时隙/s × 32 bit = 2.93 Gbps
        cfg = ORANConfig(category="B", num_ue=1, num_layers_per_ue=1, uplane=IDEAL)
        rate = fronthaul_rate(cfg, num_prb=273)
        self.assertAlmostEqual(rate.uplane_payload / 1e9, 3276 * 14 * 2000 * 32 / 1e9, places=6)
        # 协议开销约 10%
        self.assertLess(rate.uplane_total / rate.uplane_payload, 1.15)
        # 9 bit BFP 的净荷为 16 bit 的 (8 + 24·9) / 384
        cfg.uplane = CompressionConfig("bfp", 9)
        bfp = fronthaul_rate(cfg, num_prb=273)
        self.assertAlmostEqual(bfp.uplane_payload / rate.uplane_payload, 224 / 384, places=6)

    def test_option8_rate_matches_tr38801(self):
        # TR 38.801 表 11.1.4.2-1：20 MHz、2 天线、位宽 30（2×15）为 2 × 30.72 Msps × 30 bit
        cfg = ORANConfig(category="8", num_ue=1, num_oru_ant=2, uplane=CompressionConfig("fixed", 15))
        self.assertAlmostEqual(fronthaul_rate(cfg).uplane_payload / 1e9, 2 * 30.72 * 30 / 1e3, places=6)
        # 附录 A 表 A-1：100 MHz、32 天线、2×16 bit 为 157.3 Gbps，按 LTE 的 153.6 Msps 计算；
        # NR 100 MHz（273 PRB，4096 点）的采样率为 122.88 Msps，恰为其 4/5
        cfg = ORANConfig(category="8", num_ue=1, num_oru_ant=32, uplane=CompressionConfig("fixed", 16))
        rate = fronthaul_rate(cfg, num_prb=273)
        self.assertAlmostEqual(rate.uplane_payload * 1.25 / 1e9, 157.3, delta=0.05)
        self.assertEqual(rate.cplane_total, 0.0)
        self.assertLess(rate.uplane_total / rate.uplane_payload, 1.05)


if __name__ == "__main__":
    unittest.main()
