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
from oran.bler_model import mcs_entry
from oran.fronthaul import fronthaul_rate, gaussian_loading_db, tr38816_rate
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


class TestSplitsAndCellFree(unittest.TestCase):
    """option 7-1、调制压缩、cell-free（多 O-RU）、按 O-RU 的功率约束与数字增益、UMi 撒点信道。"""

    @classmethod
    def setUpClass(cls):
        sionna.phy.config.device = DEVICE

    def _all_oru_signals(self, cfg, seed=7, batch_size=2):
        sionna.phy.config.seed = seed
        sim = ORANDownlink(cfg)
        sionna.phy.config.seed = seed
        state = sim.channel.sample(batch_size)
        tx = sim.odu(batch_size, sim.channel.csi(state, sim.csi_symbol))
        out = []
        for o in range(cfg.num_oru):
            uplane, w = sim.fronthaul[o].transmit(tx.uplane[o], tx.weights[o], tx.re_map)
            out.append(sim.orus[o].frequency_domain(uplane, w))
        return torch.stack(out, 1), sim, tx

    def test_split71_equals_cat_a_without_compression(self):
        xa, _, _ = self._all_oru_signals(small_config(category="A", uplane=IDEAL))
        x71, _, _ = self._all_oru_signals(small_config(category="7-1", uplane=IDEAL))
        self.assertTrue(torch.equal(xa, x71))

    def test_cell_free_splits_identical_without_compression(self):
        kw = dict(num_oru=2, num_oru_ant=4, num_ue=2, num_ue_ant=2, num_layers_per_ue=1)
        ref, _, _ = self._all_oru_signals(small_config(category="A", uplane=IDEAL, **kw))
        for cat in ("B", "7-1", "8"):
            x, _, _ = self._all_oru_signals(small_config(category=cat, uplane=IDEAL, bfw=IDEAL, **kw))
            self.assertLess(float((x - ref).abs().max()), 1e-5, cat)

    def test_split71_fixed_point_sqnr(self):
        # 频域均匀定点：SQNR 接近高斯分量的最优均匀量化（各天线功率不同，按全链路平均功率定点时略低）
        sionna.phy.config.seed = 7
        sim = ORANDownlink(small_config(category="7-1", uplane=CompressionConfig("fixed", 8)))
        state = sim.channel.sample(4)
        tx = sim.odu(4, sim.channel.csi(state, sim.csi_symbol))
        codec = sim.fronthaul[0].uplane
        codec(tx.uplane[0])
        k = 10 ** (gaussian_loading_db(8) / 20)
        tail = (1 + k * k) * 0.5 * math.erfc(k / math.sqrt(2)) - k * math.exp(-k * k / 2) / math.sqrt(2 * math.pi)
        sqnr = -10 * np.log10((k / 128) ** 2 / 12 + 2 * tail)
        self.assertLess(codec.stats.sqnr_db, sqnr + 0.5)
        self.assertGreater(codec.stats.sqnr_db, sqnr - 3.0)
        self.assertEqual(codec.bits_per_block, 2 * 8 * 12)

    def test_modulation_compression_is_lossless(self):
        from oran.csi_rs import CSIRS
        for mcs in (5, 15, 27):                         # MCS 表 2：QPSK、64QAM、256QAM
            cfg = small_config(category="B", mcs_table=2, mcs_index=mcs, uplane=CompressionConfig("modulation", 16),
                               bfw=IDEAL)
            x, sim, tx = self._all_oru_signals(cfg)
            ideal, _, _ = self._all_oru_signals(small_config(category="B", mcs_table=2, mcs_index=mcs,
                                                             uplane=IDEAL, bfw=IDEAL))
            self.assertLess(float((x - ideal).abs().max() / ideal.abs().max()), 1e-3, mcs)
            st = sim.fronthaul[0].uplane.stats
            self.assertGreater(st.sqnr_db, 60.0)
            self.assertEqual(st.num_clipped, 0)
            # 比特数：数据符号每 PRB 24·Qm/2 bit，DMRS 符号 24 bit
            qm = mcs_entry(2, mcs)[0]
            n_dmrs = len(sim.pdsch_configs[0].dmrs_symbol_indices)
            per_slot = cfg.num_layers * cfg.num_prb * (24 * (qm // 2) * (14 - n_dmrs) + 24 * n_dmrs)
            self.assertEqual(st.num_bits, 2 * per_slot)
            rate = sim.fronthaul_rate()
            self.assertAlmostEqual(rate.uplane_payload, per_slot * cfg.slots_per_second, places=3)
        # 测量时隙：CSI-RS 与数据同段，CSI-IM 不承载，仍然无损
        csirs = CSIRS(cfg, sim.pdsch_configs[0], device=DEVICE)
        state = sim.channel.sample(1)
        tx = sim.odu(1, sim.channel.csi(state, sim.csi_symbol), csi_rs=csirs)
        x_hat, _ = sim.fronthaul[0].transmit(tx.uplane[0], tx.weights[0], tx.re_map)
        self.assertLess(float((x_hat - tx.x_layers).abs().max()), 1e-3)

    def test_modulation_only_for_cat_b(self):
        for cat in ("A", "7-1", "8"):
            with self.assertRaises(ValueError):
                ORANConfig(category=cat, uplane=CompressionConfig("modulation", 16)).validate()

    def test_split71_rate_and_tr38816(self):
        cfg = ORANConfig(category="7-1", num_ue=1, num_oru_ant=32, uplane=CompressionConfig("fixed", 16))
        rate = fronthaul_rate(cfg, num_prb=273)
        self.assertAlmostEqual(rate.uplane_payload, 32 * 3276 * 28000 * 32, places=0)
        self.assertEqual(rate.cplane_total, 0.0)
        self.assertLess(rate.uplane_total / rate.uplane_payload, 1.05)
        # TR 38.816 表 A-2，参数集 #1（100 MHz，32 天线 / 8 层，2×16 bit）
        self.assertAlmostEqual(tr38816_rate("7-1", 32, 100) / 1e9, 113.6, delta=0.15)
        self.assertAlmostEqual(tr38816_rate("7-2", 8, 100) / 1e9, 29.3, delta=0.05)
        self.assertAlmostEqual(tr38816_rate("7-1", 8, 100) / 1e9, 28.4, delta=0.05)
        # 7-1 不论调度传整个载波：与 Cat A 不压缩（16 bit）满负荷时的净荷相同
        cat_a = fronthaul_rate(ORANConfig(category="A", num_ue=1, num_oru_ant=32,
                                          uplane=CompressionConfig("none", 16)), num_prb=273)
        self.assertAlmostEqual(cat_a.uplane_payload, rate.uplane_payload, places=0)

    def test_per_oru_power_constraint(self):
        cfg = small_config(num_oru=2, num_oru_ant=4, num_ue=2, num_ue_ant=2, num_layers_per_ue=1,
                           link_gain_db=[[0.0, -15.0], [-3.0, -20.0]], per_oru_power=True)
        sionna.phy.config.seed = 2
        sim = ORANDownlink(cfg)
        state = sim.channel.sample(3)
        h = sim.channel.csi(state, sim.csi_symbol)
        w = sim.odu.precoder(h)
        p = sim.odu.oru_power(w)
        self.assertTrue(torch.allclose(p.amax(-1), torch.full((3,), 0.5, device=p.device), atol=1e-5))
        # 整体缩放不改变 BD 的零陷
        leak = (h[:, :, 1] @ w[..., :1]).abs().max() + (h[:, :, 0] @ w[..., 1:]).abs().max()
        self.assertLess(float(leak), 1e-4)

    def test_fronthaul_agc_helps_uniform_quantizer(self):
        # 两个 O-RU 增益相差 25 dB：弱 O-RU 的天线功率远低于标称 1/N_t，按标称功率定点时均匀量化浪费分辨率
        kw = dict(category="8", num_oru=2, num_oru_ant=4, num_ue=1, num_ue_ant=2, num_layers_per_ue=1,
                  link_gain_db=[[0.0, -25.0]], uplane=CompressionConfig("fixed", 8))
        sqnr = {}
        for agc in (False, True):
            sionna.phy.config.seed = 4
            sim = ORANDownlink(small_config(fronthaul_agc=agc, **kw))
            sim.run_batch(2, 30.0)
            sqnr[agc] = [fh.uplane.stats.sqnr_db for fh in sim.fronthaul]
        self.assertGreater(min(sqnr[True]), min(sqnr[False]) + 3.0)
        self.assertGreater(min(sqnr[True]), 38.0)

    def test_umi_drop(self):
        cfg = ORANConfig(channel_model="umi", num_prb=12, num_oru=4, num_oru_ant=4, num_ue=2, num_ue_ant=2,
                         num_layers_per_ue=1, per_oru_power=True, device=DEVICE)
        sim = ORANDownlink(cfg)
        ch = sim.channel.model
        np.testing.assert_allclose(ch.oru_positions[:, :2], [[100, 100], [300, 100], [100, 300], [300, 300]])
        pos = ch.drop_ues(2, 5)
        d2d = np.linalg.norm(pos[:, None, :2] - ch.oru_positions[None, :, :2], axis=-1)
        self.assertTrue(np.all(d2d >= cfg.umi.min_dist_2d))
        sim.channel.set_ue_positions(pos, seed=5)
        h1 = ch.h.clone()
        state = sim.channel.sample(3)
        self.assertEqual(tuple(state.h_freq.shape), (3, 2, 2, 4, 4, 14, 144))
        self.assertTrue(torch.equal(state.h_freq[0], state.h_freq[2]))       # 静止 UE
        # 每个 UE 以最早到达的路径为时延零点；绝对到达时间下最远 O-RU 的首径晚到 (d_max − d_min) / c 左右
        self.assertTrue(np.allclose(ch.delay_offset.min(axis=1), 0.0))
        gain = ch.link_gain_db()
        self.assertTrue(np.all((gain < -50) & (gain > -160)))
        # 同一 seed 得到同一信道；不给 seed、位置不变时沿用已有信道
        sim.channel.set_ue_positions(pos)
        self.assertTrue(torch.equal(ch.h, h1))
        sim.channel.set_ue_positions(pos, seed=5)
        self.assertTrue(torch.allclose(ch.h, h1))
        stats = sim.run_batch(3, 0.0, state=state)
        self.assertTrue(np.all(np.isfinite(stats.evm_percent)))


if __name__ == "__main__":
    unittest.main()
