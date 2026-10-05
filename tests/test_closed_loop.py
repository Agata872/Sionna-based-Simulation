"""闭环（CQI → MCS → OLLA）与射线追踪信道的测试。

运行：python -m unittest discover -s tests -v
射线追踪的冒烟测试需要 Sionna RT（Mitsuba GPU 变体），不可用时跳过。
"""

import unittest

import numpy as np
import sionna.phy
import torch

from oran import ORANConfig, ORANDownlink
from oran.bler_model import AWGNLinkModel, mcs_entry
from oran.csi import CQI_TABLES, CQIReporter, lmmse_sinr, reference_data_symbols, reference_num_re
from oran.link_adaptation import LinkAdaptation, LinkAdaptationConfig

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def small_config(**kwargs):
    base = dict(num_prb=12, device=DEVICE)
    base.update(kwargs)
    return ORANConfig(**base)


def rt_available() -> bool:
    try:
        import mitsuba as mi
        import sionna.rt  # noqa: F401
        return mi.variant() is not None and mi.variant().startswith("cuda")
    except Exception:
        return False


class TestCQIAndLinkAdaptation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sionna.phy.config.device = DEVICE
        cls.model = AWGNLinkModel(0.1, device=DEVICE)

    def test_cqi_tables_are_mcs_entries(self):
        # CQI 表 2 的序号 c >= 2 与 MCS 表 2 的序号 2c-3 是同一 (Qm, R)
        for c in range(2, 16):
            self.assertEqual(tuple(map(float, CQI_TABLES[2][c - 1])), mcs_entry(2, 2 * c - 3))
        # CQI 表 1 除序号 1（78/1024）外都在 MCS 表 1 中
        mcs1 = {mcs_entry(1, m) for m in range(29)}
        for q, r in CQI_TABLES[1][1:]:
            self.assertIn((q, float(r)), mcs1)

    def test_reference_resource(self):
        sim = ORANDownlink(small_config(num_ue=1))
        pc = sim.pdsch_configs[0]
        # DMRS 在符号 2、11：参考资源的数据符号为 3~10、12、13
        self.assertEqual(reference_data_symbols(pc), [3, 4, 5, 6, 7, 8, 9, 10, 12, 13])
        self.assertEqual(reference_num_re(pc, 12, 2), 12 * 12 * 10 * 2)

    def test_cqi_thresholds_increase(self):
        for table in (1, 2):
            rep = CQIReporter(table, 51 * 12 * 10 * 2, self.model)
            self.assertTrue(np.all(np.diff(rep.threshold_db) > 0), table)

    def test_cqi_on_awgn(self):
        # 平坦 SINR：CQI 为门限不超过该 SINR 的最大序号
        rep = CQIReporter(2, 51 * 12 * 10 * 2, self.model)
        for c in (1, 5, 10, 15):
            sinr_db = rep.threshold_db[c - 1] + 0.05
            sinr = torch.full((1, 2, 100), 10 ** (sinr_db / 10))
            self.assertEqual(int(rep(sinr)[0][0]), c)
        self.assertEqual(int(rep(torch.full((1, 2, 100), 1e-3))[0][0]), 0)

    def test_cqi_lookup_and_olla(self):
        num_re = 51 * 12 * 10 * 2
        rep = CQIReporter(2, num_re, self.model)
        la = LinkAdaptation(LinkAdaptationConfig(olla_step_db=0.5), 2, 1, num_re, rep, self.model)
        # 参考资源与实际分配相同、偏移为 0：CQI c 对应同一 (Qm, R) 的 MCS 2c-3
        for c in range(2, 16):
            self.assertEqual(int(la.select_mcs(np.array([c]))[0]), 2 * c - 3)
        # NACK 下调 δ，ACK 上调 δ·p/(1-p)
        la.update(np.array([0]), np.array([2]))
        self.assertAlmostEqual(la.offset_db[0], -1.0)
        la.update(np.array([9]), np.array([0]))
        self.assertAlmostEqual(la.offset_db[0], -0.5)
        self.assertLess(int(la.select_mcs(np.array([10]))[0]), 17)
        # 已用最高 MCS 时 ACK 不再上调
        la.reset()
        la.update(np.array([8]), np.array([0]), mcs=np.array([27]))
        self.assertEqual(la.offset_db[0], 0.0)
        # 可选 MCS 受限时取最低可用
        la = LinkAdaptation(LinkAdaptationConfig(), 2, 1, num_re, rep, self.model, allowed_mcs=range(3, 28))
        self.assertEqual(int(la.select_mcs(np.array([0]))[0]), 3)
        self.assertEqual(int(la.select_mcs(np.array([2]))[0]), 3)


class TestLinkPieces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sionna.phy.config.device = DEVICE

    def test_lmmse_sinr_matches_sionna_equalizer(self):
        sionna.phy.config.seed = 2
        cfg = small_config(channel_estimation="perfect")
        sim = ORANDownlink(cfg)
        state = sim.channel.sample(2)
        h = state.stacked()
        w = sim.odu.precoder(sim.channel.csi(state, sim.csi_symbol))
        no = 0.05
        sinr = lmmse_sinr(h, w, no, sim.odu.prg_of_sc, cfg.num_layers_per_ue)   # [B, K, ν, S, F]
        from oran.precoding import effective_channel
        h_eff = effective_channel(h, w, sim.odu.prg_of_sc).unflatten(3, (cfg.num_ue, cfg.num_layers_per_ue))
        y = torch.zeros(2, cfg.num_ue, cfg.num_ue_ant, 14, cfg.num_subcarriers, dtype=h.dtype, device=h.device)
        _, no_eff = sim.ue.equalizer(y, h_eff, torch.zeros((), device=h.device), torch.tensor(no, device=h.device))
        # 数据 RE 按（符号, 子载波）行优先排列
        data = sim.rg.build_type_grid()[0, 0].flatten() == 0
        ref = (1 / no_eff).flatten(-1)                                    # [B, K, ν, N_data]
        ours = sinr.flatten(-2)[..., data.to(sinr.device)]
        self.assertLess(float(((ours - ref).abs() / ref).max()), 1e-3)

    def test_set_mcs_switches_codewords(self):
        sionna.phy.config.seed = 4
        sim = ORANDownlink(small_config(num_ue=1, mcs_table=2))
        sim.set_mcs([5])
        tb5, enc5 = sim.tb_sizes[0], sim.odu.tb_encoders[0]
        self.assertEqual(sim.run_batch(4, 30.0).block_errors.sum(), 0)
        sim.set_mcs([15])
        self.assertGreater(sim.tb_sizes[0], tb5)
        self.assertEqual(sim.run_batch(4, 30.0).block_errors.sum(), 0)
        sim.set_mcs([5])
        self.assertIs(sim.odu.tb_encoders[0], enc5)         # 编码链被缓存
        self.assertIn(3, sim.supported_mcs())
        self.assertNotIn(0, sim.supported_mcs())             # R < 1/5，Sionna 不支持

    def test_delay_compensation(self):
        # 纯时延信道（τ = 1.5 µs，单径）：补偿后估计误差应远小于不补偿
        tau = 1.5e-6
        nmse = {}
        for comp in (False, True):
            sionna.phy.config.seed = 6
            cfg = small_config(num_ue=1, ce_delay_compensation=comp)
            sim = ORANDownlink(cfg)
            _, _, x = sim.odu.high_phy(1)                       # [1, ν, S, F]
            k = torch.arange(cfg.num_subcarriers, device=x.device)
            ramp = torch.polar(torch.ones(k.shape, device=x.device), -2 * np.pi * 30e3 * tau * k)
            g = torch.Generator().manual_seed(0)
            c = torch.complex(torch.randn(cfg.num_ue_ant, 2, generator=g),
                              torch.randn(cfg.num_ue_ant, 2, generator=g)).to(x.device)
            h = c[:, :, None, None] * ramp                      # [N_r, ν, 1, F]
            y = torch.einsum("rlsf,blsf->brsf", h.expand(-1, -1, 14, -1), x)[:, None]
            h_hat, _ = sim.ue.estimator(y, torch.tensor(1e-4, device=x.device))
            err = (h_hat[0, 0, :, 0] - h.expand(-1, -1, 14, -1)).abs() ** 2
            nmse[comp] = float(err.mean() / (h.abs() ** 2).mean())
            if comp:
                slope = float(sim.ue.estimator.phase_slope(y)[0, 0])
                self.assertAlmostEqual(slope, -2 * np.pi * 30e3 * tau, delta=1e-3)
        self.assertLess(nmse[True], 1e-3)
        self.assertGreater(nmse[False], 100 * nmse[True])


@unittest.skipUnless(rt_available(), "需要 Sionna RT 的 GPU 变体")
class TestRTClosedLoop(unittest.TestCase):
    def test_drop_and_run(self):
        from oran import ClosedLoopDownlink
        sionna.phy.config.device = DEVICE
        cfg = ORANConfig(channel_model="rt", num_ue=1, num_prb=12, device=DEVICE)
        sim = ClosedLoopDownlink(cfg)
        pos = sim.drop(3)
        self.assertEqual(pos.shape, (1, 3))
        self.assertTrue(np.all(sim.scene.is_outdoor(pos)))
        res = sim.run_drop(pos, num_batches=3, batch_size=4, warmup_batches=1)
        s = res.summary(cfg.slots_per_second)
        self.assertTrue(np.isfinite(s["snr_db"][0]))
        self.assertTrue(0 <= s["cqi"][0] <= 15)
        self.assertEqual(s["num_blocks"], 8)
        self.assertGreaterEqual(s["throughput_mbps"][0], 0.0)
        # 同一种子撒点可复现
        np.testing.assert_allclose(sim.drop(3), pos)


if __name__ == "__main__":
    unittest.main()
