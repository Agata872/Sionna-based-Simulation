"""闭环下行仿真：物理环境（Sionna RT）中的 UE 位置 → CSI / CQI → MAC 链路自适应 → PDSCH → CRC → OLLA。

每个撒点（drop）：
    在覆盖区内随机放置 K 个静止 UE，射线追踪得到各 (UE, O-RU) 链路的信道（含链路预算）；
    重复 num_batches 个批次，每批 batch_size 个时隙：
        O-DU 由 CSI（TDD 互易，假设理想）计算 BD 预编码                                    [L1, O-DU]
        UE 在 CSI 参考资源上测量等效信道，得到 CQI（'cri-RI-CQI'），经 UCI 上报（无误码、无时延）[L1, UE]
        MAC 由 CQI（+ OLLA 偏移）选 MCS，经 DCI 1_1 下发（只体现为参数）                      [MAC]
        PDSCH：O-DU → 前传（Cat A/B、压缩）→ O-RU → 信道 → UE 接收 → CRC                    [L1，原有链路]
        ACK/NACK → OLLA                                                                     [MAC]
前 warmup_batches 个批次（OLLA 收敛期）不计入统计。

UE 静止、上报无时延时，同一撒点每个批次的 CQI 相同，MCS 只随 OLLA 变化。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from sionna.phy import config as sionna_config

from .bler_model import AWGNLinkModel
from .config import ORANConfig
from .csi import CQIReporter, lmmse_sinr, reference_data_symbols, reference_num_re
from .link import ORANDownlink
from .link_adaptation import LinkAdaptation, LinkAdaptationConfig


@dataclass
class DropResult:
    """一个撒点的逐批次记录（数组形状 [num_batches, K]，除非另有说明）。"""

    positions: np.ndarray               # [K, 3]
    path_gain_db: np.ndarray            # [K]
    snr_db: np.ndarray                  # [K] 宽带 SNR：每接收天线、各发射天线等功率，不含波束赋形增益
    num_paths: np.ndarray               # [K]（所有 O-RU 之和）
    batch_size: int
    warmup_batches: int
    cqi: List[np.ndarray] = field(default_factory=list)
    sinr_eff_db: List[np.ndarray] = field(default_factory=list)   # UE 测得的有效 SINR（EESM）
    olla_offset_db: List[np.ndarray] = field(default_factory=list)  # 选 MCS 时的 OLLA 偏移
    mcs: List[np.ndarray] = field(default_factory=list)
    tb_size: List[np.ndarray] = field(default_factory=list)
    block_errors: List[np.ndarray] = field(default_factory=list)
    evm_err: List[np.ndarray] = field(default_factory=list)
    evm_ref: List[np.ndarray] = field(default_factory=list)

    def summary(self, slots_per_second: int) -> Dict:
        """warmup 之后的统计，以及全部逐批次记录（可直接写入 JSON）。"""
        m = slice(self.warmup_batches, None)
        arr = lambda x: np.asarray(x)
        n_batches = len(arr(self.mcs)[m])
        blocks = n_batches * self.batch_size
        errors = arr(self.block_errors)[m].sum(0)
        delivered = ((self.batch_size - arr(self.block_errors)[m]) * arr(self.tb_size)[m]).sum(0)
        duration = blocks / slots_per_second
        evm = 100 * np.sqrt(arr(self.evm_err)[m].sum(0) / np.maximum(arr(self.evm_ref)[m].sum(0), 1e-30))
        return {
            "positions": self.positions.tolist(),
            "path_gain_db": self.path_gain_db.tolist(),
            "snr_db": self.snr_db.tolist(),
            "num_paths": self.num_paths.tolist(),
            "cqi": arr(self.cqi)[-1].tolist(),
            "sinr_eff_db": arr(self.sinr_eff_db)[-1].tolist(),
            "mean_mcs": arr(self.mcs)[m].mean(0).tolist(),
            "final_olla_offset_db": arr(self.olla_offset_db)[-1].tolist(),
            "num_blocks": blocks,
            "bler": (errors / max(blocks, 1)).tolist(),
            "throughput_mbps": (delivered / duration / 1e6).tolist() if duration > 0 else None,
            "evm_percent": evm.tolist(),
            "history": {name: arr(getattr(self, name)).tolist()
                        for name in ("cqi", "mcs", "olla_offset_db", "tb_size", "block_errors")},
        }


class ClosedLoopDownlink:
    """射线追踪场景中的闭环（CQI → MCS → OLLA）下行仿真。

    >>> cfg = ORANConfig(channel_model="rt", num_ue=1, mcs_table=2)
    >>> sim = ClosedLoopDownlink(cfg)
    >>> results = sim.simulate(num_drops=10, seed=1)
    """

    def __init__(self, cfg: ORANConfig, la_cfg: Optional[LinkAdaptationConfig] = None,
                 precision: Optional[str] = None):
        if cfg.channel_model != "rt":
            raise ValueError("闭环仿真需要射线追踪信道（channel_model='rt'）")
        self.cfg = cfg
        self.la_cfg = la_cfg or LinkAdaptationConfig()
        self.sim = ORANDownlink(cfg, precision)
        pc = self.sim.pdsch_configs[0]
        nu = cfg.num_layers_per_ue
        self.model = AWGNLinkModel(self.la_cfg.bler_target, device=self.sim.device)
        cqi_table = self.la_cfg.cqi_table or cfg.mcs_table
        self.reporter = CQIReporter(cqi_table, reference_num_re(pc, cfg.num_prb, nu), self.model)
        num_re_alloc = self.sim.rg.num_data_symbols * nu
        self.la = LinkAdaptation(self.la_cfg, cfg.mcs_table, cfg.num_ue, int(num_re_alloc),
                                 self.reporter, self.model, self.sim.supported_mcs())
        self.ref_symbols = torch.tensor(reference_data_symbols(pc), device=self.sim.device)

    @property
    def rt_channel(self):
        return self.sim.channel.model

    @property
    def scene(self):
        return self.rt_channel.scene

    def drop(self, seed: int) -> np.ndarray:
        """随机撒点并放置 UE，返回 K 个 UE 的位置 [K, 3]。

        覆盖图与 PathSolver 都是蒙特卡洛射线追踪，个别位置在覆盖图中达到门限、PathSolver 却找不到
        足够的路径；射线追踪后路径增益比门限低 10 dB 以上时换一个种子重撒。
        """
        threshold = self.cfg.rt.drop_min_path_gain_db - 10
        for attempt in range(20):
            positions = self.scene.drop_ues(self.cfg.num_ue, seed + 1_000_003 * attempt)
            self.sim.channel.set_ue_positions(positions)
            if np.all(self.rt_channel.path_gain_db() >= threshold):
                return positions
        raise RuntimeError("撒点多次失败：射线追踪得到的路径增益都低于门限")

    @torch.no_grad()
    def measure_cqi(self, state):
        """UE 侧：在 CSI 参考资源的数据符号上测量等效信道（第一个时隙），返回 (cqi, 有效 SINR dB)。"""
        h = state.stacked()[:1].index_select(4, self.ref_symbols)      # [1, K, N_r, N_t, S_ref, F]
        h_csi = self.sim.channel.csi(state, self.sim.csi_symbol)[:1]
        w = self.sim.odu.precoder(h_csi)
        sinr = lmmse_sinr(h, w, 1.0, self.sim.odu.prg_of_sc, self.cfg.num_layers_per_ue)
        return self.reporter(sinr[0])

    def run_drop(self, positions: np.ndarray, num_batches: int = 16, batch_size: int = 8,
                 warmup_batches: int = 4) -> DropResult:
        self.sim.channel.set_ue_positions(positions)
        self.sim.reset_stats()
        self.la.reset()
        rt = self.rt_channel
        res = DropResult(np.asarray(positions), rt.path_gain_db(), rt.wideband_snr_db(),
                         rt.num_paths.sum(-1), batch_size, warmup_batches)
        for _ in range(num_batches):
            state = self.sim.channel.sample(batch_size)
            cqi, sinr_eff = self.measure_cqi(state)
            offset = self.la.offset_db.copy()
            mcs = self.la.select_mcs(cqi)
            self.sim.set_mcs(mcs)
            stats = self.sim.run_batch(batch_size, 0.0, state=state)   # 信道已含链路预算，噪声方差为 1
            nack = stats.block_errors.astype(int)
            self.la.update(batch_size - nack, nack, mcs)
            res.cqi.append(cqi)
            res.sinr_eff_db.append(sinr_eff)
            res.olla_offset_db.append(offset)
            res.mcs.append(mcs)
            res.tb_size.append(np.array(self.sim.tb_sizes))
            res.block_errors.append(nack)
            res.evm_err.append(stats.evm_err.copy())
            res.evm_ref.append(stats.evm_ref.copy())
        return res

    def simulate(self, num_drops: int, num_batches: int = 16, batch_size: int = 8,
                 warmup_batches: int = 4, seed: int = 0, verbose: bool = True) -> List[DropResult]:
        """依次仿真 num_drops 个撒点。撒点 i 的 UE 位置和比特、噪声由 seed + i 决定。"""
        results = []
        for i in range(num_drops):
            positions = self.drop(seed + i)
            sionna_config.seed = seed + i
            res = self.run_drop(positions, num_batches, batch_size, warmup_batches)
            results.append(res)
            if verbose:
                s = res.summary(self.cfg.slots_per_second)
                ue = " | ".join(
                    f"UE{k} SNR {s['snr_db'][k]:5.1f} dB CQI {s['cqi'][k]:2d} "
                    f"MCS {s['mean_mcs'][k]:4.1f} Δ {s['final_olla_offset_db'][k]:+5.2f} dB "
                    f"BLER {s['bler'][k]:.2f} {s['throughput_mbps'][k]:6.1f} Mbps"
                    for k in range(self.cfg.num_ue))
                print(f"  drop {i:3d} | {ue}", flush=True)
        return results
