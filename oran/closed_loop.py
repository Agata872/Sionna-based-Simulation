"""闭环下行仿真：物理环境（Sionna RT）中的 UE 位置 → CSI / CQI → MAC 链路自适应 → PDSCH → CRC → OLLA。

每个撒点（drop）：
    在覆盖区内随机放置 K 个静止 UE，射线追踪得到各 (UE, O-RU) 链路的信道（含链路预算）；
    重复 num_batches 个 CSI-RS 周期，每个周期 = 1 个测量时隙 + batch_size 个数据时隙：
        O-DU 由 CSI（TDD 互易，假设理想）计算 BD 预编码                                      [L1, O-DU]
        测量时隙：预编码 CSI-RS + CSI-IM 与 PDSCH 一起经前传（Cat A/B、压缩）、O-RU、信道到达 UE；
                  UE 估计 CMR / NZP-IMR / CSI-IM，得到 CQI（'cri-RI-CQI'），经 UCI 上报（无误码、无时延）
                  （cqi_source="genie" 时不发测量时隙，直接用真实等效信道算 CQI）            [L1]
        MAC 由 CQI（+ OLLA 偏移）选 MCS，经 DCI 1_1 下发（只体现为参数）                        [MAC]
        数据时隙：PDSCH：O-DU → 前传 → O-RU → 信道 → UE 接收 → CRC                           [L1，原有链路]
        ACK/NACK → OLLA                                                                       [MAC]
前 warmup_batches 个周期（OLLA 收敛期）不计入统计。默认 batch_size = 7，CSI-RS 周期为 8 个时隙（'slots8'）。
测量时隙的 PDSCH 不译码；吞吐量按数据时隙计算，即把测量时隙的 PDSCH 视为与数据时隙相同。

UE 静止、上报无时延时，同一撒点每个周期的信道相同，CQI 只随测量噪声变化，MCS 主要随 OLLA 变化。
无论用哪种 CQI，都同时记录精灵 CQI（不经前传、真实等效信道）作对照。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from sionna.phy import config as sionna_config

from .bler_model import AWGNLinkModel
from .config import CompressionConfig, ORANConfig
from .csi import CQIReporter, lmmse_sinr, reference_data_symbols, reference_num_re
from .csi_rs import CSIRS, CSIRSReceiver
from .link import ORANDownlink
from .link_adaptation import LinkAdaptation, LinkAdaptationConfig


@dataclass
class DropResult:
    """一个撒点的逐周期记录（数组形状 [num_batches, K]，除非另有说明）。"""

    positions: np.ndarray               # [K, 3]
    path_gain_db: np.ndarray            # [K]
    snr_db: np.ndarray                  # [K] 宽带 SNR：每接收天线、各发射天线等功率，不含波束赋形增益
    num_paths: np.ndarray               # [K]（所有 O-RU 之和）
    batch_size: int
    warmup_batches: int
    cqi: List[np.ndarray] = field(default_factory=list)            # 上报并用于选 MCS 的 CQI
    sinr_eff_db: List[np.ndarray] = field(default_factory=list)    # 与之对应的有效 SINR（EESM）
    cqi_genie: List[np.ndarray] = field(default_factory=list)      # 对照：精灵 CQI
    sinr_eff_genie_db: List[np.ndarray] = field(default_factory=list)
    csi_inr_db: List[np.ndarray] = field(default_factory=list)     # CSI-RS：NZP-IMR 测得的干扰噪声比
    olla_offset_db: List[np.ndarray] = field(default_factory=list)  # 选 MCS 时的 OLLA 偏移
    mcs: List[np.ndarray] = field(default_factory=list)
    tb_size: List[np.ndarray] = field(default_factory=list)
    block_errors: List[np.ndarray] = field(default_factory=list)
    evm_err: List[np.ndarray] = field(default_factory=list)
    evm_ref: List[np.ndarray] = field(default_factory=list)

    def summary(self, slots_per_second: int) -> Dict:
        """warmup 之后的统计，以及全部逐周期记录（可直接写入 JSON）。"""
        m = slice(self.warmup_batches, None)
        arr = lambda x: np.asarray(x)
        n_batches = len(arr(self.mcs)[m])
        blocks = n_batches * self.batch_size
        errors = arr(self.block_errors)[m].sum(0)
        delivered = ((self.batch_size - arr(self.block_errors)[m]) * arr(self.tb_size)[m]).sum(0)
        duration = blocks / slots_per_second
        err, ref = arr(self.evm_err)[m].sum(0), np.maximum(arr(self.evm_ref)[m].sum(0), 1e-30)
        db_mean = lambda x: (10 * np.log10(np.mean(10 ** (arr(x)[m] / 10), axis=0))).tolist()
        out = {
            "positions": self.positions.tolist(),
            "path_gain_db": self.path_gain_db.tolist(),
            "snr_db": self.snr_db.tolist(),
            "num_paths": self.num_paths.tolist(),
            "cqi": arr(self.cqi)[m].mean(0).tolist(),
            "sinr_eff_db": db_mean(self.sinr_eff_db),
            "cqi_genie": arr(self.cqi_genie)[m].mean(0).tolist(),
            "sinr_eff_genie_db": db_mean(self.sinr_eff_genie_db),
            "mean_mcs": arr(self.mcs)[m].mean(0).tolist(),
            "final_olla_offset_db": arr(self.olla_offset_db)[-1].tolist(),
            "mean_olla_offset_db": arr(self.olla_offset_db)[m].mean(0).tolist(),
            "num_blocks": blocks,
            "bler": (errors / max(blocks, 1)).tolist(),
            "throughput_mbps": (delivered / duration / 1e6).tolist() if duration > 0 else None,
            "evm_percent": (100 * np.sqrt(err / ref)).tolist(),
            # 数据时隙上实测的均衡后 SINR（发送符号功率 / 均衡误差功率，含噪声、干扰、信道估计误差和前传失真）
            "data_sinr_db": (10 * np.log10(ref / np.maximum(err, 1e-30))).tolist(),
            "history": {name: arr(getattr(self, name)).tolist()
                        for name in ("cqi", "cqi_genie", "mcs", "olla_offset_db", "tb_size",
                                     "block_errors")},
        }
        if self.csi_inr_db and len(self.positions) > 1:             # 单 UE 时没有 NZP-IMR
            out["csi_inr_db"] = db_mean(self.csi_inr_db)
        return out


class ClosedLoopDownlink:
    """射线追踪场景中的闭环（CSI-RS → CQI → MCS → OLLA）下行仿真。

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
        self.csirs, self.csi_receiver = None, None
        if cfg.cqi_source == "csi-rs":
            self.csirs = CSIRS(cfg, pc, device=self.sim.device)
            self.csi_receiver = CSIRSReceiver(self.csirs, cfg.prg_num_prb, self.sim.ue.estimator)

    @property
    def rt_channel(self):
        return self.sim.channel.model

    @property
    def scene(self):
        return self.rt_channel.scene

    def set_compression(self, uplane: CompressionConfig, bfw: Optional[CompressionConfig] = None):
        """更换前传压缩配置（同一撒点可依次比较多种压缩）。"""
        self.sim.set_compression(uplane, bfw)

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
    def measure_cqi_genie(self, state):
        """精灵 CQI：在 CSI 参考资源的数据符号上用真实等效信道（不经前传）计算，返回 (cqi, 有效 SINR dB)。"""
        h = state.stacked()[:1].index_select(4, self.ref_symbols)      # [1, K, N_r, N_t, S_ref, F]
        h_csi = self.sim.channel.csi(state, self.sim.csi_symbol)[:1]
        w = self.sim.odu.precoder(h_csi)
        sinr = lmmse_sinr(h, w, 1.0, self.sim.odu.prg_of_sc, self.cfg.num_layers_per_ue)
        return self.reporter(sinr[0])

    @torch.no_grad()
    def measure_cqi_csirs(self, state):
        """测量时隙：CSI-RS / CSI-IM 经前传发送，UE 估计后得到 (cqi, 有效 SINR dB, NZP-IMR 的 INR dB)。"""
        y = self.sim.measurement_slot(state, 0.0, self.csirs)          # 信道已含链路预算，噪声方差为 1
        sinr, diag = self.csi_receiver.sinr(y)
        cqi, eff = self.reporter(sinr[0])
        inr = 10 * np.log10(diag["inr"][0].cpu().numpy() + 1e-12)
        return cqi, eff, inr

    def run_drop(self, positions: np.ndarray, num_batches: int = 16, batch_size: int = 7,
                 warmup_batches: int = 4) -> DropResult:
        self.sim.channel.set_ue_positions(positions)
        self.sim.reset_stats()
        self.la.reset()
        rt = self.rt_channel
        res = DropResult(np.asarray(positions), rt.path_gain_db(), rt.wideband_snr_db(),
                         rt.num_paths.sum(-1), batch_size, warmup_batches)
        for _ in range(num_batches):
            state = self.sim.channel.sample(batch_size)
            cqi_g, eff_g = self.measure_cqi_genie(state)
            if self.cfg.cqi_source == "csi-rs":
                cqi, eff, inr = self.measure_cqi_csirs(state)
                res.csi_inr_db.append(inr)
            else:
                cqi, eff = cqi_g, eff_g
            offset = self.la.offset_db.copy()
            mcs = self.la.select_mcs(cqi)
            self.sim.set_mcs(mcs)
            stats = self.sim.run_batch(batch_size, 0.0, state=state)   # 信道已含链路预算，噪声方差为 1
            nack = stats.block_errors.astype(int)
            self.la.update(batch_size - nack, nack, mcs)
            res.cqi.append(cqi)
            res.sinr_eff_db.append(eff)
            res.cqi_genie.append(cqi_g)
            res.sinr_eff_genie_db.append(eff_g)
            res.olla_offset_db.append(offset)
            res.mcs.append(mcs)
            res.tb_size.append(np.array(self.sim.tb_sizes))
            res.block_errors.append(nack)
            res.evm_err.append(stats.evm_err.copy())
            res.evm_ref.append(stats.evm_ref.copy())
        return res

    def simulate(self, num_drops: int, num_batches: int = 16, batch_size: int = 7,
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
                    f"UE{k} SNR {s['snr_db'][k]:5.1f} dB CQI {s['cqi'][k]:4.1f}（精灵 {s['cqi_genie'][k]:4.1f}）"
                    f" MCS {s['mean_mcs'][k]:4.1f} Δ {s['final_olla_offset_db'][k]:+5.2f} dB "
                    f"BLER {s['bler'][k]:.2f} {s['throughput_mbps'][k]:6.1f} Mbps"
                    for k in range(self.cfg.num_ue))
                print(f"  drop {i:3d} | {ue}", flush=True)
        return results
