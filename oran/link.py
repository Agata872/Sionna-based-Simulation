"""端到端下行链路：O-DU -> 前传 -> O-RU -> 信道 -> UE，以及 BLER / EVM / 前传指标统计。

SNR 定义：每个 RE 的总发射功率为 1（所有 O-RU 所有天线之和），每对收发天线的平均信道增益为 1，
SNR = 1 / N0，N0 为每根接收天线每个 RE 的噪声方差。
"""

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from sionna.phy import config as sionna_config

from .channel import ChannelState, DownlinkChannel
from .config import CompressionConfig, ORANConfig
from .fronthaul import CodecStats, FronthaulLink, fronthaul_rate
from .odu import ODU
from .oru import ORU, make_air_numerology
from .pdsch import make_pdsch_configs, make_resource_grid
from .precoding import apply_precoder, effective_channel
from .ue import UEReceivers


@dataclass
class LinkStats:
    """一个 SNR 点上跨批次累加的统计量。"""

    num_ue: int
    blocks: np.ndarray = None
    block_errors: np.ndarray = None
    bits: np.ndarray = None
    bit_errors: np.ndarray = None
    evm_err: np.ndarray = None        # 均衡后符号误差能量（每 UE）
    evm_ref: np.ndarray = None
    tx_err: float = 0.0               # O-RU 天线口误差能量（相对理想前传）
    tx_ref: float = 0.0

    def __post_init__(self):
        z = lambda: np.zeros(self.num_ue)
        self.blocks, self.block_errors, self.bits, self.bit_errors = z(), z(), z(), z()
        self.evm_err, self.evm_ref = z(), z()

    @property
    def bler(self) -> np.ndarray:
        return self.block_errors / np.maximum(self.blocks, 1)

    @property
    def ber(self) -> np.ndarray:
        return self.bit_errors / np.maximum(self.bits, 1)

    @property
    def evm_percent(self) -> np.ndarray:
        """接收端 EVM（均衡后，数据 RE，含信道估计误差、干扰、噪声和压缩失真）。"""
        return 100 * np.sqrt(self.evm_err / np.maximum(self.evm_ref, 1e-30))

    @property
    def tx_evm_percent(self) -> float:
        """发射端 EVM：O-RU 天线口信号相对理想前传（不压缩、权值不量化）的误差。"""
        return 100 * np.sqrt(self.tx_err / self.tx_ref) if self.tx_ref > 0 else 0.0


class ORANDownlink:
    """O-RAN 7.2x 下行链路级仿真器。

    >>> sim = ORANDownlink(ORANConfig(category="B"))
    >>> res = sim.simulate([0, 5, 10], batch_size=16, max_batches=20)
    """

    def __init__(self, cfg: ORANConfig, precision: Optional[str] = None):
        cfg.validate()
        self.cfg = cfg
        device = cfg.device or sionna_config.device
        self.device = device
        self.pdsch_configs = make_pdsch_configs(cfg)
        self.rg = make_resource_grid(cfg, self.pdsch_configs, precision, device)
        self.odu = ODU(cfg, self.pdsch_configs, self.rg, precision, device)
        self.precision = precision
        self.air = None
        if cfg.domain == "time" or cfg.category == "8":     # option 8 的 O-DU 也要做 IFFT + CP
            self.air = make_air_numerology(cfg, self.pdsch_configs[0], precision, device)
        self.fronthaul = self._make_fronthaul()
        self.orus = [ORU(cfg, self.air, o, precision=precision, device=device)
                     for o in range(cfg.num_oru)]
        self.channel = DownlinkChannel(cfg, self.rg, self.air, precision, device)
        l_min = getattr(self.channel, "l_min", None)
        self.ue = UEReceivers(cfg, self.pdsch_configs, self.rg, self.odu, self.air, l_min,
                              precision=precision, device=device)
        # CSI 参考符号：第一个 DMRS 符号
        self.csi_symbol = self.pdsch_configs[0].dmrs_symbol_indices[0]
        self.mcs = [cfg.mcs_of(k) for k in range(cfg.num_ue)]

    def _make_fronthaul(self) -> List[FronthaulLink]:
        return [FronthaulLink(self.cfg, o, self.air, self.precision, self.device)
                for o in range(self.cfg.num_oru)]

    @property
    def tb_sizes(self) -> List[int]:
        return self.odu.tb_sizes

    def supported_mcs(self) -> List[int]:
        """Sionna 编码链能实现的 MCS（当前 MCS 表、当前资源分配）。

        Sionna 的 LDPC 编码器不支持重复编码：码率低于 1/5、或选到 BG1 而码率低于 1/3 的 MCS 无法实现
        （3GPP 在循环缓冲速率匹配中重复母码）。逐个尝试建立 UE 0 的编码链，成功的同时被缓存。
        """
        from .bler_model import MCS_TABLE_SIZES
        supported = []
        for m in range(MCS_TABLE_SIZES[self.cfg.mcs_table]):
            pc = make_pdsch_configs(replace(self.cfg, mcs_index=m))[0]
            key = (0, self.cfg.mcs_table, m)
            try:
                if key not in self.odu._codewords:
                    self.odu._codewords[key] = self.odu._build_codeword(pc)
                supported.append(m)
            except ValueError:
                pass
        return supported

    def set_mcs(self, mcs_index: Sequence[int]) -> None:
        """改变各 UE 的 MCS（链路自适应），之后的 run_batch 按新 MCS 编码和译码。

        MCS 表不变（cfg.mcs_table）；各 MCS 的编译码链首次用到时建立并缓存。
        """
        mcs = [int(m) for m in mcs_index]
        if mcs == self.mcs:
            return
        self.pdsch_configs = make_pdsch_configs(replace(self.cfg, mcs_index=mcs))
        self.odu.set_pdsch_configs(self.pdsch_configs)
        self.ue.set_codewords(self.odu)
        self.mcs = mcs

    def fronthaul_stats(self) -> Dict[str, CodecStats]:
        """所有 O-RU 链路合并后的压缩统计。"""
        out = {}
        for name in ("uplane", "bfw"):
            codecs = [getattr(fh, name) for fh in self.fronthaul if getattr(fh, name) is not None]
            if not codecs:
                continue
            s = CodecStats()
            for c in codecs:
                for f in ("signal_energy", "error_energy", "num_clipped", "num_blocks", "num_bits"):
                    setattr(s, f, getattr(s, f) + getattr(c.stats, f))
            out[name] = s
        return out

    def reset_stats(self) -> None:
        for fh in self.fronthaul:
            fh.reset_stats()

    def _transmit(self, batch_size: int, no: float, state: ChannelState, csi_rs=None,
                  stats: Optional[LinkStats] = None):
        """O-DU -> 前传 -> O-RU -> 信道。返回 (O-DU 输出, 接收信号, 各 O-RU 实际使用的权值)。

        :param csi_rs: 测量时隙时为 :class:`~oran.csi_rs.CSIRS`（CSI-RS 与 PDSCH 一起经前传发送）
        :param stats: 给定时累加发射 EVM（O-RU 天线口相对理想前传的误差）
        """
        cfg = self.cfg
        h_csi = self.channel.csi(state, self.csi_symbol)
        tx = self.odu(batch_size, h_csi, csi_rs=csi_rs)

        # 前传 + O-RU
        grids, w_applied = [], []
        n = cfg.num_oru_ant
        for o in range(cfg.num_oru):
            uplane_hat, w_hat = self.fronthaul[o].transmit(tx.uplane[o], tx.weights[o])
            grid = self.orus[o].frequency_domain(uplane_hat, w_hat)
            if stats is not None:
                ideal = apply_precoder(tx.x_layers, tx.w[:, :, o * n:(o + 1) * n], self.odu.prg_of_sc)
                stats.tx_err += float(((grid - ideal).abs() ** 2).sum())
                stats.tx_ref += float((ideal.abs() ** 2).sum())
            grids.append(self.orus[o].to_air(grid))
            w_applied.append(w_hat if w_hat is not None else tx.w[:, :, o * n:(o + 1) * n])
        x = torch.stack(grids, dim=1)                       # [B, num_oru, N_TRX, ...]
        return tx, self.channel(x, state, no), w_applied

    @torch.no_grad()
    def measurement_slot(self, state: ChannelState, snr_db: float, csi_rs) -> torch.Tensor:
        """CSI 测量时隙：PDSCH（打孔）+ CSI-RS / CSI-IM 经前传、O-RU、信道到达 UE。

        用 state 中第一个样本的信道发送一个时隙，返回 UE 侧的频域接收网格 [1, K, N_r, num_symbols, num_sc]。
        该时隙的 PDSCH 不译码。
        """
        one = ChannelState(state.h_freq[:1], None if state.h_time is None else state.h_time[:1])
        _, y, _ = self._transmit(1, 10 ** (-snr_db / 10), one, csi_rs=csi_rs)
        return self.ue.frequency_grid(y)

    def set_compression(self, uplane: CompressionConfig, bfw: Optional[CompressionConfig] = None) -> None:
        """更换前传压缩配置（重建各 O-RU 的前传链路，压缩统计清零）。"""
        self.cfg.uplane = uplane
        if bfw is not None:
            self.cfg.bfw = bfw
        self.fronthaul = self._make_fronthaul()

    @torch.no_grad()
    def run_batch(self, batch_size: int, snr_db: float, stats: Optional[LinkStats] = None,
                  state: Optional[ChannelState] = None) -> LinkStats:
        """仿真一个批次（batch_size 个时隙）。

        :param state: 给定的信道实现（如闭环仿真中已用于计算 CQI 的射线追踪信道），None 时重新采样
        """
        cfg = self.cfg
        stats = stats or LinkStats(cfg.num_ue)
        no = 10 ** (-snr_db / 10)

        if state is None:
            state = self.channel.sample(batch_size)
        tx, y, w_applied = self._transmit(batch_size, no, state, stats=stats)

        h_eff = None
        if cfg.channel_estimation == "perfect":
            w = torch.cat(w_applied, dim=2)                 # 各 O-RU 实际使用的权值
            h_eff = effective_channel(state.stacked(), w, self.odu.prg_of_sc)
            h_eff = h_eff.unflatten(3, (cfg.num_ue, cfg.num_layers_per_ue))

        rx = self.ue(y, no, h_eff)

        for k in range(cfg.num_ue):
            b, b_hat = tx.bits[k], rx.bits[k]
            stats.bit_errors[k] += float((b != b_hat).sum())
            stats.bits[k] += b.numel()
            stats.block_errors[k] += float((~rx.crc_ok[:, k]).sum())
            stats.blocks[k] += batch_size
            stats.evm_err[k] += float(((rx.x_hat[:, k] - tx.x_data[:, k]).abs() ** 2).sum())
            stats.evm_ref[k] += float((tx.x_data[:, k].abs() ** 2).sum())
        return stats

    def simulate(self, snr_db: Sequence[float], batch_size: int = 16, max_batches: int = 50,
                 target_block_errors: int = 100, min_bler: float = 1e-3,
                 seed: Optional[int] = None, verbose: bool = True) -> List[Dict]:
        """扫描 SNR。每个 SNR 点在累计块错误数达到 target_block_errors 或批次数达到上限时停止；
        若某点 BLER 已低于 min_bler，则跳过更高的 SNR。

        :param seed: 若给定，每个 SNR 点开始时把随机种子设为 seed + 序号。不同配置（如 Cat A / B、
            不同压缩位宽）用同一个 seed 时，同一 SNR 点上的比特、信道和噪声完全相同，便于配对比较。
        """
        results = []
        for i, snr in enumerate(snr_db):
            if seed is not None:
                sionna_config.seed = seed + i
            self.reset_stats()
            stats = LinkStats(self.cfg.num_ue)
            for _ in range(max_batches):
                self.run_batch(batch_size, snr, stats)
                if stats.block_errors.sum() >= target_block_errors:
                    break
            fh = self.fronthaul_stats()
            res = {
                "snr_db": float(snr),
                "bler": stats.bler.tolist(),
                "ber": stats.ber.tolist(),
                "evm_percent": stats.evm_percent.tolist(),
                "tx_evm_percent": stats.tx_evm_percent,
                "num_blocks": stats.blocks.tolist(),
                "throughput_mbps": [(1 - p) * tb * self.cfg.slots_per_second / 1e6
                                    for p, tb in zip(stats.bler, self.tb_sizes)],
                "uplane_sqnr_db": fh["uplane"].sqnr_db,
                "uplane_clipped": fh["uplane"].num_clipped,
            }
            if "bfw" in fh:
                res["bfw_sqnr_db"] = fh["bfw"].sqnr_db
            results.append(res)
            if verbose:
                bler = " ".join(f"{p:.3e}" for p in stats.bler)
                evm = " ".join(f"{e:5.2f}" for e in stats.evm_percent)
                print(f"  SNR {snr:6.2f} dB | BLER {bler} | EVM {evm} % | "
                      f"TX-EVM {stats.tx_evm_percent:6.3f} % | blocks {int(stats.blocks[0])}",
                      flush=True)
            if stats.bler.max() < min_bler:
                break
        return results

    def fronthaul_rate(self, num_prb: Optional[int] = None):
        return fronthaul_rate(self.cfg, num_prb)
