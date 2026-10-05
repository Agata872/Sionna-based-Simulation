"""AWGN 链路抽象：(调制阶数 Qm, 码率 R, RE 数) 达到目标 TB 误块率所需的有效 SINR，以及 EESM 有效 SINR。

UE 计算 CQI（TS 38.214 5.2.2.1）和 MAC 由 CQI 选 MCS 都要回答同一个问题："某个 (Qm, R) 在多大的
SINR 下 TBLER 恰好等于目标值（0.1）"。这里用 Sionna SYS 的 PDSCH BLER 表（AWGN 信道、5G LDPC，
按码块长度 CBS 给出码块误码率）：TB 分段得到码块数 C 和 CBS，TBLER = 1 − (1 − BLER_CB)^C。

CQI 表与 MCS 表的条目大多相同（如 CQI 表 2 的序号 2~15 与 MCS 表 2 的 1, 3, 5, …, 27），这里按 (Qm, R)
在 PDSCH MCS 表 1~3 中查找相同条目，取有 BLER 数据的那一个。

表中缺少的条目：Sionna 的表从 −5 dB 起，没有最低几档 QPSK（MCS 表 1 的 0~2、表 2 的 0~1、
CQI 的 78/1024 等）。对这些条目，用同调制下频谱效率最接近的有数据条目，按 Shannon 间隔平移：
    SINR(SE) = SINR(SE_ref) + 10·log10((2^SE − 1) / (2^SE_ref − 1))，SE = Qm·R
只影响小区边缘（SINR < −3 dB 左右）。

EESM：SINR_eff = −β·ln( (1/N)·Σ exp(−SINR_i / β) )，β 取 Sionna 的标定值（按 MCS 表 1/2 的条目；
其他条目取同调制下码率最接近的条目的 β）。
"""

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import sionna.sys
import torch
from sionna.phy.nr.utils import decode_mcs_index
from sionna.sys import PHYAbstraction

PDSCH_CATEGORY = 1          # Sionna SYS 的 MCS 类别：0 为 PUSCH，1 为 PDSCH
MCS_TABLE_SIZES = {1: 29, 2: 28, 3: 29}

Entry = Tuple[int, float]   # (Qm, 码率 × 1024)


def mcs_entry(table: int, mcs: int) -> Entry:
    """PDSCH MCS 表（TS 38.214 表 5.1.3.1-1~3）的 (Qm, R×1024)。"""
    qm, rate = decode_mcs_index(mcs, table_index=table, is_pusch=False)
    return int(qm), round(float(rate) * 1024, 1)


def spectral_efficiency(entry: Entry) -> float:
    return entry[0] * entry[1] / 1024


class AWGNLinkModel:
    """AWGN 下的 TBLER 门限与 EESM。

    :param bler_target: TB 误块率目标（CQI 表 1/2 为 0.1）
    """

    GRID_DB = (-5.0, 30.0, 0.1)          # 与 PHYAbstraction 的插值网格一致

    def __init__(self, bler_target: float = 0.1, device=None):
        self.bler_target = bler_target
        self.phy_abs = PHYAbstraction(device=device)
        self.device = self.phy_abs.device
        lo, hi, step = self.GRID_DB
        self._grid_db = torch.arange(lo, hi + step / 2, step, dtype=torch.float64)
        # (Qm, R×1024) -> [(MCS 表, MCS)]
        self._table_entries: Dict[Entry, List[Tuple[int, int]]] = {}
        for table, size in MCS_TABLE_SIZES.items():
            for mcs in range(size):
                self._table_entries.setdefault(mcs_entry(table, mcs), []).append((table, mcs))
        path = os.path.join(os.path.dirname(sionna.sys.__file__), "esm_params", "eesm_beta_table.json")
        with open(path, "r", encoding="utf-8") as f:
            beta = json.load(f)["index"]
        self._beta: Dict[Entry, float] = {}
        for table in (1, 2):
            for mcs, b in enumerate(beta[str(table)]):
                self._beta.setdefault(mcs_entry(table, mcs), float(b))
        self._cache: Dict[Tuple[Entry, int], float] = {}

    # ------------------------------------------------------------------
    def _tabulated_threshold_db(self, table: int, mcs: int, num_re: int) -> Optional[float]:
        """查表得到的门限；没有数据或门限低于网格下限时返回 None。"""
        g = self._grid_db
        n = len(g)
        shape = (n, 1)
        sinr_eff = (10 ** (g / 10)).to(self.phy_abs.dtype).to(self.device).reshape(shape)
        full = lambda v: torch.full(shape, v, dtype=torch.int32, device=self.device)
        *_, tbler, _ = self.phy_abs(full(mcs), sinr_eff=sinr_eff, num_allocated_re=full(num_re),
                                    mcs_table_index=full(table), mcs_category=full(PDSCH_CATEGORY),
                                    check_mcs_index_validity=False)
        tbler = tbler[:, 0].double().cpu()
        if not torch.isfinite(tbler).all():
            return None
        bad = torch.nonzero(tbler > self.bler_target).flatten()
        if len(bad) == 0 or bad[-1] == n - 1:
            return None
        i = int(bad[-1]) + 1                         # 此后 TBLER 都不超过目标
        b0, b1 = float(tbler[i - 1]), float(tbler[i])
        if b1 <= 0:
            return float(g[i])
        # 在 log(TBLER) 上线性插值
        frac = (np.log(b0) - np.log(self.bler_target)) / (np.log(b0) - np.log(b1))
        return float(g[i - 1] + frac * (g[i] - g[i - 1]))

    def threshold_db(self, entry: Entry, num_re: int) -> float:
        """(Qm, R×1024) 在 num_re 个 RE（含各层）上传一个 TB、TBLER = 目标值时的有效 SINR [dB]。"""
        entry = (int(entry[0]), round(float(entry[1]), 1))
        key = (entry, int(num_re))
        if key in self._cache:
            return self._cache[key]
        thr = None
        for table, mcs in self._table_entries.get(entry, []):
            thr = self._tabulated_threshold_db(table, mcs, num_re)
            if thr is not None:
                break
        if thr is None:
            thr = self._extrapolated_threshold_db(entry, num_re)
        self._cache[key] = float(thr)
        return self._cache[key]

    def _extrapolated_threshold_db(self, entry: Entry, num_re: int) -> float:
        se = spectral_efficiency(entry)
        candidates = sorted((e for e in self._table_entries if e[0] == entry[0] and e != entry),
                            key=lambda e: abs(spectral_efficiency(e) - se))
        for ref in candidates:
            thr_ref = None
            for table, mcs in self._table_entries[ref]:
                thr_ref = self._tabulated_threshold_db(table, mcs, num_re)
                if thr_ref is not None:
                    break
            if thr_ref is not None:
                se_ref = spectral_efficiency(ref)
                return thr_ref + 10 * np.log10((2 ** se - 1) / (2 ** se_ref - 1))
        raise ValueError(f"调制阶数 {entry[0]} 没有可用的 BLER 表")

    # ------------------------------------------------------------------
    def beta(self, entry: Entry) -> float:
        """EESM 参数 β。"""
        entry = (int(entry[0]), round(float(entry[1]), 1))
        if entry in self._beta:
            return self._beta[entry]
        same = [e for e in self._beta if e[0] == entry[0]]
        nearest = min(same, key=lambda e: abs(e[1] - entry[1]))
        return self._beta[nearest]

    @staticmethod
    def effective_sinr_db(sinr: torch.Tensor, betas: Sequence[float]) -> np.ndarray:
        """EESM：sinr 为线性 SINR（任意形状，全部参与平均），对每个 β 返回有效 SINR [dB]。"""
        x = sinr.flatten().double().clamp_min(1e-12)
        b = torch.as_tensor(np.asarray(betas, dtype=np.float64), device=x.device)[:, None]
        eff = -b[:, 0] * (torch.logsumexp(-x[None, :] / b, dim=1) - np.log(x.numel()))
        return 10 * np.log10(eff.cpu().numpy())
