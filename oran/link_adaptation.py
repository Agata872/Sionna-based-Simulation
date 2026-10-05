"""gNB MAC 链路自适应：由 UE 上报的 CQI 选择 PDSCH 的 MCS，并按 ACK/NACK 做外环修正（OLLA）。

3GPP 不规定 gNB 如何使用 CQI（属于调度器实现），这里采用常见做法：
  1. CQI → SINR 估计：取所报 CQI 在 CSI 参考资源上的门限 SINR（该 CQI 刚好满足 TBLER = 0.1，是真实
     SINR 的保守下界）；
  2. 叠加 OLLA 偏移 Δ（dB）；
  3. 在 MCS 表（TS 38.214 表 5.1.3.1-1/2）中选门限不超过 SINR 估计 + Δ 的最大 MCS，门限按实际 PDSCH
     分配（RE 数、TBS）计算。Δ = 0 时选出的就是与该 CQI 的 (Qm, R) 相同（或最接近）的 MCS，即"CQI 查表"。
  4. OLLA：每个 ACK 使 Δ 增加 δ·p/(1−p)，每个 NACK 使 Δ 减小 δ，稳态时首传 BLER 收敛到目标 p。
     一个批次（若干时隙）用同一个 MCS，批次结束后按该批次的 ACK/NACK 个数一次性更新，相当于 HARQ 反馈
     有一个批次的时延。已用最高 MCS 时 ACK 不再上调 Δ，避免 Δ 无意义地累积。

可选的 MCS 由仿真器给出（:meth:`~oran.link.ORANDownlink.supported_mcs`）：Sionna 的 LDPC 编码器不支持
重复编码，码率低于 1/5、或 BG1 下低于 1/3 的 MCS 无法实现（3GPP 在循环缓冲速率匹配中重复母码）。
51 PRB、2 层时 MCS 表 2 最低可用 MCS 3（QPSK 449/1024）。SINR 估计低于所有可用 MCS 的门限（包括 CQI 为 0）
时仍按最低可用 MCS 发送。没有 HARQ 重传，吞吐量只计首传成功的 TB。
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .bler_model import MCS_TABLE_SIZES, AWGNLinkModel, mcs_entry
from .csi import CQIReporter


@dataclass
class LinkAdaptationConfig:
    """链路自适应配置。

    :param cqi_table: CSI-ReportConfig 的 cqi-Table，None 时与 mcs_table 相同（1 对 1，2 对 2）
    :param bler_target: 首传 BLER 目标（CQI 表 1/2 的定义为 0.1）
    :param olla_step_db: NACK 时 OLLA 偏移的下调步长 δ；ACK 时上调 δ·p/(1−p)
    """

    cqi_table: Optional[int] = None
    bler_target: float = 0.1
    olla: bool = True
    olla_step_db: float = 0.5
    olla_min_db: float = -10.0
    olla_max_db: float = 10.0


class LinkAdaptation:
    """每个 UE 一个 OLLA 状态。

    :param mcs_table: PDSCH 的 MCS 表（1 或 2）
    :param num_re_alloc: 实际 PDSCH 分配中一个 TB 的数据 RE 数（含各层）
    :param allowed_mcs: 可选的 MCS 序号，None 时为整张表
    """

    def __init__(self, la_cfg: LinkAdaptationConfig, mcs_table: int, num_ue: int, num_re_alloc: int,
                 reporter: CQIReporter, model: AWGNLinkModel,
                 allowed_mcs: Optional[Sequence[int]] = None):
        if mcs_table not in (1, 2):
            raise ValueError("链路自适应只支持 MCS 表 1、2")
        self.cfg = la_cfg
        self.mcs_table = mcs_table
        self.num_ue = num_ue
        self.mcs_threshold_db = np.array([model.threshold_db(mcs_entry(mcs_table, m), num_re_alloc)
                                          for m in range(MCS_TABLE_SIZES[mcs_table])])
        allowed = range(MCS_TABLE_SIZES[mcs_table]) if allowed_mcs is None else allowed_mcs
        self.allowed_mcs = np.array(sorted(int(m) for m in allowed))
        if len(self.allowed_mcs) == 0:
            raise ValueError("没有可用的 MCS")
        self.cqi_sinr_db = reporter.threshold_db
        p = la_cfg.bler_target
        self.step_down = la_cfg.olla_step_db
        self.step_up = la_cfg.olla_step_db * p / (1 - p)
        self.offset_db = np.zeros(num_ue)

    def reset(self) -> None:
        self.offset_db = np.zeros(self.num_ue)

    def sinr_estimate_db(self, cqi: np.ndarray) -> np.ndarray:
        """gNB 由 CQI 得到的 SINR 估计（含 OLLA 偏移），CQI 为 0 时为 −inf。"""
        cqi = np.asarray(cqi, dtype=int)
        base = np.where(cqi > 0, self.cqi_sinr_db[np.maximum(cqi, 1) - 1], -np.inf)
        return base + (self.offset_db if self.cfg.olla else 0.0)

    def select_mcs(self, cqi: np.ndarray) -> np.ndarray:
        sinr_hat = self.sinr_estimate_db(cqi)
        mcs = np.zeros(self.num_ue, dtype=int)
        thr = self.mcs_threshold_db[self.allowed_mcs]
        for k in range(self.num_ue):
            ok = np.flatnonzero(thr <= sinr_hat[k] + 1e-9)
            mcs[k] = self.allowed_mcs[ok[-1] if len(ok) else 0]
        return mcs

    def update(self, num_ack: np.ndarray, num_nack: np.ndarray,
               mcs: Optional[np.ndarray] = None) -> None:
        """按一个批次内各 UE 的 ACK / NACK 个数更新 OLLA 偏移。

        :param mcs: 该批次使用的 MCS。已是最高 MCS 的 UE，ACK 不再上调偏移（抗积分饱和）
        """
        if not self.cfg.olla:
            return
        up = self.step_up * np.asarray(num_ack, dtype=float)
        if mcs is not None:
            up = np.where(np.asarray(mcs) >= self.allowed_mcs[-1], 0.0, up)
        self.offset_db = np.clip(self.offset_db + up - self.step_down * np.asarray(num_nack),
                                 self.cfg.olla_min_db, self.cfg.olla_max_db)
