"""O-RAN 7.2x 下行（Category A / B）链路级仿真，基于 Sionna PHY。

模块划分：
    config.py            仿真配置 ORANConfig
    pdsch.py             PDSCH 配置（复用 Sionna PUSCHConfig 的 DMRS/TBS 逻辑）与资源网格
    odu.py               O-DU High-PHY：TB 编码、调制、层映射、RE 映射、（Cat A）预编码
    precoding.py         多用户预编码（BD / 特征 ZF / RZF），按 PRG 计算
    compression/         前传 IQ 压缩算法（BFP 等），独立成包，便于扩展
    fronthaul.py         前传链路：U-plane / C-plane 压缩传输、比特率估算
    oru.py               O-RU Low-PHY：（Cat B）预编码、IFFT+CP、DAC 预留
    channel.py           多链路信道：CDL（每个 UE×O-RU 一条）或射线追踪
    rt_channel.py        Sionna RT 射线追踪：场景、撒点、链路预算、信道冲激响应
    channel_estimation.py  PRG 内线性插值器
    ue.py                UE 接收机：信道估计、LMMSE、译码
    link.py              端到端仿真器 ORANDownlink
    bler_model.py        AWGN 链路抽象：(Qm, R, RE 数) 的 TBLER 门限、EESM
    csi.py               UE 侧 CSI：CQI 表、CSI 参考资源、LMMSE SINR、宽带 CQI
    link_adaptation.py   MAC 链路自适应：CQI -> MCS、OLLA
    closed_loop.py       闭环仿真器 ClosedLoopDownlink（射线追踪场景中撒点）
"""

from .closed_loop import ClosedLoopDownlink, DropResult
from .config import CompressionConfig, ORANConfig, RTConfig
from .link import LinkStats, ORANDownlink
from .link_adaptation import LinkAdaptationConfig

__all__ = ["CompressionConfig", "ORANConfig", "RTConfig", "ORANDownlink", "LinkStats",
           "ClosedLoopDownlink", "DropResult", "LinkAdaptationConfig"]
