"""PDSCH 配置与资源网格。

Sionna 2.2 只有 PUSCH 的配置类。CP-OFDM 下（PUSCH 不做 transform precoding），
PDSCH 与 PUSCH 的以下内容一致，因此复用 :class:`~sionna.phy.nr.PUSCHConfig`：
  - DMRS：序列、Type 1/2、单/双符号、附加位置、CDM 组与 OCC（TS 38.211 7.4.1.1 / 6.4.1.1），
    DMRS 端口 p 对应 PDSCH 天线端口 1000+p；
  - DMRS 功率提升 β（TS 38.214 表 4.1-1 / 6.2.2-1）；
  - TBS 计算（TS 38.214 5.1.3.2）。
不同之处只在 TB 的信道类型（MCS 表、加扰初始化 c_init = n_RNTI·2^15 + q·2^14 + n_ID），
由 :class:`PDSCHConfig` 把 TBConfig.channel_type 设为 "PDSCH" 处理。

简化：整个时隙 14 个符号都用于 PDSCH（不预留 PDCCH/CORESET）。
"""

from typing import List

import numpy as np
from sionna.phy.nr import CarrierConfig, PUSCHConfig, PUSCHDMRSConfig, PUSCHPilotPattern, TBConfig
from sionna.phy.ofdm import ResourceGrid

from .config import ORANConfig


class PDSCHConfig(PUSCHConfig):
    """PDSCH 配置：PUSCHConfig + channel_type="PDSCH" 的 TBConfig。"""

    def __init__(self, carrier_config=None, dmrs_config=None, tb_config=None, **kwargs):
        if tb_config is None:
            tb_config = TBConfig(channel_type="PDSCH")
        super().__init__(carrier_config, dmrs_config, tb_config, **kwargs)
        self._name = "PDSCH Configuration"

    @property
    def tb(self) -> TBConfig:
        """TB 配置，channel_type 必须为 "PDSCH"。"""
        return self._tb

    @tb.setter
    def tb(self, value) -> None:
        if value is None:
            value = TBConfig(channel_type="PDSCH")
        elif not isinstance(value, TBConfig):
            raise TypeError("tb 必须是 TBConfig")
        elif value.channel_type != "PDSCH":
            raise ValueError('TBConfig 的 channel_type 必须为 "PDSCH"')
        self._tb = value

    def check_config(self) -> bool:
        # 父类只接受 channel_type 为 "PUSCH" 的 TBConfig，检查期间临时切换
        tb = self.tb
        channel_type = tb.channel_type
        tb.channel_type = "PUSCH"
        try:
            return super().check_config()
        finally:
            tb.channel_type = channel_type


def max_dmrs_ports(config_type: int, length: int) -> int:
    """单个时隙可用的 DMRS 端口数（正交端口数）。"""
    return (4 if config_type == 1 else 6) * length


def make_pdsch_configs(cfg: ORANConfig) -> List[PDSCHConfig]:
    """为每个 UE 生成 PDSCH 配置。

    UE k 使用 DMRS 端口 [k·ν, ..., k·ν+ν-1]（ν 为每 UE 层数），各 UE 共用同一组时频资源（MU-MIMO）。
    num_cdm_groups_without_data 取所有 UE 用到的 CDM 组数，保证 UE 不会在其他 UE 的 DMRS 上解数据。
    """
    nu = cfg.num_layers_per_ue
    num_ports = cfg.num_layers
    if num_ports > max_dmrs_ports(cfg.dmrs_config_type, cfg.dmrs_length):
        raise ValueError(
            f"总层数 {num_ports} 超过 DMRS Type {cfg.dmrs_config_type}、长度 {cfg.dmrs_length} "
            f"的正交端口数，请增大 dmrs_length 或改用 dmrs_config_type=2")

    # 先用全部端口建一个 DMRS 配置，得到所用 CDM 组数
    probe = PUSCHDMRSConfig(config_type=cfg.dmrs_config_type, length=cfg.dmrs_length,
                            dmrs_port_set=list(range(num_ports)))
    num_cdm_groups = int(max(probe.cdm_groups)) + 1

    configs = []
    for k in range(cfg.num_ue):
        carrier = CarrierConfig(
            n_cell_id=cfg.n_cell_id,
            subcarrier_spacing=cfg.subcarrier_spacing,
            n_size_grid=cfg.num_prb,
            slot_number=cfg.slot_number,
            frame_number=cfg.frame_number,
        )
        dmrs = PUSCHDMRSConfig(
            config_type=cfg.dmrs_config_type,
            length=cfg.dmrs_length,
            additional_position=cfg.dmrs_additional_position,
            type_a_position=cfg.dmrs_type_a_position,
            dmrs_port_set=list(range(k * nu, (k + 1) * nu)),
            num_cdm_groups_without_data=num_cdm_groups,
        )
        tb = TBConfig(channel_type="PDSCH", mcs_index=cfg.mcs_of(k), mcs_table=cfg.mcs_table)
        pc = PDSCHConfig(
            carrier_config=carrier,
            dmrs_config=dmrs,
            tb_config=tb,
            num_layers=nu,
            num_antenna_ports=nu,
            precoding="non-codebook",
            mapping_type="A",
            symbol_allocation=[0, carrier.num_symbols_per_slot],
            n_rnti=cfg.rnti_of(k),
        )
        configs.append(pc)

    # 所有 UE 的资源网格结构必须相同
    for pc in configs[1:]:
        if not np.array_equal(pc.dmrs_mask, configs[0].dmrs_mask):
            raise ValueError("各 UE 的 DMRS 图样不一致")
    return configs


def make_resource_grid(cfg: ORANConfig, pdsch_configs: List[PDSCHConfig], precision=None, device=None):
    """建立资源网格：num_tx = UE 数，num_streams_per_tx = 每 UE 层数。

    这里的 "tx" 是虚拟发射机：每个 UE 的层组看成一个发射机，经预编码后由 O-RU 的天线发出。
    这样 UE 侧可直接沿用 Sionna 的多发射机信道估计与 LMMSE 均衡。
    """
    pc = pdsch_configs[0]
    bandwidth = pc.num_subcarriers * pc.carrier.subcarrier_spacing * 1e3
    cp_len = int(np.ceil(pc.carrier.cyclic_prefix_length * bandwidth))
    pilot_pattern = PUSCHPilotPattern(pdsch_configs, precision=precision, device=device)
    return ResourceGrid(
        num_ofdm_symbols=pc.carrier.num_symbols_per_slot,
        fft_size=pc.num_subcarriers,
        subcarrier_spacing=pc.carrier.subcarrier_spacing * 1e3,
        num_tx=len(pdsch_configs),
        num_streams_per_tx=pc.num_layers,
        cyclic_prefix_length=cp_len,
        pilot_pattern=pilot_pattern,
        precision=precision,
        device=device,
    )
