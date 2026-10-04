"""O-DU：7.2x 分割点以上的 High-PHY。

处理链（每个 UE 一个码字）：
    TB CRC -> 码块分割 + CB CRC -> LDPC 编码 -> 速率匹配 -> 加扰（PDSCH）    [TBEncoder]
    -> 调制 -> 层映射 -> RE 映射（数据 + DMRS）
    -> Cat A：预编码 -> 前传发送 N_TRX 路天线端口流
       Cat B：前传发送 L 路层流 + C-plane 预编码权值

预编码与 RE 映射的先后：图中 Cat A 为"预编码 -> RE 映射"，这里先做 RE 映射（含 DMRS）再逐 RE
预编码。预编码是逐 RE 的线性运算，两种顺序结果相同，而 DMRS 本就需要与数据一起预编码。
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
from sionna.phy.mapping import BinarySource, Mapper
from sionna.phy.nr import LayerMapper, TBEncoder
from sionna.phy.ofdm import ResourceGridMapper

from .config import ORANConfig
from .pdsch import PDSCHConfig
from .precoding import apply_precoder, compute_precoder, subcarrier_to_prg


def _scalar(v):
    return v.item() if hasattr(v, "item") else v


@dataclass
class ODUOutput:
    bits: List[torch.Tensor]          # 每个 UE 的信息比特 [batch, 1, tb_size_k]
    x_data: torch.Tensor              # 层映射后的数据符号 [batch, K, ν, num_data_symbols]
    x_layers: torch.Tensor            # RE 映射后的层信号 [batch, L, num_symbols, num_sc]
    w: torch.Tensor                   # 预编码矩阵 [batch, num_prg, N_t, L]
    uplane: List[torch.Tensor]        # 每个 O-RU 的 U-plane eAxC 流 [batch, num_eaxc, num_symbols, num_sc]
    weights: List[Optional[torch.Tensor]]  # Cat B：每个 O-RU 的 C-plane 权值 [batch, num_prg, N_TRX, L]


class ODU:
    """O-DU High-PHY。

    :param cfg: 仿真配置
    :param pdsch_configs: 每个 UE 的 PDSCH 配置
    :param resource_grid: 资源网格（num_tx = UE 数，num_streams_per_tx = 每 UE 层数）
    """

    def __init__(self, cfg: ORANConfig, pdsch_configs: List[PDSCHConfig], resource_grid,
                 precision=None, device=None):
        self.cfg = cfg
        self.device = device
        self.resource_grid = resource_grid
        self.binary_source = BinarySource(precision=precision, device=device)
        self.tb_encoders, self.mappers, self.layer_mappers = [], [], []
        for pc in pdsch_configs:
            n_id = pc.carrier.n_cell_id if pc.tb.n_id is None else pc.tb.n_id
            num_bits_per_symbol = int(_scalar(pc.tb.num_bits_per_symbol))
            self.tb_encoders.append(TBEncoder(
                target_tb_size=pc.tb_size,
                num_coded_bits=pc.num_coded_bits,
                target_coderate=float(_scalar(pc.tb.target_coderate)),
                num_bits_per_symbol=num_bits_per_symbol,
                num_layers=pc.num_layers,
                n_rnti=pc.n_rnti,
                n_id=n_id,
                channel_type="PDSCH",
                codeword_index=0,
                precision=precision,
                device=device,
            ))
            self.mappers.append(Mapper("qam", num_bits_per_symbol, precision=precision, device=device))
            self.layer_mappers.append(LayerMapper(num_layers=pc.num_layers, precision=precision,
                                                  device=device))
        self.tb_sizes = [enc.tb_size for enc in self.tb_encoders]
        self.rg_mapper = ResourceGridMapper(resource_grid, precision=precision, device=device)
        self.prg_of_sc = subcarrier_to_prg(cfg.num_prb, cfg.prg_num_prb)

    def high_phy(self, batch_size: int):
        """信道编码、调制、层映射、RE 映射。返回 (bits, x_data, x_layers)。"""
        bits, x_data = [], []
        for enc, mapper, layer_mapper, tb_size in zip(
                self.tb_encoders, self.mappers, self.layer_mappers, self.tb_sizes):
            b = self.binary_source([batch_size, 1, tb_size])
            x = layer_mapper(mapper(enc(b)))              # [batch, 1, ν, num_data_symbols]
            bits.append(b)
            x_data.append(x)
        x_data = torch.cat(x_data, dim=1)                   # [batch, K, ν, num_data_symbols]
        x_grid = self.rg_mapper(x_data)                     # [batch, K, ν, num_symbols, num_sc]
        x_layers = x_grid.flatten(1, 2)                     # [batch, L, num_symbols, num_sc]
        return bits, x_data, x_layers

    def precoder(self, h_csi: torch.Tensor) -> torch.Tensor:
        """h_csi: [batch, num_prg, K, N_r, N_t] -> W: [batch, num_prg, N_t, L]"""
        return compute_precoder(h_csi, self.cfg.num_layers_per_ue, self.cfg.precoder,
                                self.cfg.rzf_alpha)

    def __call__(self, batch_size: int, h_csi: torch.Tensor) -> ODUOutput:
        bits, x_data, x_layers = self.high_phy(batch_size)
        w = self.precoder(h_csi)
        n = self.cfg.num_oru_ant
        uplane, weights = [], []
        for o in range(self.cfg.num_oru):
            w_o = w[:, :, o * n:(o + 1) * n, :]             # 该 O-RU 天线对应的权值行
            if self.cfg.category == "A":
                uplane.append(apply_precoder(x_layers, w_o, self.prg_of_sc))
                weights.append(None)
            else:
                uplane.append(x_layers)
                weights.append(w_o)
        return ODUOutput(bits, x_data, x_layers, w, uplane, weights)
