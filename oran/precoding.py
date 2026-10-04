"""下行多用户预编码（按 PRG 计算，O-DU 根据 CSI 求出）。

记号：K 个 UE，每个 UE N_r 根接收天线、ν 层；发射天线共 N_t 根（cell-free 时为所有 O-RU 天线之和）；
总层数 L = K·ν。预编码矩阵 W ∈ C^{N_t×L}，每个 PRG 一个，在整个时隙内不变。

功率约定：层符号功率为 1，W 每列（每层）功率 1/L，即 ||W||_F^2 = 1，每个 RE 的总发射功率为 1。

方法：
  - "bd"：块对角化。UE k 的预编码位于其他所有 UE 信道的零空间内，彻底消除用户间干扰；
          再在零空间内对 H_k 做 SVD，取前 ν 个右奇异向量。要求 N_t - (K-1)·N_r >= ν。
  - "ezf" / "rzf"：特征 ZF / RZF。先对每个 UE 做 SVD 得到等效信道 U_k^H H_k（ν×N_t），
          把所有 UE 的等效信道堆叠后做（正则化）迫零。只消除各 UE 主特征方向上的干扰，对天线数要求更低。
"""

import torch


def prg_center_subcarriers(num_prb: int, prg_num_prb: int) -> torch.Tensor:
    """每个 PRG 中心子载波的下标（最后一个 PRG 可能不满）。"""
    starts = torch.arange(0, num_prb, prg_num_prb) * 12
    ends = torch.clamp(starts + prg_num_prb * 12, max=num_prb * 12)
    return (starts + ends) // 2


def subcarrier_to_prg(num_prb: int, prg_num_prb: int) -> torch.Tensor:
    """每个子载波所属 PRG 的下标，长度 12·num_prb。"""
    return torch.arange(num_prb * 12) // (prg_num_prb * 12)


def _dominant_right_singular_vectors(h: torch.Tensor, n: int) -> torch.Tensor:
    """h: [..., M, N] -> 前 n 个右奇异向量 [..., N, n]。"""
    _, _, vh = torch.linalg.svd(h, full_matrices=False)
    return vh[..., :n, :].mH


def block_diagonalization(h: torch.Tensor, num_layers_per_ue: int) -> torch.Tensor:
    """h: [..., K, N_r, N_t] -> W: [..., N_t, K·ν]（各列单位范数）。"""
    num_ue, num_rx_ant, num_tx_ant = h.shape[-3:]
    nu = num_layers_per_ue
    w = []
    for k in range(num_ue):
        h_k = h[..., k, :, :]
        if num_ue > 1:
            others = torch.cat([h[..., j, :, :] for j in range(num_ue) if j != k], dim=-2)
            # 其他 UE 信道的零空间：右奇异向量中奇异值为 0 的部分
            _, _, vh = torch.linalg.svd(others, full_matrices=True)
            rank = others.shape[-2]
            v0 = vh[..., rank:, :].mH                     # [..., N_t, N_t - rank]
            v1 = _dominant_right_singular_vectors(h_k @ v0, nu)
            w.append(v0 @ v1)
        else:
            w.append(_dominant_right_singular_vectors(h_k, nu))
    return torch.cat(w, dim=-1)


def eigen_rzf(h: torch.Tensor, num_layers_per_ue: int, alpha: float = 0.0) -> torch.Tensor:
    """特征 RZF（alpha=0 时为特征 ZF）。h: [..., K, N_r, N_t] -> W: [..., N_t, K·ν]（各列单位范数）。"""
    num_ue = h.shape[-3]
    nu = num_layers_per_ue
    h_eff = []
    for k in range(num_ue):
        u, s, vh = torch.linalg.svd(h[..., k, :, :], full_matrices=False)
        # U_k^H H_k 的前 ν 行 = diag(s) V_k^H
        h_eff.append(s[..., :nu, None].to(vh.dtype) * vh[..., :nu, :])
    h_eff = torch.cat(h_eff, dim=-2)                       # [..., L, N_t]
    gram = h_eff @ h_eff.mH
    eye = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    w = h_eff.mH @ torch.linalg.inv(gram + alpha * eye)
    return w / torch.linalg.vector_norm(w, dim=-2, keepdim=True)


def compute_precoder(h_csi: torch.Tensor, num_layers_per_ue: int, method: str = "bd",
                     alpha: float = 0.0) -> torch.Tensor:
    """由 CSI 计算预编码矩阵。

    :param h_csi: [batch, num_prg, K, N_r, N_t]，各 PRG 参考位置的信道
    :return: W，[batch, num_prg, N_t, L]，||W||_F = 1
    """
    if method == "bd":
        w = block_diagonalization(h_csi, num_layers_per_ue)
    elif method == "ezf":
        w = eigen_rzf(h_csi, num_layers_per_ue, 0.0)
    elif method == "rzf":
        w = eigen_rzf(h_csi, num_layers_per_ue, alpha)
    else:
        raise ValueError(f"未知预编码方法 {method}")
    num_layers = w.shape[-1]
    return w / num_layers ** 0.5


def apply_precoder(x: torch.Tensor, w: torch.Tensor, prg_of_sc: torch.Tensor) -> torch.Tensor:
    """逐 RE 预编码。

    :param x: 层信号 [batch, L, num_symbols, num_sc]
    :param w: [batch, num_prg, N_t, L]
    :param prg_of_sc: [num_sc]，每个子载波所属 PRG
    :return: 天线信号 [batch, N_t, num_symbols, num_sc]
    """
    w_sc = w[:, prg_of_sc.to(w.device)]                    # [batch, num_sc, N_t, L]
    return torch.einsum("bkal,blsk->bask", w_sc, x)


def effective_channel(h: torch.Tensor, w: torch.Tensor, prg_of_sc: torch.Tensor) -> torch.Tensor:
    """预编码后的等效信道 H·W（用于理想 CSI 接收）。

    :param h: [batch, K, N_r, N_t, num_symbols, num_sc]
    :param w: [batch, num_prg, N_t, L]
    :return: [batch, K, N_r, L, num_symbols, num_sc]
    """
    w_sc = w[:, prg_of_sc.to(w.device)]                    # [batch, num_sc, N_t, L]
    return torch.einsum("burask,bkal->burlsk", h, w_sc)
