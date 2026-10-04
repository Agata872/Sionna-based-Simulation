"""PRG 感知的 DMRS 信道估计。

O-DU 以 PRG（若干 PRB）为粒度预编码，UE 看到的等效信道 H·W 在 PRG 边界处会跳变。
TS 38.214 5.1.2.3 允许 UE 假定同一 PRG 内预编码相同，但不能跨 PRG 插值或平滑。
Sionna 自带的估计器在整个带宽上插值，会把相邻 PRG 的 DMRS 混进来，因此这里重新实现：

1. CDM 分离（:class:`DMRSChannelEstimator`）：同一 CDM 组的端口靠 OCC 区分。
   Sionna 的做法是相邻两个导频直接平均，若信道在频域有斜率，同组另一端口会以一阶误差泄漏进来
   （误差与时延扩展成正比，CDL-C 100 ns 时约 -32 dB 的估计误差平底）。
   这里在 PRG 内对每个导频取邻近若干导频，按局部线性模型
       z_s(j) = h_s(k_j) + Σ_{s'} ρ_{s'}(j) h_{s'}(k_j)，h(k) ≈ α + β·(k - k_n)
   做最小二乘，取 α 作为该导频处的估计，一阶泄漏与位置偏差都被消掉。
   ρ_{s'} 为同组端口与本端口的导频比值（由 DMRS 配置已知）。
2. 插值（:class:`PRGLinearInterpolator`）：频域在 PRG 内相邻导频间线性插值、两端常数外推；
   时域在 DMRS 符号间线性插值、两端常数外推。
"""

import numpy as np
import torch
from sionna.phy.nr import PUSCHLSChannelEstimator
from sionna.phy.ofdm import BaseChannelInterpolator


def _to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _linear_weights(targets: np.ndarray, anchors: np.ndarray):
    """对每个 target 找左右最近的 anchor（序号）及线性权重，超出范围时常数外推。"""
    right = np.searchsorted(anchors, targets, side="left")
    right = np.clip(right, 0, len(anchors) - 1)
    left = np.where(anchors[right] > targets, right - 1, right)
    left = np.clip(left, 0, len(anchors) - 1)
    right = np.where(targets > anchors[right], left, right)  # 右侧没有 anchor 时
    span = anchors[right] - anchors[left]
    w_right = np.where(span > 0, (targets - anchors[left]) / np.where(span > 0, span, 1), 0.0)
    return left, right, 1.0 - w_right, w_right


class PRGLinearInterpolator(BaseChannelInterpolator):
    """只在 PRG 内做频域线性插值的信道插值器。

    :param pilot_pattern: 资源网格的导频图样
    :param prg_num_sc: 每个 PRG 的子载波数（宽带预编码时取全部子载波）
    """

    def __init__(self, pilot_pattern, prg_num_sc: int, precision=None, device=None):
        super().__init__(time_avg=False, precision=precision, device=device)
        mask = _to_numpy(pilot_pattern.mask).astype(bool)   # [T, S, num_symbols, num_sc]
        pilots = _to_numpy(pilot_pattern.pilots)            # [T, S, P]
        num_tx, num_streams, num_symbols, num_sc = mask.shape

        idx_l, idx_r, w_l, w_r = [], [], [], []
        dmrs_symbols = None
        for t in range(num_tx):
            for s in range(num_streams):
                pos = np.argwhere(mask[t, s])               # 与导频向量顺序一致（先符号后子载波）
                valid = np.abs(pilots[t, s]) > 0
                syms = np.unique(pos[valid, 0])
                if dmrs_symbols is None:
                    dmrs_symbols = syms
                elif not np.array_equal(dmrs_symbols, syms):
                    raise ValueError("各数据流的 DMRS 符号位置必须相同")
                il, ir, wl, wr = [], [], [], []
                for d in syms:
                    sel = np.flatnonzero(valid & (pos[:, 0] == d))   # 该符号上本流的导频序号
                    ks = pos[sel, 1]
                    a_l = np.zeros(num_sc, int)
                    a_r = np.zeros(num_sc, int)
                    b_l = np.zeros(num_sc)
                    b_r = np.zeros(num_sc)
                    for start in range(0, num_sc, prg_num_sc):
                        sc = np.arange(start, min(start + prg_num_sc, num_sc))
                        in_prg = (ks >= sc[0]) & (ks <= sc[-1])
                        if not in_prg.any():
                            raise ValueError("某个 PRG 内没有 DMRS")
                        l, r, wl_, wr_ = _linear_weights(sc, ks[in_prg])
                        a_l[sc], a_r[sc] = sel[in_prg][l], sel[in_prg][r]
                        b_l[sc], b_r[sc] = wl_, wr_
                    il.append(a_l); ir.append(a_r); wl.append(b_l); wr.append(b_r)
                idx_l.append(np.concatenate(il)); idx_r.append(np.concatenate(ir))
                w_l.append(np.concatenate(wl)); w_r.append(np.concatenate(wr))

        shape = [num_tx, num_streams, -1]
        dev = self.device
        self._idx_l = torch.as_tensor(np.stack(idx_l).reshape(shape), device=dev)
        self._idx_r = torch.as_tensor(np.stack(idx_r).reshape(shape), device=dev)
        self._w_l = torch.as_tensor(np.stack(w_l).reshape(shape), dtype=self.dtype, device=dev)
        self._w_r = torch.as_tensor(np.stack(w_r).reshape(shape), dtype=self.dtype, device=dev)
        self._num_dmrs = len(dmrs_symbols)
        self._num_sc = num_sc

        t_l, t_r, v_l, v_r = _linear_weights(np.arange(num_symbols), dmrs_symbols)
        self._t_l = torch.as_tensor(t_l, device=dev)
        self._t_r = torch.as_tensor(t_r, device=dev)
        self._v_l = torch.as_tensor(v_l, dtype=self.dtype, device=dev)[:, None]
        self._v_r = torch.as_tensor(v_r, dtype=self.dtype, device=dev)[:, None]

    def _interpolate(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, R, Ra, T, S, P] -> [B, R, Ra, T, S, num_symbols, num_sc]
        lead = x.shape[:3]
        idx_l = self._idx_l.expand(*lead, *self._idx_l.shape)
        idx_r = self._idx_r.expand(*lead, *self._idx_r.shape)
        xf = self._w_l * x.gather(-1, idx_l) + self._w_r * x.gather(-1, idx_r)
        xf = xf.reshape(*xf.shape[:-1], self._num_dmrs, self._num_sc)
        return (self._v_l * xf.index_select(-2, self._t_l) +
                self._v_r * xf.index_select(-2, self._t_r))

    def __call__(self, h_hat: torch.Tensor, err_var: torch.Tensor):
        err_var = err_var.broadcast_to(h_hat.shape).to(self.dtype)
        return self._interpolate(h_hat), self._interpolate(err_var)


class DMRSChannelEstimator(PUSCHLSChannelEstimator):
    """DMRS 信道估计：LS + PRG 内局部线性模型的 CDM 分离 + PRG 内线性插值。

    导频提取、调用插值器等流程沿用 :class:`~sionna.phy.nr.PUSCHLSChannelEstimator`，
    只替换"导频处的估计"这一步。

    :param prg_num_sc: 每个 PRG 的子载波数
    :param window: 估计每个导频时使用的邻近导频数（同一 PRG、同一 DMRS 符号、同一 CDM 组内）。
        窗口越大噪声越小，但信道曲率带来的偏差越大
    """

    def __init__(self, resource_grid, dmrs_length: int, dmrs_additional_position: int,
                 num_cdm_groups_without_data: int, prg_num_sc: int, window: int = 6,
                 precision=None, device=None):
        interpolator = PRGLinearInterpolator(resource_grid.pilot_pattern, prg_num_sc,
                                             precision=precision, device=device)
        super().__init__(resource_grid, dmrs_length, dmrs_additional_position,
                         num_cdm_groups_without_data, interpolator=interpolator,
                         precision=precision, device=device)
        mask = _to_numpy(self._pilot_pattern.mask).astype(bool)   # [T, S, num_symbols, num_sc]
        pilots = _to_numpy(self._pilot_pattern.pilots)            # [T, S, P]
        num_tx, num_streams, _, _ = mask.shape
        num_pilots = pilots.shape[-1]
        for t in range(num_tx):
            for s in range(num_streams):
                if not np.array_equal(mask[t, s], mask[0, 0]):
                    raise ValueError("各数据流的 DMRS 掩码必须相同")
        pos = np.argwhere(mask[0, 0])
        sym, sc = pos[:, 0], pos[:, 1]

        # 双符号 DMRS：先对成对的两个 DMRS 符号取平均（时域 OCC 解扩），partner 为同子载波的另一符号
        partner = np.arange(num_pilots)
        if dmrs_length == 2:
            syms = np.unique(sym)
            for a, b in zip(syms[0::2], syms[1::2]):
                ia, ib = np.flatnonzero(sym == a), np.flatnonzero(sym == b)
                if not np.array_equal(sc[ia], sc[ib]):
                    raise ValueError("双符号 DMRS 的两个符号导频位置不一致")
                partner[ia], partner[ib] = ib, ia

        p = pilots.reshape(num_tx * num_streams, num_pilots)
        valid = np.abs(p) > 0
        idx = np.tile(np.arange(num_pilots)[None, :, None], [len(p), 1, window])
        wgt = np.zeros([len(p), num_pilots, window], complex)
        noise_gain = np.zeros([len(p), num_pilots])
        for u in range(len(p)):
            safe = np.where(valid[u], p[u], 1)
            # 同一 CDM 组的其他端口及其（时域平均后的）导频比值 ρ
            rhos = []
            for v in range(len(p)):
                if v == u or not np.array_equal(valid[v], valid[u]):
                    continue
                rho = np.where(valid[u], p[v] / safe, 0)
                rho = (rho + rho[partner]) / 2
                if np.abs(rho).max() > 1e-6:
                    rhos.append(rho)
            for d in np.unique(sym[valid[u]]):
                on_sym = np.flatnonzero(valid[u] & (sym == d))
                for start in range(0, sc.max() + 1, prg_num_sc):
                    group = on_sym[(sc[on_sym] >= start) & (sc[on_sym] < start + prg_num_sc)]
                    for n in group:
                        dist = np.abs(sc[group] - sc[n])
                        taps = group[np.argsort(dist, kind="stable")[:window]]
                        x = (sc[taps] - sc[n]) / 12.0
                        cols = [np.ones_like(x), x]
                        for rho in rhos:
                            cols += [rho[taps], rho[taps] * x]
                        a = np.stack(cols, axis=1)
                        if len(taps) < a.shape[1]:          # 导频太少时退化为不含斜率的模型
                            a = a[:, 0::2]
                        coef = np.linalg.pinv(a)[0]
                        idx[u, n, :len(taps)] = taps
                        wgt[u, n, :len(taps)] = coef
                        # 时域平均使噪声方差减半
                        noise_gain[u, n] = np.sum(np.abs(coef) ** 2 / np.abs(safe[taps]) ** 2) \
                            / (2 if dmrs_length == 2 else 1)

        shape = [num_tx, num_streams, num_pilots]
        dev = self.device
        self._partner = torch.as_tensor(partner, device=dev)
        self._valid = torch.as_tensor(valid.reshape(shape), device=dev)
        self._pilots = torch.as_tensor(pilots, dtype=self.cdtype, device=dev)
        self._tap_idx = torch.as_tensor(idx.reshape(num_tx, num_streams, -1), device=dev)
        self._tap_wgt = torch.as_tensor(wgt.reshape(*shape, window), dtype=self.cdtype, device=dev)
        self._noise_gain = torch.as_tensor(noise_gain.reshape(shape), dtype=self.dtype, device=dev)
        self._window = window

    def estimate_at_pilot_locations(self, y_pilots: torch.Tensor, no: torch.Tensor):
        # y_pilots: [B, R, Ra, T, S, P]
        safe = torch.where(self._valid, self._pilots, torch.ones_like(self._pilots))
        z = torch.where(self._valid, y_pilots / safe, torch.zeros_like(y_pilots))
        if self._dmrs_length == 2:
            z = (z + z.index_select(-1, self._partner)) / 2
        lead = z.shape[:3]
        taps = z.gather(-1, self._tap_idx.expand(*lead, *self._tap_idx.shape))
        taps = taps.unflatten(-1, (-1, self._window))
        h_hat = (taps * self._tap_wgt).sum(-1)
        no = no.to(self.dtype)
        while no.dim() < 3:
            no = no.unsqueeze(-1)
        err_var = no[..., None, None, None] * self._noise_gain
        return h_hat, err_var.broadcast_to(h_hat.shape)
