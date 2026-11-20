"""ZF / RZF 参照：活跃天线数 K = 2M，所有活跃天线联合服务全部 UE（[10] 的 ZF 规则：ZF 的 AP 服务所有 UE）。

  W = H_S^H (H_S H_S^H + alpha I)^{-1}，ZF 时 alpha = 0，RZF 时 alpha = U * noise_var / P_total。
  功率仿照 [10] 式 (18)：所有 UE 共用一个缩放因子 rho，取满足约束的最大值（由最吃紧的约束决定）。
    'ant'：每根活跃天线 rho * ||w_m||^2 <= P_max / M（M = 1 时就是 [10] 式 (18)），且总功率 <= P_total
    'ap' ：每个 AP rho * sum_{m in AP n} ||w_m||^2 <= P_max，且总功率 <= P_total
  E = H_S W，SINR_u = rho |E_uu|^2 / (rho sum_{v != u} |E_uv|^2 + noise_var)，sum_SE = sum_u log2(1 + SINR_u)。
  ZF 在理想 CSI 下 E = I，没有多用户干扰。

  非理想 CSI：W 与 rho 只用估计信道计算，候选选择按估计信道下的 sum SE，返回值按真实信道评估。

  候选：A = 任选 K/M 个整 AP（含 AP 1 + AP 2，因为 ZF 不需要 AP–UE 映射）；C = 任选 K 根天线。
  天线全局下标 = n * M + m（AP n 的第 m 根）。
"""

from functools import lru_cache
from itertools import combinations

import numpy as np

KINDS = ('zf', 'rzf')
VARIANTS = ('ant', 'ap')


@lru_cache(maxsize=None)
def zf_candidates(N, M, K):
    """返回 (C 的候选 [Nc, N*M] 布尔矩阵, A 候选在 C 中的下标)。"""
    nm = N * M
    comb = np.array(list(combinations(range(nm), K)), dtype=np.int64)
    bits = np.zeros((len(comb), nm), dtype=bool)
    bits[np.arange(len(comb))[:, None], comb] = True
    per_ap = bits.reshape(len(comb), N, M).sum(2)
    is_a = np.all((per_ap == 0) | (per_ap == M), axis=1)
    return bits, np.flatnonzero(is_a)


def _outer(x, y):
    """x, y [U, NM] → [NM, U, U]，第 m 个为 x_m y_m^H。"""
    return np.einsum('im,jm->mij', x, np.conj(y))


def _inv2(m):
    """批量 2x2 矩阵求逆（闭式）。"""
    a, b, c, d = m[:, 0, 0], m[:, 0, 1], m[:, 1, 0], m[:, 1, 1]
    det = a * d - b * c
    out = np.empty_like(m)
    out[:, 0, 0], out[:, 0, 1], out[:, 1, 0], out[:, 1, 1] = d / det, -b / det, -c / det, a / det
    return out


def _evaluate(bits, H, He, M, N, P_total, P_max, noise_vars, kind):
    """bits [Nc, NM] → est、true 的 sum SE，各为 [2 variants, Nc, nS]。"""
    U = H.shape[0]
    nc = len(bits)
    b = bits.astype(float)
    oe = _outer(He, He)                                                  # [NM, U, U]
    ge = (b @ oe.reshape(-1, U * U)).reshape(-1, U, U)                   # 估计 Gram
    gx = ge if He is H else (b @ _outer(H, He).reshape(-1, U * U)).reshape(-1, U, U)  # H_S He_S^H
    oe_t = np.swapaxes(oe, 1, 2).reshape(-1, U * U).T                    # [(i,j), m] = oe[m, j, i]
    inv = _inv2 if U == 2 else np.linalg.inv
    eye = np.eye(U)
    ns = len(noise_vars)
    est = np.zeros((2, nc, ns))
    true = np.zeros((2, nc, ns))
    alphas = [0.0] * ns if kind == 'zf' else [U * s2 / P_total for s2 in noise_vars]
    cache = {}
    for si, (s2, alpha) in enumerate(zip(noise_vars, alphas)):
        if alpha not in cache:
            a = inv(ge + alpha * eye)                                        # [Nc, U, U]，Hermitian
            q = np.real((a @ a).reshape(nc, U * U) @ oe_t) * bits            # ||w_m||^2 = h_m^H A^2 h_m，仅活跃天线
            q_tot = q.sum(1)
            q_ap = q.reshape(nc, N, M).sum(2)
            rho_ant = np.minimum((P_max / M) / q.max(1), P_total / q_tot)
            rho_ap = np.minimum(P_max / q_ap.max(1), P_total / q_tot)
            pw = []
            for e in (ge @ a, gx @ a):
                p2 = np.abs(e) ** 2
                d = np.einsum('cuu->cu', p2)
                pw.append((d, p2.sum(-1) - d))
            cache[alpha] = (np.stack([rho_ant, rho_ap])[:, :, None], pw)
        rho, pw = cache[alpha]
        for out, (d, i) in zip((est, true), pw):
            out[:, :, si] = np.log2(1.0 + rho * d[None] / (rho * i[None] + s2)).sum(-1)
    return est, true


def solve_zf(h, K, P_total, P_max, noise_vars, kind, h_est=None, chunk=50_000):
    """一个信道样本（显式信道 h [U, N, M]，已含 beta）→ sum SE [2 variants, 2 schemes (A, C), nS]
    及最优候选的每 AP 激活数 [2, 2, nS, N]。"""
    U, N, M = h.shape
    bits, a_pos = zf_candidates(N, M, K)
    H = h.reshape(U, N * M)
    He = H if h_est is None else h_est.reshape(U, N * M)
    ns = len(noise_vars)
    out_v = np.zeros((2, 2, ns))
    out_k = np.zeros((2, 2, ns, N), dtype=np.int64)
    for ci, cand in enumerate((bits[a_pos], bits)):
        best_est = np.full((2, ns), -np.inf)
        best_true = np.zeros((2, ns))
        best_k = np.zeros((2, ns, N), dtype=np.int64)
        for start in range(0, len(cand), chunk):
            part = cand[start:start + chunk]
            est, true = _evaluate(part, H, He, M, N, P_total, P_max, noise_vars, kind)
            j = np.argmax(est, axis=1)                                       # [2, nS]
            v = np.take_along_axis(est, j[:, None, :], axis=1)[:, 0]
            upd = v > best_est
            best_est[upd] = v[upd]
            tv = np.take_along_axis(true, j[:, None, :], axis=1)[:, 0]
            best_true[upd] = tv[upd]
            kk = part[j].reshape(2, ns, N, M).sum(-1)
            best_k[upd] = kk[upd]
        out_v[:, ci] = best_true
        out_k[:, ci] = best_k
    return out_v, out_k
