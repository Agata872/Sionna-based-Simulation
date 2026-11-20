"""第二轮受控 ZF：同一预编码器下的 AP 级与天线级选择（cell_free_round2_zf_simulation.txt）。

记号
  H 的第 u 行是 UE u 的下行信道行，接收信号 = H @ W。由第一轮的列信道 h[u, n, m]（接收项 h^H w）组装：
  H[u, n*M + m] = conj(h[u, n, m])。估计信道 H_hat 由 h_hat 同样组装。
  对候选天线集合 S（恰好 K 根）：
    G = H_hat_S H_hat_S^H，W0 = H_hat_S^H G^{-1}
    q[n] = W0 中属于 AP n 的行的功率和，Q = sum_n q[n]
    c2 = min(P_total / Q, P_max / max_n q[n])，W = sqrt(c2) W0，P_used = c2 Q
  估计 sum SE = U log2(1 + c2 / noise)（H_hat_S W = sqrt(c2) I），对 c2 严格递增，所以选择与 SNR 无关。
  真实：F = H_S W，SINR_u = |F_uu|^2 / (noise + sum_{v != u} |F_uv|^2)。

方案（全部是集中式 ZF，所有活跃天线服务两个 UE）
  A      恰好 K/M 个完整 AP
  B      穷举各 AP 活跃数 k[n]（sum k = K），每个 AP 取估计 sum_u |H_hat[u, m]|^2 最大的 k[n] 根（并列按天线编号）
  C      全部 K 子集
  Ceq    同实际发射功率：P_ref = P_used(A)，c2 = P_ref / Q，只保留所有 AP 都满足 c2 q[n] <= P_max 的 C 候选
  A_AP3、C_AP3   匹配 MRT 候选集合的对照：在 A、C 中要求 AP 3 至少一根天线激活
  选择一律按估计 sum SE（即按 c2）取最大；并列时取全局候选编号（字典序）最小者，各方案规则相同。
"""

from functools import lru_cache
from itertools import product

import numpy as np

import antsel_zf

SCHEMES = ('A', 'B', 'C', 'Ceq', 'A_AP3', 'C_AP3')
RANK_TOL = 1e-12          # det(G) / (tr(G)/2)^2 低于此值视为秩不足
POWER_TOL = 1e-12


@lru_cache(maxsize=None)
def cand(N, M, K, ap_required=None):
    """C 的候选（布尔 [Nc, NM]）及各子集的下标。ap_required 为对照中必须激活的 AP（0 起）。"""
    bits, a_pos = antsel_zf.zf_candidates(N, M, K)
    nm = N * M
    keys = bits.astype(np.int64) @ (np.int64(1) << np.arange(nm, dtype=np.int64))
    order = np.argsort(keys, kind='stable')
    per_ap = bits.reshape(len(bits), N, M).sum(2)
    need = ap_required if ap_required is not None else N - 1
    ap_ok = per_ap[:, need] > 0
    kvecs = np.array([k for k in product(range(M + 1), repeat=N) if sum(k) == K], dtype=np.int64)
    return {'bits': bits, 'a_pos': a_pos, 'keys': keys, 'order': order, 'keys_sorted': keys[order],
            'per_ap': per_ap, 'c_ap3': np.flatnonzero(ap_ok), 'a_ap3': a_pos[ap_ok[a_pos]], 'kvecs': kvecs}


def assemble(h):
    """列信道 h [U, N, M] → 行信道矩阵 H [U, N*M]，H[u, n*M+m] = conj(h[u, n, m])。"""
    U = h.shape[0]
    return np.conj(h).reshape(U, -1)


def _outer(x):
    return np.einsum('im,jm->mij', x, np.conj(x))


def score(bits, He, N, M, chunk=100_000):
    """全部候选的 Q [Nc]、q_ap [Nc, N]、valid [Nc]（G 满秩）。He [U, NM]。"""
    U = He.shape[0]
    nc = len(bits)
    oe = _outer(He)                                                   # [NM, U, U]，第 m 个为 x_m x_m^H
    oe_flat = oe.reshape(-1, U * U)
    oe_t = np.swapaxes(oe, 1, 2).reshape(-1, U * U).T
    Q = np.empty(nc)
    qap = np.empty((nc, N))
    valid = np.empty(nc, dtype=bool)
    for s in range(0, nc, chunk):
        b = bits[s:s + chunk].astype(float)
        g = (b @ oe_flat).reshape(-1, U, U)
        if U == 2:
            det = np.real(g[:, 0, 0] * g[:, 1, 1] - g[:, 0, 1] * g[:, 1, 0])
            tr = np.real(g[:, 0, 0] + g[:, 1, 1])
            ok = det > RANK_TOL * (tr / 2) ** 2
            a = antsel_zf._inv2(np.where(ok[:, None, None], g, np.eye(U)))
        else:
            ev = np.linalg.eigvalsh(g)
            ok = ev[:, 0] > RANK_TOL * ev[:, -1]
            a = np.linalg.inv(np.where(ok[:, None, None], g, np.eye(U)))
        q = np.real((a @ a).reshape(len(b), U * U) @ oe_t) * b       # ||W0 第 m 行||^2 = x_m^H G^{-2} x_m
        qa = q.reshape(len(b), N, M).sum(2)
        qap[s:s + chunk] = qa
        Q[s:s + chunk] = qa.sum(1)
        valid[s:s + chunk] = ok
    return Q, qap, valid


def b_indices(He, c, N, M):
    """方案 B 的候选在 C 列表中的下标（升序、去重）。"""
    U = He.shape[0]
    strength = np.sum(np.abs(He) ** 2, axis=0).reshape(N, M)
    top = np.zeros((N, M + 1), dtype=np.int64)
    for n in range(N):
        order = np.argsort(-strength[n], kind='stable')              # 并列时编号小的在前
        for k in range(1, M + 1):
            top[n, k] = top[n, k - 1] | (np.int64(1) << np.int64(n * M + order[k - 1]))
    keys = np.zeros(len(c['kvecs']), dtype=np.int64)
    for n in range(N):
        keys |= top[n, c['kvecs'][:, n]]
    pos = np.searchsorted(c['keys_sorted'], keys)
    assert np.all(c['keys_sorted'][pos] == keys)
    return np.unique(c['order'][pos])


def _argmax_in(values, subset):
    """在 subset（升序下标）中取最大；并列取最小下标。不可行处 values 为 -inf。"""
    j = subset[np.argmax(values[subset])]
    return int(j) if np.isfinite(values[j]) else -1


def select(H, He, N, M, K, P_total, P_max):
    """返回 {方案: (候选下标, c2)}，以及全部候选的诊断量。"""
    c = cand(N, M, K)
    Q, qap, valid = score(c['bits'], He, N, M)
    maxq = qap.max(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        c2 = np.where(valid, np.minimum(P_total / Q, P_max / maxq), -np.inf)
    win = {}
    for name, sub in (('A', c['a_pos']), ('B', b_indices(He, c, N, M)), ('C', np.arange(len(Q))),
                      ('A_AP3', c['a_ap3']), ('C_AP3', c['c_ap3'])):
        j = _argmax_in(c2, sub)
        win[name] = (j, c2[j] if j >= 0 else np.nan)
    ja, c2a = win['A']
    p_ref = c2a * Q[ja]
    with np.errstate(divide='ignore', invalid='ignore'):
        c2e = np.where(valid, p_ref / Q, -np.inf)
    feas = valid & (c2e * maxq <= P_max * (1 + POWER_TOL))
    c2e = np.where(feas, c2e, -np.inf)
    je = _argmax_in(c2e, np.arange(len(Q)))
    win['Ceq'] = (je, c2e[je])
    return win, {'Q': Q, 'qap': qap, 'valid': valid, 'c2': c2, 'p_ref': p_ref, 'n_invalid': int((~valid).sum()),
                 'n_ceq_feasible': int(feas.sum())}


def evaluate(H, He, mask, c2, N, M, noise_vars):
    """对一个胜出候选显式求 W0（线性求解），返回真实/估计的逐 UE SINR [nS, U]、每 AP 功率 [N]、
    估计信道上的非对角残差（相对）、显式 q 与快速 q 的核对用量。"""
    S = np.flatnonzero(mask)
    Hs, Hes = H[:, S], He[:, S]
    U = H.shape[0]
    g = Hes @ Hes.conj().T
    W0 = Hes.conj().T @ np.linalg.solve(g, np.eye(U))                # [K, U]
    row_pw = np.sum(np.abs(W0) ** 2, axis=1)
    q_ap = np.zeros(N)
    np.add.at(q_ap, S // M, row_pw)
    W = np.sqrt(c2) * W0
    out = {}
    for tag, Hx in (('true', Hs), ('est', Hes)):
        F = Hx @ W
        pw = np.abs(F) ** 2
        d = np.diag(pw)
        i = pw.sum(1) - d
        out[f'sinr_{tag}'] = d[None, :] / (i[None, :] + np.asarray(noise_vars)[:, None])   # [nS, U]
        out[f'intf_{tag}'] = i                                       # 各 UE 的多用户干扰功率（与噪声无关）
    Fe = Hes @ W0
    off = Fe - np.diag(np.diag(Fe))
    out['resid_est'] = float(np.max(np.abs(off)) / np.max(np.abs(np.diag(Fe))))
    out['q_ap'] = q_ap
    out['p_ap'] = c2 * q_ap
    out['p_used'] = float(c2 * q_ap.sum())
    out['n_active'] = len(S)
    out['counts'] = np.bincount(S // M, minlength=N)
    return out
