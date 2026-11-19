"""逐 PRB 的 AP 级与天线级选择（MRT），单 PRB、固定 AP–UE 映射。

记号（与任务说明一致）
  h[u, n, m]   UE u 与 AP n 第 m 根天线之间的信道，接收项为 h^H w。
  ap_ue[n]     AP n 固定服务的 UE。
  S[n]         AP n 激活的天线子集，用 M 位掩码表示。
  p[n] = P_total * |S[n]| / K，要求 p[n] <= P_max。
  w[n] = sqrt(p[n]) * h[ap_ue[n], n, S[n]] / ||h[ap_ue[n], n, S[n]]||，空集为 0。
  a[u, v] = sum_{n: ap_ue[n] = v, S[n] 非空} h[u, n, S[n]]^H w[n]   （先加复数幅度）
  D[u] = |a[u, u]|^2，I[u] = sum_{v != u} |a[u, v]|^2
  SINR[u] = D[u] / (I[u] + noise_var)，sum_SE = sum_u log2(1 + SINR[u])

快速实现
  每个 AP 预先算出 2^M 个掩码对每个 UE 的复幅度 c[n][mask, u]，候选 (S[0], ..., S[N-1]) 的
  a 由这些表相加得到，与直接按子集重算 MRT 等价（test_antsel.py 逐样本核对）。
  干扰链路 h[u, n] = sqrt(beta) * g[u, n]（ap_ue[n] != u），服务链路 h = g。
  a[u, v]（v != u）只含干扰链路，所以 I(beta) = beta * I(1)，D 与 beta 无关。
  同一样本只枚举一次，12 组 (beta, SNR) 共用 D、I，但每组参数分别取 argmax。

可行性只取决于每个 AP 的激活数 k[n]：sum k = K，p[n] <= P_max，每个 UE 至少有一个服务 AP 的 k > 0。
"""

from functools import lru_cache
from itertools import product

import numpy as np

TOL_POWER = 1e-12


def popcount(x):
    return bin(int(x)).count('1')


def mask_bits(M):
    """[2^M, M] 的 0/1 矩阵，第 s 行是掩码 s 的各位。"""
    s = np.arange(2 ** M)
    return ((s[:, None] >> np.arange(M)[None, :]) & 1).astype(float)


def feasible_counts(N, M, K, ap_ue, num_ue, P_total, P_max):
    """所有可行的激活数向量 (k[0], ..., k[N-1])。"""
    out = []
    for k in product(range(M + 1), repeat=N):
        if sum(k) != K:
            continue
        if any(P_total * kn / K > P_max + TOL_POWER for kn in k):
            continue
        if not all(any(k[n] > 0 for n in range(N) if ap_ue[n] == u) for u in range(num_ue)):
            continue
        out.append(k)
    return np.array(out, dtype=np.int64)


@lru_cache(maxsize=None)
def candidates(N, M, K, ap_ue, num_ue, P_total, P_max):
    """返回 (C 的候选掩码 [Nc, N], 每个候选的激活数 [Nc, N], A 候选在 C 中的下标, 可行激活数向量)。"""
    ap_ue = tuple(ap_ue)
    by_pc = {k: np.array([s for s in range(2 ** M) if popcount(s) == k], dtype=np.int64)
             for k in range(M + 1)}
    kvecs = feasible_counts(N, M, K, ap_ue, num_ue, P_total, P_max)
    blocks = []
    for k in kvecs:
        grids = np.meshgrid(*[by_pc[int(kn)] for kn in k], indexing='ij')
        blocks.append(np.stack([g.ravel() for g in grids], axis=1))
    idx = np.concatenate(blocks, axis=0) if blocks else np.zeros((0, N), dtype=np.int64)
    full = 2 ** M - 1
    pc = np.array([popcount(s) for s in range(2 ** M)], dtype=np.int64)
    is_a = np.all((idx == 0) | (idx == full), axis=1)
    return idx, pc[idx], np.flatnonzero(is_a), kvecs


def ap_tables(h, ap_ue, K, P_total):
    """每个 AP 的复幅度表 c[n] [2^M, U]：AP n 激活掩码 s 时 UE u 收到的 h[u,n,S]^H w[n]。"""
    num_ue, N, M = h.shape
    bits = mask_bits(M)
    size = bits.sum(1)
    p = P_total * size / K
    tabs = []
    for n in range(N):
        hv = h[ap_ue[n], n]                                   # [M] 服务链路
        norm2 = bits @ (np.abs(hv) ** 2)                      # [2^M]
        inner = bits @ (np.conj(h[:, n, :]) * hv[None, :]).T  # [2^M, U]，h_u,S^H h_v,S
        scale = np.zeros_like(norm2)
        nz = size > 0
        scale[nz] = np.sqrt(p[nz] / norm2[nz])
        tabs.append(inner * scale[:, None])
    return tabs


def amplitudes(tabs, idx, ap_ue, num_ue):
    """候选 idx [Nc, N] → a [Nc, U, U]，a[:, u, v] 是 UE u 收到的发给 UE v 的复幅度和。"""
    a = np.zeros((idx.shape[0], num_ue, num_ue), complex)
    for n, c in enumerate(tabs):
        a[:, :, ap_ue[n]] += c[idx[:, n]]
    return a


def signal_interference(a):
    """a [Nc, U, U] → D [Nc, U]、I [Nc, U]。"""
    pw = np.abs(a) ** 2
    d = np.einsum('cuu->cu', pw)
    return d, pw.sum(-1) - d


def sum_se(d, i, betas, noise_vars):
    """D、I（beta = 1 时的干扰）→ sum SE [Nc, nB, nS]。"""
    b = np.asarray(betas, float)[None, None, :, None]
    s2 = np.asarray(noise_vars, float)[None, None, None, :]
    sinr = d[:, :, None, None] / (b * i[:, :, None, None] + s2)
    return np.log2(1.0 + sinr).sum(1)


def best(tabs, idx, ap_ue, num_ue, betas, noise_vars, chunk=100_000):
    """在候选 idx 中逐 (beta, SNR) 取最大 sum SE。返回 (最大值 [nB, nS], 最优候选下标 [nB, nS])。"""
    best_v = np.full((len(betas), len(noise_vars)), -np.inf)
    best_i = np.zeros((len(betas), len(noise_vars)), dtype=np.int64)
    for start in range(0, idx.shape[0], chunk):
        part = idx[start:start + chunk]
        d, i = signal_interference(amplitudes(tabs, part, ap_ue, num_ue))
        se = sum_se(d, i, betas, noise_vars)
        j = np.argmax(se, axis=0)
        v = np.take_along_axis(se, j[None], axis=0)[0]
        upd = v > best_v
        best_v[upd] = v[upd]
        best_i[upd] = j[upd] + start
    return best_v, best_i


def b_candidates(h, ap_ue, kvecs):
    """方案 B：每个激活数向量下，各 AP 取对其服务 UE |h|^2 最大的 k[n] 根天线。返回 [len(kvecs), N] 掩码。"""
    num_ue, N, M = h.shape
    top = np.zeros((N, M + 1), dtype=np.int64)
    for n in range(N):
        order = np.argsort(-np.abs(h[ap_ue[n], n]) ** 2, kind='stable')
        for k in range(1, M + 1):
            top[n, k] = top[n, k - 1] | (1 << int(order[k - 1]))
    return top[np.arange(N)[None, :], kvecs]


def solve(h, ap_ue, K, P_total, P_max, betas, noise_vars):
    """一个信道样本 → A、B、C 的 sum SE [3, nB, nS] 及各自的最优激活数 [3, nB, nS, N]。

    h 中干扰链路按 beta = 1 给出，betas 只缩放干扰项。要直接用给定信道（如确定性例子），传 betas = [1]。
    """
    num_ue, N, M = h.shape
    ap_ue = tuple(int(x) for x in ap_ue)
    idx, counts, a_pos, kvecs = candidates(N, M, K, ap_ue, num_ue, float(P_total), float(P_max))
    tabs = ap_tables(h, ap_ue, K, P_total)
    idx_b = b_candidates(h, ap_ue, kvecs)
    out_v, out_k = [], []
    for cand, cnt in ((idx[a_pos], counts[a_pos]), (idx_b, kvecs), (idx, counts)):
        v, j = best(tabs, cand, ap_ue, num_ue, betas, noise_vars)
        out_v.append(v)
        out_k.append(cnt[j] if len(cand) else np.full(v.shape + (N,), -1, dtype=np.int64))
    return np.stack(out_v), np.stack(out_k)                    # 无可行候选时 sum SE = -inf，激活数 = -1
