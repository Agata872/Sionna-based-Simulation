"""第二轮代码的验收检查。直接运行：python test_round2.py

1. 非理想 CSI 的 MRT：h_est = h 时与理想 CSI 一致；M = 2 时快速算法与直接穷举逐样本一致
   （按估计信道选择与计算权值，按真实信道评估）。
2. 最强 UE 映射：映射与逐 AP 手算一致；映射改变的样本上，快速算法与直接穷举一致。
3. ZF / RZF：M = 2 时快速算法与逐子集显式求 W、显式检查功率约束的直接实现一致（两种功率约束，理想与非理想 CSI）；
   M = 1、'ant' 约束与 [10] 式 (18) 一致；理想 CSI 的 ZF 没有多用户干扰。
"""

from itertools import combinations

import numpy as np

import antsel
import antsel_zf
from test_antsel import BETAS, NOISE, TOL, cgauss, direct_abc, feasible, scale_links

from itertools import product


def direct_se_mis(h, hw, ap_ue, subsets, K, P_total, noise_var):
    """权值由 hw 计算，接收经过 h。"""
    U, N, M = h.shape
    a = np.zeros((U, U), complex)
    for n in range(N):
        S = list(subsets[n])
        if not S:
            continue
        p = P_total * len(S) / K
        hv = hw[ap_ue[n], n, S]
        w = np.sqrt(p) * hv / np.linalg.norm(hv)
        for u in range(U):
            a[u, ap_ue[n]] += np.vdot(h[u, n, S], w)
    return sum(np.log2(1 + abs(a[u, u]) ** 2 / (noise_var + sum(abs(a[u, v]) ** 2 for v in range(U) if v != u)))
               for u in range(U))


def direct_abc_mis(h, he, ap_ue, K, P_total, P_max, noise_var):
    """按估计 sum SE 选择，返回所选方案的真实 sum SE。"""
    U, N, M = h.shape
    est = lambda s: direct_se_mis(he, he, ap_ue, s, K, P_total, noise_var)
    true = lambda s: direct_se_mis(h, he, ap_ue, s, K, P_total, noise_var)
    out = []
    pools = [
        [[list(range(M)) if o else [] for o in on] for on in product((0, 1), repeat=N)],
        [[list(np.argsort(-np.abs(he[ap_ue[n], n]) ** 2, kind='stable')[:k[n]]) for n in range(N)]
         for k in product(range(M + 1), repeat=N)],
        [[[x % M for x in sel if x // M == n] for n in range(N)] for sel in combinations(range(N * M), K)],
    ]
    for pool in pools:
        cand = [s for s in pool if feasible(s, ap_ue, U, K, P_total, P_max)]
        best = max(cand, key=est)
        out.append(true(best))
    return np.array(out)


def test_mrt_csi():
    rng = np.random.default_rng(11)
    ap_ue = (0, 0, 1)
    worst_same, worst_direct = 0.0, 0.0
    for _ in range(30):
        g = cgauss(rng, (2, 3, 2))
        v0, _ = antsel.solve(g, ap_ue, 4, 2, 1, BETAS, NOISE)
        v1, _ = antsel.solve(g, ap_ue, 4, 2, 1, BETAS, NOISE, h_est=g.copy())
        worst_same = max(worst_same, np.max(np.abs(v0 - v1)))
        ge = g + np.sqrt(0.1) * cgauss(rng, g.shape)
        v, _ = antsel.solve(g, ap_ue, 4, 2, 1, BETAS, NOISE, h_est=ge)
        for bi, beta in enumerate(BETAS):
            h, he = scale_links(g, ap_ue, beta), scale_links(ge, ap_ue, beta)
            for si, nv in enumerate(NOISE):
                d = direct_abc_mis(h, he, ap_ue, 4, 2, 1, nv)
                worst_direct = max(worst_direct, np.max(np.abs(v[:, bi, si] - d)))
    assert worst_same < TOL and worst_direct < TOL, (worst_same, worst_direct)
    print(f'MRT imperfect CSI: h_est = h max diff {worst_same:.1e}; fast vs direct (M = 2) max diff {worst_direct:.1e}')


def test_mapping():
    rng = np.random.default_rng(12)
    changed, worst = 0, 0.0
    for _ in range(400):
        g = cgauss(rng, (2, 3, 1 + rng.integers(2)))
        h = scale_links(g, (0, 0, 1), 0.8)
        mp = antsel.strongest_mapping(h)
        manual = tuple(int(np.argmax([np.sum(np.abs(h[u, n]) ** 2) for u in range(2)])) for n in range(3))
        assert mp == manual
        if mp == (0, 0, 1) or len(set(mp)) < 2:
            continue
        changed += 1
        M = h.shape[2]
        v, _ = antsel.solve(h, mp, 2 * M, 2, 1, [1.0], NOISE)
        for si, nv in enumerate(NOISE):
            worst = max(worst, np.max(np.abs(v[:, 0, si] - direct_abc(h, mp, 2 * M, 2, 1, nv))))
    assert changed > 10 and worst < TOL, (changed, worst)
    print(f'strongest-UE mapping: {changed} changed-and-feasible samples checked, max diff {worst:.1e}')


def direct_zf(h, he, K, P_total, P_max, noise_var, kind, variant):
    """逐子集显式求 W，显式检查约束，返回 (A, C) 的真实 sum SE（按估计 sum SE 选择）。"""
    U, N, M = h.shape
    H, He = h.reshape(U, -1), he.reshape(U, -1)
    alpha = 0.0 if kind == 'zf' else U * noise_var / P_total

    def se_pair(S):
        Hs, Hes = H[:, S], He[:, S]
        W = Hes.conj().T @ np.linalg.inv(Hes @ Hes.conj().T + alpha * np.eye(U))     # [K, U]
        pw = np.sum(np.abs(W) ** 2, axis=1)
        if variant == 'ant':
            rho = min((P_max / M) / pw.max(), P_total / pw.sum())
        else:
            ap = np.zeros(N)
            for k, m in enumerate(S):
                ap[m // M] += pw[k]
            rho = min(P_max / ap.max(), P_total / pw.sum())
        # 检查约束确实满足
        assert pw.sum() * rho <= P_total * (1 + 1e-9)
        res = []
        for Hx in (Hes, Hs):
            E = Hx @ W
            res.append(sum(np.log2(1 + rho * abs(E[u, u]) ** 2 /
                                   (rho * sum(abs(E[u, v]) ** 2 for v in range(U) if v != u) + noise_var))
                           for u in range(U)))
        return res

    out = []
    allsets = [list(c) for c in combinations(range(N * M), K)]
    apsets = [S for S in allsets if all(sum(1 for m in S if m // M == n) in (0, M) for n in range(N))]
    for pool in (apsets, allsets):
        vals = [se_pair(S) for S in pool]
        out.append(max(vals, key=lambda x: x[0])[1])
    return np.array(out)


def test_zf():
    rng = np.random.default_rng(13)
    worst = 0.0
    for _ in range(15):
        g = cgauss(rng, (2, 3, 2))
        h = scale_links(g, (0, 0, 1), 0.25)
        he = h + np.sqrt(0.05) * cgauss(rng, h.shape) * np.sqrt(np.where(np.abs(scale_links(np.ones_like(g), (0, 0, 1), 0.25)) > 0.9, 1, 0.25))
        for kind in antsel_zf.KINDS:
            for est in (None, he):
                v, _ = antsel_zf.solve_zf(h, 4, 2, 1, NOISE, kind, h_est=est)
                for vi, var in enumerate(antsel_zf.VARIANTS):
                    for si, nv in enumerate(NOISE):
                        d = direct_zf(h, h if est is None else est, 4, 2, 1, nv, kind, var)
                        worst = max(worst, np.max(np.abs(v[vi, :, si] - d)))
    assert worst < 1e-8, worst
    print(f'ZF/RZF fast vs direct (M = 2, both power rules, perfect and imperfect CSI): max diff {worst:.1e}')


def test_zf_paper_eq18_and_no_interference():
    rng = np.random.default_rng(14)
    for _ in range(50):
        h = scale_links(cgauss(rng, (2, 3, 1)), (0, 0, 1), 0.8)
        v, _ = antsel_zf.solve_zf(h, 2, 2, 1, NOISE, 'zf')
        best = -np.inf
        for S in combinations(range(3), 2):
            Hs = h.reshape(2, 3)[:, list(S)]
            W = Hs.conj().T @ np.linalg.inv(Hs @ Hs.conj().T)
            p = 1.0 / np.max(np.sum(np.abs(W) ** 2, axis=1))        # [10] 式 (18)，P_max = 1
            best = max(best, 2 * np.log2(1 + p / NOISE[2]))
        assert abs(v[0, 1, 2] - best) < 1e-9
        # 理想 CSI 下 ZF 的 sum SE = U log2(1 + rho / noise)，与 SNR 的关系只经过 rho / noise
    print('ZF M = 1 per-antenna rule matches [10] eq. (18) on 50 samples')


if __name__ == '__main__':
    test_mrt_csi()
    test_mapping()
    test_zf()
    test_zf_paper_eq18_and_no_interference()
    print('all round-2 checks passed')
