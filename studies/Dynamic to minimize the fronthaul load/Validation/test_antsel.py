"""antsel.py 的验收检查。直接运行：python test_antsel.py

1. 4 AP 确定性例子：A = 4.2309544348，C = 8.7846348456，指定的天线级可行解 = 8.7846348456。
2. 可行候选数：M = 1, 2, 4, 6, 8 时 C 为 2, 14, 494, 18563, 735470，A 为 2。
3. 快速算法与直接穷举逐样本一致（M = 2，带复数相位的随机信道，12 组 beta/SNR；另加一个功率上限起作用的配置）。
4. M = 1 时 A、B、C 逐样本完全相等。
5. C >= B >= A 逐样本成立（M = 2, 4）。
"""

from itertools import combinations, product

import numpy as np

import antsel

BETAS = [0.05, 0.25, 0.8]
SNRS = [0, 10, 20, 30]
NOISE = [10 ** (-s / 10) for s in SNRS]
TOL = 1e-9


def cgauss(rng, shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def scale_links(g, ap_ue, beta):
    """服务链路 beta = 1，干扰链路 beta。"""
    h = g.copy()
    for n, v in enumerate(ap_ue):
        for u in range(g.shape[0]):
            if u != v:
                h[u, n] *= np.sqrt(beta)
    return h


def direct_se(h, ap_ue, subsets, K, P_total, noise_var):
    """不用查表，按定义逐 AP 重算 MRT 权值和 SINR。subsets[n] 是 AP n 的天线下标列表。"""
    U, N, M = h.shape
    a = np.zeros((U, U), complex)
    for n in range(N):
        S = list(subsets[n])
        if not S:
            continue
        p = P_total * len(S) / K
        hv = h[ap_ue[n], n, S]
        w = np.sqrt(p) * hv / np.linalg.norm(hv)
        for u in range(U):
            a[u, ap_ue[n]] += np.vdot(h[u, n, S], w)          # h^H w
    se = 0.0
    for u in range(U):
        sig = abs(a[u, u]) ** 2
        intf = sum(abs(a[u, v]) ** 2 for v in range(U) if v != u)
        se += np.log2(1 + sig / (noise_var + intf))
    return se


def feasible(subsets, ap_ue, U, K, P_total, P_max):
    if sum(len(s) for s in subsets) != K:
        return False
    if any(P_total * len(s) / K > P_max + 1e-12 for s in subsets):
        return False
    return all(any(len(subsets[n]) > 0 for n in range(len(ap_ue)) if ap_ue[n] == u) for u in range(U))


def direct_abc(h, ap_ue, K, P_total, P_max, noise_var):
    """直接穷举 A、B、C。"""
    U, N, M = h.shape
    val = lambda subs: direct_se(h, ap_ue, subs, K, P_total, noise_var)
    # A：整 AP 开或关
    a_best = -np.inf
    for on in product((0, 1), repeat=N):
        subs = [list(range(M)) if o else [] for o in on]
        if feasible(subs, ap_ue, U, K, P_total, P_max):
            a_best = max(a_best, val(subs))
    # B：穷举激活数，按服务链路 |h|^2 取最强
    b_best = -np.inf
    for k in product(range(M + 1), repeat=N):
        subs = [list(np.argsort(-np.abs(h[ap_ue[n], n]) ** 2, kind='stable')[:k[n]]) for n in range(N)]
        if feasible(subs, ap_ue, U, K, P_total, P_max):
            b_best = max(b_best, val(subs))
    # C：全部天线子集
    c_best = -np.inf
    for sel in combinations(range(N * M), K):
        subs = [[x % M for x in sel if x // M == n] for n in range(N)]
        if feasible(subs, ap_ue, U, K, P_total, P_max):
            c_best = max(c_best, val(subs))
    return np.array([a_best, b_best, c_best])


def test_deterministic():
    ap_ue = (0, 0, 1, 1)
    h = np.zeros((2, 4, 2), complex)
    for n, v in enumerate(ap_ue):
        h[v, n] = [1, 1]
        h[1 - v, n] = [1, 0]
    v, k = antsel.solve(h, ap_ue, K=4, P_total=2, P_max=1, betas=[1], noise_vars=[0.1])
    a, b, c = v[:, 0, 0]
    assert abs(a - 4.2309544348) < 1e-9, a
    assert abs(c - 8.7846348456) < 1e-9, c
    spec = direct_se(h, ap_ue, [[1], [1], [1], [1]], 4, 2, 0.1)
    assert abs(spec - 8.7846348456) < 1e-9, spec
    assert c >= spec - TOL
    d = direct_abc(h, ap_ue, 4, 2, 1, 0.1)
    assert np.allclose(v[:, 0, 0], d, atol=TOL), (v[:, 0, 0], d)
    print(f'deterministic: A = {a:.10f}, B = {b:.10f}, C = {c:.10f}, C counts = {k[2, 0, 0]}')


def test_counts():
    expect = {1: 2, 2: 14, 4: 494, 6: 18563, 8: 735470}
    for M, n in expect.items():
        idx, counts, a_pos, _ = antsel.candidates(3, M, 2 * M, (0, 0, 1), 2, 2.0, 1.0)
        assert len(idx) == n, (M, len(idx))
        assert len(a_pos) == 2, (M, len(a_pos))
        assert np.all(counts[:, 2] >= 1) and np.all(counts.sum(1) == 2 * M)
    print('candidate counts:', expect, 'A = 2 for all M')


def test_fast_vs_direct(M=2, samples=40):
    rng = np.random.default_rng(1)
    ap_ue = (0, 0, 1)
    worst = 0.0
    for _ in range(samples):
        g = cgauss(rng, (2, 3, M))
        v, _ = antsel.solve(g, ap_ue, 2 * M, 2, 1, BETAS, NOISE)
        for bi, beta in enumerate(BETAS):
            h = scale_links(g, ap_ue, beta)
            for si, nv in enumerate(NOISE):
                d = direct_abc(h, ap_ue, 2 * M, 2, 1, nv)
                worst = max(worst, np.max(np.abs(v[:, bi, si] - d)))
    assert worst < TOL, worst
    print(f'fast vs direct, M = {M}, {samples} samples x 12 settings: max |diff| = {worst:.2e}')


def test_power_cap_config(samples=20):
    """功率上限起作用的另一配置：4 AP、映射 (0, 1, 0, 1)、M = 2、K = 2、P_max = 1.5，只允许 k[n] <= 1。"""
    rng = np.random.default_rng(2)
    ap_ue = (0, 1, 0, 1)
    worst = 0.0
    for _ in range(samples):
        g = cgauss(rng, (2, 4, 2))
        v, k = antsel.solve(g, ap_ue, 2, 2, 1.5, [0.5], [0.1])
        assert np.all(k <= 1)
        d = direct_abc(scale_links(g, ap_ue, 0.5), ap_ue, 2, 2, 1.5, 0.1)
        # 该配置下没有整 AP 可行解，A 为 -inf
        assert np.isneginf(v[0, 0, 0]) and np.isneginf(d[0])
        worst = max(worst, np.max(np.abs(v[1:, 0, 0] - d[1:])))
    assert worst < TOL, worst
    print(f'power-cap config: max |diff| = {worst:.2e}')


def test_m1_equal(samples=300):
    rng = np.random.default_rng(3)
    for _ in range(samples):
        g = cgauss(rng, (2, 3, 1))
        v, _ = antsel.solve(g, (0, 0, 1), 2, 2, 1, BETAS, NOISE)
        assert np.array_equal(v[0], v[1]) and np.array_equal(v[1], v[2])
    print(f'M = 1: A, B, C bitwise equal on {samples} samples x 12 settings')


def test_order(samples=200):
    rng = np.random.default_rng(4)
    for M in (2, 4):
        worst = 0.0
        for _ in range(samples):
            g = cgauss(rng, (2, 3, M))
            v, _ = antsel.solve(g, (0, 0, 1), 2 * M, 2, 1, BETAS, NOISE)
            worst = max(worst, np.max(v[0] - v[1]), np.max(v[1] - v[2]))
        assert worst < TOL, (M, worst)
        print(f'M = {M}: C >= B >= A on {samples} samples, max violation {worst:.2e}')


if __name__ == '__main__':
    test_deterministic()
    test_counts()
    test_fast_vs_direct()
    test_power_cap_config()
    test_m1_equal()
    test_order()
    print('all checks passed')
