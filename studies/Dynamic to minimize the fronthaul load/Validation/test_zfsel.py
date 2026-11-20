"""第二轮受控 ZF 的验收（说明第六节）。直接运行：python test_zfsel.py

1. 对角矩阵验算：H_hat = H = diag(1, 2)、两根天线各属一个 AP、noise = 0.1 → q = [1, 0.25]、c2 = 1、P_used = 1.25、
   单 AP 上限先生效、sum SE = 6.9188632373。
2. 误差模型：g_hat = (1-eps) g + sqrt(eps(1-eps)) e 的集合 NMSE ≈ eps，E|g_hat|^2 ≈ 1 - eps。
3. M = 1：完美与不完美 CSI 下 A、B、C、Ceq 的选优目标与实际 SE 相等。
4. 每个样本估计 SE 满足 C >= B >= A、C >= Ceq >= A，完美 CSI 下实际 SE 也满足；固定天线数、功率约束、
   Ceq 功率一致、估计信道上 ZF 非对角残差、显式求解与快速 q 一致。
5. M = 2 复数随机样本：逐候选与直接伪逆一致；四种方案的胜出集合与真实 SE 与直接穷举一致。
6. B 的并列规则：强度相同时按天线编号取小者。
7. 复共轭约定：H = conj(h) 与试跑的 H = h 结果逐样本相同（ZF 对整体共轭不变），且与试跑 ZFap_A/ZFap_C 一致。
"""

from itertools import combinations, product

import numpy as np

import antsel_zf
import zfsel
from run_round1 import NOISE
from run_round2 import scale_links
from run_zf_controlled import CSI, SCH, g_hat, work

P_TOTAL, P_MAX = 2.0, 1.0


def cgauss(rng, shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def test_diagonal():
    H = np.diag([1.0, 2.0]).astype(complex)
    win, dg = zfsel.select(H, H, 2, 1, 2, P_TOTAL, P_MAX)
    j, c2 = win['C']
    assert np.allclose(dg['qap'][j], [1.0, 0.25]) and abs(c2 - 1.0) < 1e-12
    ev = zfsel.evaluate(H, H, zfsel.cand(2, 1, 2)['bits'][j], c2, 2, 1, [0.1])
    se = np.log2(1 + ev['sinr_true']).sum()
    assert abs(ev['p_used'] - 1.25) < 1e-12 and abs(se - 6.9188632373) < 1e-9, (ev['p_used'], se)
    assert P_MAX / dg['qap'][j].max() < P_TOTAL / dg['Q'][j]               # 单 AP 上限先生效
    assert np.max(ev['intf_true']) < 1e-24
    print(f'diagonal: q = {dg["qap"][j]}, c2 = {c2}, P_used = {ev["p_used"]}, sum SE = {se:.10f}, per-AP limit binds')


def test_nmse():
    rng = np.random.default_rng(21)
    g, e = cgauss(rng, (400_000,)), cgauss(rng, (400_000,))
    for eps in (1e-2, 1e-1):
        gh = g_hat(g, e, eps)
        nmse = np.mean(np.abs(gh - g) ** 2) / np.mean(np.abs(g) ** 2)
        assert abs(nmse / eps - 1) < 0.02 and abs(np.mean(np.abs(gh) ** 2) - (1 - eps)) < 0.01, (eps, nmse)
        print(f'error model eps = {eps}: NMSE = {nmse:.5f}, E|g_hat|^2 = {np.mean(np.abs(gh) ** 2):.4f}')


def test_m1_equal(samples=60):
    worst = 0.0
    for s in range(samples):
        _, _, r = work((1, s))
        for key in ('se_true', 'se_est'):
            x = r[key][:, :, :4]                                              # A, B, C, Ceq
            worst = max(worst, float(np.max(np.abs(x - x[:, :, :1]))))
        assert np.all(r['key'][:, :, :4] == r['key'][:, :, :1])
    assert worst < 1e-12, worst
    print(f'M = 1: A = B = C = Ceq (estimated and true SE, all CSI) on {samples} samples, max diff {worst:.1e}')


def test_order_and_checks(samples=40):
    ix = {s: i for i, s in enumerate(SCH)}
    for M in (2, 4):
        worst = 0.0
        for s in range(samples):
            _, _, r = work((M, s))
            ch = r['chk']
            assert ch['resid_est'] < 1e-9 and ch['q_rel'] < 1e-9 and ch['p_over'] == 0 and ch['n_active'] == 0, ch
            assert ch['ceq_p_rel'] < 1e-12 and ch['est_formula'] < 1e-9, ch
            for ei, (_, eps) in enumerate(CSI):
                keys = ('se_est', 'se_true') if eps == 0 else ('se_est',)
                for key in keys:
                    x = r[key][:, ei]
                    for a, b in (('C', 'B'), ('B', 'A'), ('C', 'Ceq'), ('Ceq', 'A')):
                        worst = max(worst, float(np.max(x[:, ix[b]] - x[:, ix[a]])))
        assert worst < 1e-9, (M, worst)
        print(f'M = {M}: C >= B >= A and C >= Ceq >= A on {samples} samples (estimated; true under perfect CSI); '
              f'max violation {worst:.1e}; power, antenna-count, Ceq-power and residual checks pass')


def direct_select(H, He, N, M, K, noise):
    """逐候选直接伪逆，自行实现 A/B/C/Ceq 的选择，返回 {方案: (天线集合, 真实 sum SE [nS])}。"""
    subsets = [list(S) for S in combinations(range(N * M), K)]
    info = []
    for S in subsets:
        W0 = np.linalg.pinv(He[:, S])
        q = np.zeros(N)
        for k, m in enumerate(S):
            q[m // M] += np.sum(np.abs(W0[k]) ** 2)
        info.append((S, W0, q, min(P_TOTAL / q.sum(), P_MAX / q.max())))

    def true_se(S, W0, c2):
        F = H[:, S] @ (np.sqrt(c2) * W0)
        pw = np.abs(F) ** 2
        d = np.diag(pw)
        return np.array([np.sum(np.log2(1 + d / (pw.sum(1) - d + s2))) for s2 in noise])

    def pick(pool, c2fun):
        best = max(range(len(pool)), key=lambda t: (c2fun(pool[t]), -t))    # 并列取最小下标
        return pool[best]

    full = lambda S: all(sum(1 for m in S if m // M == n) in (0, M) for n in range(N))
    strength = np.sum(np.abs(He) ** 2, axis=0).reshape(N, M)
    bsets = set()
    for k in product(range(M + 1), repeat=N):
        if sum(k) == K:
            bsets.add(tuple(sorted(n * M + int(m) for n in range(N)
                                   for m in np.argsort(-strength[n], kind='stable')[:k[n]])))
    out = {}
    for name, pool in (('A', [x for x in info if full(x[0])]), ('B', [x for x in info if tuple(x[0]) in bsets]),
                       ('C', info)):
        S, W0, q, c2 = pick(pool, lambda x: x[3])
        out[name] = (S, true_se(S, W0, c2), c2)
    Sa, W0a, qa, c2a = pick([x for x in info if full(x[0])], lambda x: x[3])
    p_ref = c2a * qa.sum()
    feas = [x for x in info if (p_ref / x[2].sum()) * x[2].max() <= P_MAX * (1 + 1e-12)]
    S, W0, q, _ = pick(feas, lambda x: p_ref / x[2].sum())
    out['Ceq'] = (S, true_se(S, W0, p_ref / q.sum()), p_ref / q.sum())
    return out, info


def test_direct_m2(samples=40):
    rng = np.random.default_rng(22)
    N, M, K = 3, 2, 4
    worst_q, worst_se = 0.0, 0.0
    for _ in range(samples):
        g, e = cgauss(rng, (2, N, M)), cgauss(rng, (2, N, M))
        for eps in (0.0, 0.1):
            beta = rng.choice([0.05, 0.25, 0.8])
            H = zfsel.assemble(scale_links(g, beta))
            He = zfsel.assemble(scale_links(g_hat(g, e, eps), beta))
            win, dg = zfsel.select(H, He, N, M, K, P_TOTAL, P_MAX)
            ref, info = direct_select(H, He, N, M, K, NOISE)
            for t, (S, W0, q, c2) in enumerate(info):
                worst_q = max(worst_q, float(np.max(np.abs(dg['qap'][t] - q)) / q.sum()))
            bits = zfsel.cand(N, M, K)['bits']
            for name in ('A', 'B', 'C', 'Ceq'):
                j, c2 = win[name]
                S = list(np.flatnonzero(bits[j]))
                assert S == ref[name][0], (name, S, ref[name][0])
                ev = zfsel.evaluate(H, He, bits[j], c2, N, M, NOISE)
                worst_se = max(worst_se, float(np.max(np.abs(np.log2(1 + ev['sinr_true']).sum(1) - ref[name][1]))))
    assert worst_q < 1e-10 and worst_se < 1e-9, (worst_q, worst_se)
    print(f'M = 2 direct pseudo-inverse: per-candidate q max rel diff {worst_q:.1e}; '
          f'winners identical for A/B/C/Ceq; true SE max diff {worst_se:.1e}')


def test_b_ties():
    N, M, K = 3, 2, 4
    He = np.array([[1, 1j, 1, -1, 2, 2], [1j, 1, -1, 1, 0, 0]], dtype=complex) / np.sqrt(2)  # AP1、AP2 两根等强
    c = zfsel.cand(N, M, K)
    idx = zfsel.b_indices(He, c, N, M)
    masks = {tuple(np.flatnonzero(c['bits'][i])) for i in idx}
    assert (0, 2, 4, 5) in masks and (1, 2, 4, 5) not in masks and (0, 3, 4, 5) not in masks   # k = (1, 1, 2) 取编号小者
    print('B tie rule: equal-strength antennas are taken in index order')


def test_conj_and_trial():
    rng = np.random.default_rng(23)
    worst = 0.0
    for _ in range(30):
        g = cgauss(rng, (2, 3, 2))
        for beta in (0.05, 0.8):
            h = scale_links(g, beta)
            old, _ = antsel_zf.solve_zf(h, 4, P_TOTAL, P_MAX, NOISE, 'zf')       # 试跑：H = h，'ap' 规则
            H = zfsel.assemble(h)
            win, _ = zfsel.select(H, H, 3, 2, 4, P_TOTAL, P_MAX)
            bits = zfsel.cand(3, 2, 4)['bits']
            for ci, name in enumerate(('A', 'C')):
                j, c2 = win[name]
                se = np.log2(1 + zfsel.evaluate(H, H, bits[j], c2, 3, 2, NOISE)['sinr_true']).sum(1)
                worst = max(worst, float(np.max(np.abs(se - old[1, ci]))))
    assert worst < 1e-9, worst
    print(f'conjugation: new (H = conj h) vs trial (H = h, per-AP rule) A/C perfect-CSI SE max diff {worst:.1e}')


if __name__ == '__main__':
    test_diagonal()
    test_nmse()
    test_b_ties()
    test_direct_m2()
    test_conj_and_trial()
    test_m1_equal()
    test_order_and_checks()
    print('all controlled-ZF checks passed')
