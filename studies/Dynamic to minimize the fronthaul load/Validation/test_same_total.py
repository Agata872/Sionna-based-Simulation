"""E8 验收（同总天线数）。直接运行：python test_same_total.py

1. 布局：16 个格点就是 cfsim（EuCNC 8 × 8 布局）左下 4 × 4 块，最近邻 40 m；各 K 的 RU 站点为编号连续 K 个格点的质心。
2. 大尺度：路损函数与 cfsim._pathloss_m2135_db 逐点相同；同一随机流下 large_scale 与 cfsim.drop 的 β 逐位相同；
   常数（频率、高度、天线增益、P_max）与 cfsim 相同。
3. 小尺度随机数在各 K 之间共用：H / sqrt(β) 与 K 无关。
4. 单 UE 解析验算：|h_j| = a 时，任意 m 根天线的 ZF（即 MRT）c² = P_max m² a²（两种功率规则相同）。
5. 手算例子：H 只有 h_0 = [1, 0]、h_1 = [0, 2] 非零 → 含天线 0、1 的子集 q = [1, 0.25]，'ant' 规则 c² = P_max，
   'ru' 规则 c² = min(m P_max / 1.25, K P_max / 1.25)；其余子集秩不足（个数 49,135）。
6. 快速穷举与逐候选直接伪逆一致（抽 3000 个候选），各 (规则, 粒度, m) 的最大值与全部候选直接计算一致。
7. 20 个撒点 × 各 K：天线级 ≥ RU 级；K = 1 时两者逐位相等；M̂_ant ≤ M̂_RU；M̂ 处达标、M̂ − 1 处不达标；
   胜出子集显式验算的功率约束（'ant' 每根 ≤ P_max；'ru' 每 RU ≤ K P_max；两者总功率 ≤ m P_max）。
"""

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import sys

import numpy as np

import same_total as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..', 'dynamic_split')))
import cfsim  # noqa: E402

KS = (1, 2, 4, 8)
U = 4
NOISE_MW = [1e-9, 1e-7, 1e-5]          # −90 / −70 / −50 dBm
TOL = 1e-9


def test_geometry():
    pos = st.antenna_positions()
    ref = cfsim.antenna_positions()
    for j in range(st.M_TOT):
        block, i = divmod(j, 2 * st.N_COL)
        col, sub = divmod(i, 2)
        assert np.allclose(pos[j], ref[16 * block + 2 * col + sub]), j
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1) + np.eye(st.M_TOT) * 1e9
    assert np.allclose(d.min(1), 40.0, atol=0.01), d.min(1)
    assert np.all(pos > 0) and np.all(pos < np.array(st.AREA))
    assert np.allclose(st.ru_sites(1), pos)
    assert np.allclose(st.ru_sites(2)[0], (20.0, 34.64))
    assert np.allclose(st.ru_sites(4), [(40, 34.64), (120, 34.64), (40, 103.92), (120, 103.92)])
    assert np.allclose(st.ru_sites(8), [(80, 34.64), (80, 103.92)])
    print(f'geometry: lower-left 4x4 block of the cfsim lattice, nearest neighbour 40 m, area {st.AREA}; '
          'K = 4 sites form a 2x2 grid')


def test_large_scale():
    rng = np.random.default_rng(7)
    d2d = rng.uniform(10, 400, 20_000)
    los = rng.random(20_000) < 0.5
    diff = float(np.max(np.abs(st.pathloss_m2135_db(d2d, los) - cfsim._pathloss_m2135_db(d2d, los))))
    assert diff == 0.0, diff
    assert (st.FC_GHZ, st.H_BS, st.H_UT, st.ANT_GAIN_DBI, st.P_MAX_MW) == \
        (cfsim.FC_GHZ, cfsim.H_BS, cfsim.H_UT, cfsim.ANT_GAIN_DBI, cfsim.P_MAX_PRB_MW)
    worst = 0.0
    for seed in range(5):
        _, beta_ref, ue = cfsim.drop(np.random.default_rng(seed), 6, 'm2135')
        rng = np.random.default_rng(seed)
        ue2 = rng.uniform((0, 0), cfsim.AREA, size=(6, 2))                # cfsim.drop 先抽 UE 位置
        beta, _ = st.large_scale(rng, ue2, cfsim.antenna_positions())
        assert np.array_equal(ue, ue2)
        worst = max(worst, float(np.max(np.abs(beta / beta_ref - 1))))
    assert worst < 1e-12, worst
    print(f'large scale: path loss identical to cfsim (max diff {diff}), beta vs cfsim.drop max rel diff {worst:.1e}, '
          'constants identical')


def test_common_small_scale():
    f = None
    for K in KS:
        H, beta, _, _ = st.drop(20261124, 3, K, U, 4)
        fk = H / np.sqrt(np.repeat(beta, K, axis=1))[None]
        if f is None:
            f = fk
        assert np.allclose(fk, f, rtol=1e-12, atol=0), K
    print('small scale: the same CN(0,1) draws for antenna j under every K')


def test_single_ue():
    rng = np.random.default_rng(11)
    a = 3e-5
    H = a * np.exp(2j * np.pi * rng.random((1, st.M_TOT)))
    worst = 0.0
    for K in KS:
        _, _, size, _, _ = st.candidates(K, 1)
        c2, _, ninv = st.score(H, K)
        expect = st.P_MAX_MW * size ** 2 * a ** 2
        worst = max(worst, float(np.max(np.abs(c2 / expect[None] - 1))))
        assert ninv == 0
    assert worst < 1e-10, worst
    print(f'single UE: c2 = P_max m^2 a^2 for every subset and both rules (max rel err {worst:.1e})')


def test_hand_example():
    H = np.zeros((2, st.M_TOT), complex)
    H[0, 0], H[1, 1] = 1.0, 2.0
    P = st.P_MAX_MW
    for K in KS:
        keys, _, size, _, _ = st.candidates(K, 2)
        c2, q, ninv = st.score(H, K)
        assert ninv == 49_135, ninv
        ok = (keys & 3) == 3
        assert np.all(np.isfinite(c2[:, ok])) and np.all(np.isinf(c2[:, ~ok]))
        assert np.allclose(q[ok][:, :2], [1.0, 0.25]) and np.allclose(q[ok][:, 2:], 0)
        # K ≥ 2 时天线 0、1 同属 RU 0（q_RU0 = 1.25）；K = 1 时各自成 RU，max q_n = 1，'ru' 规则即 'ant' 规则
        assert np.allclose(c2[0, ok], P)
        expect_ru = np.minimum(size[ok] * P / 1.25, K * P / 1.25) if K > 1 else np.full(ok.sum(), P)
        assert np.allclose(c2[1, ok], expect_ru), K
        qx, c2x, resid = st.explicit(H, 3, K)
        assert np.allclose(qx[:2], [1, 0.25]) and resid < 1e-12
        assert np.allclose(c2x, [P, min(2 * P / 1.25, K * P / 1.25) if K > 1 else P])
    print("hand example: q = [1, 0.25]; 'ant' c2 = P_max; 'ru' c2 = min(m P_max/1.25, K P_max/1.25); "
          '49,135 rank-deficient subsets')


def test_vs_direct():
    worst_c, worst_v = 0.0, 0.0
    rng = np.random.default_rng(13)
    for K in KS:
        H = st.drop(555, 0, K, U, 1)[0][0]
        keys, _, size, is_ru, _ = st.candidates(K, U)
        c2, q, ninv = st.score(H, K)
        assert ninv == 0
        direct = np.array([st.explicit(H, k, K)[1] for k in keys]).T             # [2, Nc]
        pick = rng.choice(len(keys), 3000, replace=False)
        worst_c = max(worst_c, float(np.max(np.abs(c2[:, pick] / direct[:, pick] - 1))))
        val, _ = st.best_per_m(c2, K, U)
        for gi, g in enumerate(st.GRANS):
            for m in range(U, st.M_TOT + 1):
                sel = (size == m) & (is_ru if g == 'RU' else True)
                for ri in range(2):
                    ref = direct[ri, sel].max() if sel.any() else -np.inf
                    if np.isfinite(ref):
                        worst_v = max(worst_v, abs(val[ri, gi, m] / ref - 1))
                    else:
                        assert np.isinf(val[ri, gi, m])
    assert worst_c < 1e-8 and worst_v < 1e-8, (worst_c, worst_v)
    print(f'fast vs direct pseudo-inverse: per-candidate c2 max rel diff {worst_c:.1e}, '
          f'per-(rule, granularity, m) maxima over all 64,839 subsets max rel diff {worst_v:.1e}')


def test_order_and_mhat(drops=20):
    n_chk, worst_p = 0, 0.0
    for d in range(drops):
        for K in KS:
            H = st.drop(777, d, K, U, 1)[0][0]
            keys = st.candidates(K, U)[0]
            c2, _, _ = st.score(H, K)
            val, idx = st.best_per_m(c2, K, U)
            ru, ant = val[:, 0], val[:, 1]
            fin = np.isfinite(ru)
            assert np.all(ant[fin] >= ru[fin] * (1 - TOL))
            if K == 1:
                assert np.array_equal(ru, ant) and np.array_equal(idx[:, 0], idx[:, 1])
            for ri in range(2):
                for noise in NOISE_MW:
                    mh, c2max = st.m_hat(val[ri], noise)
                    assert mh[1] <= mh[0]
                    target = st.KAPPA * st.sum_se(c2max, noise, U)
                    for gi in range(2):
                        assert st.sum_se(val[ri, gi, mh[gi]], noise, U) >= target * (1 - 1e-12)
                        below = val[ri, gi, U:mh[gi]]
                        below = below[np.isfinite(below)]                    # RU 级只在 m 为 K 的倍数时有候选
                        assert np.all(st.sum_se(below, noise, U) < target)
                for gi in range(2):
                    for m in range(U, st.M_TOT + 1):
                        if idx[ri, gi, m] < 0:
                            continue
                        q, c2x, _ = st.explicit(H, keys[idx[ri, gi, m]], K)
                        p = c2x[ri] * q
                        assert p.sum() <= m * st.P_MAX_MW * (1 + TOL)
                        if ri == 0:
                            assert p.max() <= st.P_MAX_MW * (1 + TOL)
                        else:
                            assert p.reshape(-1, K).sum(1).max() <= K * st.P_MAX_MW * (1 + TOL)
                            worst_p = max(worst_p, p.max() / st.P_MAX_MW)
                        n_chk += 1
    print(f'{drops} drops x K in {KS}: antenna level >= RU level, K = 1 identical, M̂_ant <= M̂_RU, target met at M̂ '
          f'and missed below it, power limits hold for {n_chk} winners; under the paper rule one antenna reaches up to '
          f'{worst_p:.2f} P_max')


if __name__ == '__main__':
    test_geometry()
    test_large_scale()
    test_common_small_scale()
    test_single_ue()
    test_hand_example()
    test_vs_direct()
    test_order_and_mhat()
    print('all same-total checks passed')
