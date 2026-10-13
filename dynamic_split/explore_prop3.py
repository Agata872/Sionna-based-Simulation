"""S6：命题 3（逐 RU 位宽的注水分配）检验。

同一服务集合上比较三种逐 RU 配置方法（P1：SE ≥ κ SE_max 时的最小负载）：
  CF-U   全网一个位宽 w（7-1 的 BFP = 7-2 的权值），切分按命题 2
  CF-WF  命题 3：w_r = t + ½ log₂(G_r / (336 n_r))，b_r = t + ½ log₂(G'_r / (2 n_r U))，取整并限制在 4..12，
         G_r、G'_r 为位宽 4 时该 RU 的和速率加权失真（线性化），t 为全网一个标量；切分按命题 2（用各自位宽）
  OPT    逐 RU 拉格朗日枚举（阶段 3 的优化器）
服务集合两条：'10'（[10] 的启发式，heuristics10）与 'se'（和速率贪心），各取 κ_s 网格。MCS 截顶。

用法：
    .venv/bin/python dynamic_split/explore_prop3.py results/dynamic_split_20261012/stage0_repro \
        --out results/dynamic_split_20261012/stage6_prop3
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import cfsim  # noqa: E402
import faithful  # noqa: E402
import heuristics10 as H10  # noqa: E402
import joint as J  # noqa: E402
import run_stage3 as R3  # noqa: E402
import summarize_stage0 as s0  # noqa: E402
from runio import Run, read_rows  # noqa: E402

PATHS = ['10', 'se']
METHODS = ['CF-U', 'CF-WF', 'OPT']


def payload71(n, w):
    return n * 168 * (2 * w + 2 / 3)


def payload72(qm_sum, n_ue, n, b):
    return 168 * qm_sum + n_ue * (2 * b * n + 24)


def evaluate(model, st, opts, choose, k):
    """choose(r, n_r, ro) -> 选项键；返回 (和速率, 总负载 Gbit/s)。"""
    ru = st['serving'] // k
    d = np.zeros_like(st['sig'])
    total = 0.0
    for r, ro in opts:
        key = choose(r, int((ru == r).sum()), ro)
        o = next(o for o in ro if o[0] == key)
        d += o[3]
        total += o[1]
    return model.se(st, d), total / 1e9


def methods_for_state(model, st, k, cal, se_max):
    """一个服务集合、一个 K 下三种方法的最小可行负载。"""
    opts = model.ru_options(st, k)
    u_ = model.num_ue
    qm = st['qm']
    out = {}
    # CF-U
    best = np.inf
    for w in range(4, 13):
        def ch(r, n, ro, w=w):
            if model.bf != 'zf':
                return ('7-2', None)
            return ('7-2', w) if payload72(qm.sum(), u_, n, w) < payload71(n, w) else ('7-1', f'bfp{w}')
        se, tot = evaluate(model, st, opts, ch, k)
        if se >= J.KAPPA * se_max - 1e-9:
            best = min(best, tot)
    out['CF-U'] = best
    # CF-WF
    ref = cal['r71']['bfp4']
    x71, x72 = {}, {}
    ru = st['serving'] // k
    for r, ro in opts:
        n = int((ru == r).sum())
        g71 = next(o for o in ro if o[0] == ('7-1', 'bfp4'))[4] / ref
        x71[r] = 0.5 * np.log2(max(g71, 1e-300) / (336 * n))
        if model.bf == 'zf':
            g72 = next(o for o in ro if o[0] == ('7-2', 4))[4] / ref
            x72[r] = 0.5 * np.log2(max(g72, 1e-300) / (2 * n * u_))
    best = np.inf
    if model.bf != 'zf':
        se, tot = evaluate(model, st, opts, lambda r, n, ro: ('7-2', None), k)
        best = tot if se >= J.KAPPA * se_max - 1e-9 else np.inf
    else:
        xs = list(x71.values()) + list(x72.values())
        for t in np.arange(np.floor(4 - max(xs)) - 0.5, np.ceil(12 - min(xs)) + 0.75, 0.25):
            def ch(r, n, ro, t=t):
                w = int(np.clip(np.round(t + x71[r]), 4, 12))
                b = int(np.clip(np.round(t + x72[r]), 4, 12))
                return ('7-2', b) if payload72(qm.sum(), u_, n, b) < payload71(n, w) else ('7-1', f'bfp{w}')
            se, tot = evaluate(model, st, opts, ch, k)
            if se >= J.KAPPA * se_max - 1e-9:
                best = min(best, tot)
    out['CF-WF'] = best
    # OPT
    best = np.inf
    for lam in J.LAMBDAS:
        se, tot, _, _, _ = model.lagrange(st, opts, lam)
        if se >= J.KAPPA * se_max - 1e-9:
            best = min(best, tot)
    out['OPT'] = best
    return out


def task(item):
    (si, d), (num_ue, noise_dbm, bf, row, cal, model_name, se_model) = item
    faithful.setup(model_name)
    h, beta, _ = cfsim.drop(np.random.default_rng([2024, d, num_ue]), num_ue, 'm2135')
    model = J.DropModel(h, bf, faithful.noise(model_name, d, num_ue, noise_dbm), se_model, cal)
    path10 = H10.path(h, bf, model.noise, se_model)
    # κ 的参照：fixed + mcs 时沿用 S6（阶段 0 近似路径的最大值），其他口径取 [10] 路径自身的最大值（同 S7）
    se_max = float(np.max(row['curve'])) if (model_name == 'fixed' and se_model == 'mcs') else max(p[2] for p in path10)
    res = np.full((len(R3.KS), len(PATHS), len(METHODS)), np.nan)
    for pi, name in enumerate(PATHS):
        if name == '10':
            path = path10
        else:
            path = J.la_path(model, cfsim.antenna_order(beta), 1, se_model, score_mode='se')
        states = []
        for ks in J.KAPPA_S:
            n = R3.first_index([p[2] for p in path], ks * se_max)
            states.append(None if n is None else model.state(path[n][0], path[n][1]))
        for ki, k in enumerate(R3.KS):
            best = {m: np.inf for m in METHODS}
            for st in states:
                if st is None:
                    continue
                r = methods_for_state(model, st, k, cal, se_max)
                for m in METHODS:
                    best[m] = min(best[m], r[m])
            res[ki, pi] = [best[m] if np.isfinite(best[m]) else np.nan for m in METHODS]
    return (si, d), res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage0')
    p.add_argument('--out', required=True)
    p.add_argument('--drops', type=int, default=100)
    p.add_argument('--workers', type=int, default=48)
    p.add_argument('--model', choices=faithful.MODELS, default='fixed')
    p.add_argument('--se-model', choices=['mcs', 'mcs4', 'shannon'], default='mcs')
    args = p.parse_args()
    here = Path(__file__).parent
    run = Run(args.out, [here / f for f in ('explore_prop3.py', 'heuristics10.py', 'joint.py', 'faithful.py', 'cfsim.py')],
              dict(purpose='S6: Proposition 3 (per-RU width water-filling) vs one width vs per-RU Lagrangian',
                   args=vars(args), model=args.model, se_model=args.se_model, paths=PATHS, methods=METHODS, t_grid='adaptive: [4 - max x, 12 - min x], step 0.25'))
    rows0 = read_rows(args.stage0)
    states = s0.select_states(rows0)
    keys = sorted({(u, n) for (u, n, _) in states})
    cal = J.calibrate()
    if args.model == 'fixed' and args.se_model == 'mcs':
        bfmap = {(u, n): states[(u, n, 'mcs')]['bf'] for (u, n) in keys}
    else:
        bfmap, _ = faithful.select_bf(keys, args.drops, args.model, args.se_model, args.workers)
    items = []
    for si, (u, n) in enumerate(keys):
        bf = bfmap[(u, n)]
        for r in rows0:                      # 阶段 0 的记录只用来取 MCS 口径下的 SE_max 参照（fixed + mcs）
            if (r['num_ue'], r['noise_dbm'], r['se_model'], r['bf']) == (u, n, 'mcs', states[(u, n, 'mcs')]['bf']) \
                    and r['drop'] < args.drops:
                items.append(((si, r['drop']), (u, n, bf, r, cal, args.model, args.se_model)))
    arr = np.full((len(keys), args.drops, len(R3.KS), len(PATHS), len(METHODS)), np.nan)
    with ProcessPoolExecutor(args.workers) as ex:
        for (si, d), res in ex.map(task, items, chunksize=1):
            arr[si, d] = res
    np.savez_compressed(run.path / 'prop3.npz', states=np.array(keys), ks=R3.KS, paths=PATHS, methods=METHODS, arr=arr)
    zf = np.array([bfmap[(u, n)] == 'zf' for (u, n) in keys])
    for pi, name in enumerate(PATHS):
        print(f'== serving set: {name}')
        for ki, k in enumerate(R3.KS):
            a = arr[:, :, ki, pi]
            m = ~np.isnan(a).any(-1)
            mz = m & zf[:, None]
            row = dict(path=name, k=k, drops=int(m.sum()), **{m_: float(a[..., j][m].mean()) for j, m_ in enumerate(METHODS)},
                       **{f'zf_{m_}': float(a[..., j][mz].mean()) for j, m_ in enumerate(METHODS)})
            run.add(row)
            print(f"  K={k:2d}: drops {row['drops']:3d} | all states: CF-U {row['CF-U']:.2f} ({100 * (row['CF-U'] / row['OPT'] - 1):+.1f}%) "
                  f"CF-WF {row['CF-WF']:.2f} ({100 * (row['CF-WF'] / row['OPT'] - 1):+.1f}%) OPT {row['OPT']:.2f} | ZF states: "
                  f"CF-U {100 * (row['zf_CF-U'] / row['zf_OPT'] - 1):+.1f}% CF-WF {100 * (row['zf_CF-WF'] / row['zf_OPT'] - 1):+.1f}%")
    run.finish()


if __name__ == '__main__':
    main()
