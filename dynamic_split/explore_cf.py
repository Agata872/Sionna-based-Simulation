"""S4c：闭式策略与优化器的对比（理论笔记第 4 节第 4 条）。

闭式策略 CF（每个撒点）：
  1. 服务集合：和速率贪心路径（ablate_path 的 'se'），在 κ_s 网格上取位置；
  2. 位宽：全网一个 w（7-1 的 BFP 宽度 = 7-2 的权值宽度），在 4..12 中搜索；
  3. 每个 RU 的切分：命题 2（ZF：n_r > n*(w) 选 7-2，否则 7-1）/ 推论 2（MRT：7-2）。
取满足 SE ≥ κ SE_max 的最小负载。对照：同一路径上的逐 RU 拉格朗日优化（S4b 的 'se' 列），
以及 S4b 三条路径取最好（'all'）。

用法：
    .venv/bin/python dynamic_split/explore_cf.py results/dynamic_split_20261012/stage0_repro \
        results/dynamic_split_20261012/stage4_path_ablation --out results/dynamic_split_20261012/stage4_cf
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
import joint as J  # noqa: E402
import run_stage3 as R3  # noqa: E402
import summarize_stage0 as s0  # noqa: E402
from runio import Run, read_rows  # noqa: E402

WIDTHS = list(range(4, 13))


def n_star(n_ue, qm_mean, w):
    return n_ue * (168 * qm_mean + 24) / (168 * (2 * w + 2 / 3) - 2 * w * n_ue)


def cf_point(model, st, k, w):
    opts = model.ru_options(st, k)
    ru = st['serving'] // k
    d = np.zeros_like(st['sig'])
    total, n72 = 0.0, 0
    for r, ro in opts:
        n_r = int((ru == r).sum())
        if model.bf == 'zf':
            use72 = n_r > n_star(model.num_ue, st['qm'].mean(), w)
            key = ('7-2', w) if use72 else ('7-1', f'bfp{w}')
        else:
            key = ('7-2', None)
        o = next(o for o in ro if o[0] == key)
        d += o[3]
        total += o[1]
        n72 += key[0] == '7-2'
    return model.se(st, d), total / 1e9, n72 / len(opts)


def task(item):
    (si, d), (num_ue, noise_dbm, bf, row, cal) = item
    h, _, _ = cfsim.drop(np.random.default_rng([2024, d, num_ue]), num_ue, 'm2135')
    model = J.DropModel(h, bf, 10 ** (noise_dbm / 10), 'mcs', cal)
    se_max = float(np.max(row['curve']))
    path = J.la_path(model, np.asarray(row['order']), 1, 'mcs', score_mode='se')
    states = []
    for ks in J.KAPPA_S:
        n = R3.first_index([p[2] for p in path], ks * se_max)
        states.append(None if n is None else model.state(path[n][0], path[n][1]))
    out = np.full((len(R3.KS), 3), np.nan)          # 负载、所用宽度、7-2 RU 比例
    for ki, k in enumerate(R3.KS):
        best = None
        for st in states:
            if st is None:
                continue
            for w in WIDTHS:
                se, tot, f72 = cf_point(model, st, k, w)
                if se >= J.KAPPA * se_max - 1e-9 and (best is None or tot < best[0]):
                    best = (tot, w, f72)
        if best is not None:
            out[ki] = best
    return (si, d), out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage0')
    p.add_argument('ablation')
    p.add_argument('--out', required=True)
    p.add_argument('--drops', type=int, default=100)
    p.add_argument('--workers', type=int, default=48)
    args = p.parse_args()
    here = Path(__file__).parent
    run = Run(args.out, [here / f for f in ('explore_cf.py', 'joint.py', 'run_stage3.py')],
              dict(purpose='S4c: closed-form policy (SE-greedy serving set + one width + Proposition-2 split per RU) '
                           'vs per-RU Lagrangian optimizer', args=vars(args), widths=WIDTHS, ks=R3.KS))
    rows0 = read_rows(args.stage0)
    states = s0.select_states(rows0)
    keys = sorted({(u, n) for (u, n, _) in states})
    cal = J.calibrate()
    items = []
    for si, (u, n) in enumerate(keys):
        bf = states[(u, n, 'mcs')]['bf']
        for r in rows0:
            if (r['num_ue'], r['noise_dbm'], r['se_model'], r['bf']) == (u, n, 'mcs', bf) and r['drop'] < args.drops:
                items.append(((si, r['drop']), (u, n, bf, r, cal)))
    arr = np.full((len(keys), args.drops, len(R3.KS), 3), np.nan)
    with ProcessPoolExecutor(args.workers) as ex:
        for (si, d), out in ex.map(task, items, chunksize=2):
            arr[si, d] = out
    np.savez_compressed(run.path / 'cf.npz', states=np.array(keys), ks=R3.KS, arr=arr)
    abl = np.load(Path(args.ablation) / 'ablation.npz')['arr'][..., 0]
    abl[~np.isfinite(abl)] = np.nan
    for ki, k in enumerate(R3.KS):
        cf = arr[:, :, ki, 0]
        opt_se = abl[:, :, ki, 1]
        opt_all = np.fmin(np.fmin(abl[:, :, ki, 0], abl[:, :, ki, 1]), abl[:, :, ki, 2])
        m = ~np.isnan(cf) & ~np.isnan(opt_se)
        row = dict(k=k, drops_common=int(m.sum()), cf=float(np.nanmean(cf[m])), opt_same_path=float(np.nanmean(opt_se[m])),
                   opt_all_paths=float(np.nanmean(opt_all[m])), paper_path_opt=float(np.nanmean(abl[:, :, ki, 0][m])),
                   cf_width_mean=float(np.nanmean(arr[:, :, ki, 1][m])), cf_share72=float(np.nanmean(arr[:, :, ki, 2][m])),
                   cf_infeasible=int(np.isnan(cf).sum()))
        run.add(row)
        print(f"K={k:2d}: drops {row['drops_common']}  CF {row['cf']:.2f}  optimizer (same path) {row['opt_same_path']:.2f} "
              f"({100 * (row['cf'] / row['opt_same_path'] - 1):+.1f}%)  optimizer (best path) {row['opt_all_paths']:.2f} "
              f"({100 * (row['cf'] / row['opt_all_paths'] - 1):+.1f}%)  paper path + optimizer {row['paper_path_opt']:.2f}  "
              f"| CF width {row['cf_width_mean']:.1f}, 7-2 share {row['cf_share72']:.2f}, infeasible {row['cf_infeasible']}")
    run.finish()


if __name__ == '__main__':
    main()
