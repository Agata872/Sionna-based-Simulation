"""S4b：服务集合路径的消融。三条路径都接同样的逐 RU 拉格朗日选择（κ_s × λ 网格），比较逐撒点 P1 负载：
  paper  论文的逐根考察（按大尺度增益排序、能提高和速率才加入；阶段 0 记录）
  se     贪心：每步加入理想和速率增量最大的天线（MRT 时为 (天线, UE)），不看负载
  la     负载感知贪心：每步取 和速率增量 / 参考负载增量 最大者（阶段 3 的 P 用的路径）

用法：
    .venv/bin/python dynamic_split/ablate_path.py results/dynamic_split_20261012/stage0_repro \
        --out results/dynamic_split_20261012/stage4_path_ablation
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

PATHS = ['paper', 'se', 'la']


def best_load(model, k, path_states, se_max):
    """一条路径上满足 SE ≥ κ SE_max 的最小负载，以及对应的服务天线数。"""
    best = (np.inf, np.nan)
    for st in path_states:
        if st is None:
            continue
        opts = model.ru_options(st, k)
        for lam in J.LAMBDAS:
            se, tot, _, _, _ = model.lagrange(st, opts, lam)
            if se >= J.KAPPA * se_max - 1e-9 and tot < best[0]:
                best = (tot, len(st['serving']))
    return best if np.isfinite(best[0]) else (np.nan, np.nan)      # 这条路径达不到 κ 时记为 NaN


def task(item):
    (si, d), (num_ue, noise_dbm, bf, row, cal) = item
    h, _, _ = cfsim.drop(np.random.default_rng([2024, d, num_ue]), num_ue, 'm2135')
    model = J.DropModel(h, bf, 10 ** (noise_dbm / 10), 'mcs', cal)
    curve = np.asarray(row['curve'])
    se_max = float(curve.max())
    ppath = J.paper_path(row)
    pstates = [model.state(*ppath[R3.first_index(curve, ks * se_max)]) for ks in J.KAPPA_S]
    out = np.full((len(R3.KS), len(PATHS), 2), np.nan)
    se_states = None
    for ki, k in enumerate(R3.KS):
        for pi, name in enumerate(PATHS):
            if name == 'paper':
                states = pstates
            elif name == 'se':
                if se_states is None:                 # 只看和速率时路径与 K 无关
                    path = J.la_path(model, np.asarray(row['order']), 1, 'mcs', score_mode='se')
                    se_states = []
                    for ks in J.KAPPA_S:
                        n = R3.first_index([p[2] for p in path], ks * se_max)
                        se_states.append(None if n is None else model.state(path[n][0], path[n][1]))
                states = se_states
            else:
                path = J.la_path(model, np.asarray(row['order']), k, 'mcs')
                states = []
                for ks in J.KAPPA_S:
                    n = R3.first_index([p[2] for p in path], ks * se_max)
                    states.append(None if n is None else model.state(path[n][0], path[n][1]))
            out[ki, pi] = best_load(model, k, states, se_max)
    return (si, d), out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage0')
    p.add_argument('--out', required=True)
    p.add_argument('--drops', type=int, default=100)
    p.add_argument('--workers', type=int, default=48)
    args = p.parse_args()
    here = Path(__file__).parent
    run = Run(args.out, [here / f for f in ('ablate_path.py', 'joint.py', 'run_stage3.py')],
              dict(purpose='S4b: serving-set path ablation (paper order vs SE-greedy vs load-aware), same per-RU '
                           'Lagrangian; per-drop P1 load, MCS SE', args=vars(args), paths=PATHS, ks=R3.KS))
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
    arr = np.full((len(keys), args.drops, len(R3.KS), len(PATHS), 2), np.nan)
    with ProcessPoolExecutor(args.workers) as ex:
        for i, ((si, d), out) in enumerate(ex.map(task, items, chunksize=1)):
            arr[si, d] = out
            if (i + 1) % 100 == 0:
                print(f'{i + 1}/{len(items)}', flush=True)
    np.savez_compressed(run.path / 'ablation.npz', states=np.array(keys), ks=R3.KS, paths=PATHS, arr=arr)
    for si, (u, n) in enumerate(keys):
        for ki, k in enumerate(R3.KS):
            row = dict(num_ue=u, noise_dbm=n, bf=states[(u, n, 'mcs')]['bf'], k=k)
            for pi, name in enumerate(PATHS):
                row[f'load_{name}'] = float(np.nanmean(arr[si, :, ki, pi, 0]))
                row[f'antennas_{name}'] = float(np.nanmean(arr[si, :, ki, pi, 1]))
            run.add(row)
    run.finish()
    print('done', run.path)


if __name__ == '__main__':
    main()
