"""S7（方向 3）：用三步闭式方法刻画"动态"的价值从哪来。

每个 (状态, 撒点, K)：服务集合 = [10] 路径，κ 的参照 SE_max = 该路径的最大理想和速率，κ_s 网格取位置；
记录
  grid[κ_s, 模式, τ]  = (和速率 / SE_max, 总负载 Gbit/s, 7-2 RU 比例)，模式 ∈ {rule, all71, all72}，τ 为名义位宽
  uni[κ_s, 切分, w]   = 全网同一切分、同一位宽（阶段 1b 的口径）的 (和速率 / SE_max, 总负载)
分析见 analyse_stage7.py。--model 选口径（faithful.py；fixed 为第一版），--se-model 选 mcs（表 2）、mcs4（表 4，含 1024QAM）或 shannon；
fixed + mcs 时 BF 沿用阶段 0 的选择，其他组合按论文规则在 [10] 路径的平均曲线上重新选。

用法：
    .venv/bin/python dynamic_split/run_stage7.py results/dynamic_split_20261012/stage0_repro \
        --out results/dynamic_split_20261012/stage7_dynamics
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
import closed_form as CF  # noqa: E402
import faithful  # noqa: E402
import heuristics10 as H10  # noqa: E402
import joint as J  # noqa: E402
import run_stage3 as R3  # noqa: E402
import summarize_stage0 as s0  # noqa: E402
from runio import Run, read_rows  # noqa: E402

MODES = ['rule', 'all71', 'all72']
SPLITS = ['7-1', '7-2']


def task(item):
    (si, d), (num_ue, noise_dbm, bf, cal, model_name, se_model) = item
    faithful.setup(model_name)
    h, _, _ = cfsim.drop(np.random.default_rng([2024, d, num_ue]), num_ue, 'm2135')
    model = J.DropModel(h, bf, faithful.noise(model_name, d, num_ue, noise_dbm), se_model, cal)
    path = H10.path(h, bf, model.noise, se_model)
    curve = [p[2] for p in path]
    se_max = float(max(curve))
    n_k, n_ks = len(R3.KS), len(J.KAPPA_S)
    grid = np.full((n_k, n_ks, len(MODES), len(CF.TAU), 3), np.nan, np.float32)
    uni = np.full((n_k, n_ks, len(SPLITS), len(CF.UNIFORM_W), 2), np.nan, np.float32)
    if se_max <= 0:          # 例如信道求逆式 ZF 下某个 UE 深衰落、全部 UE 低于最低 MCS：撒点无效，分析时剔除
        return (si, d), np.nan, grid, uni
    sts = [model.state(*path[R3.first_index(curve, ks * se_max)][:2]) for ks in J.KAPPA_S]
    for ki, k in enumerate(R3.KS):
        for ks_i, st in enumerate(sts):
            cs = CF.CFState(model, st, k, cal)
            for mi, mode in enumerate(MODES):
                for ti, tau in enumerate(CF.TAU):
                    idx, _ = cs.choose(tau, mode)
                    se, load, f72 = cs.evaluate(idx)
                    grid[ki, ks_i, mi, ti] = (se / se_max, load, f72)
            for pi, split in enumerate(SPLITS):
                for wi, w in enumerate(CF.UNIFORM_W):
                    se, load, _ = cs.evaluate(cs.uniform(split, w))
                    uni[ki, ks_i, pi, wi] = (se / se_max, load)
    return (si, d), se_max, grid, uni


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
    run = Run(args.out, [here / f for f in ('run_stage7.py', 'closed_form.py', 'heuristics10.py', 'joint.py', 'faithful.py',
                                            'cfsim.py', 'distortion.py')],
              dict(purpose='S7 (direction 3): where the value of dynamic adaptation comes from, with the closed-form '
                           'pipeline ([10] serving set, Proposition 3 widths, Proposition 2 split)', args=vars(args),
                   modes=MODES, tau=CF.TAU.tolist(), uniform_w=CF.UNIFORM_W, ks=R3.KS, kappa_s=J.KAPPA_S,
                   reference='SE_max = max ideal sum SE on the [10] path of the drop', se_model=args.se_model,
                   model=args.model))
    rows0 = read_rows(args.stage0)
    states = s0.select_states(rows0)
    keys = sorted({(u, n) for (u, n, _) in states})
    if args.model == 'fixed' and args.se_model == 'mcs':
        bfs = [states[(u, n, 'mcs')]['bf'] for (u, n) in keys]      # 与 S7 第一版相同（阶段 0 的选择）
        bf_info = None
    else:
        bfmap, bf_info = faithful.select_bf(keys, args.drops, args.model, args.se_model, args.workers)
        bfs = [bfmap[(int(u), float(n))] for (u, n) in keys]
        bf_info = {f'{int(u)},{int(n)}': v for (u, n), v in bf_info.items()}
    print('BF per state:', dict(zip([f'{int(u)},{int(n)}' for u, n in keys], bfs)), flush=True)
    cal = J.calibrate()
    items = [((si, d), (u, n, bfs[si], cal, args.model, args.se_model))
             for si, (u, n) in enumerate(keys) for d in range(args.drops)]
    shp = (len(keys), args.drops)
    arr = dict(se_max=np.zeros(shp),
               grid=np.full(shp + (len(R3.KS), len(J.KAPPA_S), len(MODES), len(CF.TAU), 3), np.nan, np.float32),
               uni=np.full(shp + (len(R3.KS), len(J.KAPPA_S), len(SPLITS), len(CF.UNIFORM_W), 2), np.nan, np.float32))
    with ProcessPoolExecutor(args.workers) as ex:
        for i, ((si, d), se_max, grid, uni) in enumerate(ex.map(task, items, chunksize=2)):
            arr['se_max'][si, d], arr['grid'][si, d], arr['uni'][si, d] = se_max, grid, uni
            if (i + 1) % 150 == 0:
                print(f'{i + 1}/{len(items)}', flush=True)
    np.savez_compressed(run.path / 'stage7.npz', states=np.array(keys), bfs=bfs, ks=R3.KS, **arr)
    for si, (u, n) in enumerate(keys):
        run.add(dict(num_ue=u, noise_dbm=n, bf=bfs[si], se_max_mean=float(np.nanmean(arr['se_max'][si])),
                     invalid_drops=int(np.isnan(arr['se_max'][si]).sum())))
    run.finish(bfs=bfs, bf_info=bf_info)
    print('done', run.path)


if __name__ == '__main__':
    main()
