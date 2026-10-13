"""S5：按 [10] 忠实实现服务 AP 选择后，重做两件事。

A. 复现检查：用 [10] 的启发式重算阶段 0 的服务天线曲线（两种 BF、两种频谱效率模型），
   得到 M̂（κ = 0.98）与 κ 处的和速率，与论文图 5 和阶段 0（我们的近似实现）对照。
B. 服务集合对比（MCS，BF 沿用阶段 0 的选择，κ 的参照 SE_max 与阶段 3 / S4b 相同）：
   [10] 路径 + 逐 RU 拉格朗日，与 S4b 中"近似论文路径"和"和速率贪心路径"的结果在共同可行撒点上比较。

用法：
    .venv/bin/python dynamic_split/compare_paths10.py results/dynamic_split_20261012/stage0_repro \
        results/dynamic_split_20261012/stage4_path_ablation --out results/dynamic_split_20261012/stage5_paths10
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
import ablate_path as AB  # noqa: E402
import cfsim  # noqa: E402
import heuristics10 as H10  # noqa: E402
import joint as J  # noqa: E402
import run_stage3 as R3  # noqa: E402
import summarize_stage0 as s0  # noqa: E402
from runio import Run, read_rows  # noqa: E402

MODELS = ['shannon', 'mcs']


def task(item):
    (si, d), (num_ue, noise_dbm, bf_mcs, row_mcs, cal) = item
    h, _, _ = cfsim.drop(np.random.default_rng([2024, d, num_ue]), num_ue, 'm2135')
    noise = 10 ** (noise_dbm / 10)
    curves = np.zeros((len(MODELS), 2, cfsim.NUM_ANT))           # [模型, (zf, mrt), 考察数]
    for mi, sm in enumerate(MODELS):
        for bi, bf in enumerate(('zf', 'mrt')):
            curves[mi, bi] = [p[2] for p in H10.path(h, bf, noise, sm)]
    # B：MCS，阶段 0 的 BF
    model = J.DropModel(h, bf_mcs, noise, 'mcs', cal)
    se_max = float(np.max(row_mcs['curve']))
    path = H10.path(h, bf_mcs, noise, 'mcs')
    states = []
    for ks in J.KAPPA_S:
        n = R3.first_index([p[2] for p in path], ks * se_max)
        states.append(None if n is None else model.state(path[n][0], path[n][1]))
    loads = np.full((len(R3.KS), 2), np.nan)
    for ki, k in enumerate(R3.KS):
        loads[ki] = AB.best_load(model, k, states, se_max)
    return (si, d), curves, loads


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage0')
    p.add_argument('ablation')
    p.add_argument('--out', required=True)
    p.add_argument('--drops', type=int, default=100)
    p.add_argument('--workers', type=int, default=48)
    args = p.parse_args()
    here = Path(__file__).parent
    run = Run(args.out, [here / f for f in ('compare_paths10.py', 'heuristics10.py', 'joint.py', 'ablate_path.py')],
              dict(purpose='S5: faithful [10] AP-selection heuristics; (A) reproduction of EuCNC Fig. 5, '
                           '(B) serving-set comparison with per-RU Lagrangian, MCS', args=vars(args),
                   deviations_from_10=['Serving set common to all PRBs (paper load formula), ranking by PRB-averaged '
                                       '|h|^2; [10] selects per PRB (Assumption 1).',
                                       'ZF power normalisation as stage 0 (unit-norm columns, common scaling); [10] '
                                       'eq. (18) scales the unnormalised pseudo-inverse.',
                                       'ZF: Algorithm 2 order + EuCNC rule (select only if sum SE improves).']))
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
    curves = np.zeros((len(keys), args.drops, len(MODELS), 2, cfsim.NUM_ANT))
    loads = np.full((len(keys), args.drops, len(R3.KS), 2), np.nan)
    with ProcessPoolExecutor(args.workers) as ex:
        for (si, d), c, l in ex.map(task, items, chunksize=1):
            curves[si, d], loads[si, d] = c, l
    np.savez_compressed(run.path / 'paths10.npz', states=np.array(keys), curves=curves, loads=loads, ks=R3.KS)

    print('A. Reproduction (kappa 0.98 on mean curves): [10]-faithful | stage-0 approximation | paper Fig. 5')
    for mi, sm in enumerate(MODELS):
        for si, (u, n) in enumerate(keys):
            st0 = states[(u, n, sm)]
            res = {}
            for bi, bf in enumerate(('zf', 'mrt')):
                mean = curves[si, :, mi, bi].mean(0)
                m_hat = int(np.argmax(mean >= s0.KAPPA * mean.max())) + 1
                res[bf] = (m_hat, float(mean[m_hat - 1]))
            bf10 = max(res, key=lambda b: res[b][1])
            paper = s0.PAPER_FIG5.get((u, n)) if sm == 'shannon' else None
            row = dict(part='A', se_model=sm, num_ue=u, noise_dbm=n, bf_10=bf10,
                       zf_m_hat=res['zf'][0], zf_se=res['zf'][1], mrt_m_hat=res['mrt'][0], mrt_se=res['mrt'][1],
                       bf_stage0=st0['bf'], stage0_zf=[st0['zf']['m_hat'], st0['zf']['se']],
                       stage0_mrt=[st0['mrt']['m_hat'], st0['mrt']['se']],
                       paper=None if paper is None else [paper[0], paper[1]])
            run.add(row)
            print(f"  {sm:7s} {u:2d} {int(n):4d} | ZF {res['zf'][0]:2d}/{res['zf'][1]:6.1f} MRT {res['mrt'][0]:2d}/{res['mrt'][1]:6.1f} "
                  f"-> {bf10.upper():3s} | stage0 ZF {st0['zf']['m_hat']:2d}/{st0['zf']['se']:6.1f} MRT {st0['mrt']['m_hat']:2d}/"
                  f"{st0['mrt']['se']:6.1f} -> {st0['bf'].upper():3s} | paper {'' if paper is None else (paper[0].upper(), paper[1])}")

    print('\nB. Per-drop P1 load (MCS) on drops where all three paths are feasible [Gbit/s], mean over states')
    abl = np.load(Path(args.ablation) / 'ablation.npz')['arr'][..., 0]
    abl[~np.isfinite(abl)] = np.nan
    for ki, k in enumerate(R3.KS):
        f10, appr, seg = loads[:, :, ki, 0], abl[:, :, ki, 0], abl[:, :, ki, 1]
        m = ~np.isnan(f10) & ~np.isnan(appr) & ~np.isnan(seg)
        row = dict(part='B', k=k, drops=int(m.sum()), faithful10=float(f10[m].mean()), approx=float(appr[m].mean()),
                   se_greedy=float(seg[m].mean()), infeasible10=int(np.isnan(f10).sum()),
                   antennas10=float(np.nanmean(loads[:, :, ki, 1])))
        run.add(row)
        print(f"  K={k:2d}: drops {row['drops']:3d} | [10] {row['faithful10']:.2f} | approx {row['approx']:.2f} | "
              f"SE-greedy {row['se_greedy']:.2f} ({100 * (row['se_greedy'] / row['faithful10'] - 1):+.1f}% vs [10]) | "
              f"[10] infeasible {row['infeasible10']}")
    run.finish()


if __name__ == '__main__':
    main()
