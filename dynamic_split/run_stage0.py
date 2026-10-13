"""阶段 0：复现 Girycki 等（EuCNC 2024）的服务天线选择曲线。

每个（撒点, UE 数）生成一次信道；对每个噪声、波束赋形方式（ZF/MRT）和频谱效率模型
（论文的 Shannon，或按 MCS 表 2 截顶）各跑一遍逐根考察，记录整条曲线。
服务天线数 M̂、波束赋形选择和前传负载都在 summarize_stage0.py 里由这些曲线算出。

用法：
    .venv/bin/python dynamic_split/run_stage0.py --out results/dynamic_split_20261012/stage0_repro
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import cfsim  # noqa: E402
from runio import Run  # noqa: E402


def task(args_tuple):
    seed, d, num_ue, noises, se_models, pathloss = args_tuple
    rng = np.random.default_rng([seed, d, num_ue])
    h, beta, ue = cfsim.drop(rng, num_ue, pathloss)
    order = cfsim.antenna_order(beta)
    rows, per_ue = [], []
    for noise_dbm in noises:
        noise_mw = 10 ** (noise_dbm / 10)
        for bf, fn in (('zf', cfsim.zf_sequence), ('mrt', cfsim.mrt_sequence)):
            for se_model in se_models:
                curve, pu, incl, assign = fn(h, order, noise_mw, se_model)
                rows.append(dict(drop=d, num_ue=num_ue, noise_dbm=noise_dbm, bf=bf, se_model=se_model,
                                 order=order.tolist(), curve=np.round(curve, 4).tolist(),
                                 incl=incl.astype(int).tolist(), assign=assign.tolist()))
                per_ue.append(pu.astype(np.float32))
    return rows, per_ue


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--drops', type=int, default=100)
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--ues', type=int, nargs='+', default=[5, 10, 20])
    p.add_argument('--noise', type=float, nargs='+', default=[-90., -70., -50.])
    p.add_argument('--se-models', nargs='+', default=['shannon', 'mcs'])
    p.add_argument('--pathloss', choices=['m2135', 'm2412'], default='m2135')
    p.add_argument('--workers', type=int, default=48)
    args = p.parse_args()
    here = Path(__file__).parent
    run = Run(args.out, [here / 'run_stage0.py', here / 'cfsim.py', here / 'runio.py'],
              dict(purpose='Stage 0: reproduce Girycki et al. EuCNC 2024 serving-antenna curves',
                   args=vars(args), assumptions=cfsim.ASSUMPTIONS,
                   paper_targets='Fig. 3 (10 UEs, -90 dBm: ZF 32 antennas / 127 bit/s/Hz, MRT 23 / 71) and Fig. 5'))
    tasks = [(args.seed, d, u, args.noise, args.se_models, args.pathloss) for u in args.ues for d in range(args.drops)]
    per_ue = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (rows, pu) in enumerate(ex.map(task, tasks, chunksize=1)):
            for r, a in zip(rows, pu):
                key = f"d{r['drop']}_u{r['num_ue']}_n{int(r['noise_dbm'])}_{r['bf']}_{r['se_model']}"
                r['per_ue_key'] = key
                per_ue[key] = a
                run.add(r)
            if (i + 1) % 25 == 0:
                print(f'{i + 1}/{len(tasks)} tasks', flush=True)
    np.savez_compressed(run.path / 'per_ue_se.npz', **per_ue)
    run.finish()
    print('done', run.path)


if __name__ == '__main__':
    main()
