"""阶段 2a：双 O-RU（各 32 阵元）、简化 UMi 簇信道。对每个撒点、DL 噪声底、频谱效率模型，
记录 51 个配置（7-1 / 7-2 集中式 × 位宽 × CI 方案 / 7-2 本地预编码）的和速率、每 O-RU 下行负载与每次 SRS 的上行 CI 比特。
SRS 周期（移动性）只影响上行 CI，在分析时折算。

用法：
    .venv/bin/python dynamic_split/run_stage2a.py --out results/dynamic_split_20261012/stage2a_two_ru
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import cfsim  # noqa: E402
import geo2ru as g  # noqa: E402
import stage2_eval as e2  # noqa: E402
from runio import Run  # noqa: E402

MODELS = ['mcs', 'shannon']


def task(item):
    seed, num_ue, d, noises = item
    rng = np.random.default_rng([seed, num_ue, d])
    h, rays, _ = g.drop(rng, num_ue)
    ci = e2.ci_reconstructions(rng, h, rays, g.CI_NOISE_MW)
    ci_bits = {k: v[1] for k, v in ci.items()}
    n_cfg = len(e2.CONFIGS)
    shape = (len(MODELS), len(noises))
    se = np.zeros(shape + (n_cfg,))
    dl = np.zeros(shape + (n_cfg, g.L))
    ul = np.zeros(shape + (n_cfg, g.L))
    ideal = np.zeros(shape)
    for mi, model in enumerate(MODELS):
        for ni, nd in enumerate(noises):
            ev = e2.Drop2(h, ci, np.random.default_rng([seed, num_ue, d, mi, ni]), 10 ** (nd / 10), model)
            res = ev.evaluate()
            ideal[mi, ni] = ev.se_ideal.sum()
            for ci_i, cfg in enumerate(e2.CONFIGS):
                se_ue = res[cfg]
                se[mi, ni, ci_i] = se_ue.sum()
                dl[mi, ni, ci_i], ul[mi, ni, ci_i] = e2.loads(cfg, num_ue, cfsim.qm_from_se(se_ue), ci_bits)
    return (num_ue, d), ideal, se, dl, ul


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--ues', type=int, nargs='+', default=[8, 16, 32])
    p.add_argument('--noise', type=float, nargs='+', default=[-110., -90., -70.])
    p.add_argument('--drops', type=int, default=100)
    p.add_argument('--seed', type=int, default=20261012)
    p.add_argument('--workers', type=int, default=48)
    args = p.parse_args()
    here = Path(__file__).parent
    run = Run(args.out, [here / f for f in ('run_stage2a.py', 'stage2_eval.py', 'geo2ru.py', 'distortion.py',
                                            'fhload.py', 'cfsim.py', 'runio.py')],
              dict(purpose='Stage 2a: two 32-element O-RUs, stochastic clustered channel; joint choice of '
                           'central (7-1 / 7-2) vs local precoding, U-plane/weight widths and CI scheme',
                   args=vars(args), models=MODELS, configs=e2.CONFIGS, ci_noise_mw=g.CI_NOISE_MW,
                   p_ru_prb_mw=g.P_RU_PRB_MW, thermal_dbm_prb=g.N_TH_PRB_DBM,
                   assumptions=g.ASSUMPTIONS + [
                       'Network-wide configuration (both O-RUs identical); full buffer, every UE scheduled on all PRBs.',
                       'Central RZF (alpha = K N / (L P_RU)), unit-norm columns, UE power ~ network gain^nu with nu in '
                       '{0, 0.5} picked by the DU from its own CI (predicted SE), common scaling to the per-O-RU limit; '
                       'local RZF per O-RU with power share g_kl^nu / sum_j g_jl^nu, nu in {0, 0.5}.',
                       'Reference: central RZF on raw noisy CI with an ideal fronthaul; twin-subspace (KLT) CI also '
                       'denoises, so its SE ratio can exceed 1.',
                       'Weights per slot (SE-11, bundle = 2 PRB, one block = 32 weights); modulation compression lossless.',
                       'UL CI: raw = antenna-level SRS IQ BFP9 (16 UEs per SRS symbol); ST 6 BFP W per UE/group/TRX; '
                       'KLT proxy bits from noise-matched reverse water-filling; C-plane headers per 1450 bytes.',
                       'Each 2-PRB group simulated as one PRB slot (168 REs) at the group-centre channel; loads for 52 PRB.']))
    tasks = [(args.seed, u, d, args.noise) for u in args.ues for d in range(args.drops)]
    shape = (len(args.ues), args.drops, len(MODELS), len(args.noise))
    n_cfg = len(e2.CONFIGS)
    arr = dict(ideal=np.zeros(shape), se=np.zeros(shape + (n_cfg,)),
               dl=np.zeros(shape + (n_cfg, g.L)), ul_bits=np.zeros(shape + (n_cfg, g.L)))
    with ProcessPoolExecutor(args.workers) as ex:
        for i, ((u, d), ideal, se, dl, ul) in enumerate(ex.map(task, tasks, chunksize=1)):
            ui = args.ues.index(u)
            arr['ideal'][ui, d], arr['se'][ui, d], arr['dl'][ui, d], arr['ul_bits'][ui, d] = ideal, se, dl, ul
            if (i + 1) % 30 == 0:
                print(f'{i + 1}/{len(tasks)} drops', flush=True)
    np.savez_compressed(run.path / 'per_drop.npz', ues=args.ues, noise=args.noise, models=MODELS, **arr)
    for ui, u in enumerate(args.ues):
        for mi, model in enumerate(MODELS):
            for ni, nd in enumerate(args.noise):
                ideal = float(arr['ideal'][ui, :, mi, ni].mean())
                for c, cfg in enumerate(e2.CONFIGS):
                    run.add(dict(se_model=model, num_ue=u, noise_dbm=nd, family=cfg[0], uplane=cfg[1], bfw=cfg[2],
                                 ci=cfg[3], se=float(arr['se'][ui, :, mi, ni, c].mean()), se_ideal=ideal,
                                 se_ratio=float(arr['se'][ui, :, mi, ni, c].mean()) / ideal,
                                 dl_gbps=float(arr['dl'][ui, :, mi, ni, c].sum(-1).mean()) / 1e9,
                                 ul_kbit_per_srs=float(arr['ul_bits'][ui, :, mi, ni, c].sum(-1).mean()) / 1e3))
    run.finish()
    print('done', run.path)


if __name__ == '__main__':
    main()
