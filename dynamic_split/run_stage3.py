"""阶段 3（方向 ②+③）：服务集合 × 每 RU 切分 × 压缩的联合选择，在论文场景（阶段 0）上逐撒点运行。

每个（频谱效率模型, 状态, 撒点, K）记录：
  pts[path, κ_s, λ]  = (和速率, 总负载 Gbit/s, 最忙链路 Gbit/s, 7-2 RU 比例)，path ∈ {paper, la}
  b1[κ_s, 配置]       = 全网统一配置（论文服务集合）的 (和速率, 总负载, 最忙链路)
  b0                  = 论文原方法：状态级 M̂、按论文公式（q = 32、γ 常数）选全网切分、不压缩，用显式口径计负载
  se_max              = 论文服务集合路径上的最大理想和速率（κ 的参照）
波束赋形方式与 M̂ 取自阶段 0（与论文相同的按状态选择）。

用法：
    .venv/bin/python dynamic_split/run_stage3.py results/dynamic_split_20261012/stage0_repro \
        --out results/dynamic_split_20261012/stage3_joint
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
import fhload  # noqa: E402
import joint as J  # noqa: E402
import summarize_stage0 as s0  # noqa: E402
from runio import Run, read_rows  # noqa: E402

KS = [1, 2, 4, 8, 16, 32, 64]
MODELS = ['mcs', 'shannon']
B1_CFGS = [('7-1', o) for o in J.UP71] + [('7-2', b) for b in J.BFW]
CAL = None


def first_index(values, target):
    idx = np.nonzero(np.asarray(values) >= target - 1e-9)[0]
    return int(idx[0]) if len(idx) else None


def b0_point(model, st, k):
    """论文原方法：按论文公式（q = 32，γ 常数，全部 RU 计入）在全网 7-1 / 7-2 间选，再按显式口径、不压缩计负载。"""
    serving, assign = st['serving'], st['assign']
    ru = serving // k
    n_ru = cfsim.NUM_ANT // k
    n_ant = np.bincount(ru, minlength=n_ru)
    if model.bf == 'zf':
        n_ues = np.where(n_ant > 0, model.num_ue, 0)
    else:
        n_ues = np.array([len(set(assign[ru == r].tolist())) for r in range(n_ru)])
    gamma = fhload.RE_PER_S * 32
    r71 = gamma * n_ant.sum() + fhload.GAMMA_71 * n_ru
    r72 = gamma * n_ues.sum() + fhload.GAMMA_72 * n_ru
    split = '7-1' if r71 <= r72 else '7-2'
    total, link = 0.0, 0.0
    d = np.zeros_like(st['sig'])
    for r in np.unique(ru):
        idx = serving[ru == r]
        ues = list(range(model.num_ue)) if model.bf == 'zf' else sorted(set(assign[ru == r].tolist()))
        if split == '7-1':
            dl, ul = J.ru_load('7-1', 'none', len(idx), ues, st['qm'], model.bf, model.num_ue)
            d += np.einsum('bun->bu', model.h2[:, :, idx]) * cfsim.P_MAX_PRB_MW * CAL['floor']
        else:
            dl, ul = J.ru_load('7-2', 16, len(idx), ues, st['qm'], model.bf, model.num_ue, uplane72='none')
        total += dl + ul
        link = max(link, dl, ul)
    return model.se(st, d), total / 1e9, link / 1e9, float(split == '7-2')


def task(item):
    (mi, si, d), (num_ue, noise_dbm, se_model, bf, m_hat, row) = item
    h, beta, _ = cfsim.drop(np.random.default_rng([2024, d, num_ue]), num_ue, 'm2135')
    assert (cfsim.antenna_order(beta) == np.asarray(row['order'])).all()
    model = J.DropModel(h, bf, 10 ** (noise_dbm / 10), se_model, CAL)
    curve = np.asarray(row['curve'])
    se_max = float(curve.max())
    ppath = J.paper_path(row)
    n_k, n_ks, n_l = len(KS), len(J.KAPPA_S), len(J.LAMBDAS)
    pts = np.full((n_k, 2, n_ks, n_l, 4), np.nan)
    b1 = np.full((n_k, n_ks, len(B1_CFGS), 3), np.nan)
    b0 = np.full((n_k, 4), np.nan)
    pstates = {}

    def paper_state(n):
        if n not in pstates:
            pstates[n] = model.state(*ppath[n])
        return pstates[n]

    for ki, k in enumerate(KS):
        lpath = J.la_path(model, np.asarray(row['order']), k, se_model)
        b0[ki] = b0_point(model, paper_state(m_hat - 1), k)
        for pi in range(2):
            for ks_i, ks in enumerate(J.KAPPA_S):
                if pi == 0:
                    n = first_index(curve, ks * se_max)
                    if n is None:
                        continue
                    st = paper_state(n)
                else:
                    n = first_index([p[2] for p in lpath], ks * se_max)
                    if n is None:
                        continue
                    st = model.state(lpath[n][0], lpath[n][1])
                opts = model.ru_options(st, k)
                for li, lam in enumerate(J.LAMBDAS):
                    se, tot, link, f72, _ = model.lagrange(st, opts, lam)
                    pts[ki, pi, ks_i, li] = (se, tot, link, f72)
                if pi == 0:
                    for ci, key in enumerate(B1_CFGS):
                        if key[0] == '7-2' and bf == 'mrt':
                            key = ('7-2', None)
                            if ci != len(J.UP71):
                                continue
                        b1[ki, ks_i, ci] = model.uniform(st, opts, key)
    return (mi, si, d), se_max, pts, b1, b0


def init_worker(cal):
    global CAL
    CAL = cal


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage0')
    p.add_argument('--out', required=True)
    p.add_argument('--drops', type=int, default=100)
    p.add_argument('--workers', type=int, default=48)
    args = p.parse_args()
    here = Path(__file__).parent
    stage0 = Path(args.stage0)
    cal = J.calibrate()
    run = Run(args.out, [here / f for f in ('run_stage3.py', 'joint.py', 'distortion.py', 'fhload.py', 'cfsim.py',
                                            'summarize_stage0.py', 'runio.py')],
              dict(purpose='Stage 3 (directions 2+3): joint serving set, per-RU split (7-1/7-2) and compression; '
                           'P1 min load s.t. SE >= 0.98 SE_max, P2 max SE under fronthaul capacity; per-drop and per-state',
                   stage0_run=str(stage0), args=vars(args), ks=KS, models=MODELS, kappa=J.KAPPA,
                   kappa_s=J.KAPPA_S, lambdas=J.LAMBDAS.tolist(), b1_cfgs=B1_CFGS,
                   calibration=dict(r71_db={k: 10 * np.log10(v) for k, v in cal['r71'].items()}),
                   assumptions=['Scenario, BF type per state and M-hat from stage 0 (paper method); MCS-capped and Shannon SE.',
                                'Per-RU options: 7-1 U-plane BFP4-12/16 bit; 7-2 modulation compression + SE-11 weights '
                                'BFP4-12/16 bit (ZF, per slot, bundle 1 PRB) or RU-local weights (MRT).',
                                'Optimizer distortion model: 7-1 white per-antenna noise = power x r71(w) (codec-calibrated); '
                                '7-2 weight errors from the BFP rule per block (analytic), leakage to other UEs only.',
                                'Per-RU choice by Lagrangian relaxation with Shannon-linearized SE loss; feasibility with the '
                                'true SE model. Real-codec validation in validate_stage3.py.',
                                'CI in the load as stage 1a (antenna-level SRS IQ BFP9 every 10 ms; ZF 7-2 min with ST 6).',
                                'B0 = paper: state-level M-hat, network-wide split by the paper formula (q = 32, gamma), no '
                                'compression, load evaluated with the explicit O-RAN accounting.']))
    rows0 = read_rows(stage0)
    states = s0.select_states(rows0)
    keys = sorted({(u, n) for (u, n, _) in states})
    items = []
    for mi, sm in enumerate(MODELS):
        for si, (u, n) in enumerate(keys):
            st = states[(u, n, sm)]
            bf, m_hat = st['bf'], st[st['bf']]['m_hat']
            for r in rows0:
                if (r['num_ue'], r['noise_dbm'], r['se_model'], r['bf']) == (u, n, sm, bf) and r['drop'] < args.drops:
                    items.append(((mi, si, r['drop']), (u, n, sm, bf, m_hat, r)))
    shp = (len(MODELS), len(keys), args.drops)
    arr = dict(se_max=np.full(shp, np.nan),
               pts=np.full(shp + (len(KS), 2, len(J.KAPPA_S), len(J.LAMBDAS), 4), np.nan, np.float32),
               b1=np.full(shp + (len(KS), len(J.KAPPA_S), len(B1_CFGS), 3), np.nan, np.float32),
               b0=np.full(shp + (len(KS), 4), np.nan))
    with ProcessPoolExecutor(args.workers, initializer=init_worker, initargs=(cal,)) as ex:
        for i, (idx, se_max, pts, b1, b0) in enumerate(ex.map(task, items, chunksize=1)):
            arr['se_max'][idx], arr['pts'][idx], arr['b1'][idx], arr['b0'][idx] = se_max, pts, b1, b0
            if (i + 1) % 100 == 0:
                print(f'{i + 1}/{len(items)} drops', flush=True)
    np.savez_compressed(run.path / 'stage3.npz', states=np.array(keys), models=MODELS, ks=KS, **arr)
    for mi, sm in enumerate(MODELS):
        for si, (u, n) in enumerate(keys):
            run.add(dict(se_model=sm, num_ue=u, noise_dbm=n, bf=states[(u, n, sm)]['bf'],
                         m_hat=states[(u, n, sm)][states[(u, n, sm)]['bf']]['m_hat'],
                         se_max_mean=float(np.nanmean(arr['se_max'][mi, si]))))
    run.finish()
    print('done', run.path)


if __name__ == '__main__':
    main()
