"""S8：用命题 2 预测"动态切分何时不再需要"的 K 边界，并与 S7 的观察对照。

对每个状态取 S7 的逐状态最优 (κ_s, τ)，在 [10] 服务集合上统计：
  M̂(s)        服务天线数（撒点平均）
  Q̄m(s)       UE 的平均调制阶数
  n*(s)        命题 2 的门限，w = b = τ(s)（名义位宽）
  n̄_r(s, K)   每个激活 RU 的平均服务天线数；P(n_r > n*) 为按门限应选 7-2 的 RU 比例（半解析预测）
闭式边界（命题 4 草稿）：平均每个激活 RU 的服务天线数为 M̂ K / N（RU 都被激活时），
  当所有 ZF 状态都满足 M̂(s) K / N > n*(s) 时 7-2 处处最优，切分不需随状态变化，即
  K* = max_{ZF 状态} N n*(s) / M̂(s)。

用法：
    .venv/bin/python dynamic_split/predict_boundary.py results/dynamic_split_20261012/stage7_dynamics \
        --out results/dynamic_split_20261012/stage8_boundary
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import analyse_stage7 as A7  # noqa: E402
import cfsim  # noqa: E402
import faithful  # noqa: E402
import heuristics10 as H10  # noqa: E402
import joint as J  # noqa: E402
import run_stage3 as R3  # noqa: E402
from closed_form import TAU  # noqa: E402
from runio import Run  # noqa: E402

N_ANT = cfsim.NUM_ANT


def n_star(u, qm_mean, w, b):
    return u * (168 * qm_mean + 24) / (168 * (2 * w + 2 / 3) - 2 * b * u)


def task(item):
    (si, d), (u, nd, bf, ks_i, model_name, se_model) = item
    faithful.setup(model_name)
    h, _, _ = cfsim.drop(np.random.default_rng([2024, d, u]), u, 'm2135')
    model = J.DropModel(h, bf, faithful.noise(model_name, d, u, nd), se_model, None)
    path = H10.path(h, bf, model.noise, se_model)
    curve = [p[2] for p in path]
    serving, assign = path[R3.first_index(curve, J.KAPPA_S[ks_i] * max(curve))][:2]
    st = model.state(serving, assign)
    per_k = []
    for k in R3.KS:
        ru = np.asarray(serving) // k
        _, counts = np.unique(ru, return_counts=True)
        per_k.append(counts.tolist())
    return (si, d), len(serving), float(st['qm'].mean()), per_k


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage7')
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=48)
    args = p.parse_args()
    here = Path(__file__).parent
    run = Run(args.out, [here / f for f in ('predict_boundary.py', 'heuristics10.py', 'analyse_stage7.py')],
              dict(purpose='S8: predict the K boundary of split dynamics from Proposition 2 and compare with S7',
                   stage7=args.stage7))
    z = dict(np.load(Path(args.stage7) / 'stage7.npz'))
    summ = json.loads((Path(args.stage7) / 'stage7_summary.json').read_text())
    states = [tuple(s) for s in z['states']]
    bfs = list(z['bfs'])
    n_drops = z['se_max'].shape[1]
    # 每个状态的最优 (κ_s, τ)：取 K = 16 处的逐状态最优（τ 在各 K 间几乎不变，见 S7 报告）
    ki16 = list(z['ks']).index(16)
    g = z['grid'][:, :, ki16, :, 0]                          # [S, D, KS, T, 3]
    s_, d_ = z['se_max'].shape
    _, js = A7.per_state_best(g[..., 0].reshape(s_, d_, -1), g[..., 1].reshape(s_, d_, -1), z['se_max'])
    ks_idx = [int(j // len(TAU)) for j in js]
    tau = [float(TAU[j % len(TAU)]) for j in js]
    meta = json.loads((Path(args.stage7) / 'metadata.json').read_text())
    model_name, se_model = meta.get('model', 'fixed'), meta.get('se_model', 'mcs')
    items = [((si, d), (int(u), float(n), bfs[si], ks_idx[si], model_name, se_model))
             for si, (u, n) in enumerate(states) for d in range(n_drops) if np.isfinite(z['se_max'][si, d])]
    m_hat = np.full((s_, d_), np.nan)
    qm = np.full((s_, d_), np.nan)
    counts = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for (si, d), m, q, per_k in ex.map(task, items, chunksize=4):
            m_hat[si, d], qm[si, d] = m, q
            for ki, c in enumerate(per_k):
                counts.setdefault((si, ki), []).extend(c)
    print('Per-state quantities (ZF states; MRT states always choose 7-2 by Corollary 2)')
    k_star = []
    for si, (u, n) in enumerate(states):
        if bfs[si] != 'zf':
            continue
        ns = n_star(u, np.nanmean(qm[si]), tau[si], tau[si])
        ks = N_ANT * ns / np.nanmean(m_hat[si])
        k_star.append(ks)
        row = dict(num_ue=int(u), noise_dbm=float(n), m_hat=float(np.nanmean(m_hat[si])), qm_mean=float(np.nanmean(qm[si])),
                   tau=tau[si], n_star=float(ns), k_star=float(ks))
        obs = {}
        for ki, k in enumerate(R3.KS):
            c = np.array(counts[(si, ki)])
            obs[k] = dict(pred_share72=float(np.mean(c > ns)), mean_n_r=float(c.mean()),
                          observed_share72=float(next(e for e in summ if e['k'] == k)['per_state_f72'][si]))
        row['per_k'] = obs
        run.add(row)
        print(f"  ({int(u):2d},{int(n)}): M̂ {row['m_hat']:5.1f}  Q̄m {row['qm_mean']:.2f}  τ {tau[si]:.2f}  n* {ns:5.2f}  "
              f"K* = N n*/M̂ = {ks:5.1f} | P(n_r>n*) pred/obs: " +
              '  '.join(f"K{k}:{obs[k]['pred_share72']:.2f}/{obs[k]['observed_share72']:.2f}" for k in R3.KS))
    kb = max(k_star)
    print(f'\nPredicted boundary K* = max over ZF states = {kb:.1f} antennas per RU '
          f'(S7: dynamic gain 10.8% at K = 32, 0.8% at K = 64)')
    run.finish(k_star=kb)


if __name__ == '__main__':
    main()
