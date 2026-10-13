"""S7 复核：三步闭式方法（规则模式）逐撒点选出的配置经过真实编解码器（oran/compression）后的和速率。

用法：
    .venv/bin/python dynamic_split/validate_cf.py results/dynamic_split_20261012/stage7_dynamics --drops 20
"""

import argparse
import json
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
import closed_form as CF  # noqa: E402
import heuristics10 as H10  # noqa: E402
import joint as J  # noqa: E402
import run_stage3 as R3  # noqa: E402
from validate_stage3 import real_se  # noqa: E402


def task(item):
    (si, d, ki), (num_ue, noise_dbm, bf, ks_i, tau, pred, cal, model_name, se_model) = item
    faithful.setup(model_name)
    h, _, _ = cfsim.drop(np.random.default_rng([2024, d, num_ue]), num_ue, 'm2135')
    model = J.DropModel(h, bf, faithful.noise(model_name, d, num_ue, noise_dbm), se_model, cal)
    path = H10.path(h, bf, model.noise, se_model)
    curve = [p[2] for p in path]
    se_max = float(max(curve))
    st = model.state(*path[R3.first_index(curve, J.KAPPA_S[ks_i] * se_max)][:2])
    k = R3.KS[ki]
    cs = CF.CFState(model, st, k, cal)
    idx, picks = cs.choose(tau, 'rule')
    se_model = cs.evaluate(idx)[0]
    choice = {r: key for r, key in zip(cs.rus, picks)}
    se = real_se(model, st, choice, k, np.random.default_rng([d, ki, 9]))
    return dict(state=[num_ue, noise_dbm], drop=d, k=k, pred_ratio=se_model / se_max, stored_ratio=pred,
                real_ratio=se / se_max)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage7')
    p.add_argument('--drops', type=int, default=20)
    p.add_argument('--ks', type=int, nargs='+', default=[1, 4, 16, 64])
    p.add_argument('--workers', type=int, default=48)
    args = p.parse_args()
    z = dict(np.load(Path(args.stage7) / 'stage7.npz'))
    cal = J.calibrate()
    meta = json.loads((Path(args.stage7) / 'metadata.json').read_text())
    model_name, se_model = meta.get('model', 'fixed'), meta.get('se_model', 'mcs')
    items = []
    for si, (u, n) in enumerate(z['states']):
        for ki, k in enumerate(R3.KS):
            if k not in args.ks:
                continue
            g = z['grid'][si, :, ki, :, 0]                        # [D, KS, T, 3]（规则模式）
            for d in range(args.drops):
                if not np.isfinite(z['se_max'][si, d]):          # 无效撒点
                    continue
                r, l = g[d, ..., 0], g[d, ..., 1]
                masked = np.where(r >= J.KAPPA - 1e-9, l, np.inf)
                ks_i, ti = np.unravel_index(int(masked.argmin()), masked.shape)
                items.append(((si, d, ki), (int(u), float(n), str(z['bfs'][si]), int(ks_i), float(CF.TAU[ti]),
                                            float(r[ks_i, ti]), cal, model_name, se_model)))
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(task, items, chunksize=2))
    (Path(args.stage7) / 'validation_real_codec.json').write_text(json.dumps(res, indent=1))
    real = np.array([x['real_ratio'] for x in res])
    pred = np.array([x['pred_ratio'] for x in res])
    print(f'n = {len(res)}  predicted mean {pred.mean():.4f}  real mean {real.mean():.4f}  '
          f'real < {J.KAPPA}: {np.mean(real < J.KAPPA - 1e-9):.1%}  real < 0.97: {np.mean(real < 0.97):.1%}  min {real.min():.3f}')
    for k in args.ks:
        rr = np.array([x['real_ratio'] for x in res if x['k'] == k])
        pp = np.array([x['pred_ratio'] for x in res if x['k'] == k])
        print(f'   K={k:2d}: predicted {pp.mean():.4f}  real {rr.mean():.4f}  real < {J.KAPPA}: {np.mean(rr < J.KAPPA - 1e-9):.1%}')


if __name__ == '__main__':
    main()
