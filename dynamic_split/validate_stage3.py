"""阶段 3 复核：把逐撒点 P1 选出的配置（服务集合 + 每 RU 的切分与位宽）原样重建，
让信号经过 oran/compression 的真实编解码器，用 UE 端 LMMSE 得到和速率，检查是否仍 ≥ κ·SE_max。

7-1 RU：x = W s 逐天线、逐 PRB 符号 12 RE 为块做 U-plane 压缩；
7-2 RU（ZF）：每 (PRB, UE) 的 n_r 个权值为一块做 BFP，x = Ŵ s（层符号调制压缩无损）；7-2 RU（MRT）：权值本地，不量化。

用法：
    .venv/bin/python dynamic_split/validate_stage3.py results/dynamic_split_20261012/stage3_joint \
        results/dynamic_split_20261012/stage0_repro --drops 20
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
import analyse_stage3 as A  # noqa: E402
import cfsim  # noqa: E402
import distortion as dz  # noqa: E402
import joint as J  # noqa: E402
import run_stage3 as R3  # noqa: E402
import summarize_stage0 as s0  # noqa: E402
from runio import read_rows  # noqa: E402

N_RE = dz.N_RE


def real_se(model, st, choice, k, rng):
    sinr = real_sinr(model, st, choice, k, rng)
    return float(dz.se_per_ue(sinr, model.model).sum())


def real_sinr(model, st, choice, k, rng):
    """经真实编解码器后的每 PRB 每 UE 有效 SINR [B, U]（UE 端按 PRB 做 LMMSE）。"""
    h, w = model.h, st['w']
    b_, u_, _ = h.shape
    s = np.stack([dz.qam(rng, st['qm'][u], (b_, N_RE)) for u in range(u_)], axis=1)
    y = np.zeros((b_, u_, N_RE), complex)
    ru = st['serving'] // k
    for r in np.unique(ru):
        idx = st['serving'][ru == r]
        split, opt = choice[int(r)]
        wr = w[:, idx, :]
        if split == '7-1':
            x = dz.apply(opt, wr @ s, cfsim.P_MAX_PRB_MW, 12)
        elif opt is None:
            x = wr @ s
        else:
            o = 'none' if opt >= 16 else f'bfp{opt}'
            w_hat = np.swapaxes(dz.apply(o, np.swapaxes(wr, 1, 2), cfsim.P_MAX_PRB_MW / u_, len(idx)), 1, 2)
            x = w_hat @ s
        y += h[:, :, idx] @ x
    ys = np.sum(y * np.conj(s), axis=2)
    yy = np.sum(np.abs(y) ** 2, axis=2)
    ss = np.sum(np.abs(s) ** 2, axis=2)
    return dz.sinr_from_stats(ys, yy, ss, model.noise)


def task(item):
    (mi, si, d, ki), (num_ue, noise_dbm, se_model, bf, row, picks, se_max, cal) = item
    h, _, _ = cfsim.drop(np.random.default_rng([2024, d, num_ue]), num_ue, 'm2135')
    model = J.DropModel(h, bf, 10 ** (noise_dbm / 10), se_model, cal)
    k = R3.KS[ki]
    curve = np.asarray(row['curve'])
    ppath = J.paper_path(row)
    lpath = None
    out = {}
    for method, (pi, ks_i, li_or_ci) in picks.items():
        target = J.KAPPA_S[ks_i] * se_max
        if pi == 0:
            n = R3.first_index(curve, target)
            st = model.state(*ppath[n])
        else:
            if lpath is None:
                lpath = J.la_path(model, np.asarray(row['order']), k, se_model)
            n = R3.first_index([p[2] for p in lpath], target)
            st = model.state(lpath[n][0], lpath[n][1])
        opts = model.ru_options(st, k)
        if method == 'B1':
            key = R3.B1_CFGS[li_or_ci]
            if key[0] == '7-2' and bf == 'mrt':
                key = ('7-2', None)
            choice = {r: key for r, _ in opts}
            se_model_pred = model.uniform(st, opts, key)[0]
        else:
            se_model_pred, _, _, _, choice = model.lagrange(st, opts, J.LAMBDAS[li_or_ci])
        se_real = real_se(model, st, choice, k, np.random.default_rng([d, ki, 5]))
        out[method] = (se_model_pred / se_max, se_real / se_max)
    return (mi, si, d, ki), out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage3')
    p.add_argument('stage0')
    p.add_argument('--drops', type=int, default=20)
    p.add_argument('--ks', type=int, nargs='+', default=[1, 4, 16, 64])
    p.add_argument('--workers', type=int, default=48)
    args = p.parse_args()
    z = dict(np.load(Path(args.stage3) / 'stage3.npz'))
    rows0 = read_rows(args.stage0)
    states = s0.select_states(rows0)
    keys = [tuple(s) for s in z['states']]
    cal = J.calibrate()
    items = []
    for mi, sm in enumerate(z['models']):
        for si, (u, n) in enumerate(keys):
            u, n = int(u), float(n)
            bf = states[(u, n, sm)]['bf']
            rows = {r['drop']: r for r in rows0 if (r['num_ue'], r['noise_dbm'], r['se_model'], r['bf']) == (u, n, sm, bf)}
            for ki, k in enumerate(R3.KS):
                if k not in args.ks:
                    continue
                for m in ('P', 'B2', 'B1'):
                    se, load, link, f72 = A.candidates(z, m, mi, si, ki)
                    ok = se >= A.KAPPA * z['se_max'][mi, si][:, None] - 1e-9
                    z.setdefault('_pick', {})[(mi, si, ki, m)] = np.argmin(np.where(ok, load, np.inf), axis=1), ok.any(axis=1)
                for d in range(args.drops):
                    picks = {}
                    for m in ('P', 'B2', 'B1'):
                        j, feas = z['_pick'][(mi, si, ki, m)]
                        if not feas[d]:
                            continue
                        if m == 'P':
                            picks[m] = np.unravel_index(j[d], (2, len(J.KAPPA_S), len(J.LAMBDAS)))
                        elif m == 'B2':
                            a, b = np.unravel_index(j[d], (len(J.KAPPA_S), len(J.LAMBDAS)))
                            picks[m] = (0, a, b)
                        else:
                            a, b = np.unravel_index(j[d], (len(J.KAPPA_S), len(R3.B1_CFGS)))
                            picks[m] = (0, a, b)
                        picks[m] = tuple(int(v) for v in picks[m])
                    items.append(((mi, si, d, ki), (u, n, str(sm), bf, rows[d], picks,
                                                    float(z['se_max'][mi, si, d]), cal)))
    results = []
    with ProcessPoolExecutor(args.workers) as ex:
        for idx, out in ex.map(task, items, chunksize=2):
            for m, (pred, real) in out.items():
                results.append(dict(model=str(z['models'][idx[0]]), state=list(keys[idx[1]]), drop=idx[2],
                                    k=R3.KS[idx[3]], method=m, pred_ratio=pred, real_ratio=real))
    path = Path(args.stage3) / 'validation_real_codec.json'
    path.write_text(json.dumps(results, indent=1, default=float))
    for model in sorted({r['model'] for r in results}):
        print(f'== {model}: real-codec SE / SE_max for the per-drop P1 picks (target {A.KAPPA})')
        for m in ('P', 'B2', 'B1'):
            rs = [r for r in results if r['model'] == model and r['method'] == m]
            real = np.array([r['real_ratio'] for r in rs])
            pred = np.array([r['pred_ratio'] for r in rs])
            print(f'   {m}: n={len(rs)}  predicted mean {pred.mean():.4f}  real mean {real.mean():.4f}  '
                  f'real < {A.KAPPA}: {np.mean(real < A.KAPPA - 1e-9):.1%}  real < 0.97: {np.mean(real < 0.97):.1%}  '
                  f'min {real.min():.3f}')
    print('saved', path)


if __name__ == '__main__':
    main()
