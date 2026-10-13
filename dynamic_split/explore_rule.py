"""S4：用阶段 3 的逐撒点 P1 选择检验理论笔记的命题 2 / 推论 2，并记录服务集合结构（n_r、激活 RU 数）。

对每个 (状态, 撒点, K) 重建方法 P 与 B2 选出的配置（服务集合 + 每 RU 的切分与位宽），逐 RU 记录：
  n_r、|U_r|、U_r 内的 Qm、所选切分与位宽；
  命题 2 的预测（取所选位宽，另一种切分假设同样位宽 w = b）；
  同样位宽下两种切分的精确负载（joint.ru_load，含报头与 CI），看优化器是否按负载在选。

用法：
    .venv/bin/python dynamic_split/explore_rule.py results/dynamic_split_20261012/stage3_joint \
        results/dynamic_split_20261012/stage0_repro --out results/dynamic_split_20261012/stage4_rule --drops 30
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
import analyse_stage3 as A  # noqa: E402
import cfsim  # noqa: E402
import joint as J  # noqa: E402
import run_stage3 as R3  # noqa: E402
import summarize_stage0 as s0  # noqa: E402
from runio import Run, read_rows  # noqa: E402


def payload71(n, w):
    return n * 168 * (2 * w + 2 / 3)


def payload72(qms, n, b, zf):
    return 168 * float(np.sum(qms)) + (len(qms) * (2 * b * n + 24) if zf else 0.0)


def width_of(split, opt):
    if split == '7-1':
        return 16 if opt == 'none' else int(opt[3:])
    return 16 if opt is None else int(opt)


def task(item):
    (si, d, ki), (num_ue, noise_dbm, se_model, bf, row, picks, se_max, cal) = item
    h, _, _ = cfsim.drop(np.random.default_rng([2024, d, num_ue]), num_ue, 'm2135')
    model = J.DropModel(h, bf, 10 ** (noise_dbm / 10), se_model, cal)
    k = R3.KS[ki]
    curve = np.asarray(row['curve'])
    ppath = J.paper_path(row)
    lpath = None
    rows = []
    for method, (pi, ks_i, li) in picks.items():
        target = J.KAPPA_S[ks_i] * se_max
        if pi == 0:
            st = model.state(*ppath[R3.first_index(curve, target)])
        else:
            if lpath is None:
                lpath = J.la_path(model, np.asarray(row['order']), k, se_model)
            n = R3.first_index([p[2] for p in lpath], target)
            st = model.state(lpath[n][0], lpath[n][1])
        opts = model.ru_options(st, k)
        se, tot, link, f72, choice = model.lagrange(st, opts, J.LAMBDAS[li])
        ru = st['serving'] // k
        for r, (split, opt) in choice.items():
            m = ru == r
            n_r = int(m.sum())
            ues = list(range(num_ue)) if bf == 'zf' else sorted(set(st['assign'][m].tolist()))
            qms = st['qm'][ues]
            w = width_of(split, opt) if bf == 'zf' or split == '7-1' else 6
            l71, l72 = payload71(n_r, w), payload72(qms, n_r, w, bf == 'zf')
            e71 = sum(J.ru_load('7-1', 'none' if w >= 16 else f'bfp{min(w, 12)}', n_r, ues, st['qm'], bf, num_ue))
            e72 = sum(J.ru_load('7-2', w if bf == 'zf' else None, n_r, ues, st['qm'], bf, num_ue))
            n_star = (len(ues) * (168 * qms.mean() + 24) / (168 * (2 * w + 2 / 3) - 2 * w * len(ues))
                      if bf == 'zf' else 0.0)
            rows.append(dict(method=method, state=[num_ue, noise_dbm], drop=d, k=k, bf=bf, ru=int(r), n_r=n_r,
                             n_ue_r=len(ues), qm_mean=float(qms.mean()), split=split, width=w,
                             rule_split='7-2' if l72 < l71 else '7-1', exact_split='7-2' if e72 < e71 else '7-1',
                             n_star=float(n_star), n_active=len(choice), n_serving=int(len(st['serving'])),
                             load_gbps=tot, se_ratio=se / se_max))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage3')
    p.add_argument('stage0')
    p.add_argument('--out', required=True)
    p.add_argument('--drops', type=int, default=30)
    p.add_argument('--workers', type=int, default=48)
    args = p.parse_args()
    here = Path(__file__).parent
    run = Run(args.out, [here / f for f in ('explore_rule.py', 'joint.py', 'run_stage3.py', 'analyse_stage3.py')],
              dict(purpose='S4: test theory_notes Proposition 2 / Corollary 2 on stage-3 per-RU choices; '
                           'serving-set structure of P vs B2', stage3=args.stage3, args=vars(args), se_model='mcs'))
    z = dict(np.load(Path(args.stage3) / 'stage3.npz'))
    rows0 = read_rows(args.stage0)
    states = s0.select_states(rows0)
    keys = [tuple(s) for s in z['states']]
    cal = J.calibrate()
    mi, sm = 0, 'mcs'
    items = []
    for si, (u, n) in enumerate(keys):
        u, n = int(u), float(n)
        bf = states[(u, n, sm)]['bf']
        rows = {r['drop']: r for r in rows0 if (r['num_ue'], r['noise_dbm'], r['se_model'], r['bf']) == (u, n, sm, bf)}
        for ki in range(len(R3.KS)):
            pick = {}
            for m in ('P', 'B2'):
                se, load, link, f72 = A.candidates(z, m, mi, si, ki)
                ok = se >= A.KAPPA * z['se_max'][mi, si][:, None] - 1e-9
                pick[m] = (np.argmin(np.where(ok, load, np.inf), axis=1), ok.any(axis=1))
            for d in range(args.drops):
                picks = {}
                for m in ('P', 'B2'):
                    j, feas = pick[m]
                    if not feas[d]:
                        continue
                    if m == 'P':
                        picks[m] = tuple(int(v) for v in np.unravel_index(j[d], (2, len(J.KAPPA_S), len(J.LAMBDAS))))
                    else:
                        a, b = np.unravel_index(j[d], (len(J.KAPPA_S), len(J.LAMBDAS)))
                        picks[m] = (0, int(a), int(b))
                items.append(((si, d, ki), (u, n, sm, bf, rows[d], picks, float(z['se_max'][mi, si, d]), cal)))
    with ProcessPoolExecutor(args.workers) as ex:
        for rows in ex.map(task, items, chunksize=2):
            for r in rows:
                run.add(r)
    run.finish()
    print('done', run.path)


if __name__ == '__main__':
    main()
