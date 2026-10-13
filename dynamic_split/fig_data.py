"""S14：论文插图的数据（忠实口径 paper10、MCS 表 2；动态分解另有 1024QAM）。输出带表头、空格分隔的 .dat，供 pgfplots 读取。

  fig_prop1.dat        命题 1：(10 UE, −90 dBm, ZF, K = 8)，有效 SINR 随位宽：理想、命题 1、真实编解码器（7-1 / 7-2 权值）、上限 G/ρ(w)
  fig_prop2.dat        命题 2：逐 RU 拉格朗日优化器（OPT）的选择中，ZF 状态下 7-2 的比例随 log2(n_r / n*)
  fig_prop3.dat        命题 3：OPT 所选位宽与注水预测（各自减去同一撒点内的均值），分箱平均
  fig_dyn_mcs.dat / fig_dyn_mcs4.dat   动态分解（S10 / S11 的 stage7_summary.json）
  fig_boundary.dat     命题 4：各 ZF 状态 7-2 比例随 K，预测 / 观测（S13）；fig_kstar.dat 为各状态的 N n*/M̂
  fig_p2_k{4,16}.dat   P2：平均 SE/SE_max 随前传容量 C：闭式方法（逐撒点）、全网统一配置、论文不压缩方法
另存 summary.json（各部分的统计量，含论文不压缩方法在忠实口径下的负载）。

用法：
    .venv/bin/python dynamic_split/fig_data.py --out results/dynamic_split_20261012/stage14_figdata \
        --copy-to /home/user/Dynamic-Compression-With-Fronthaul-Load-Estimation/figures/data
"""

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import cfsim  # noqa: E402
import faithful  # noqa: E402
import heuristics10 as H10  # noqa: E402
import joint as J  # noqa: E402
import run_stage3 as R3  # noqa: E402
from runio import Run  # noqa: E402
from validate_stage3 import real_sinr  # noqa: E402

MODEL, SE = 'paper10', 'mcs'
R = Path('results/dynamic_split_20261012')
WIDTHS = list(range(4, 13)) + [16]
KS_OPT = [1, 2, 4, 8, 16, 32]
KS_P2 = [4, 16]
CAPS = np.logspace(np.log10(0.3), np.log10(40), 60)


def prep(u, nd, d, bf, cal):
    faithful.setup(MODEL)
    h, _, _ = cfsim.drop(np.random.default_rng([2024, d, u]), u, 'm2135')
    model = J.DropModel(h, bf, faithful.noise(MODEL, d, u, nd), SE, cal)
    path = H10.path(h, bf, model.noise, SE)
    curve = [p[2] for p in path]
    return model, path, curve, float(max(curve))


def db(x):
    return float(np.mean(10 * np.log10(np.maximum(x, 1e-30))))


# ---------------------------------------------------------------- A：命题 1
def task_prop1(item):
    d, cal = item
    model, path, curve, se_max = prep(10, -90.0, d, 'zf', cal)
    if se_max <= 0:
        return None
    st = model.state(*path[R3.first_index(curve, 0.98 * se_max)][:2])
    k = 8
    opts = model.ru_options(st, k)
    base = st['intf'] + model.noise
    ideal = st['sig'] / base
    inc = np.einsum('bun,bn->bu', model.h2[:, :, st['serving']], st['p_ant'][:, st['serving']])
    g = st['sig'] / inc                                              # 相干增益比 G_ub
    out = []
    for w in WIDTHS:
        row = [w]
        for split in ('7-1', '7-2'):
            key = (split, ('none' if w >= 16 else f'bfp{w}')) if split == '7-1' else ('7-2', w)
            d_tot = sum(o[3] for _, ro in opts for o in ro if o[0] == key)
            sim = real_sinr(model, st, {r: key for r, _ in opts}, k, np.random.default_rng([d, w, 3]))
            row += [db(st['sig'] / (base + d_tot)), db(sim)]
        rho = cal['r71']['none' if w >= 16 else f'bfp{w}']
        row += [db(g / rho)]
        out.append(row)
    return db(ideal), out


# ---------------------------------------------------------------- B/C：OPT 的逐 RU 选择
def task_opt(item):
    (si, d), (u, nd, bf, cal) = item
    model, path, curve, se_max = prep(u, nd, d, bf, cal)
    if se_max <= 0:
        return []
    sts = [model.state(*path[R3.first_index(curve, ks * se_max)][:2]) for ks in J.KAPPA_S]
    ref = cal['r71']['bfp4']
    rows = []
    for k in KS_OPT:
        best = None
        for st in sts:
            opts = model.ru_options(st, k)
            for lam in J.LAMBDAS:
                se, tot, _, _, choice = model.lagrange(st, opts, lam)
                if se >= J.KAPPA * se_max - 1e-9 and (best is None or tot < best[0]):
                    best = (tot, st, opts, choice)
        if best is None:
            continue
        _, st, opts, choice = best
        ru = st['serving'] // k
        qm_mean = float(st['qm'].mean())
        for r, ro in opts:
            n = int((ru == r).sum())
            split, opt = choice[r]
            gd = {o[0]: o[4] for o in ro}
            x71 = 0.5 * np.log2(max(gd[('7-1', 'bfp4')] / ref, 1e-300) / (336 * n))
            rec = dict(state=[u, nd], drop=d, k=k, bf=bf, ru=int(r), n_r=n, split=split, x71=float(x71))
            if bf == 'zf':
                width = (16 if opt == 'none' else int(opt[3:])) if split == '7-1' else int(opt)
                rec['width'] = width
                rec['x72'] = float(0.5 * np.log2(max(gd[('7-2', 4)] / ref, 1e-300) / (2 * n * u)))
                rec['n_star'] = float(u * (168 * qm_mean + 24) / (168 * (2 * width + 2 / 3) - 2 * width * u))
            rows.append(rec)
    return rows


# ---------------------------------------------------------------- F：论文不压缩方法（B0）
def task_b0(item):
    (si, d), (u, nd, bf, m_hat, cal) = item
    R3.CAL = cal
    model, path, curve, se_max = prep(u, nd, d, bf, cal)
    out = np.full((len(KS_P2), 2), np.nan)
    if se_max <= 0:
        return (si, d), out
    st = model.state(*path[min(m_hat, len(path)) - 1][:2])
    for ki, k in enumerate(KS_P2):
        se, tot, _, _ = R3.b0_point(model, st, k)
        out[ki] = (se / se_max, tot)
    return (si, d), out


def write_dat(path, header, rows):
    with open(path, 'w') as f:
        f.write(' '.join(header) + '\n')
        for r in rows:
            f.write(' '.join('nan' if (isinstance(v, float) and not np.isfinite(v)) else
                             (f'{v:.6g}' if isinstance(v, float) else str(v)) for v in r) + '\n')


def frontier(ratio, load):
    """每个撒点：容量 C 下可行点的最大 SE 比例；ratio / load: [D, P]。返回 [C] 的撒点平均。"""
    out = []
    for c in CAPS:
        v = np.where(load <= c, ratio, -np.inf).max(axis=1)
        v = np.where(np.isfinite(v), v, 0.0)
        out.append(float(np.nanmean(v)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--copy-to', default='')
    p.add_argument('--drops-prop1', type=int, default=40)
    p.add_argument('--drops-opt', type=int, default=30)
    p.add_argument('--workers', type=int, default=48)
    args = p.parse_args()
    here = Path(__file__).parent
    run = Run(args.out, [here / f for f in ('fig_data.py', 'joint.py', 'heuristics10.py', 'faithful.py', 'cfsim.py',
                                            'validate_stage3.py', 'run_stage3.py')],
              dict(purpose='S14: data for the paper figures (paper10 model, MCS table 2)', args=vars(args),
                   model=MODEL, se_model=SE))
    out = run.path
    z10 = dict(np.load(R / 'stage10_paper10/stage7.npz'))
    comp = json.loads((R / 'stage10_paper10/complete.json').read_text())
    keys = [(int(a), float(b)) for a, b in z10['states']]
    bfs = comp['bfs']
    bf_info = comp['bf_info']
    cal = J.calibrate()
    summary = {}

    # A
    with ProcessPoolExecutor(args.workers) as ex:
        res = [r for r in ex.map(task_prop1, [(d, cal) for d in range(args.drops_prop1)]) if r is not None]
    ideal = float(np.mean([r[0] for r in res]))
    arr = np.mean([r[1] for r in res], axis=0)
    write_dat(out / 'fig_prop1.dat', ['w', 'ideal', 'model71', 'sim71', 'model72', 'sim72', 'ceil71'],
              [[int(a[0]), ideal] + [float(x) for x in a[1:]] for a in arr])
    summary['prop1'] = dict(drops=len(res), ideal_db=ideal, rows=arr.tolist())
    print('A done: ideal SINR', round(ideal, 2), 'dB', flush=True)

    # B / C
    items = [((si, d), (u, n, bfs[si], cal)) for si, (u, n) in enumerate(keys) for d in range(args.drops_opt)]
    recs = []
    with ProcessPoolExecutor(args.workers) as ex:
        for rows in ex.map(task_opt, items, chunksize=2):
            recs += rows
    for r in recs:
        run.add(r)
    zf = [r for r in recs if r['bf'] == 'zf']
    lr = np.array([np.log2(r['n_r'] / r['n_star']) for r in zf])
    is72 = np.array([r['split'] == '7-2' for r in zf])
    rule = np.array([(r['n_r'] > r['n_star']) == (r['split'] == '7-2') for r in zf])
    mrt72 = float(np.mean([r['split'] == '7-2' for r in recs if r['bf'] == 'mrt']))
    edges = np.arange(-4.0, 3.01, 0.25)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (lr >= a) & (lr < b)
        if m.sum():
            rows.append([0.5 * (a + b), float(is72[m].mean()), int(m.sum())])
    write_dat(out / 'fig_prop2.dat', ['logratio', 'share72', 'count'], rows)
    summary['prop2'] = dict(zf_ru_decisions=int(len(zf)), rule_agreement=float(rule.mean()), mrt_share72=mrt72,
                            all_ru_decisions=len(recs))
    print(f'B done: {len(zf)} ZF RU decisions, rule agreement {rule.mean():.4f}, MRT 7-2 share {mrt72:.3f}', flush=True)
    # C：逐撒点去均值
    groups = {}
    for r in zf:
        key = (tuple(r['state']), r['drop'], r['k'], r['split'])
        groups.setdefault(key, []).append(r)
    pts = {'7-1': [], '7-2': []}
    for (_, _, _, split), rs in groups.items():
        if len(rs) < 2:
            continue
        x = np.array([q['x71'] if split == '7-1' else q['x72'] for q in rs])
        w = np.array([q['width'] for q in rs], float)
        keep = (w >= 4) & (w <= 12)
        if keep.sum() < 2:
            continue
        pts[split] += list(zip(x[keep] - x[keep].mean(), w[keep] - w[keep].mean()))
    rows = []
    edges = np.arange(-3.0, 3.01, 0.5)
    stats = {}
    for split in ('7-1', '7-2'):
        a = np.array(pts[split]) if pts[split] else np.zeros((0, 2))
        if len(a) > 2:
            slope = float(np.polyfit(a[:, 0], a[:, 1], 1)[0])
            stats[split] = dict(points=int(len(a)), slope=slope)
    for lo, hi in zip(edges[:-1], edges[1:]):
        row = [0.5 * (lo + hi)]
        for split in ('7-1', '7-2'):
            a = np.array(pts[split]) if pts[split] else np.zeros((0, 2))
            m = (a[:, 0] >= lo) & (a[:, 0] < hi) if len(a) else np.zeros(0, bool)
            row += [float(a[m, 1].mean()) if m.sum() >= 5 else float('nan'),
                    float(a[m, 1].std()) if m.sum() >= 5 else float('nan'), int(m.sum())]
        rows.append(row)
    write_dat(out / 'fig_prop3.dat', ['dx', 'dw71', 'sd71', 'n71', 'dw72', 'sd72', 'n72'], rows)
    summary['prop3'] = stats
    print('C done:', stats, flush=True)

    # D：动态分解
    for name, run_dir in (('mcs', 'stage10_paper10'), ('mcs4', 'stage11_paper10_mcs4')):
        sm = json.loads((R / run_dir / 'stage7_summary.json').read_text())
        write_dat(out / f'fig_dyn_{name}.dat', ['K', 'ST', 'SU', 'F', 'G', 'S', 'D'],
                  [[e['k'], e['ST'], e['SU'], e['F'], e['G'], e['S'], e['D']] for e in sm])

    # E：边界
    b = [json.loads(line) for line in open(R / 'stage13_boundary_paper10/rows.jsonl')]
    hdr, cols = ['K'], []
    for r in b:
        tag = f"u{r['num_ue']}n{abs(int(r['noise_dbm']))}"
        hdr += [f'p_{tag}', f'o_{tag}']
        cols.append(r)
    rows = []
    for k in R3.KS:
        row = [k]
        for r in cols:
            row += [r['per_k'][str(k)]['pred_share72'], r['per_k'][str(k)]['observed_share72']]
        rows.append(row)
    write_dat(out / 'fig_boundary.dat', hdr, rows)
    write_dat(out / 'fig_kstar.dat', ['state', 'kstar'],
              [[f"u{r['num_ue']}n{abs(int(r['noise_dbm']))}", r['k_star']] for r in b])

    # F：P2
    items = []
    for si, (u, n) in enumerate(keys):
        bf = bfs[si]
        m_hat = bf_info[f'{u},{int(n)}'][bf][0]
        items += [((si, d), (u, n, bf, m_hat, cal)) for d in range(z10['se_max'].shape[1])]
    b0 = np.full((len(keys), z10['se_max'].shape[1], len(KS_P2), 2), np.nan)
    with ProcessPoolExecutor(args.workers) as ex:
        for (si, d), o in ex.map(task_b0, items, chunksize=4):
            b0[si, d] = o
    ks_all = list(z10['ks'])
    summary['b0'] = {}
    for ki, k in enumerate(KS_P2):
        kk = ks_all.index(k)
        g = z10['grid'][:, :, kk, :, 0]                    # 规则模式 [S, D, KS, T, 3]
        u_ = z10['uni'][:, :, kk]                          # [S, D, KS, 2, W, 2]
        cf_curves, uni_curves, b0_curves = [], [], []
        for si in range(len(keys)):
            valid = np.isfinite(z10['se_max'][si])
            rr, ll = g[si][valid][..., 0].reshape(valid.sum(), -1), g[si][valid][..., 1].reshape(valid.sum(), -1)
            cf_curves.append(frontier(rr, ll))
            ur, ul = u_[si][valid][..., 0].reshape(valid.sum(), -1), u_[si][valid][..., 1].reshape(valid.sum(), -1)
            uni_curves.append(frontier(ur, ul))
            br, bl = b0[si][valid][:, ki, 0][:, None], b0[si][valid][:, ki, 1][:, None]
            b0_curves.append(frontier(br, bl))
        write_dat(out / f'fig_p2_k{k}.dat', ['C', 'cf', 'uni', 'b0'],
                  [[float(c), a, bb, cc] for c, a, bb, cc in zip(CAPS, np.mean(cf_curves, 0), np.mean(uni_curves, 0),
                                                                 np.mean(b0_curves, 0))])
        summary['b0'][k] = dict(mean_load=float(np.nanmean(b0[:, :, ki, 1])),
                                mean_se_ratio=float(np.nanmean(b0[:, :, ki, 0])))
    print('F done:', summary['b0'], flush=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    if args.copy_to:
        dst = Path(args.copy_to)
        dst.mkdir(parents=True, exist_ok=True)
        for f in out.glob('fig_*.dat'):
            shutil.copy(f, dst / f.name)
    run.finish()
    print('done', out)


if __name__ == '__main__':
    main()
