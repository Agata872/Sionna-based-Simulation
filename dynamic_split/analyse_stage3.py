"""阶段 3 分析：P1（和速率 ≥ 0.98 SE_max 时的最小前传负载）与 P2（给定前传容量时的最大和速率），
逐撒点与逐状态两种粒度，方法 P / B2 / B1 / B0。

方法：
  P   本文：两条服务集合路径（论文、负载感知）× κ_s × λ，每个 RU 用拉格朗日法选切分与位宽
  B2  论文服务集合路径 × κ_s × λ，每个 RU 用拉格朗日法选切分与位宽
  B1  论文服务集合路径 × κ_s，全网统一的切分与位宽（阶段 1b 的口径）
  B0  论文原方法（状态级 M̂、论文公式选全网切分、不压缩）
粒度：
  逐撒点  每个撒点在候选点里选满足 SE ≥ κ·SE_max(撒点) 的最小负载
  逐状态  每个状态固定一组参数（路径、κ_s、λ 或统一配置），满足 mean SE ≥ κ·mean SE_max，最小化平均负载

用法：
    .venv/bin/python dynamic_split/analyse_stage3.py results/dynamic_split_20261012/stage3_joint
"""

import json
import warnings
import sys
from pathlib import Path

import numpy as np

KAPPA = 0.98
warnings.filterwarnings('ignore', category=RuntimeWarning)   # 空切片（某些路径达不到 κ_s）
METHODS = ['P', 'B2', 'B1', 'B0']
CAPS = np.logspace(np.log10(0.3), np.log10(60), 41)


def candidates(z, method, mi, si, ki):
    """返回该方法在每个撒点上的候选点：se [D, N]、load [D, N]、link [D, N]、f72 [D, N]。"""
    pts = z['pts'][mi, si, :, ki]                     # [D, path, ks, lam, 4]
    d = pts.shape[0]
    if method == 'P':
        x = pts.reshape(d, -1, 4)
    elif method == 'B2':
        x = pts[:, 0].reshape(d, -1, 4)
    elif method == 'B1':
        b1 = z['b1'][mi, si, :, ki].reshape(d, -1, 3)
        x = np.concatenate([b1, np.full(b1.shape[:2] + (1,), np.nan)], axis=-1)
    else:
        x = z['b0'][mi, si, :, ki][:, None, :]
    return x[..., 0], x[..., 1], x[..., 2], x[..., 3]


def per_drop(se, load, link, f72, se_max):
    ok = se >= KAPPA * se_max[:, None] - 1e-9
    masked = np.where(ok, load, np.inf)
    j = np.argmin(masked, axis=1)
    rows = np.arange(len(j))
    feas = np.isfinite(masked[rows, j])
    return dict(load=np.where(feas, load[rows, j], np.nan), link=np.where(feas, link[rows, j], np.nan),
                se_ratio=np.where(feas, se[rows, j] / se_max, np.nan), f72=np.where(feas, f72[rows, j], np.nan),
                infeasible=int((~feas).sum()))


def per_state(se, load, link, f72, se_max):
    """参数（候选下标）对所有撒点固定：要求每个撒点都有该点，mean SE ≥ κ mean SE_max。"""
    valid = ~np.isnan(se).any(axis=0)
    mse, mload = np.nanmean(se, axis=0), np.nanmean(load, axis=0)
    ok = valid & (mse >= KAPPA * se_max.mean() - 1e-9)
    if not ok.any():
        return None
    j = int(np.argmin(np.where(ok, mload, np.inf)))
    return dict(load=float(mload[j]), link=float(np.nanmean(link[:, j])), se_ratio=float(mse[j] / se_max.mean()),
                f72=float(np.nanmean(f72[:, j])) if not np.isnan(f72[:, j]).all() else float('nan'), index=j)


def capacity_curve(se, load, se_max, granularity):
    """P2：每个容量 C 下的平均和速率 / SE_max。"""
    out = []
    for c in CAPS:
        if granularity == 'drop':
            v = np.where(load <= c, se, -np.inf).max(axis=1)
            out.append(float(np.mean(np.where(np.isfinite(v), v, 0) / se_max)))
        else:
            valid = ~np.isnan(se).any(axis=0)
            ok = valid & (np.nanmean(load, axis=0) <= c)
            out.append(float(np.nanmean(se, axis=0)[ok].max() / se_max.mean()) if ok.any() else 0.0)
    return out


def analyse(z):
    states = [tuple(s) for s in z['states']]
    models, ks = list(z['models']), list(z['ks'])
    res = []
    for mi, model in enumerate(models):
        for ki, k in enumerate(ks):
            for si, st in enumerate(states):
                se_max = z['se_max'][mi, si]
                entry = dict(se_model=model, k=int(k), state=[int(st[0]), float(st[1])], methods={})
                for m in METHODS:
                    se, load, link, f72 = candidates(z, m, mi, si, ki)
                    pd = per_drop(se, load, link, f72, se_max)
                    if m == 'B0':
                        ps = dict(load=float(np.nanmean(load)), link=float(np.nanmean(link)),
                                  se_ratio=float(np.nanmean(se) / se_max.mean()), f72=float(np.nanmean(f72)))
                        pd = dict(load=load[:, 0], link=link[:, 0], se_ratio=se[:, 0] / se_max, f72=f72[:, 0],
                                  infeasible=int((se[:, 0] < KAPPA * se_max - 1e-9).sum()))
                    else:
                        ps = per_state(se, load, link, f72, se_max)
                    entry['methods'][m] = dict(
                        drop=dict(load=float(np.nanmean(pd['load'])), link=float(np.nanmean(pd['link'])),
                                  se_ratio=float(np.nanmean(pd['se_ratio'])), f72=float(np.nanmean(pd['f72'])),
                                  infeasible=pd['infeasible']),
                        state=ps,
                        cap_drop=capacity_curve(se, load, se_max, 'drop'),
                        cap_state=capacity_curve(se, load, se_max, 'state'))
                res.append(entry)
    return res


def table(res, model='mcs'):
    print(f'\n=== {model} SE, P1: mean fronthaul load (DL+UL) over the 9 states [Gbit/s]  '
          f'(per-drop | per-state); saving vs B0 / vs B1 per-state')
    ks = sorted({e['k'] for e in res})
    for k in ks:
        es = [e for e in res if e['se_model'] == model and e['k'] == k]
        cell = {}
        for m in METHODS:
            dvals = [e['methods'][m]['drop']['load'] for e in es]
            svals = [(e['methods'][m]['state'] or {}).get('load', np.nan) for e in es]
            cell[m] = (np.mean(dvals), np.mean(svals))
        b0, b1s = cell['B0'][1], cell['B1'][1]
        txt = ' | '.join(f"{m} {cell[m][0]:6.2f} {cell[m][1]:6.2f}" for m in METHODS)
        f72 = np.mean([e['methods']['P']['drop']['f72'] for e in es])
        print(f"K={k:2d} | {txt} | P drop vs B0 {100 * (1 - cell['P'][0] / b0):5.1f}%, vs B1-state "
              f"{100 * (1 - cell['P'][0] / b1s):5.1f}%, P state vs B1-state {100 * (1 - cell['P'][1] / b1s):5.1f}% "
              f"| P: 7-2 RU share {f72:.2f}")


def plot(res, run_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ks = sorted({e['k'] for e in res})
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, model in zip(axes, ('mcs', 'shannon')):
        for m, c in zip(METHODS, ('C3', 'C0', 'C1', 'k')):
            for gran, ls in (('drop', '-'), ('state', '--')):
                ys = []
                for k in ks:
                    es = [e for e in res if e['se_model'] == model and e['k'] == k]
                    vals = [(e['methods'][m][gran] or {}).get('load', np.nan) for e in es]
                    ys.append(np.mean(vals))
                if m == 'B0' and gran == 'drop':
                    continue
                ax.plot(ks, ys, ls, color=c, marker='o', ms=3, label=f'{m} ({"per drop" if gran == "drop" else "per state"})')
        ax.set_xscale('log', base=2)
        ax.set_xlabel('antennas per RU K')
        ax.set_title(f'P1: min load s.t. SE >= 0.98 SE_max ({model} SE)', fontsize=9)
        ax.grid(alpha=.3)
    axes[0].set_ylabel('mean fronthaul load DL+UL [Gbit/s]')
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(run_dir / 'fig_p1_load_vs_k.png', dpi=150)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), sharey=True)
    for ax, k in zip(axes, (1, 4, 16, 64)):
        es = [e for e in res if e['se_model'] == 'mcs' and e['k'] == k]
        for m, c in zip(METHODS, ('C3', 'C0', 'C1', 'k')):
            for gran, ls in (('drop', '-'), ('state', '--')):
                if m == 'B0' and gran == 'state':
                    continue
                y = np.mean([e['methods'][m][f'cap_{gran}'] for e in es], axis=0)
                ax.plot(CAPS, y, ls, color=c, label=f'{m} ({gran})')
        ax.set_xscale('log')
        ax.axhline(KAPPA, color='grey', lw=.8, ls=':')
        ax.set_title(f'P2, K = {k} (mcs SE, mean over states)', fontsize=9)
        ax.set_xlabel('fronthaul capacity C [Gbit/s]')
        ax.grid(alpha=.3)
    axes[0].set_ylabel('mean SE / SE_max')
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(run_dir / 'fig_p2_capacity_curves.png', dpi=150)


def main(run_dir):
    run_dir = Path(run_dir)
    z = dict(np.load(run_dir / 'stage3.npz'))
    res = analyse(z)
    (run_dir / 'stage3_summary.json').write_text(json.dumps(res, indent=1, default=float))
    table(res, 'mcs')
    table(res, 'shannon')
    plot(res, run_dir)


if __name__ == '__main__':
    main(sys.argv[1])
