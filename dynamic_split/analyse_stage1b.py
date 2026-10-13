"""阶段 1b 分析：峰值约束下最小化平均负载，比较四类策略。

每个状态 s（UE 数 × 噪声）、配置 c（切分、U-plane 压缩、权值位宽）有：
  L(s,c)  网络总下行负载（撒点平均），link(s,c) 最忙 RU 链路的下行负载（撒点平均），
  ρ(s,c)  和速率 / 理想前传和速率（撒点平均）。可行：ρ ≥ κ。
策略类：
  static     一个配置用于所有状态；
  dyn_width  切分固定，每个状态选位宽（U-plane、权值）；
  dyn_split  位宽固定（7-1 一个、7-2 一个 U-plane + 一个权值位宽），每个状态选切分；
  dyn_full   每个状态联合选切分和位宽。
问题：min mean_s L(s,c(s))  s.t.  max_s link(s,c(s)) ≤ C，ρ(s,c(s)) ≥ κ ∀s。
C 取"最佳静态设计能做到的最小链路峰值"（按静态设计配置链路容量），各类在同一容量下比平均负载；
另给每类自己能做到的最小链路峰值，以及 C 扫描曲线。

用法：
    .venv/bin/python dynamic_split/analyse_stage1b.py results/dynamic_split_20261012/stage1b_distortion
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from runio import read_rows  # noqa: E402

KAPPAS = [0.98, 0.95, 0.99]
CLASSES = ['static', 'dyn_width', 'dyn_split', 'dyn_full']


def label(c, bf='zf'):
    """MRT 时 7-2 的权值由 RU 本地计算，不显示权值位宽。"""
    split, up, b = c
    return f'{split}/{up}' + ('' if split == '7-1' or bf == 'mrt' else f'/w{b}')


def tables(rows, model, k):
    rs = [r for r in rows if r['se_model'] == model and r['k'] == k]
    states = sorted({(r['num_ue'], r['noise_dbm']) for r in rs})
    cfgs = sorted({(r['split'], r['uplane'], r['bfw']) for r in rs}, key=lambda c: (c[0], c[1], c[2] or 0))
    si = {s: i for i, s in enumerate(states)}
    ci = {c: i for i, c in enumerate(cfgs)}
    shape = (len(states), len(cfgs))
    L, link, rho = np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan)
    bf = {}
    for r in rs:
        i, j = si[(r['num_ue'], r['noise_dbm'])], ci[(r['split'], r['uplane'], r['bfw'])]
        L[i, j], link[i, j], rho[i, j] = r['dl_gbps'], r['link_gbps'], r['se_ratio']
        bf[(r['num_ue'], r['noise_dbm'])] = r['bf']
    return states, cfgs, L, link, rho, bf


def per_state_choice(L, link, ok, cap, allowed):
    """每个状态在 allowed 列中选可行且 link ≤ cap 的最小负载配置；不可行返回 None。"""
    choice = []
    for s in range(L.shape[0]):
        cand = [j for j in allowed if ok[s, j] and link[s, j] <= cap + 1e-12]
        if not cand:
            return None
        choice.append(min(cand, key=lambda j: L[s, j]))
    return choice


def solve(cls, cfgs, L, link, ok, cap):
    """返回 (平均负载, 每状态配置下标) 的最优解，不可行返回 None。"""
    n_s, n_c = L.shape
    idx71 = [j for j, c in enumerate(cfgs) if c[0] == '7-1']
    idx72 = [j for j, c in enumerate(cfgs) if c[0] == '7-2']
    best = None

    def consider(choice):
        nonlocal best
        if choice is None:
            return
        m = float(np.mean([L[s, j] for s, j in enumerate(choice)]))
        if best is None or m < best[0] - 1e-12:
            best = (m, choice)

    if cls == 'static':
        for j in range(n_c):
            if ok[:, j].all() and link[:, j].max() <= cap + 1e-12:
                consider([j] * n_s)
    elif cls == 'dyn_width':
        for allowed in (idx71, idx72):
            consider(per_state_choice(L, link, ok, cap, allowed))
    elif cls == 'dyn_split':
        for a, b in itertools.product(idx71, idx72):
            consider(per_state_choice(L, link, ok, cap, [a, b]))
    else:
        consider(per_state_choice(L, link, ok, cap, list(range(n_c))))
    return best


def min_peak(cls, cfgs, L, link, ok):
    """该类能做到的最小链路峰值（在候选容量值上二分）。"""
    caps = np.unique(link[np.isfinite(link)])
    lo, hi = 0, len(caps) - 1
    if solve(cls, cfgs, L, link, ok, caps[hi]) is None:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if solve(cls, cfgs, L, link, ok, caps[mid]) is None:
            lo = mid + 1
        else:
            hi = mid
    return float(caps[lo])


def analyse(rows, models, ks):
    out = []
    for model in models:
        for k in ks:
            states, cfgs, L, link, rho, bf = tables(rows, model, k)
            for kappa in KAPPAS:
                ok = rho >= kappa
                cap_static = min_peak('static', cfgs, L, link, ok)
                entry = dict(se_model=model, k=k, kappa=kappa, states=[list(s) for s in states],
                             bf=[bf[s] for s in states], cap_gbps=cap_static, classes={})
                for cls in CLASSES:
                    own = min_peak(cls, cfgs, L, link, ok)
                    res = {}
                    for name, cap in (('at_static_cap', cap_static), ('unconstrained', np.inf)):
                        sol = solve(cls, cfgs, L, link, ok, cap) if cap is not None else None
                        if sol is None:
                            res[name] = None
                            continue
                        m, ch = sol
                        res[name] = dict(mean_gbps=m,
                                         peak_total_gbps=float(max(L[s, j] for s, j in enumerate(ch))),
                                         peak_link_gbps=float(max(link[s, j] for s, j in enumerate(ch))),
                                         choice=[label(cfgs[j], bf[states[s]]) for s, j in enumerate(ch)])
                    res['own_min_link_peak_gbps'] = own
                    entry['classes'][cls] = res
                base = entry['classes']['static']['at_static_cap']
                for cls in CLASSES:
                    r = entry['classes'][cls]['at_static_cap']
                    if base and r:
                        r['saving_vs_static'] = 1 - r['mean_gbps'] / base['mean_gbps']
                out.append(entry)
    return out


def report(summary, model='mcs', kappa=0.98):
    print(f'\n=== {model} SE, kappa {kappa}: mean DL load [Gbit/s] at the static link capacity C '
          f'(saving vs static) | own min link peak')
    for e in summary:
        if e['se_model'] != model or e['kappa'] != kappa:
            continue
        cells = []
        for cls in CLASSES:
            r = e['classes'][cls]['at_static_cap']
            own = e['classes'][cls]['own_min_link_peak_gbps']
            cells.append('   n/a' if r is None else
                         f"{r['mean_gbps']:6.2f} ({100 * r.get('saving_vs_static', 0):4.1f}%) pk {own:5.2f}")
        print(f"K={e['k']:2d} C={e['cap_gbps']:5.2f} | " + ' | '.join(f'{c}: {x}' for c, x in zip(CLASSES, cells)))
        st = e['classes']['static']['at_static_cap']
        full = e['classes']['dyn_full']['at_static_cap']
        print(f"      static: {st['choice'][0] if st else 'n/a'}")
        if full:
            print('      dyn_full: ' + ', '.join(f"{tuple(s)}{b[0].upper()}:{c}" for s, b, c in
                                                  zip(e['states'], e['bf'], full['choice'])))


def plot(summary, rows, run_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # 1) 各类策略在静态容量下的平均负载节省
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, model in zip(axes, ('mcs', 'shannon')):
        es = sorted((e for e in summary if e['se_model'] == model and e['kappa'] == 0.98), key=lambda e: e['k'])
        for cls in CLASSES[1:]:
            ys = [100 * (e['classes'][cls]['at_static_cap'] or {}).get('saving_vs_static', np.nan) for e in es]
            ax.plot([e['k'] for e in es], ys, marker='o', ms=3, label=cls)
        ax.set_xscale('log', base=2)
        ax.set_xlabel('antennas per RU K')
        ax.set_title(f'{model} SE, kappa 0.98, link capacity = best static', fontsize=9)
        ax.grid(alpha=.3)
    axes[0].set_ylabel('mean DL load saving vs static [%]')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / 'fig_policy_saving_vs_k.png', dpi=150)
    # 2) 失真曲线：和速率比例随每 RE 比特数（K = 8）
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey='row')
    for i, model in enumerate(('mcs', 'shannon')):
        for ax, (u, n) in zip(axes[i], ((10, -90.0), (20, -70.0), (10, -50.0))):
            rs = [r for r in rows if r['se_model'] == model and r['k'] == 8 and (r['num_ue'], r['noise_dbm']) == (u, n)]
            bfw_ref = 16
            for split, style in (('7-1', 'C0o-'), ('7-2', 'C1s-')):
                pts = [r for r in rs if r['split'] == split and r['uplane'].startswith('bfp')
                       and (split == '7-1' or r['bfw'] == bfw_ref)]
                pts.sort(key=lambda r: int(r['uplane'][3:]))
                ax.plot([2 * int(r['uplane'][3:]) for r in pts], [r['se_ratio'] for r in pts], style, ms=3,
                        label=f'{split} U-plane BFP' + ('' if split == '7-1' else ', 16-bit weights'))
            if rs and rs[0]['bf'] == 'zf':
                pts = sorted((r for r in rs if r['split'] == '7-2' and r['uplane'] == 'modcomp'),
                             key=lambda r: r['bfw'])
                ax.plot([2 * r['bfw'] for r in pts], [r['se_ratio'] for r in pts], 'C2^--', ms=3,
                        label='7-2 modcomp, weight BFP (x = weight bits)')
            ax.axhline(0.98, color='k', lw=.8, ls=':')
            ax.set_title(f"{model}: {u} UEs, {int(n)} dBm, {rs[0]['bf'].upper() if rs else ''}, K = 8", fontsize=9)
            ax.set_ylim(0.5, 1.02)
            ax.grid(alpha=.3)
            if i == 1:
                ax.set_xlabel('bits per I/Q pair (2 x width)')
        axes[i, 0].set_ylabel('sum SE / ideal')
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(run_dir / 'fig_distortion_curves_k8.png', dpi=150)


def main(run_dir):
    run_dir = Path(run_dir)
    rows = read_rows(run_dir)
    models = sorted({r['se_model'] for r in rows})
    ks = sorted({r['k'] for r in rows})
    summary = analyse(rows, models, ks)
    (run_dir / 'stage1b_summary.json').write_text(json.dumps(summary, indent=1))
    for model in ('mcs', 'shannon'):
        if model in models:
            report(summary, model, 0.98)
    for kappa in (0.95, 0.99):
        report(summary, 'mcs', kappa)
    plot(summary, rows, run_dir)


if __name__ == '__main__':
    main(sys.argv[1])
