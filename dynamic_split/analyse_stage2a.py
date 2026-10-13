"""阶段 2a 分析：峰值（链路）约束下最小化平均前传负载（下行 + 上行），比较策略类和可选方案集合。

状态 s = (UE 数, 下行噪声底, SRS 周期)。每个配置 c 的
  total(s,c) = 撒点平均的 Σ_O-RU (下行 + 上行 CI)，link(s,c) = 撒点平均的 max_O-RU max(下行, 上行)，
  ρ(s,c) = 和速率 / 理想参考（集中式 RZF、原始含噪 CI、无失真前传）。可行：ρ ≥ κ。
策略类：static（一个配置）、dyn_width（方案族固定，每状态调位宽/CI/功率规则）、
dyn_mode（每个方案族的参数固定，每状态选方案族：7-1 / 7-2 集中式 / 7-2 本地）、dyn_full（每状态任选）。
方案集合：all、no_klt（只用 O-RAN 已有的 CI 方式）、no_local（不允许 O-RU 本地预编码）、oran_central（两者都不要）。
链路容量 C 取该方案集合内最佳静态设计能做到的最小链路峰值。

用法：
    .venv/bin/python dynamic_split/analyse_stage2a.py results/dynamic_split_20261012/stage2a_two_ru
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import stage2_eval as e2  # noqa: E402

T_SRS_MS = [5.0, 10.0, 20.0, 40.0]
KAPPAS = [0.98, 0.95]
CLASSES = ['static', 'dyn_width', 'dyn_mode', 'dyn_full']
SUBSETS = {
    'all': lambda c: True,
    'no_klt': lambda c: not (c[3] or '').startswith('klt'),
    'no_local': lambda c: c[0] != '7-2l',
    'oran_central': lambda c: c[0] != '7-2l' and not (c[3] or '').startswith('klt'),
}


def label(c):
    fam, up, b, ci = c
    if fam == '7-1':
        return f'7-1/{up}'
    if fam == '7-2l':
        return f'7-2 local/{ci}'
    return f'7-2 central/w{b}/{ci}'


def build(npz, model):
    """返回 states、total [S, C]、link [S, C]、rho [S, C]（Gbit/s）、上行占比 [S, C]。"""
    ues, noise, models = list(npz['ues']), list(npz['noise']), list(npz['models'])
    mi = models.index(model)
    states, tot, lnk, rho, ush = [], [], [], [], []
    for ui, u in enumerate(ues):
        for ni, nd in enumerate(noise):
            ideal = npz['ideal'][ui, :, mi, ni].mean()
            ratio = npz['se'][ui, :, mi, ni].mean(0) / ideal                     # [C]
            dl = npz['dl'][ui, :, mi, ni]                                         # [D, C, L]
            ulb = npz['ul_bits'][ui, :, mi, ni]
            for t in T_SRS_MS:
                ul = ulb * 1000.0 / t
                states.append((int(u), float(nd), t))
                tot.append((dl + ul).sum(-1).mean(0) / 1e9)
                lnk.append(np.maximum(dl, ul).max(-1).mean(0) / 1e9)
                rho.append(ratio)
                ush.append(ul.sum(-1).mean(0) / np.maximum((dl + ul).sum(-1).mean(0), 1e-30))
    return states, np.array(tot), np.array(lnk), np.array(rho), np.array(ush)


def per_state(total, link, ok, cap, allowed):
    choice = []
    for s in range(total.shape[0]):
        cand = [j for j in allowed if ok[s, j] and link[s, j] <= cap + 1e-12]
        if not cand:
            return None
        choice.append(min(cand, key=lambda j: total[s, j]))
    return choice


def solve(cls, fam, total, link, ok, cap, idx):
    best = None

    def consider(ch):
        nonlocal best
        if ch is not None:
            m = float(np.mean([total[s, j] for s, j in enumerate(ch)]))
            if best is None or m < best[0] - 1e-12:
                best = (m, ch)

    groups = {f: [j for j in idx if fam[j] == f] for f in sorted(set(fam[j] for j in idx))}
    if cls == 'static':
        for j in idx:
            if ok[:, j].all() and link[:, j].max() <= cap + 1e-12:
                consider([j] * total.shape[0])
    elif cls == 'dyn_width':
        for js in groups.values():
            consider(per_state(total, link, ok, cap, js))
    elif cls == 'dyn_mode':
        for combo in itertools.product(*groups.values()):
            consider(per_state(total, link, ok, cap, list(combo)))
    else:
        consider(per_state(total, link, ok, cap, idx))
    return best


def min_peak(cls, fam, total, link, ok, idx):
    caps = np.unique(link[:, idx])
    if solve(cls, fam, total, link, ok, caps[-1], idx) is None:
        return None
    lo, hi = 0, len(caps) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if solve(cls, fam, total, link, ok, caps[mid], idx) is None:
            lo = mid + 1
        else:
            hi = mid
    return float(caps[lo])


def analyse(npz):
    cfgs = [tuple(c) for c in e2.CONFIGS]
    fam = [c[0] for c in cfgs]
    out = []
    for model in ('mcs', 'shannon'):
        states, total, link, rho, ush = build(npz, model)
        for kappa in KAPPAS:
            ok = rho >= kappa
            for name, keep in SUBSETS.items():
                idx = [j for j, c in enumerate(cfgs) if keep(c)]
                cap = min_peak('static', fam, total, link, ok, idx)
                entry = dict(se_model=model, kappa=kappa, subset=name, cap_gbps=cap,
                             states=[list(s) for s in states], classes={})
                for cls in CLASSES:
                    sol = solve(cls, fam, total, link, ok, cap, idx) if cap is not None else None
                    if sol is None:
                        entry['classes'][cls] = None
                        continue
                    m, ch = sol
                    entry['classes'][cls] = dict(
                        mean_gbps=m, peak_link_gbps=float(max(link[s, j] for s, j in enumerate(ch))),
                        own_min_link_peak_gbps=min_peak(cls, fam, total, link, ok, idx),
                        choice=[label(cfgs[j]) for j in ch],
                        ul_share=float(np.mean([ush[s, j] for s, j in enumerate(ch)])))
                out.append(entry)
    return out


def report(summary, model='mcs', kappa=0.98):
    print(f'\n=== {model} SE, kappa {kappa}: mean fronthaul load (DL + UL) [Gbit/s] at the best-static link capacity')
    for e in summary:
        if e['se_model'] != model or e['kappa'] != kappa:
            continue
        st = e['classes']['static']
        cells = []
        for cls in CLASSES:
            r = e['classes'][cls]
            cells.append('n/a' if r is None else
                         f"{r['mean_gbps']:5.2f} ({100 * (1 - r['mean_gbps'] / st['mean_gbps']):4.1f}%, "
                         f"UL {100 * r['ul_share']:.1f}%)")
        print(f"{e['subset']:13s} C={e['cap_gbps'] if e['cap_gbps'] is None else round(e['cap_gbps'], 2)} | "
              + ' | '.join(f'{c}: {x}' for c, x in zip(CLASSES, cells)))
        if st:
            print(f"   static: {st['choice'][0]}")
    for e in summary:
        if e['se_model'] == model and e['kappa'] == kappa and e['subset'] == 'all' and e['classes']['dyn_full']:
            print('   dyn_full choices (K, noise, T_SRS):')
            for s, c in zip(e['states'], e['classes']['dyn_full']['choice']):
                if s[2] in (5.0, 40.0):
                    print(f'     {tuple(s)}: {c}')


def plot(summary, run_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, model in zip(axes, ('mcs', 'shannon')):
        es = [e for e in summary if e['se_model'] == model and e['kappa'] == 0.98]
        ref = next(e for e in es if e['subset'] == 'oran_central')['classes']['static']['mean_gbps']
        x = np.arange(len(SUBSETS))
        for i, cls in enumerate(CLASSES):
            vals = [(next(e for e in es if e['subset'] == sub)['classes'][cls] or {}).get('mean_gbps', np.nan)
                    for sub in SUBSETS]
            ax.bar(x + (i - 1.5) * 0.2, vals, 0.2, label=cls)
        ax.axhline(ref, color='k', lw=.8, ls=':')
        ax.set_xticks(x, list(SUBSETS), fontsize=8)
        ax.set_title(f'{model} SE, kappa 0.98 (dotted: static O-RAN central)', fontsize=9)
        ax.grid(alpha=.3, axis='y')
    axes[0].set_ylabel('mean fronthaul load DL+UL [Gbit/s]')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / 'fig_stage2a_policies.png', dpi=150)


def main(run_dir):
    run_dir = Path(run_dir)
    npz = dict(np.load(run_dir / 'per_drop.npz'))
    summary = analyse(npz)
    (run_dir / 'stage2a_summary.json').write_text(json.dumps(summary, indent=1))
    report(summary, 'mcs', 0.98)
    report(summary, 'mcs', 0.95)
    report(summary, 'shannon', 0.98)
    plot(summary, run_dir)


if __name__ == '__main__':
    main(sys.argv[1])
