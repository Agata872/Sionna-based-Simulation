"""阶段 1a：只看负载的筛查。在阶段 0 复现的服务天线集合上，换不同计费口径，看动态切分还有多少收益。

和速率与服务天线由阶段 0 决定（论文的方法），这里不考虑压缩失真对和速率的影响（留给阶段 1b）。
收益的定义（每个 K、口径、目标分别计算，L̄ 为对撒点平均的负载）：
  - 按状态动态 vs 最佳静态（平均）：1 − mean_s min_c L̄(s,c) / min_c mean_s L̄(s,c)
  - 峰值（前传按最坏状态配置容量）：1 − max_s min_c L̄(s,c) / min_c max_s L̄(s,c)
  - 按撒点动态：每个撒点单独选（需要更快的切换）
  - 按 RU 动态：每个 RU 每个撒点单独选（负载在 RU 间可加，这是本模型下的上界）
状态 s = (UE 数, 噪声) 共 9 个，等权。

用法：
    .venv/bin/python dynamic_split/screen_stage1a.py results/dynamic_split_20261012/stage0_repro \
        --out results/dynamic_split_20261012/stage1a_load_screen
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fhload  # noqa: E402
import summarize_stage0 as s0  # noqa: E402
from fhload import Profile  # noqa: E402
from runio import Run, read_rows  # noqa: E402

KS = [1, 2, 4, 8, 16, 32, 64]
SPLITS = ('7-1', '7-2')
PROFILES = [
    Profile('P0_paper'),                                              # 论文：q = 32，γ 常数，无 CI
    Profile('E0_headers'),                                            # 显式 O-RAN 报头，q = 32
    Profile('E1_bfp9', comp_71='bfp9', comp_72='bfp9'),               # 两边都 BFP9
    Profile('E2_modcomp', comp_71='bfp9', comp_72='modcomp'),         # 7-2 用调制压缩
    Profile('E3_modcomp_ci', comp_71='bfp9', comp_72='modcomp', ci='srs'),            # + CI，权值每时隙
    Profile('E4_modcomp_ci_bfwsrs', comp_71='bfp9', comp_72='modcomp', ci='srs', bfw_update='srs'),
    Profile('E5_modcomp_ci_fast', comp_71='bfp9', comp_72='modcomp', ci='srs', t_srs_ms=2.5),  # 高移动性
    Profile('E6_bfp9_ci', comp_71='bfp9', comp_72='bfp9', ci='srs'),  # 无调制压缩，只加 CI
]


def loads_for(usage, k, prof):
    """{split: {'dl': 每 RU 数组, 'ul': 每 RU 数组}}"""
    if prof.name == 'P0_paper':
        return {s: dict(dl=v, ul=np.zeros_like(v)) for s, v in fhload.paper_load(usage, k).items()}
    return {s: dict(dl=v['dl'], ul=v['ul']) for s, v in fhload.explicit_load(usage, k, prof).items()}


def state_task(item):
    (u, n, sm), us = item
    rows = []
    for d, x in enumerate(us):
        for k in KS:
            for prof in PROFILES:
                ld = loads_for(x, k, prof)
                row = dict(se_model=sm, num_ue=u, noise_dbm=n, drop=d, k=k, profile=prof.name,
                           bf=x.bf, serving=int(x.ants.sum()))
                for obj in ('dl', 'dlul'):
                    per = {s: ld[s]['dl'] + (ld[s]['ul'] if obj == 'dlul' else 0) for s in SPLITS}
                    for s in SPLITS:
                        row[f'{obj}_{s}'] = float(per[s].sum()) / 1e9
                    row[f'{obj}_perru'] = float(np.minimum(per['7-1'], per['7-2']).sum()) / 1e9
                for s in SPLITS:
                    row[f'ul_{s}'] = float(ld[s]['ul'].sum()) / 1e9
                rows.append(row)
    return rows


def analyse(rows):
    out = []
    idx = {}
    for r in rows:
        idx.setdefault((r['se_model'], r['k'], r['profile']), []).append(r)
    for (sm, k, prof), rs in sorted(idx.items()):
        states = sorted({(r['num_ue'], r['noise_dbm']) for r in rs})
        for obj in ('dl', 'dlul'):
            lbar = np.array([[np.mean([r[f'{obj}_{s}'] for r in rs if (r['num_ue'], r['noise_dbm']) == st])
                              for s in SPLITS] for st in states])                     # [S, 2]
            per_drop = np.array([np.mean([min(r[f'{obj}_7-1'], r[f'{obj}_7-2']) for r in rs
                                          if (r['num_ue'], r['noise_dbm']) == st]) for st in states])
            per_ru = np.array([np.mean([r[f'{obj}_perru'] for r in rs if (r['num_ue'], r['noise_dbm']) == st])
                               for st in states])
            static_mean = lbar.mean(0).min()
            static_peak = lbar.max(0).min()
            choice = ''.join('12'[i] for i in lbar.argmin(1))
            out.append(dict(
                se_model=sm, k=k, profile=prof, objective=obj,
                static_split_mean=SPLITS[int(lbar.mean(0).argmin())],
                static_split_peak=SPLITS[int(lbar.max(0).argmin())],
                per_state_choice=choice, states=[list(s) for s in states],
                static_mean_gbps=float(static_mean), static_peak_gbps=float(static_peak),
                gain_state_mean=float(1 - lbar.min(1).mean() / static_mean),
                gain_state_peak=float(1 - lbar.min(1).max() / static_peak),
                gain_drop_mean=float(1 - per_drop.mean() / static_mean),
                gain_ru_mean=float(1 - per_ru.mean() / static_mean),
                gain_ru_peak=float(1 - per_ru.max() / static_peak),
                lbar_gbps=lbar.tolist()))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage0')
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=18)
    args = p.parse_args()
    here = Path(__file__).parent
    stage0 = Path(args.stage0)
    run = Run(args.out, [here / f for f in ('screen_stage1a.py', 'fhload.py', 'summarize_stage0.py',
                                            'cfsim.py', 'runio.py')],
              dict(purpose='Stage 1a: load-only screen of dynamic split selection with compression, '
                           'explicit O-RAN headers, CI and beamforming weights',
                   stage0_run=str(stage0), args=vars(args), kappa=s0.KAPPA, ks=KS,
                   profiles=[vars(pr) for pr in PROFILES],
                   assumptions=['Serving sets, beamforming choice and per-UE SE come from stage 0 (paper method).',
                                'No compression distortion in SE (stage 1b); BFP9 assumed transparent.',
                                'Modulation compression bits/RE = Qm of the UE from MCS table 2 at its mean SE.',
                                'SRS: comb-4 x 4 cyclic shifts = 16 UEs per SRS symbol, antenna-level IQ in BFP9.',
                                'CI from serving antennas only (lower bound; candidate antennas ignored).',
                                'Weights: SE-11, BFP9, PRG = 2 PRB; per slot (O-RAN default) or per SRS occasion.',
                                '7-2 + MRT: O-RU computes weights from its own SRS, no CI and no weights.']))
    rows0 = read_rows(stage0)
    per_ue = np.load(stage0 / 'per_ue_se.npz')
    states = s0.select_states(rows0)
    use = s0.usages(rows0, states, per_ue)
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for rs in ex.map(state_task, sorted(use.items())):
            for r in rs:
                run.add(r)
            rows += rs
    summary = analyse(rows)
    (run.path / 'stage1a_summary.json').write_text(json.dumps(summary, indent=1))
    run.finish()
    report(summary)


def report(summary):
    for obj in ('dl',):
        for sm in ('shannon', 'mcs'):
            print(f'\n=== SE model {sm}, objective {obj}: per-state choice (U=5,10,20 x noise -90,-70,-50; '
                  f'1 = 7-1, 2 = 7-2) | static split | gains: state-mean / state-peak / per-drop / per-RU [%]')
            for prof in [pr.name for pr in PROFILES]:
                print(f'-- {prof}')
                for e in summary:
                    if e['se_model'] == sm and e['objective'] == obj and e['profile'] == prof:
                        print(f"   K={e['k']:2d}  {e['per_state_choice']}  static {e['static_split_mean']} "
                              f"({e['static_mean_gbps']:6.1f} Gbit/s)  "
                              f"{100 * e['gain_state_mean']:5.1f} / {100 * e['gain_state_peak']:5.1f} / "
                              f"{100 * e['gain_drop_mean']:5.1f} / {100 * e['gain_ru_mean']:5.1f}")


if __name__ == '__main__':
    main()


def plot(run_dir):
    """动态收益随 K 的变化（按状态平均收益、峰值收益、按 RU 收益），两种频谱效率模型各一行。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / 'stage1a_summary.json').read_text())
    shown = ['P0_paper', 'E0_headers', 'E1_bfp9', 'E2_modcomp', 'E3_modcomp_ci']
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
    for i, sm in enumerate(('shannon', 'mcs')):
        for j, (key, title) in enumerate((('gain_state_mean', 'per-state, mean load'),
                                          ('gain_state_peak', 'per-state, peak load'),
                                          ('gain_ru_mean', 'per-RU per-drop, mean load'))):
            ax = axes[i, j]
            for prof in shown:
                es = sorted((e for e in summary if e['se_model'] == sm and e['objective'] == 'dl'
                             and e['profile'] == prof), key=lambda e: e['k'])
                ax.plot([e['k'] for e in es], [100 * e[key] for e in es], marker='o', ms=3, label=prof)
            ax.set_xscale('log', base=2)
            ax.set_title(f'{title} ({sm} SE)', fontsize=9)
            ax.grid(alpha=.3)
            if i == 1:
                ax.set_xlabel('antennas per RU K')
            if j == 0:
                ax.set_ylabel('saving vs best static split [%]')
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(run_dir / 'fig_dynamic_gain_vs_k.png', dpi=150)
