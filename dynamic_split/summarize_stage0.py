"""阶段 0 汇总：由逐根考察曲线得到 M̂、波束赋形选择和论文口径的前传负载，并与论文图 3–5 对照。

用法：
    .venv/bin/python dynamic_split/summarize_stage0.py results/dynamic_split_20261012/stage0_repro
输出（写在运行目录里）：stage0_summary.json、fig_se_curves.png、fig_load_vs_k.png
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import cfsim  # noqa: E402
import fhload  # noqa: E402
from runio import read_rows  # noqa: E402

KAPPA = 0.98
KS = [1, 2, 4, 8, 16, 32, 64]
# 论文图 5：(UE 数, 噪声) -> (波束赋形, 和速率, {K: (切分, Gbit/s)})
PAPER_FIG5 = {
    (20, -90): ('zf', 238, {1: ('7-1', 33.6), 4: ('7-1', 27.7), 8: ('7-1', 26.3)}),
    (10, -90): ('zf', 127, {1: ('7-1', 25.0), 4: ('7-1', 19.1), 8: ('7-1', 17.7)}),
    (5, -90): ('zf', 67, {1: ('7-1', 18.0), 4: ('7-1', 12.2), 8: ('7-1', 10.7)}),
    (20, -70): ('zf', 109, {1: ('7-1', 36.2), 4: ('7-1', 30.4), 8: ('7-1', 29.0)}),
    (10, -70): ('zf', 62, {1: ('7-1', 28.2), 4: ('7-1', 22.4), 8: ('7-1', 20.9)}),
    (5, -70): ('mrt', 37, {1: ('7-1', 29.2), 4: ('7-1', 23.4), 8: ('7-2', 10.9)}),
    (20, -50): ('mrt', 33, {1: ('7-1', 40.4), 4: ('7-2', 33.1), 8: ('7-2', 18.6)}),
    (10, -50): ('mrt', 22, {1: ('7-1', 40.5), 4: ('7-2', 31.6), 8: ('7-2', 15.2)}),
    (5, -50): ('mrt', 14, {1: ('7-1', 40.0), 4: ('7-2', 29.4), 8: ('7-2', 11.6)}),
}


def select_states(rows, kappa=KAPPA):
    """每个 (UE 数, 噪声, 频谱效率模型)：两种波束赋形的 M̂、κ 处的和速率，以及选中的波束赋形。"""
    curves = defaultdict(list)
    for r in rows:
        curves[(r['num_ue'], r['noise_dbm'], r['se_model'], r['bf'])].append(r['curve'])
    states = {}
    for (u, n, sm, bf), cs in curves.items():
        mean = np.mean(cs, 0)
        m_hat = int(np.argmax(mean >= kappa * mean.max())) + 1
        states.setdefault((u, n, sm), {})[bf] = dict(m_hat=m_hat, se=float(mean[m_hat - 1]),
                                                     se_max=float(mean.max()), curve=mean.tolist())
    for st in states.values():
        st['bf'] = max(('zf', 'mrt'), key=lambda b: st[b]['se'])
    return states


def usages(rows, states, per_ue):
    """每个 (UE 数, 噪声, 模型) 下、选中波束赋形的逐撒点服务情况。"""
    out = defaultdict(list)
    for r in rows:
        key = (r['num_ue'], r['noise_dbm'], r['se_model'])
        st = states[key]
        if r['bf'] != st['bf']:
            continue
        m_hat = st[r['bf']]['m_hat']
        se_ue = per_ue[r['per_ue_key']][m_hat - 1]
        out[key].append(fhload.Usage.from_row(r, m_hat, cfsim.qm_from_se(se_ue)))
    return out


def main(run_dir):
    run_dir = Path(run_dir)
    rows = read_rows(run_dir)
    per_ue = np.load(run_dir / 'per_ue_se.npz')
    states = select_states(rows)
    use = usages(rows, states, per_ue)
    summary = dict(kappa=KAPPA, states=[])
    for (u, n, sm), st in sorted(states.items()):
        loads = {k: {s: float(np.mean([fhload.paper_load(x, k)[s].sum() for x in use[(u, n, sm)]])) / 1e9
                     for s in ('7-1', '7-2')} for k in KS}
        n_serv = float(np.mean([x.ants.sum() for x in use[(u, n, sm)]]))
        entry = dict(num_ue=u, noise_dbm=n, se_model=sm, bf=st['bf'],
                     zf=dict(m_hat=st['zf']['m_hat'], se=st['zf']['se']),
                     mrt=dict(m_hat=st['mrt']['m_hat'], se=st['mrt']['se']),
                     serving_antennas=n_serv, load_gbps=loads,
                     best_split={k: min(loads[k], key=loads[k].get) for k in KS})
        if sm == 'shannon' and (u, n) in PAPER_FIG5:
            pb, pse, pk = PAPER_FIG5[(u, n)]
            entry['paper'] = dict(bf=pb, se=pse, fig5={k: list(v) for k, v in pk.items()})
        summary['states'].append(entry)
    (run_dir / 'stage0_summary.json').write_text(json.dumps(summary, indent=1))

    # 文字对照表
    print('model    U  noise | ours: bf  SE  M̂  serv | 7-1/7-2 Gbit/s at K=1,4,8        | paper: bf SE  K=1,4,8')
    for e in summary['states']:
        lk = ' '.join(f"{e['load_gbps'][k]['7-1']:5.1f}/{e['load_gbps'][k]['7-2']:5.1f}" for k in (1, 4, 8))
        pp = ''
        if 'paper' in e:
            p = e['paper']
            pp = f"{p['bf']:3s} {p['se']:3d}  " + ' '.join(f"{v[0]} {v[1]}" for v in p['fig5'].values())
        b = e['bf']
        print(f"{e['se_model']:7s} {e['num_ue']:2d} {int(e['noise_dbm']):4d} | {b:3s} {e[b]['se']:5.1f} {e[b]['m_hat']:2d} "
              f"{e['serving_antennas']:4.1f} | {lk} | {pp}")
    plot(summary, run_dir)


def plot(summary, run_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    st = {(e['num_ue'], e['noise_dbm'], e['se_model']): e for e in summary['states']}
    # 图 4 的对应：10 UE，负载随 K
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, n in zip(axes, (-90, -70, -50)):
        for sm, ls in (('shannon', '-'), ('mcs', '--')):
            e = st[(10, float(n), sm)]
            for s, c in (('7-1', 'C0'), ('7-2', 'C1')):
                ax.plot(KS, [e['load_gbps'][k][s] for k in KS], ls, color=c, marker='o', ms=3,
                        label=f"{s}, {sm} SE")
        ax.set_xscale('log', base=2)
        bf = '/'.join(sorted({st[(10, float(n), sm)]['bf'].upper() for sm in ('shannon', 'mcs')}))
        ax.set_title(f'10 UEs, noise {n} dBm/PRB, {bf}', fontsize=10)
        ax.set_xlabel('antennas per RU K')
        ax.grid(alpha=.3)
    axes[0].set_ylabel('fronthaul load [Gbit/s]')
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(run_dir / 'fig_load_vs_k.png', dpi=150)


if __name__ == '__main__':
    main(sys.argv[1])
