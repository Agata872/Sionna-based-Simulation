#!/usr/bin/env python3
"""Figures for the stage-4 SRS-BF convergence note (static PNG, light surface).

Palette: documented default categorical slots (validated adjacent-pair order);
the validator could not be run here (no node), so every multi-series figure
also carries marker shapes and direct labels as secondary encoding.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from summarize_srsbf import load, per_drop

R = Path('results/stage4_srsbf_20261007')
SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
BLUE = SERIES[0]
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.facecolor': SURFACE,
                     'figure.facecolor': SURFACE, 'savefig.facecolor': SURFACE, 'axes.titlesize': 10,
                     'axes.titlecolor': INK, 'axes.spines.top': False, 'axes.spines.right': False})
_cache = {}


def metric(run, policy, key):
    if run not in _cache:
        meta, rows = load(R/run)
        _cache[run] = per_drop(meta, rows)
    d = _cache[run][policy]
    return float(np.mean([d[x][key] for x in sorted(d)]))


def ladder():
    rows = [('Nominal-noise query (as specified)', {3: ('loop_v3_8', 'q_nom'), 30: ('loop_v30_8', 'q_nom')}),
            ('Legacy: SU-CQI + OLLA', {3: ('main_v3_8', 'su_olla'), 30: ('main_v30_8', 'su_olla')}),
            ('Legacy + age-grouped OLLA', {3: ('lat_v3_8', 'su_olla+A'), 30: ('aging_v30_srs10_8', 'su_olla+A')}),
            ('Beam-power prior (0 queries)', {3: ('main_v3_8', 'beam2r_olla'), 30: ('main_v30_8', 'beam2r_olla')}),
            ('Planned-set queries (4 per TDD period)', {3: ('main_v3_8', 'set_qr'), 30: ('main_v30_8', 'set_qr')}),
            ('Calibrated queries, unlimited*', {3: ('main_v3_8', 'qr_cqi'), 30: ('aging_v30_srs10_8', 'qr_cqi+A')}),
            ('Causal-ideal O-DU (reference)', {3: ('main_v3_8', 'ref_dec'), 30: ('main_v30_8', 'ref_dec')})]
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 5.6), sharey=True)
    for r, speed in enumerate([3, 30]):
        for c, (key, title) in enumerate([('thr', 'Sum goodput, Mbit/s'), ('util', 'PF utility, sum log(Mbit/s)')]):
            ax = axes[r, c]
            vals = [metric(*m[speed], key) for _, m in rows]
            ok = [v for v in vals if v > -20]
            lo = 0 if key == 'thr' else min(ok)-1.5
            hi = max(ok)*1.12 if key == 'thr' else max(ok)+1.2
            y = np.arange(len(rows))[::-1]
            for yi, v, (lab, _) in zip(y, vals, rows):
                color = MUTED if 'reference' in lab else BLUE
                if v <= lo+1e-9:
                    ax.text(lo+.01*(hi-lo), yi, 'collapses (MCS selection fails)' if key == 'thr' else f'{v:.0f}: collapses',
                            va='center', ha='left', color=INK2, fontsize=8)
                    continue
                ax.barh(yi, v-lo, left=lo, height=.56, color=color, edgecolor=SURFACE, linewidth=2)
                ax.text(v+.01*(hi-lo), yi, f'{v:.0f}' if key == 'thr' else f'{v:.1f}', va='center', ha='left',
                        color=INK2, fontsize=8)
            ax.set_xlim(lo, hi)
            ax.set_yticks(y)
            ax.set_yticklabels([lab for lab, _ in rows], color=INK2)
            ax.set_title(f'{title} — {speed} km/h', loc='left')
            ax.grid(axis='x', color=GRID, linewidth=.6)
            ax.set_axisbelow(True)
    fig.suptitle('O-DU information policy under O-RU-side SRS beamforming (TR 38.901 UMi, 32-element O-RU, 8 UEs, '
                 'SRS every 10 slots, 8 paired drops)', x=.01, ha='left', color=INK, fontsize=9.5)
    fig.text(.01, .01, '* with age-grouped OLLA at 30 km/h; at 3 km/h age grouping hurts (no aging to learn).',
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, .03, 1, .95))
    fig.savefig(R/'fig_policy_ladder.png', dpi=160)


def aging():
    periods = [5, 10, 20]
    runs = {5: 'srs5plus_v30_8', 10: 'aging_v30_srs10_8', 20: 'aging_v30_srs20_8'}
    alt = {5: 'srs5_v30_8'}
    series = [('Legacy + age-grouped OLLA', 'su_olla+A', 'o'), ('Calibrated queries + age-grouped OLLA', 'qr_cqi+A', 's'),
              ('Per-UE aging statistics (statistical genie)', 'qs_cqi', '^'),
              ('Perfect info at decision time (no aging model)', 'ref_dec', 'D')]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for i, (lab, pol, mk) in enumerate(series):
        ys = []
        for p in periods:
            run = runs[p]
            try:
                ys.append(metric(run, pol, 'thr'))
            except (KeyError, FileNotFoundError):
                ys.append(metric(alt[p], pol, 'thr'))
        ax.plot(periods, ys, color=SERIES[i], marker=mk, markersize=8, linewidth=2,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=lab)
        ax.text(periods[-1]+.6, ys[-1], lab.split(' (')[0], color=INK2, fontsize=8, va='center')
    legacy = []
    for p in periods:
        run = runs[p]
        try:
            legacy.append(metric(run, 'su_olla', 'thr'))
        except (KeyError, FileNotFoundError):
            legacy.append(metric(alt[p], 'su_olla', 'thr'))
    ax.plot(periods, legacy, color=MUTED, linewidth=1.5, linestyle=(0, (3, 3)), marker='x', label='Legacy SU-CQI + OLLA')
    ax.set_xticks(periods)
    ax.set_xlabel('SRS period, slots (0.5 ms)')
    ax.set_ylabel('Sum goodput, Mbit/s')
    ax.set_xlim(4, 29)
    ax.grid(color=GRID, linewidth=.6)
    ax.legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(.42, -.16), ncol=2, labelcolor=INK2)
    ax.set_title('Aging at 30 km/h: what per-UE decorrelation statistics are worth', loc='left')
    fig.tight_layout()
    fig.savefig(R/'fig_aging_srs_period.png', dpi=160)


def fronthaul():
    cost = json.loads((R/'fronthaul_cost.json').read_text())['cases']
    pretty = {'ST6': 'ST 6 SRS CI to O-DU (BFP9)', 'ST13_Tmatrix': 'ST 13 T-matrix + beam power',
              'ST13_CI': 'ST 13 CI coherence (proposed)', 'ST11+13': 'ST 11/13 post-EQ SINR queries',
              'reference': 'Reference: DL layer U-plane payload'}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2), sharey=True)
    for i, (ax, name) in enumerate(zip(axes, cost)):
        items = list(cost[name].items())
        y = np.arange(len(items))[::-1]
        for yi, (k, v) in zip(y, items):
            ref = k.startswith('reference')
            ax.barh(yi, v, color=MUTED if ref else BLUE, height=.56, edgecolor=SURFACE, linewidth=2)
            note = ''
            if k.startswith('ST11'):
                note = f"  ({k.split('_')[-3][1:]}/TDD period)"
            ax.text(v*1.15, yi, (f'{v:.1f}' if v >= 1 else f'{v:.2f}')+note, va='center', color=INK2, fontsize=8)
        ax.set_xscale('log')
        if i == 0:
            ax.set_yticks(y)
            ax.set_yticklabels([pretty[next(p for p in pretty if k.startswith(p))] for k, _ in items], color=INK2)
        ax.set_title(name, loc='left', fontsize=9)
        ax.set_xlabel('Wire rate, Mbit/s (log scale)')
        ax.grid(axis='x', color=GRID, linewidth=.6)
        ax.set_axisbelow(True)
        ax.set_xlim(.3, 3e5)
    fig.text(.01, .01, 'Blue bars: O-RU→O-DU information per option. Gray bar: DU→RU layer U-plane payload, for scale.',
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(R/'fig_fronthaul_cost.png', dpi=160)


if __name__ == '__main__':
    for job in sys.argv[1:] or ['ladder', 'aging', 'fronthaul']:
        globals()[job]()
        print('wrote', job)
