#!/usr/bin/env python3
"""Paper-1 manuscript figures (vector PDF, IEEE column widths) from results/paper1_20261008.

Data access reuses scripts/summarize_paper1.py (same tables, same bootstrap intervals, same scheme
colours: the documented, validated default categorical palette, one slot per scheme). Output goes to
paper1/manuscript/figures/. Run from the repository root: python scripts/paper1_figures.py
"""
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import summarize_paper1 as S

OUT = Path('paper1/manuscript/figures')
OUT.mkdir(parents=True, exist_ok=True)
INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif',
                     'font.size': 7.5, 'axes.titlesize': 7.5, 'axes.labelsize': 7.5, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.facecolor': 'white', 'figure.facecolor': 'white',
                     'savefig.facecolor': 'white', 'axes.titlecolor': INK, 'axes.spines.top': False,
                     'axes.spines.right': False, 'lines.linewidth': 1.4, 'lines.markersize': 4.2})
W1, W2 = 3.5, 7.16                                          # IEEE single / double column width, inches
LABEL = {'urts_vq/task': 'Proposed', 'urts_code/task': 'Proposed, fixed precision', 'twin_res/uniform': 'Point twin + residual',
         'site_klt/uniform': 'Site-wide twin KLT', 'ad_dft/uniform': 'Angle-delay DFT', 'grid_vq/task': 'Grid cell (64 pts) + proposed coding',
         'cai4:grid_vq/task': 'Grid cell (4 vertices) + proposed coding', 'twin_aep/uniform': 'Twin-trained AE, position input',
         'twin_ae/uniform': 'Twin-trained AE', 'grid_ecb/uniform': 'Grid-cell basis (64 pts)', 'urts/uniform': 'Region KLT, truncation only'}


def style(key):
    lab, c, m, ls = S.STYLE[key]
    return dict(color=S.SLOT[c], marker=m, linestyle=ls, label=LABEL.get(key, lab), markeredgecolor='white',
                markeredgewidth=.6)


def finish(fig, name):
    fig.savefig(OUT/f'{name}.pdf', bbox_inches='tight', pad_inches=.02)
    fig.savefig(OUT/f'{name}.png', dpi=200, bbox_inches='tight', pad_inches=.02)
    plt.close(fig)


def curves(ax, d, keys, gram=None, ci=True):
    for z, key in enumerate(keys):
        if key not in d:
            continue
        cv = d[key]
        b = sorted(cv, key=lambda x: cv[x]['kbits'])
        x = [cv[k]['kbits'] for k in b]
        ax.plot(x, [100*cv[k]['frac'] for k in b], zorder=20-z, **style(key))
        if ci:
            ax.fill_between(x, [100*cv[k]['frac_lo'] for k in b], [100*cv[k]['frac_hi'] for k in b],
                            color=S.SLOT[S.STYLE[key][1]], alpha=.12, linewidth=0, zorder=1)
    if gram:
        ax.axvline(gram, color=INK2, linewidth=.9, linestyle=(0, (1, 1.5)))
    ax.axhline(100, color=MUTED, linewidth=.8, linestyle=(0, (3, 2)))
    ax.set_xscale('log')
    ax.grid(color=GRID, linewidth=.5)
    ax.set_ylim(0, 106)


def legend_below(fig, axes, ncol, y=-.02):
    h, l = [], []
    for ax in np.ravel(axes):
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    fig.legend(h, l, frameon=False, loc='upper center', ncol=ncol, bbox_to_anchor=(.5, y), labelcolor=INK2,
               handlelength=2.2, columnspacing=1.2)


def fig_main(rep):
    keys = ['urts_vq/task', 'urts_code/task', 'twin_res/uniform', 'site_klt/uniform', 'ad_dft/uniform']
    fig, axes = plt.subplots(2, 2, figsize=(W2, 3.9), sharey=True, sharex=True)
    for i, real in enumerate(['nominal', 'diffuse']):
        e = rep[real]['8']
        for j, s in enumerate(['0', '2']):
            ax = axes[i, j]
            curves(ax, e['curves'][s], keys, gram=e['gram_ul_kbits'])
            ax.set_title(f"({'abcd'[2*i+j]}) {'Reality = twin' if real == 'nominal' else 'Reality with diffuse scattering'}, "
                         f"$\\sigma$ = {s} m", loc='left')
        axes[0, 1].text(e['gram_ul_kbits']*.93, 4, 'Gram (UL)', color=INK2, fontsize=6.5, ha='right')
    fig.supylabel('Sum SE (% of uncompressed CI)', fontsize=7.5, color=INK2, x=.005)
    fig.supxlabel('Fronthaul per O-RU per SRS occasion (kbit)', fontsize=7.5, color=INK2, y=.0)
    legend_below(fig, axes, 5, y=-.035)
    fig.tight_layout(h_pad=.6, w_pad=.8)
    finish(fig, 'fig_main_k8')


def fig_kscale(rep):
    keys = ['urts_vq/task', 'urts_code/task', 'grid_vq/task']
    panels = [('nominal', '0', .95, '(a) Reality = twin, $\\sigma$ = 0, 95 %'), ('nominal', '2', .95, '(b) Reality = twin, $\\sigma$ = 2 m, 95 %'),
              ('diffuse', '0', .9, '(c) With diffuse scattering, $\\sigma$ = 0, 90 %')]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.15), sharey=True)
    for ax, (real, s, t, title) in zip(axes, panels):
        Ks = sorted(int(k) for k in rep[real])
        for key in keys:
            xs, ys = [], []
            for K in Ks:
                v = rep[real][str(K)]['bits_to'].get(f'{key} sigma={s}', {}).get(f'{t:g}', (None, 0))[0]
                if v is not None:
                    xs.append(K)
                    ys.append(v)
            ax.plot(xs, ys, **style(key))
        ax.plot(Ks, [rep[real][str(K)]['gram_ul_kbits'] for K in Ks], color=INK2, linestyle=(0, (1, 1.5)), marker='x',
                markersize=4, label='Local Gram + central inverse (UL only)')
        ax.set_yscale('log')
        ax.set_xscale('log', base=2)
        ax.set_xticks(Ks)
        ax.set_xticklabels([str(k) for k in Ks])
        ax.minorticks_off()
        ax.set_title(title, loc='left')
        ax.set_xlabel('Number of UEs $K$')
        ax.grid(color=GRID, linewidth=.5)
    axes[0].set_ylabel('kbit per O-RU per SRS occasion')
    legend_below(fig, axes, 4, y=-.08)
    fig.tight_layout(w_pad=.8)
    finish(fig, 'fig_kscale')


def fig_ablation(rep):
    keys = ['urts_vq/task', 'grid_vq/task', 'cai4:grid_vq/task', 'twin_aep/uniform', 'site_klt/uniform']
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.25), sharey=True)
    e = rep['nominal']['8']
    for ax, s, tag in zip(axes, ['0', '2', '5'], 'abc'):
        curves(ax, e['curves'][s], keys, gram=e['gram_ul_kbits'])
        ax.set_title(f'({tag}) Reality = twin, $\\sigma$ = {s} m', loc='left')
    axes[0].set_ylabel('Sum SE (% of uncompressed CI)')
    fig.supxlabel('Fronthaul per O-RU per SRS occasion (kbit)', fontsize=7.5, color=INK2, y=.0)
    legend_below(fig, axes, 3, y=-.06)
    fig.tight_layout(w_pad=.8)
    finish(fig, 'fig_ablation')


def fig_ae_study():
    rows = S.rows('ae_study_ru01')
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = r['model'] if r['model'] != 'ae' else f"ae{r['steps']}"
        g[key][r['d']].append(r['nmse_mean_db'])
    spec = [('pca', 'Site-wide twin KLT (linear AE)', 3, 'D', '-'), ('ae4000', 'Twin-trained AE, nested dropout', 6, 'X', '-'),
            ('ae_d', 'Twin-trained AE, one model per $d$', 6, 's', 'none'), ('pca_res', 'KLT + learned correction', 0, 'o', (0, (4, 2)))]
    fig, ax = plt.subplots(figsize=(W1, 2.1))
    for key, lab, c, m, ls in spec:
        d = sorted(g[key])
        ax.plot(d, [np.mean(g[key][x]) for x in d], color=S.SLOT[c], marker=m, linestyle=ls, label=lab,
                markeredgecolor='white', markeredgewidth=.6)
    ax.set_xscale('log', base=2)
    ax.set_xticks([4, 8, 16, 32, 64, 128])
    ax.set_xticklabels(['4', '8', '16', '32', '64', '128'])
    ax.minorticks_off()
    ax.set_xlabel('Latent / KLT dimensions per link $d$')
    ax.set_ylabel('Held-out shape NMSE (dB)')
    ax.grid(color=GRID, linewidth=.5)
    ax.legend(frameon=False, labelcolor=INK2, loc='upper right')
    fig.tight_layout()
    finish(fig, 'fig_ae_study')


def fig_mismatch(mm):
    keys = ['urts_vq/task', 'grid_vq/task', 'urts_code/task', 'ad_dft/uniform']
    tags = [t for t, _ in S.MM if t in mm]
    lab = {'nominal': 'None', 'geom05': 'Geometry\n0.5 m', 'mat03': 'Materials\n±30 %', 'cars40': '40 cars',
           'diffusetw0': 'Diffuse', 'all': 'All'}
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.2), sharey=True)
    x = np.arange(len(tags))
    for ax, budget, tag in zip(axes, (32, 128), 'ab'):
        for i, key in enumerate(keys):
            y, lo, hi = [], [], []
            for t in tags:
                c = mm[t]['0'][key][str(budget)]
                y.append(100*c['frac'])
                lo.append(100*c['frac_lo'])
                hi.append(100*c['frac_hi'])
            st = style(key)
            st.update(linestyle='none', markersize=5.5, markeredgewidth=.5)
            ax.errorbar(x+(i-1.5)*.17, y, yerr=[np.array(y)-lo, np.array(hi)-y], ecolor=st['color'], elinewidth=.9, **st)
        ax.set_xticks(x)
        ax.set_xticklabels([lab[t] for t in tags])
        ax.axhline(100, color=MUTED, linewidth=.8, linestyle=(0, (3, 2)))
        ax.set_title(f'({tag}) {budget} coefficients per link budget, $\\sigma$ = 0', loc='left')
        ax.grid(axis='y', color=GRID, linewidth=.5)
    axes[0].set_ylabel('Sum SE (% of uncompressed CI)')
    legend_below(fig, axes, 4, y=-.1)
    fig.tight_layout(w_pad=.8)
    finish(fig, 'fig_mismatch')


def fig_tracking():
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.15))
    c = 0
    for run, real in [('track_final_nominal_8', 'reality = twin'), ('track_final_diffuse_8', 'diffuse')]:
        rep = S.tracking_report(run)
        for s0, m in [(5, 'o'), (10, 's')]:
            occ = [1, 2, 3]
            err = [rep[f'tracked s0={s0} t={t}']['loc_err_median'] for t in occ]
            p90 = [rep[f'tracked s0={s0} t={t}']['loc_err_p90'] for t in occ]
            col = S.SLOT[[0, 1][c//2]]
            ls = '-' if s0 == 5 else (0, (4, 2))
            axes[0].plot(occ, err, color=col, marker=m, linestyle=ls, label=f'{real}, $\\sigma_0$ = {s0} m',
                         markeredgecolor='white', markeredgewidth=.6)
            orc = rep['oracle s0=0 t=0']['se']['32']['frac']
            axes[1].plot(occ, [100*rep[f'tracked s0={s0} t={t}']['se']['32']['frac']/orc for t in occ], color=col, marker=m,
                         linestyle=ls, label=f'{real}, $\\sigma_0$ = {s0} m', markeredgecolor='white', markeredgewidth=.6)
            c += 1
    axes[0].set_yscale('log')
    axes[0].set_yticks([.2, .3, .5, 1, 2])
    axes[0].set_yticklabels(['0.2', '0.3', '0.5', '1', '2'])
    axes[0].minorticks_off()
    axes[0].set_ylabel('Median localization error (m)')
    axes[0].set_title('(a) O-DU position estimate', loc='left')
    axes[1].set_ylabel('Sum SE (% of known position)')
    axes[1].axhline(100, color=MUTED, linewidth=.8, linestyle=(0, (3, 2)))
    axes[1].set_title('(b) 32 coefficients per link (about 3–4 kbit)', loc='left')
    for ax in axes:
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel('SRS occasion')
        ax.grid(color=GRID, linewidth=.5)
    legend_below(fig, axes, 4, y=-.08)
    fig.tight_layout(w_pad=1.2)
    finish(fig, 'fig_tracking')


def fig_dimension(dim):
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    order = [('4x4_52', 0, '-', 'o', '4×4, 19 MHz'), ('4x4_104', 0, (0, (4, 2)), 'o', '4×4, 37 MHz'),
             ('8x8_52', 1, '-', 's', '8×8, 19 MHz'), ('8x8_104', 1, (0, (4, 2)), 's', '8×8, 37 MHz')]
    for key, c, ls, m, lab in order:
        d = dim[key]
        s = sorted(d, key=float)
        ax.plot([float(x) for x in s], [d[x]['d_noise_median'] for x in s], color=S.SLOT[c], linestyle=ls, marker=m,
                label=lab, markeredgecolor='white', markeredgewidth=.6)
    ax.set_xscale('log')
    ax.set_xticks([.25, .5, 1, 2, 3, 5, 8])
    ax.set_xticklabels(['0.25', '0.5', '1', '2', '3', '5', '8'])
    ax.minorticks_off()
    ax.set_xlabel('Region standard deviation $s$ (m)')
    ax.set_ylabel('KLT dimensions above CI noise')
    ax.grid(color=GRID, linewidth=.5)
    ax.legend(frameon=False, labelcolor=INK2, title='Array (cross-pol.), bandwidth', title_fontsize=6.5)
    fig.tight_layout()
    finish(fig, 'fig_dimension')


def main():
    rep = json.loads((S.R/'paper1_summary.json').read_text())
    fig_main(rep['main'])
    fig_kscale(rep['main'])
    fig_ablation(rep['main'])
    fig_ae_study()
    fig_mismatch(rep['mismatch'])
    fig_tracking()
    fig_dimension(rep['dimension'])
    print('wrote', sorted(p.name for p in OUT.glob('*.pdf')))


if __name__ == '__main__':
    main()
