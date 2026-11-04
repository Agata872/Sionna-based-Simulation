#!/usr/bin/env python3
"""Stage-5 summaries and figures (twin-aided CI compression). Static PNG, light surface.

Palette: documented default categorical slots (adjacent-pair validated order); the
palette validator needs node, which is unavailable here, so every multi-series figure
also uses marker shapes and direct labels as secondary encoding.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

R = Path('results/stage5_twin_20261007')
SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
MARK = ['o', 's', '^', 'D', 'v']
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.facecolor': SURFACE,
                     'figure.facecolor': SURFACE, 'savefig.facecolor': SURFACE, 'axes.titlesize': 9.5,
                     'axes.titlecolor': INK, 'axes.spines.top': False, 'axes.spines.right': False})


def rows(run):
    return [json.loads(x) for x in (R/run/'rows.jsonl').open()]


def single_ru_table(run):
    g = defaultdict(list)
    for r in rows(run):
        g[(r['sigma'], r['scheme'], r['budget'])].append(r)
    out = {}
    for (s, sch, b), v in g.items():
        out.setdefault(str(s), {}).setdefault(sch, {})[b] = dict(
            nmse_median=float(np.median([x['nmse_db'] for x in v])),
            nmse_p90=float(np.percentile([x['nmse_db'] for x in v], 90)),
            coeffs=float(np.mean([x['coeffs'] for x in v])),
            side_bits=float(np.mean([x['side_bits'] for x in v])), n=len(v))
    return out


def fig_single(runs, name):
    schemes = [('ad_dft', 'Angle-delay DFT (generic)'), ('twin_path', 'Twin path atoms (point)'),
               ('twin_adapt', 'Mismatch-aware twin atoms'), ('cov_klt', 'Uncertainty-region twin subspace')]
    sigmas = [0., 2., 5.]
    fig, axes = plt.subplots(len(runs), len(sigmas), figsize=(10.2, 3.1*len(runs)), sharey=True)
    axes = np.atleast_2d(axes)
    for i, (run, title) in enumerate(runs):
        tab = single_ru_table(run)
        for j, s in enumerate(sigmas):
            ax = axes[i, j]
            for c, (sch, lab) in enumerate(schemes):
                d = tab[str(s)].get(sch, {})
                b = sorted(d)
                x = [d[k]['coeffs'] for k in b]
                y = [d[k]['nmse_median'] for k in b]
                ax.plot(x, y, color=SERIES[c], marker=MARK[c], markersize=6, linewidth=2,
                        markeredgecolor=SURFACE, markeredgewidth=1.2, label=lab)
            ax.set_xscale('log', base=2)
            ax.set_title(f'{title}, position error σ = {s:g} m', loc='left')
            ax.grid(color=GRID, linewidth=.6)
            ax.set_xlabel('Complex coefficients sent per UE port')
            if j == 0:
                ax.set_ylabel('Median CI reconstruction NMSE, dB')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, loc='lower center', ncol=4, labelcolor=INK2,
               bbox_to_anchor=(.5, .03))
    fig.text(.01, .005, 'Raw CI = 64 elements x 26 PRB groups = 1664 complex values. Angle-delay DFT also sends '
             '11 index bits per coefficient; twin atoms need no index bits.', color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(R/name, dpi=160)


def cellfree_table(run):
    g = defaultdict(list)
    for r in rows(run):
        g[(r['sigma'], r['scheme'], r['alloc'], r['budget'])].append(r)
    out = {}
    for (s, sch, al, b), v in g.items():
        out.setdefault(str(s), {}).setdefault(f'{sch}/{al}', {})[b] = dict(
            kbits=float(np.mean([x['kbits_per_ru'] for x in v])), sum_se=float(np.mean([x['sum_se'] for x in v])),
            min_se=float(np.mean([x['min_se'] for x in v])), nmse=float(np.mean([x['nmse_db'] for x in v])),
            ideal=float(np.mean([x['sum_se_ideal'] for x in v])), noisy=float(np.mean([x['sum_se_noisy'] for x in v])),
            raw_kbits=float(v[0]['raw_kbits_per_ru']), drops=len({x['drop'] for x in v}))
    return out


def fig_cellfree(panels, name, series=None, gram=False):
    series = series or [('ad_dft/uniform', 'Angle-delay DFT'), ('twin_res/twin', 'Point twin + residual'),
                        ('urts/uniform', 'Region twin subspace, uniform'), ('urts/twin', 'Region twin subspace, twin-guided')]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6*len(panels), 3.7), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (run, s, title) in zip(axes, panels):
        tab = cellfree_table(run)[str(s)]
        ref = next(iter(tab.values()))
        any_b = next(iter(ref.values()))
        ax.axhline(any_b['ideal'], color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
        ax.text(.5, any_b['ideal'], ' ideal CI', color=MUTED, fontsize=7.5, va='bottom')
        for c, (key, lab) in enumerate(series):
            d = tab.get(key, {})
            b = sorted(d)
            ax.plot([d[k]['kbits'] for k in b], [d[k]['sum_se'] for k in b], color=SERIES[c], marker=MARK[c],
                    markersize=6, linewidth=2, markeredgecolor=SURFACE, markeredgewidth=1.2, label=lab)
        ax.plot([any_b['raw_kbits']], [any_b['noisy']], marker='x', color=INK2, markersize=8, linestyle='none',
                label='Uncompressed CI (BFP9)')
        if gram:
            a_ = json.loads((R/run/'metadata.json').read_text())['args']
            gk = a_['num_ue']**2*9*a_['groups']/1e3                 # Hermitian KxK per PRB group, 9-bit reals
            ax.plot([gk], [any_b['noisy']], marker='P', color=INK2, markersize=7, linestyle='none',
                    label='Local Gram + central inverse (exact RZF; UL shown, same again on DL)')
        ax.set_xscale('log')
        ax.set_title(title, loc='left')
        ax.set_xlabel('Fronthaul per O-RU per SRS occasion, kbit (log)')
        ax.grid(color=GRID, linewidth=.6)
    axes[0].set_ylabel('Sum SE, bit/s/Hz (centralized RZF)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc='lower center', ncol=3, labelcolor=INK2,
               bbox_to_anchor=(.5, .0))
    fig.tight_layout(rect=(0, .13, 1, 1))
    fig.savefig(R/name, dpi=160)


def fig_proxy(runs, name, gamma=2.7):
    """Measured vs predicted sum-SE loss (first-order proxy with fitted leakage factor), every row."""
    fig, axes = plt.subplots(1, len(runs), figsize=(3.9*len(runs), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    groups = [('ad_dft', 'Angle-delay DFT'), ('twin_res', 'Point twin + residual'), ('urts', 'Region subspace (plain KLT)'),
              ('urts_code', 'URTS-CODE')]
    for ax, (run, title) in zip(axes, runs):
        rs = rows(run)
        for c, (sch, lab) in enumerate(groups):
            x = np.array([r['sum_se_ideal']-r['sum_se'] for r in rs if r['scheme'] == sch])
            y = np.array([r['proxy_sum_se_ideal']-r['proxy_gamma_sum_se'] for r in rs if r['scheme'] == sch])
            m = (x >= 1) & (y >= 1)
            ax.plot(x[m], y[m], linestyle='none', marker=MARK[c], color=SERIES[c], markersize=3.8, alpha=.7,
                    markeredgecolor=SURFACE, markeredgewidth=.5, label=lab)
        xn = np.array([r['sum_se_ideal']-r['sum_se_noisy'] for r in rs])
        yn = np.array([r['proxy_sum_se_ideal']-r['proxy_sum_se_noisy'] for r in rs])
        ax.plot(xn, yn, linestyle='none', marker='x', color=INK2, markersize=5, label='Uncompressed CI (white noise, γ = 1)')
        ax.plot([1, 150], [1, 150], color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(1, 150); ax.set_ylim(1, 150)
        ax.set_title(title, loc='left')
        ax.set_xlabel('Measured sum-SE loss vs ideal CI, bit/s/Hz')
        ax.grid(color=GRID, linewidth=.6)
    axes[0].set_ylabel(f'Predicted loss, first-order proxy (γ = {gamma:g})')
    axes[-1].legend(frameon=False, fontsize=7.5, loc='lower right', labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(R/name, dpi=160)


def bits_to(run, sigma, key, frac):
    """kbit per O-RU at which a scheme reaches frac x the uncompressed sum SE (log-bit interpolation);
    None if it never does within the run's budgets. Also returns the largest fraction reached."""
    tab = cellfree_table(run)[str(sigma)].get(key, {})
    b = sorted(tab, key=lambda x: tab[x]['kbits'])
    kb = np.array([tab[x]['kbits'] for x in b])
    fr = np.array([tab[x]['sum_se']/tab[x]['noisy'] for x in b])
    i = np.flatnonzero(fr >= frac)
    if not len(i):
        return None, float(fr.max())
    j = i[0]
    if j == 0:
        return float(kb[0]), float(fr.max())
    return float(np.exp(np.interp(frac, [fr[j-1], fr[j]], [np.log(kb[j-1]), np.log(kb[j])]))), float(fr.max())


KSCALE = {8: ('cellfree_code_nominal_16', 'cellfree_code_diffuse_16'), 16: ('cellfree_k16_nominal_8', 'cellfree_k16_diffuse_8'),
          32: ('cellfree_k32_nominal_6', 'cellfree_k32_diffuse_6')}
KSCALE_VQ = {8: ('cellfree_vq_nominal_16', 'cellfree_vq_diffuse_16'), 16: ('cellfree_vq_k16_nominal_8', 'cellfree_vq_k16_diffuse_8'),
             32: ('cellfree_vq_k32_nominal_6', 'cellfree_vq_k32_diffuse_6')}


def kscale_report(targets=(.8, .9, .95)):
    out = {}
    for K, runs in KSCALE.items():
        for real, run in zip(['nominal', 'diffuse'], runs):
            if not (R/run/'complete.json').exists():
                continue
            a_ = json.loads((R/run/'metadata.json').read_text())['args']
            M, F = 2*a_['rows']*a_['cols'], a_['groups']
            e = out.setdefault(real, {}).setdefault(str(K), dict(run=run, raw_kbits=K*M*F*18/1e3, gram_ul_kbits=K*K*9*F/1e3))
            for sigma in [0., 2.]:
                for key in ['urts_code/task', 'ad_dft/uniform']:
                    e[f'{key} sigma={sigma:g}'] = {f'{t:g}': bits_to(run, sigma, key, t) for t in targets}
                vq = KSCALE_VQ[K][['nominal', 'diffuse'].index(real)]
                if (R/vq/'complete.json').exists():
                    e[f'urts_vq/task sigma={sigma:g}'] = {f'{t:g}': bits_to(vq, sigma, 'urts_vq/task', t) for t in targets}
    return out


def fig_kscale(rep, name, target='0.8', series=(('urts_code/task', 0., 'URTS-CODE 18-bit, σ = 0'), ('urts_code/task', 2., 'URTS-CODE 18-bit, σ = 2 m'),
                                                ('urts_vq/task', 0., 'Twin-set precision, σ = 0'), ('urts_vq/task', 2., 'Twin-set precision, σ = 2 m'))):
    reals = [r for r in ['nominal', 'diffuse'] if r in rep]
    fig, axes = plt.subplots(1, len(reals), figsize=(4.4*len(reals), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, real in zip(axes, reals):
        Ks = sorted(int(k) for k in rep[real])
        ax.plot(Ks, [rep[real][str(k)]['raw_kbits'] for k in Ks], color=INK2, linestyle=(0, (4, 2)), linewidth=1.5,
                marker='x', markersize=6, label='Uncompressed CI (BFP9)')
        ax.plot(Ks, [rep[real][str(k)]['gram_ul_kbits'] for k in Ks], color=INK2, linewidth=1.5, marker='P',
                markersize=6, label='Local Gram, exact RZF (UL only)')
        for c, (key, sigma, lab) in enumerate(series):
            xs, ys = [], []
            for k in Ks:
                v = rep[real][str(k)].get(f'{key} sigma={sigma:g}', {}).get(target, (None, 0))[0]
                if v is not None:
                    xs.append(k)
                    ys.append(v)
            if xs:
                ax.plot(xs, ys, color=SERIES[c], marker=MARK[c], markersize=6, linewidth=2, markeredgecolor=SURFACE,
                        markeredgewidth=1.2, label=lab, linestyle='-' if key == 'urts_code/task' else (0, (4, 2)))
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks(Ks)
        ax.set_xticklabels([str(k) for k in Ks])
        ax.set_xlabel('UEs served by the 4 O-RUs, K')
        ax.set_title({'nominal': 'Twin scene', 'diffuse': 'Reality + diffuse'}[real] + f': bits for {float(target)*100:.0f}% of uncompressed SE',
                     loc='left')
        ax.grid(color=GRID, linewidth=.6)
    axes[0].set_ylabel('Fronthaul per O-RU per SRS occasion, kbit')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc='lower center', ncol=3, labelcolor=INK2, bbox_to_anchor=(.5, .04))
    fig.text(.01, .005, 'Missing points: target not reached within 128 coefficients per link.', color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, .19, 1, 1))
    fig.savefig(R/name, dpi=160)


MISMATCH = [('nominal', 'Twin = reality'), ('geom02', 'Buildings shifted, 0.2 m'), ('geom05', 'Buildings shifted, 0.5 m'),
            ('mat03', 'Materials ±30%'), ('cars40', '40 cars added'), ('all', 'All + diffuse')]
TWINSCAT = [('nominal', 'Twin = reality (no diffuse)'), ('diffuse_tw0', 'Diffuse; twin without diffuse'),
            ('diffuse_tw02', 'Diffuse 0.3; twin models 0.2'), ('diffuse_tw03', 'Diffuse 0.3; twin models 0.3')]


def mismatch_report(series):
    out = {}
    for tag, _ in series:
        run = f'mismatch_{tag}_3'
        if (R/run/'complete.json').exists():
            out[tag] = cellfree_table(run)
    return out


def fig_mismatch(series, name, key='urts_code/task'):
    rep = mismatch_report(series)
    sigmas = ['0.0', '2.0']
    fig, axes = plt.subplots(1, len(sigmas), figsize=(4.6*len(sigmas), 3.7), sharey=True)
    for ax, sg in zip(axes, sigmas):
        c = 0
        for tag, lab in series:
            if tag not in rep or key not in rep[tag].get(sg, {}):
                continue
            d = rep[tag][sg][key]
            b = sorted(d)
            x = [d[k]['kbits'] for k in b]
            y = [100*d[k]['sum_se']/d[k]['noisy'] for k in b]
            if tag == 'nominal':
                ax.plot(x, y, color=INK2, linestyle=(0, (4, 2)), linewidth=1.5, marker='o', markersize=4, label=lab)
            else:
                ax.plot(x, y, color=SERIES[c], marker=MARK[c], markersize=6, linewidth=2, markeredgecolor=SURFACE,
                        markeredgewidth=1.2, label=lab)
                c += 1
        ax.axhline(100, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.set_xticklabels(['1', '2', '4', '8', '16'])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(f'URTS-CODE, position error σ = {float(sg):g} m', loc='left')
        ax.set_xlabel('Fronthaul per O-RU per SRS occasion, kbit (log)')
        ax.grid(color=GRID, linewidth=.6)
    axes[0].set_ylabel('Sum SE, % of uncompressed CI in the same reality')
    axes[0].legend(frameon=False, fontsize=7.5, loc='lower right', labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(R/name, dpi=160)
    return rep


def tracking_report(run):
    rows_ = rows(run)
    se, loc = defaultdict(list), defaultdict(list)
    for r in rows_:
        k = (r['setting'], r['sigma0'], r['occasion'])
        if r['budget'] < 0:
            loc[k].append(r)
        else:
            se[k+(r['budget'],)].append(r)
    out = {}
    for k, v in loc.items():
        e = np.concatenate([x['loc_err'] for x in v])
        centre = np.concatenate([[x['centre_err_median']] for x in se[k+(min(b for (*kk, b) in se if tuple(kk) == k),)]])
        out[f'{k[0]} s0={k[1]:g} t={k[2]}'] = dict(
            loc_err_median=float(np.median(e)), loc_err_p90=float(np.percentile(e, 90)), n=len(e),
            se={str(b): dict(kbits=float(np.mean([x['kbits_per_ru'] for x in se[k+(b,)]])),
                             sum_se=float(np.mean([x['sum_se'] for x in se[k+(b,)]])),
                             min_se=float(np.mean([x['min_se'] for x in se[k+(b,)]])),
                             noisy=float(np.mean([x['sum_se_noisy'] for x in se[k+(b,)]])))
                for b in sorted({b for (*kk, b) in se if tuple(kk) == k})})
    return out


def fig_tracking(runs, name, budget='32'):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    c = 0
    for run, real in runs:
        rep = tracking_report(run)
        T = sorted({int(k.split('t=')[1]) for k in rep if k.startswith('tracked')})
        oracle = rep['oracle s0=0 t=0']
        for s0 in sorted({float(k.split('s0=')[1].split()[0]) for k in rep if k.startswith('tracked')}):
            med = [rep[f'tracked s0={s0:g} t={t}']['loc_err_median'] for t in T]
            p90 = [rep[f'tracked s0={s0:g} t={t}']['loc_err_p90'] for t in T]
            lab = f'{real}, prior σ0 = {s0:g} m'
            axes[0].plot(T, med, color=SERIES[c], marker=MARK[c], markersize=6, linewidth=2, markeredgecolor=SURFACE,
                         markeredgewidth=1.2, label=lab)
            axes[0].plot(T, p90, color=SERIES[c], linewidth=1, linestyle=(0, (2, 2)))
            rel = [100*rep[f'tracked s0={s0:g} t={t}']['se'][budget]['sum_se']/oracle['se'][budget]['sum_se'] for t in T]
            axes[1].plot(T, rel, color=SERIES[c], marker=MARK[c], markersize=6, linewidth=2, markeredgecolor=SURFACE,
                         markeredgewidth=1.2, label=lab)
            c += 1
    axes[0].set_yscale('log')
    axes[0].set_xlabel('SRS occasion (O-DU posterior update after each)')
    axes[0].set_ylabel('Position error after the update, m')
    axes[0].set_title('Median (solid), 90th percentile (dashed); colours as on the right', loc='left', fontsize=8.5)
    axes[1].axhline(100, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    axes[1].set_xlabel('SRS occasion')
    axes[1].set_ylabel(f'Sum SE, % of known-position URTS-CODE')
    axes[1].set_title(f'URTS-CODE at {budget} coefficients per link', loc='left')
    for ax in axes:
        ax.set_xticks(T)
        ax.grid(color=GRID, linewidth=.6)
    axes[1].legend(frameon=False, fontsize=7.5, loc='lower right', labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(R/name, dpi=160)


PRECISION = {'nominal': [('cellfree_code_nominal_16', 'ad_dft/uniform', 'Angle-delay DFT'),
                         ('cellfree_code_nominal_16', 'urts_code/task', 'URTS-CODE, 18-bit coefficients'),
                         ('cellfree_vq_nominal_16', 'urts_vq/task', 'URTS-CODE, twin-set precision')],
             'diffuse': [('cellfree_code_diffuse_16', 'ad_dft/uniform', 'Angle-delay DFT'),
                         ('cellfree_code_diffuse_16', 'urts_code/task', 'URTS-CODE, 18-bit coefficients'),
                         ('cellfree_vq_diffuse_16', 'urts_vq/task', 'URTS-CODE, twin-set precision')]}


def fig_precision(name, sigmas=('0.0', '2.0')):
    panels = [(real, sg) for real in ['nominal', 'diffuse'] for sg in sigmas]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6*len(panels), 3.7), sharey=True)
    out = {}
    for ax, (real, sg) in zip(axes, panels):
        for c, (run, key, lab) in enumerate(PRECISION[real]):
            if not (R/run/'complete.json').exists():
                continue
            d = cellfree_table(run)[sg].get(key, {})
            b = sorted(d)
            x = [d[k]['kbits'] for k in b]
            y = [100*d[k]['sum_se']/d[k]['noisy'] for k in b]
            out.setdefault(f'{real} sigma={sg}', {})[key] = list(zip(x, y))
            ax.plot(x, y, color=SERIES[c], marker=MARK[c], markersize=6, linewidth=2, markeredgecolor=SURFACE,
                    markeredgewidth=1.2, label=lab)
        ax.axhline(100, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
        ax.set_xscale('log')
        ax.set_title({'nominal': 'Twin scene', 'diffuse': 'Reality + diffuse'}[real]+f', σ = {float(sg):g} m', loc='left')
        ax.set_xlabel('Fronthaul per O-RU per SRS occasion, kbit (log)')
        ax.grid(color=GRID, linewidth=.6)
    axes[0].set_ylabel('Sum SE, % of uncompressed CI')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc='lower center', ncol=3, labelcolor=INK2, bbox_to_anchor=(.5, 0))
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(R/name, dpi=160)
    return out


FAIR = {'nominal': [('bfp_k8_nominal_16', 'ad_dft/uniform', 'Angle-delay DFT, O-RAN BFP9'),
                     ('nm_k8_nominal_16', 'ad_dft/uniform', 'Angle-delay DFT, noise-matched BFP'),
                     ('nm_k8_nominal_16', 'urts_code/task', 'URTS-CODE, noise-matched BFP'),
                     ('final_k8_nominal_16', 'urts_vq/task', 'Final: twin-set widths, measured scale')],
        'diffuse': [('bfp_k8_diffuse_16', 'ad_dft/uniform', 'Angle-delay DFT, O-RAN BFP9'),
                    ('nm_k8_diffuse_16', 'ad_dft/uniform', 'Angle-delay DFT, noise-matched BFP'),
                    ('nm_k8_diffuse_16', 'urts_code/task', 'URTS-CODE, noise-matched BFP'),
                    ('final_k8_diffuse_16', 'urts_vq/task', 'Final: twin-set widths, measured scale')]}
KSCALE_FAIR = {'urts_code/task': {8: ('nm_k8_nominal_16', 'nm_k8_diffuse_16'), 16: ('nm_k16_nominal_8', 'nm_k16_diffuse_8'),
                                  32: ('nm_k32_nominal_6', 'nm_k32_diffuse_6')}}
KSCALE_FAIR['urts_vq/task'] = {8: ('final_k8_nominal_16', 'final_k8_diffuse_16'), 16: ('final_k16_nominal_8', 'final_k16_diffuse_8'),
                                32: ('final_k32_nominal_6', 'final_k32_diffuse_6')}
MISMATCH_FINAL = [('nominal', 'Twin = reality'), ('geom05', 'Buildings shifted, 0.5 m'), ('mat03', 'Materials ±30%'),
                  ('cars40', '40 cars added'), ('diffusetw0', 'Diffuse, not in twin'), ('all', 'All of the above')]


def fig_precision_fair(name, sigmas=('0.0', '2.0')):
    global PRECISION
    keep = PRECISION
    PRECISION = FAIR
    try:
        return fig_precision(name, sigmas)
    finally:
        PRECISION = keep


def kscale_fair_report(targets=(.7, .8, .9, .95)):
    out = {}
    for key, runs in KSCALE_FAIR.items():
        for K, pair in runs.items():
            for real, run in zip(['nominal', 'diffuse'], pair):
                if not (R/run/'complete.json').exists():
                    continue
                a_ = json.loads((R/run/'metadata.json').read_text())['args']
                M, F = 2*a_['rows']*a_['cols'], a_['groups']
                e = out.setdefault(real, {}).setdefault(str(K), dict(run=run, raw_kbits=K*M*F*18/1e3, gram_ul_kbits=K*K*9*F/1e3))
                for sigma in [0., 2.]:
                    e[f'{key} sigma={sigma:g}'] = {f'{t:g}': bits_to(run, sigma, key, t) for t in targets}
    return out


def mismatch_final(name):
    keep = None
    rep = {}
    for tag, _ in MISMATCH_FINAL:
        run = f'mmfinal_{tag}_3'
        if (R/run/'complete.json').exists():
            rep[tag] = cellfree_table(run)
    sigmas = ['0.0', '2.0']
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharey=True)
    for ax, sg in zip(axes, sigmas):
        c = 0
        for tag, lab in MISMATCH_FINAL:
            if tag not in rep:
                continue
            d = rep[tag][sg]['urts_vq/task']
            b = sorted(d)
            x, y = [d[k]['kbits'] for k in b], [100*d[k]['sum_se']/d[k]['noisy'] for k in b]
            if tag == 'nominal':
                ax.plot(x, y, color=INK2, linestyle=(0, (4, 2)), linewidth=1.5, marker='o', markersize=4, label=lab)
            else:
                ax.plot(x, y, color=SERIES[c], marker=MARK[c], markersize=6, linewidth=2, markeredgecolor=SURFACE,
                        markeredgewidth=1.2, label=lab)
                c += 1
        ax.axhline(100, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
        ax.set_xscale('log')
        ax.set_xticks([.5, 1, 2, 4, 8, 16])
        ax.set_xticklabels(['0.5', '1', '2', '4', '8', '16'])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(f'Final method, position error σ = {float(sg):g} m', loc='left')
        ax.set_xlabel('Fronthaul per O-RU per SRS occasion, kbit (log)')
        ax.grid(color=GRID, linewidth=.6)
    axes[0].set_ylabel('Sum SE, % of uncompressed CI in the same reality')
    axes[0].legend(frameon=False, fontsize=7.5, loc='lower right', labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(R/name, dpi=160)
    return rep


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('jobs', nargs='*', default=['single', 'cellfree'])
    a = p.parse_args()
    out = R/'stage5_summary.json'
    report = json.loads(out.read_text()) if out.exists() else {}
    if 'single' in a.jobs:
        runs = [(r, t) for r, t in [('feas_all_nominal_60', 'Reality = twin scene'), ('feas_all_diffuse_60', 'Reality + diffuse')]
                if (R/r/'complete.json').exists()]
        for extra in ['feas_nominal_120', 'feas_diffuse_120', 'feas_stat_search_60']:
            if (R/extra/'complete.json').exists():
                report[extra] = single_ru_table(extra)
        for r, _ in runs:
            report[r] = single_ru_table(r)
        fig_single(runs, 'fig_single_ru_nmse.png')
        if (R/'feas_stat_search_60'/'complete.json').exists():
            report['feas_stat_search_60'] = single_ru_table('feas_stat_search_60')
    if 'cellfree' in a.jobs:
        panels = []
        for extra in ['cellfree_nominal_10', 'cellfree_diffuse_10']:
            if (R/extra/'complete.json').exists():
                report[extra] = cellfree_table(extra)
        for run, s, t in [('cellfree_urts_nominal_8', 0., 'Twin scene, σ = 0 m'), ('cellfree_urts_nominal_8', 2., 'Twin scene, σ = 2 m'),
                          ('cellfree_urts_diffuse_8', 2., 'Reality + diffuse, σ = 2 m'), ('cellfree_urts_diffuse_8', 5., 'Reality + diffuse, σ = 5 m')]:
            if (R/run/'complete.json').exists():
                panels.append((run, s, t))
                report.setdefault(run, cellfree_table(run))
        if panels:
            fig_cellfree(panels, 'fig_cellfree_urts_klt_only.png')
        code = []
        for run, s, t in [('cellfree_code_nominal_16', 0., 'Twin scene, σ = 0 m'), ('cellfree_code_nominal_16', 2., 'Twin scene, σ = 2 m'),
                          ('cellfree_code_diffuse_16', 2., 'Reality + diffuse, σ = 2 m'), ('cellfree_code_diffuse_16', 5., 'Reality + diffuse, σ = 5 m')]:
            if (R/run/'complete.json').exists():
                code.append((run, s, t))
                report.setdefault(run, cellfree_table(run))
        if code:
            fig_cellfree(code, 'fig_cellfree_se_vs_fronthaul.png',
                         [('ad_dft/uniform', 'Angle-delay DFT'), ('twin_res/uniform', 'Point twin + residual'),
                          ('urts/uniform', 'Region twin subspace (plain KLT)'), ('urts_code/task', 'URTS-CODE (full method)')], gram=True)
            fig_proxy([(r, t) for r, t in [('cellfree_code_nominal_16', 'Twin scene'), ('cellfree_code_diffuse_16', 'Reality + diffuse')]
                       if (R/r/'complete.json').exists()], 'fig_cellfree_proxy.png')
    if 'mismatch' in a.jobs:
        report['mismatch'] = fig_mismatch(MISMATCH, 'fig_cellfree_mismatch.png')
        report['twin_scattering'] = fig_mismatch(TWINSCAT, 'fig_cellfree_twin_scattering.png')
    if 'tracking' in a.jobs:
        runs = [(r, t) for r, t in [('tracking_nominal_8', 'Twin scene'), ('tracking_diffuse_8', 'Diffuse')] if (R/r/'complete.json').exists()]
        if runs:
            report['tracking'] = {r: tracking_report(r) for r, _ in runs}
            fig_tracking(runs, 'fig_tracking.png')
    if 'precision' in a.jobs:
        report['precision'] = fig_precision('fig_cellfree_precision.png')
    if 'fair' in a.jobs:
        report['precision_fair'] = fig_precision_fair('fig_fair_precision.png')
        report['kscale_fair'] = kscale_fair_report()
        fig_kscale(report['kscale_fair'], 'fig_fair_kscale.png', '0.8',
                   (('urts_code/task', 0., 'URTS-CODE noise-matched, σ = 0'), ('urts_code/task', 2., 'URTS-CODE noise-matched, σ = 2 m'),
                    ('urts_vq/task', 0., 'Final method, σ = 0'), ('urts_vq/task', 2., 'Final method, σ = 2 m')))
        report['mismatch_final'] = mismatch_final('fig_fair_mismatch.png')
    if 'kscale' in a.jobs:
        report['kscale'] = kscale_report()
        fig_kscale(report['kscale'], 'fig_cellfree_kscale.png')
    out.write_text(json.dumps(report, indent=1))
    print('wrote', list(report))


if __name__ == '__main__':
    main()
