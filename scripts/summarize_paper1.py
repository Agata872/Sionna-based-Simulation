#!/usr/bin/env python3
"""Paper-1 summaries and figures from results/paper1_20261008 (see paper1/experiments.md).

Writes results/paper1_20261008/paper1_summary.json and fig_*.png. Only complete runs are used unless
--partial is given (then in-progress runs are read too, for a preview; the JSON records which).

Metric: sum SE with compressed CI as a fraction of sum SE with the uncompressed noisy CI, averaged over
drops as a ratio of means (as in the stage-5..7 summaries); 95 % intervals by a paired drop bootstrap;
the 5 %-user ratio pools per-UE SE over drops. Bits to a target fraction: log-bit interpolation.

Colour: the documented, validated default categorical palette (dataviz reference instance), one fixed
slot per scheme across all figures; variants of a scheme share its slot and differ by dash. The palette
validator needs node, which is unavailable here, so every figure also uses marker shapes and a legend.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

R = Path('results/paper1_20261008')
SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
SLOT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
# scheme key -> (label, colour slot, marker, dash)
STYLE = {
    'urts_vq/task':     ('Proposed (region twin, final)', 0, 'o', '-'),
    'urts/uniform':     ('Region twin KLT, truncation only', 0, 'o', (0, (4, 2))),
    'ad_dft/uniform':   ('Angle-delay DFT (O-RAN beamspace type)', 1, 's', '-'),
    'grid_vq/task':     ('Grid-cell twin + proposed coding', 2, '^', '-'),
    'site_klt/uniform': ('Site-wide twin KLT (no position)', 3, 'D', '-'),
    'twin_res/uniform': ('Point twin + residual', 4, 'v', '-'),
    'urts_code/task':   ('Proposed, fixed precision', 5, 'P', '-'),
    'twin_aep/uniform': ('Twin-trained AE, position-conditioned', 6, 'X', '-'),
    'twin_ae/uniform':  ('Twin-trained AE (no position)', 6, 'X', (0, (4, 2))),
    'grid_ecb/uniform': ('Grid-cell basis, 64 points (Cai et al. style)', 7, '<', '-'),
    'cai4:grid_ecb/uniform': ('Grid-cell basis, 4 vertices (Cai et al.)', 7, '<', (0, (4, 2))),
    'cai4:grid_vq/task': ('Grid 4 vertices + proposed coding', 2, '^', (0, (4, 2))),
}
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.facecolor': SURFACE,
                     'figure.facecolor': SURFACE, 'savefig.facecolor': SURFACE, 'axes.titlesize': 9.5,
                     'axes.titlecolor': INK, 'axes.spines.top': False, 'axes.spines.right': False})
PARTIAL = False
USED = {}


def ok(run):
    p = R/run
    if (p/'complete.json').exists():
        USED[run] = 'complete'
        return True
    if PARTIAL and (p/'rows.jsonl').exists() and (p/'rows.jsonl').stat().st_size > 0:
        USED[run] = 'partial'
        return True
    return False


def rows(run):
    return [json.loads(x) for x in (R/run/'rows.jsonl').open() if x.strip()]


def meta(run):
    return json.loads((R/run/'metadata.json').read_text())['args']


def cf_table(run, prefix=''):
    """{sigma: {scheme/alloc: {budget: stats}}}; only drops for which every scheme row exists are used."""
    g = defaultdict(list)
    rr = rows(run)
    for r in rr:
        g[(r['sigma'], f"{prefix}{r['scheme']}/{r['alloc']}", r['budget'])].append(r)
    out = {}
    rng = np.random.default_rng(0)
    for (s, key, b), v in g.items():
        se = np.array([x['sum_se'] for x in v])
        nz = np.array([x['sum_se_noisy'] for x in v])
        boot = []
        for _ in range(400):
            i = rng.integers(len(v), size=len(v))
            boot.append(se[i].sum()/nz[i].sum())
        e = dict(kbits=float(np.mean([x['kbits_per_ru'] for x in v])), frac=float(se.mean()/nz.mean()),
                 frac_lo=float(np.percentile(boot, 2.5)), frac_hi=float(np.percentile(boot, 97.5)),
                 sum_se=float(se.mean()), noisy=float(nz.mean()), nmse=float(np.mean([x['nmse_db'] for x in v])),
                 drops=len(v), raw_kbits=float(v[0]['raw_kbits_per_ru']))
        if 'se_ue' in v[0]:
            a = np.concatenate([x['se_ue'] for x in v])
            n = np.concatenate([x['se_ue_noisy'] for x in v])
            e['p5_frac'] = float(np.percentile(a, 5)/max(np.percentile(n, 5), 1e-12))
        out.setdefault(f'{s:g}', {}).setdefault(key, {})[b] = e
    return out


def bits_to(curve, frac):
    """kbit at which the curve reaches frac (log-bit interpolation); None if never. Also the best fraction."""
    b = sorted(curve, key=lambda x: curve[x]['kbits'])
    kb = np.array([curve[x]['kbits'] for x in b])
    fr = np.array([curve[x]['frac'] for x in b])
    i = np.flatnonzero(fr >= frac)
    if not len(i):
        return None, float(fr.max())
    j = i[0]
    if j == 0:
        return float(kb[0]), float(fr.max())
    return float(np.exp(np.interp(frac, [fr[j-1], fr[j]], [np.log(kb[j-1]), np.log(kb[j])]))), float(fr.max())


def gram_kbits(run):
    a = meta(run)
    return a['num_ue']**2*9*a['groups']/1e3                  # Hermitian KxK per PRB group, 9-bit reals (UL only)


MAIN = {8: ('main_k8_nominal_16', 'main_k8_diffuse_16'), 16: ('main_k16_nominal_8', 'main_k16_diffuse_8'),
        32: ('main_k32_nominal_6', 'main_k32_diffuse_6')}
TARGETS = (.7, .8, .9, .95)


def main_report():
    out = {}
    for K, pair in MAIN.items():
        for real, run in zip(('nominal', 'diffuse'), pair):
            if not ok(run):
                continue
            tab = cf_table(run)
            for extra, pre in ((f'cai4_k8_{real}_16', 'cai4:'), (f'ae_k8_{real}_16', '')):
                if K == 8 and ok(extra):
                    for s, d in cf_table(extra, pre).items():
                        for key, curve in d.items():
                            if key.startswith(('twin_ae', 'twin_aep', 'cai4:')):
                                tab.setdefault(s, {})[key] = curve
            e = out.setdefault(real, {}).setdefault(str(K), dict(run=run, gram_ul_kbits=gram_kbits(run), curves=tab, bits_to={}))
            for s, d in tab.items():
                for key, curve in d.items():
                    e['bits_to'][f'{key} sigma={s}'] = {f'{t:g}': bits_to(curve, t) for t in TARGETS}
    return out


def _style(ax, key):
    lab, c, m, ls = STYLE[key]
    return dict(color=SLOT[c], marker=m, linestyle=ls, label=lab, markersize=6, linewidth=2,
                markeredgecolor=SURFACE, markeredgewidth=1.1)


def fig_curves(rep, name, keys, sigmas=('0', '2'), K='8', title='', ci=True):
    reals = [r for r in ('nominal', 'diffuse') if K in rep.get(r, {})]
    if not reals:
        return
    fig, axes = plt.subplots(len(reals), len(sigmas), figsize=(4.3*len(sigmas), 3.4*len(reals)), sharey=True, squeeze=False)
    for i, real in enumerate(reals):
        e = rep[real][K]
        for j, s in enumerate(sigmas):
            ax = axes[i, j]
            d = e['curves'].get(s, {})
            for z, key in enumerate(keys):
                if key not in d:
                    continue
                cv = d[key]
                b = sorted(cv, key=lambda x: cv[x]['kbits'])
                x = [cv[k]['kbits'] for k in b]
                ax.plot(x, [100*cv[k]['frac'] for k in b], zorder=20-z, **_style(ax, key))   # first key on top
                if ci:
                    ax.fill_between(x, [100*cv[k]['frac_lo'] for k in b], [100*cv[k]['frac_hi'] for k in b],
                                    color=SLOT[STYLE[key][1]], alpha=.10, linewidth=0)
            ax.axvline(e['gram_ul_kbits'], color=INK2, linewidth=1, linestyle=(0, (1, 2)))
            ax.text(e['gram_ul_kbits'], 3, ' Gram UL\n (exact RZF)', color=INK2, fontsize=7, va='bottom')
            ax.axhline(100, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
            ax.set_xscale('log')
            ax.set_ylim(0, 108)
            real_lab = 'reality = twin' if real == 'nominal' else 'reality + diffuse scattering'
            ax.set_title(f'K = {K}, {real_lab}, σ = {s} m', loc='left')
            ax.set_xlabel('Fronthaul per O-RU per SRS occasion, kbit (log)')
            ax.grid(color=GRID, linewidth=.6)
        axes[i, 0].set_ylabel('Sum SE, % of uncompressed CI')
    h, l = [], []
    for ax in axes.ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    fig.legend(h, l, frameon=False, fontsize=7.8, loc='lower center', ncol=min(3, len(l)), labelcolor=INK2)
    rows_leg = (len(l)+2)//3
    fig.tight_layout(rect=(0, .045*rows_leg+.01, 1, 1))
    if title:
        fig.suptitle(title, x=.01, ha='left', fontsize=10, color=INK)
    fig.savefig(R/name, dpi=170)
    plt.close(fig)


def fig_kscale(rep, name):
    keys = ['urts_vq/task', 'urts_code/task', 'grid_vq/task']
    panels = [('nominal', '0', .95), ('nominal', '2', .95), ('diffuse', '0', .8)]
    panels = [p for p in panels if p[0] in rep]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.7*len(panels), 3.4), sharey=True, squeeze=False)
    for ax, (real, s, t) in zip(axes[0], panels):
        Ks = sorted(int(k) for k in rep[real])
        for key in keys:
            xs, ys = [], []
            for K in Ks:
                v = rep[real][str(K)]['bits_to'].get(f'{key} sigma={s}', {}).get(f'{t:g}', (None, 0))[0]
                if v is not None:
                    xs.append(K)
                    ys.append(v)
            if xs:
                ax.plot(xs, ys, **_style(ax, key))
        g = [rep[real][str(K)]['gram_ul_kbits'] for K in Ks]
        ax.plot(Ks, g, color=INK2, linestyle=(0, (1, 2)), marker='x', linewidth=1.5, label='Local Gram, UL only (exact RZF)')
        ax.set_yscale('log')
        ax.set_xscale('log', base=2)
        ax.set_xticks(Ks)
        ax.set_xticklabels([str(k) for k in Ks])
        ax.set_title(f'{"reality = twin" if real == "nominal" else "+ diffuse"}, σ = {s} m, {int(100*t)} % target', loc='left')
        ax.set_xlabel('UEs K')
        ax.grid(color=GRID, linewidth=.6)
    axes[0, 0].set_ylabel('kbit per O-RU per SRS occasion (log)')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7.8, loc='lower center', ncol=4, labelcolor=INK2)
    fig.tight_layout(rect=(0, .1, 1, 1))
    fig.savefig(R/name, dpi=170)
    plt.close(fig)


MM = [('nominal', 'Twin = reality'), ('geom05', 'Buildings shifted 0.5 m'), ('mat03', 'Materials ±30 %'),
      ('cars40', '40 cars added'), ('diffusetw0', 'Diffuse, not in twin'), ('all', 'All combined')]


def mismatch_report():
    out = {}
    for tag, _ in MM:
        run = f'mm_{tag}_6'
        if ok(run):
            out[tag] = cf_table(run)
    return out


def fig_mismatch(rep, name, budgets=(32, 128), sigma='0'):
    if not rep:
        return
    keys = ['urts_vq/task', 'grid_vq/task', 'urts_code/task', 'ad_dft/uniform']
    tags = [t for t, _ in MM if t in rep]
    labels = {'nominal': 'Twin =\nreality', 'geom05': 'Buildings\nshifted 0.5 m', 'mat03': 'Materials\n±30 %',
              'cars40': '40 cars\nadded', 'diffusetw0': 'Diffuse,\nnot in twin', 'all': 'All\ncombined'}
    fig, axes = plt.subplots(1, len(budgets), figsize=(5.2*len(budgets), 3.9), sharey=True, squeeze=False)
    x = np.arange(len(tags))
    for ax, budget in zip(axes[0], budgets):
        for i, key in enumerate(keys):
            y, lo, hi = [], [], []
            for t in tags:
                c = rep[t].get(sigma, {}).get(key, {}).get(budget)
                y.append(100*c['frac'] if c else np.nan)
                lo.append(100*c['frac_lo'] if c else np.nan)
                hi.append(100*c['frac_hi'] if c else np.nan)
            st = _style(ax, key)
            st['linestyle'] = 'none'
            ax.errorbar(x+(i-1.5)*.16, y, yerr=[np.array(y)-np.array(lo), np.array(hi)-np.array(y)], ecolor=st['color'],
                        elinewidth=1.2, capsize=0, **st)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[t] for t in tags], fontsize=7.8)
        ax.axhline(100, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
        ax.set_title(f'{budget} coefficients per link, σ = {sigma} m', loc='left')
        ax.grid(axis='y', color=GRID, linewidth=.6)
    axes[0, 0].set_ylabel('Sum SE, % of uncompressed CI')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7.8, loc='lower center', ncol=4, labelcolor=INK2)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(R/name, dpi=170)
    plt.close(fig)


def tracking_report(run):
    se, loc = defaultdict(list), defaultdict(list)
    for r in rows(run):
        k = (r['setting'], r['sigma0'], r['occasion'])
        (loc if r['budget'] < 0 else se)[k if r['budget'] < 0 else k+(r['budget'],)].append(r)
    out = {}
    for k in sorted({kk[:3] for kk in se}):
        bud = sorted({kk[3] for kk in se if kk[:3] == k})
        e = dict(se={str(b): dict(kbits=float(np.mean([x['kbits_per_ru'] for x in se[k+(b,)]])),
                                  frac=float(np.mean([x['sum_se'] for x in se[k+(b,)]])/np.mean([x['sum_se_noisy'] for x in se[k+(b,)]])))
                     for b in bud})
        if k in loc:
            err = np.concatenate([x['loc_err'] for x in loc[k]])
            e.update(loc_err_median=float(np.median(err)), loc_err_p90=float(np.percentile(err, 90)), n=int(len(err)))
        out[f'{k[0]} s0={k[1]:g} t={k[2]}'] = e
    return out


def dimension_report(run='dim_region_16'):
    if not ok(run):
        return None
    g = defaultdict(list)
    for r in rows(run):
        if r['energy'] > 10*r['n_elements']*r['n_freq']:   # links clearly above the CI noise
            g[(r['array'], r['prb'], r['std'])].append(r)
    out = {}
    for (a, p, s), v in g.items():
        dn = np.array([x['d_noise'] for x in v])
        out.setdefault(f'{a}_{p}', {})[f'{s:g}'] = dict(d_noise_median=float(np.median(dn)), d_noise_p75=float(np.percentile(dn, 75)),
                                                        d99_median=float(np.median([x['d99'] for x in v])), links=len(v),
                                                        saturated=int((dn >= .9*meta(run)['region_samples']).sum()))
    return out


def fig_dimension(rep, name):
    if not rep:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    order = [('4x4_52', 0, '-', 'o'), ('4x4_104', 0, (0, (4, 2)), 'o'), ('8x8_52', 1, '-', 's'), ('8x8_104', 1, (0, (4, 2)), 's')]
    for key, c, ls, m in order:
        if key not in rep:
            continue
        d = rep[key]
        s = sorted(d, key=float)
        arr, prb = key.split('_')
        ax.plot([float(x) for x in s], [d[x]['d_noise_median'] for x in s], color=SLOT[c], linestyle=ls, marker=m, linewidth=2,
                markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.1,
                label=f'{arr} cross-pol, {int(prb)//2} PRB groups ({float(prb)*0.36:.0f} MHz)')
    ax.set_xscale('log')
    ax.set_xticks([.25, .5, 1, 2, 3, 5, 8])
    ax.set_xticklabels(['0.25', '0.5', '1', '2', '3', '5', '8'])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('Position uncertainty s, m (log)')
    ax.set_ylabel('KLT dimensions above CI noise (median link)')
    ax.set_title('Effective dimension of the region twin subspace', loc='left')
    ax.grid(color=GRID, linewidth=.6)
    ax.legend(frameon=False, fontsize=7.8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(R/name, dpi=170)
    plt.close(fig)


def main():
    global PARTIAL
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--partial', action='store_true', help='also read runs that are still in progress (preview)')
    PARTIAL = p.parse_args().partial
    rep = dict(main=main_report())
    fig_curves(rep['main'], 'fig_main_k8.png',
               ['urts_vq/task', 'urts_code/task', 'twin_res/uniform', 'site_klt/uniform', 'ad_dft/uniform'])
    fig_curves(rep['main'], 'fig_ablation_location.png',
               ['urts_vq/task', 'grid_vq/task', 'twin_aep/uniform', 'twin_ae/uniform', 'site_klt/uniform'], sigmas=('0', '2', '5'))
    fig_curves(rep['main'], 'fig_ablation_grid.png',
               ['urts_vq/task', 'urts/uniform', 'grid_vq/task', 'cai4:grid_vq/task', 'grid_ecb/uniform', 'cai4:grid_ecb/uniform'],
               sigmas=('0', '2', '5'))
    fig_kscale(rep['main'], 'fig_kscale.png')
    perru = {}
    for real in ('nominal', 'diffuse'):
        run = f'perru_k8_{real}_16'
        if ok(run):
            t = cf_table(run)
            perru[real] = {f'{key} sigma={s}': {f'{x:g}': bits_to(c, x) for x in TARGETS} for s, d in t.items() for key, c in d.items()}
    rep['perru'] = perru
    rep['mismatch'] = mismatch_report()
    fig_mismatch(rep['mismatch'], 'fig_mismatch.png')
    rep['tracking'] = {run: tracking_report(run) for run in ('track_final_nominal_8', 'track_final_diffuse_8') if ok(run)}
    rep['dimension'] = dimension_report()
    fig_dimension(rep['dimension'], 'fig_dimension.png')
    rep['runs_used'] = USED
    for real in rep['main'].values():                          # curves are bulky; keep them, but round
        for e in real.values():
            for s in e['curves'].values():
                for c in s.values():
                    for b in c.values():
                        for k, v in b.items():
                            if isinstance(v, float):
                                b[k] = round(v, 5)
    (R/'paper1_summary.json').write_text(json.dumps(rep, indent=1))
    print('runs used:', USED)


if __name__ == '__main__':
    main()
