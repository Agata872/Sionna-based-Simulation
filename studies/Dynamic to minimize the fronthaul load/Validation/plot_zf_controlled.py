"""受控 ZF 实验作图：python plot_zf_controlled.py [结果目录 ...]

fig_gain_M{4,6,8}：行 = beta_cross，列 = CSI 质量，横轴参考 SNR，纵轴真实信道上的逐样本配对平均增益，阴影为 95% t 区间。
fig_power：完美 CSI 下各方案的平均实际发射功率 P_used 与单 AP 上限决定缩放的样本比例。
颜色取参考调色板的前四个类别色（按固定顺序），每条曲线另有不同标记形状，身份不只靠颜色。
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
SERIES = [('C-A', 'C − A  (antenna vs AP level)', '#2a78d6', 'o'),
          ('Ceq-A', 'C_eq − A  (same transmit power)', '#eb6834', 's'),
          ('B-A', 'B − A  (strength sorting)', '#1baf7a', '^'),
          ('C-B', 'C − B  (beyond sorting)', '#eda100', 'D')]
CSI = [('perfect', 'perfect CSI'), ('nmse_-20dB', 'NMSE −20 dB'), ('nmse_-10dB', 'NMSE −10 dB')]
BETAS = [0.05, 0.25, 0.8]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.facecolor': SURFACE, 'figure.facecolor': SURFACE,
                     'savefig.facecolor': SURFACE, 'axes.titlecolor': INK, 'axes.titlesize': 9})


def load(dirs):
    rows = []
    for d in dirs:
        p = os.path.join(d, 'rows.jsonl')
        if os.path.exists(p):
            rows += [json.loads(l) for l in open(p)]
    return rows


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.axhline(0, color=AXIS, linewidth=0.8)


def fig_gain(rows, M, out):
    sub = [r for r in rows if r['M'] == M]
    if not sub:
        return
    fig, axes = plt.subplots(3, 3, figsize=(10, 8.2), sharex=True, sharey=True)
    for i, beta in enumerate(BETAS):
        for j, (csi, csi_name) in enumerate(CSI):
            ax = axes[i, j]
            style(ax)
            pts = sorted((r for r in sub if r['beta_cross'] == beta and r['csi'] == csi), key=lambda r: r['snr_db'])
            x = [r['snr_db'] for r in pts]
            for key, label, color, marker in SERIES:
                m = [r['diff_true'][key]['mean'] for r in pts]
                lo = [r['diff_true'][key]['ci95'][0] for r in pts]
                hi = [r['diff_true'][key]['ci95'][1] for r in pts]
                ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
                ax.plot(x, m, color=color, linewidth=1.5, marker=marker, markersize=5.5,
                        markeredgecolor=SURFACE, markeredgewidth=1.0, label=label)
            ax.set_title(f'β_cross = {beta} · {csi_name}')
            ax.set_xticks([0, 10, 20, 30])
            if i == 2:
                ax.set_xlabel('reference SNR (dB)')
            if j == 0:
                ax.set_ylabel('sum-SE gain on true channel\n(bit/s/Hz)')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, frameon=False, fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f'Controlled ZF, M = {M} antennas per AP, K = {2 * M} active: paired mean gain with 95% CI '
                 f'(1000 channel samples per point)', y=1.035, color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'fig_gain_M{M}.{ext}'), dpi=160, bbox_inches='tight')
    plt.close(fig)


def fig_power(rows, out):
    sub = [r for r in rows if r['csi'] == 'perfect' and r['snr_db'] == 0]
    Ms = sorted({r['M'] for r in sub})
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for i, beta in enumerate(BETAS):
        ax = axes[i]
        style(ax)
        pts = sorted((r for r in sub if r['beta_cross'] == beta), key=lambda r: r['M'])
        ax.axhline(2.0, color=MUTED, linewidth=0.8, linestyle='--', label='P_total = 2 (MRT always uses it)')
        for key, label, color, marker in (('A', 'A and C_eq (C_eq matches A by design)', '#2a78d6', 'o'),
                                          ('C', 'C', '#eb6834', 's')):
            ax.plot([r['M'] for r in pts], [r['power'][key]['P_used_mean'] for r in pts], color=color, linewidth=1.5,
                    marker=marker, markersize=5.5, markeredgecolor=SURFACE, label=label)
        for r in pts:
            ax.annotate(f"{100 * r['power']['C']['frac_ap_limited']:.0f}%", (r['M'], r['power']['C']['P_used_mean']),
                        textcoords='offset points', xytext=(0, 7), ha='center', fontsize=7, color=INK2)
        ax.set_title(f'β_cross = {beta}')
        ax.set_xticks(Ms)
        ax.set_xlabel('antennas per AP, M')
        if i == 0:
            ax.set_ylabel('mean transmit power P_used')
    axes[0].set_ylim(1.4, 2.12)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, frameon=False, fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle('Perfect CSI: A is per-AP-limited in 100% of samples; labels on C = share of samples where '
                 'the per-AP cap binds for C', y=1.12, color=INK, fontsize=10)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'fig_power.{ext}'), dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    dirs = sys.argv[1:] or [os.path.join(HERE, 'results', 'round2_zf_controlled_20261120'),
                            os.path.join(HERE, 'results', 'round2_zf_controlled_M8_20261120')]
    out = dirs[0]
    rows = load(dirs)
    for M in (4, 6, 8):
        fig_gain(rows, M, out)
    fig_power(rows, out)
    print('figures written to', out)


if __name__ == '__main__':
    main()
