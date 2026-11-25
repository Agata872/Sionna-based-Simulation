"""E8 作图：python plot_same_total.py [结果目录]

fig_mhat_vs_K：行 = 功率规则（'ant' 逐天线 P_max；'ru' 论文式 (norm)），列 = 噪声。横轴每 RU 天线数 K（总天线数 16 不变，
  N = 16/K），纵轴平均 M̂（达到 0.98·SE_max 的最少激活天线数）。灰色细线为 K = 1（单天线分布式）的 M̂，浅色细线为全开 16。
  按撒点的 95% 区间半宽 ≤ 0.15 根，比标记小，不另画。
fig_se_vs_m：'ant' 规则；行 = 噪声，列 = K = 2 / 4 / 8。横轴激活天线数 m，纵轴各 m 下最大 sum SE 的平均；灰线为 K = 1。
颜色：参考调色板（dataviz palette.md）类别色 1（蓝，RU 级）与 2（橙，天线级），按原顺序、原色值使用（该文件记录前三个色
  两种模式下全部配对都通过校验；本机没有 node，校验脚本未运行）；两者另用不同标记形状，并有图例和直接标注。
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
SERIES = [('RU', 'RU level (whole RU on/off)', '#2a78d6', 'o'),
          ('ant', 'antenna level (any subset)', '#eb6834', 's')]
KS = (1, 2, 4, 8)
NOISE = (-90, -70, -50)
RULES = [('ant', "rule 'ant': per-antenna P_max"), ('ru', "rule 'ru': paper eq. (norm), K·P_max per RU")]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.facecolor': SURFACE, 'figure.facecolor': SURFACE,
                     'savefig.facecolor': SURFACE, 'axes.titlecolor': INK, 'axes.titlesize': 9})


def load(d):
    return [json.loads(line) for line in open(os.path.join(d, 'rows.jsonl'))]


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def fig_mhat(rows, out):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.4), sharex=True, sharey=True)
    x = list(range(len(KS)))
    for i, (rule, rule_name) in enumerate(RULES):
        for j, nd in enumerate(NOISE):
            ax = axes[i, j]
            style(ax)
            sub = {r['K']: r for r in rows if r['rule'] == rule and r['noise_dbm'] == nd}
            d1 = sub[1]['mhat']['ant']['mean']
            ax.axhline(16, color=AXIS, linewidth=1.0, zorder=1)
            ax.axhline(d1, color=MUTED, linewidth=1.2, zorder=1)
            for g, label, color, marker in SERIES:
                y = [sub[K]['mhat'][g]['mean'] for K in KS]
                ax.plot(x, y, color=color, linewidth=2, marker=marker, markersize=7, markeredgecolor=SURFACE,
                        markeredgewidth=1.5, label=label, zorder=3, solid_capstyle='round', solid_joinstyle='round')
            k4 = KS.index(4)
            lo, hi = sub[4]['mhat']['ant']['mean'], sub[4]['mhat']['RU']['mean']
            # 括号与标注放在 K = 4 左侧或右侧中两线空隙较大的一侧（标注约占 0.65 个横轴单位），取空隙中点，避免压线
            gaps = []
            for side, kn in ((1, 8), (-1, 2)):
                an, rn = sub[kn]['mhat']['ant']['mean'], sub[kn]['mhat']['RU']['mean']
                top_orange = max(lo + 0.85 * (an - lo), lo + 0.2 * (an - lo))
                bottom_blue = min(hi + 0.2 * (rn - hi), hi + 0.85 * (rn - hi))
                gaps.append((bottom_blue - top_orange, side, (top_orange + bottom_blue) / 2))
            _, side, y_mid = max(gaps)
            y_lab = min(max(y_mid, lo + 0.3), hi - 0.3)
            ax.annotate('', xy=(k4 + 0.13 * side, lo), xytext=(k4 + 0.13 * side, hi),
                        arrowprops={'arrowstyle': '<->', 'color': INK2, 'linewidth': 0.9, 'shrinkA': 0, 'shrinkB': 0})
            ax.text(k4 + 0.2 * side, y_lab, f"ΔM̂ = {sub[4]['delta_mhat']['mean']:.1f}", color=INK2, fontsize=8,
                    va='center', ha='left' if side > 0 else 'right')
            if i == 0 and j == 0:
                ax.text(x[-1] + 0.05, d1 - 0.35, 'K = 1 (single-antenna RUs)', color=MUTED, fontsize=7.5, ha='right', va='top')
                ax.text(x[0], 16.25, 'all 16 antennas on', color=MUTED, fontsize=7.5, ha='left', va='bottom')
            if i == 0:
                ax.set_title(f'noise {nd} dBm/PRB')
            if j == 0:
                ax.set_ylabel(f'{rule_name}\n\nM̂ (active antennas)')
            if i == 1:
                ax.set_xticks(x, [f'{K}\n{16 // K} RUs' for K in KS])
                ax.set_xlabel('antennas per RU K (16 antennas in total)')
    axes[0, 0].set_ylim(4, 17.2)
    axes[0, 0].set_xlim(-0.25, len(KS) - 0.45)
    h, l = axes[0, 0].get_legend_handles_labels()
    ref = plt.Line2D([], [], color=MUTED, linewidth=1.2)
    fig.legend(h + [ref], l + ['K = 1 reference (single-antenna distributed RUs)'], loc='upper center', ncol=3,
               frameon=False, fontsize=8.5, labelcolor=INK2, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle('Active antennas needed for 98% of SE_max, same 16 antennas, ZF, U = 4 (500 drops x 4 PRBs)',
                 y=1.04, color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out, f'fig_mhat_vs_K.{ext}'), dpi=160, bbox_inches='tight')
    plt.close(fig)


def fig_se(rows, out, rule='ant'):
    kk = (2, 4, 8)
    fig, axes = plt.subplots(3, 3, figsize=(10, 8), sharex=True, sharey='row')
    for i, nd in enumerate(NOISE):
        ref = next(r for r in rows if r['rule'] == rule and r['noise_dbm'] == nd and r['K'] == 1)
        for j, K in enumerate(kk):
            ax = axes[i, j]
            style(ax)
            r = next(r for r in rows if r['rule'] == rule and r['noise_dbm'] == nd and r['K'] == K)
            m_all = [m for m, v in enumerate(ref['curve_mean_se']['ant']) if v is not None]
            ax.plot(m_all, [ref['curve_mean_se']['ant'][m] for m in m_all], color=MUTED, linewidth=1.2, zorder=2,
                    label='K = 1 (single-antenna RUs)')
            for g, label, color, marker in SERIES:
                cv = r['curve_mean_se'][g]
                ms = [m for m, v in enumerate(cv) if v is not None]
                kw = {'marker': marker, 'markersize': 7, 'markeredgecolor': SURFACE, 'markeredgewidth': 1.5} if g == 'RU' else {}
                ax.plot(ms, [cv[m] for m in ms], color=color, linewidth=2, zorder=3, label=label,
                        solid_capstyle='round', solid_joinstyle='round', **kw)
            if i == 0:
                ax.set_title(f'K = {K} ({16 // K} RUs)')
            if j == 0:
                ax.set_ylabel(f'noise {nd} dBm/PRB\n\nmean max sum SE (bit/s/Hz)')
            if i == 2:
                ax.set_xlabel('active antennas m')
                ax.set_xticks([4, 8, 12, 16])
            ax.set_ylim(bottom=0)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, frameon=False, fontsize=8.5, labelcolor=INK2, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Best sum SE at a given number of active antennas (rule 'ant', ZF, U = 4)", y=1.035, color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out, f'fig_se_vs_m.{ext}'), dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results', 'same_total_20261124')
    rows = load(d)
    fig_mhat(rows, d)
    fig_se(rows, d)
    print(f'figures written to {d}')


if __name__ == '__main__':
    main()
