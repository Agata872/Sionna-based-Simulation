#!/usr/bin/env python3
"""汇总 run_cqi_bler.py 的结果：各 Cat × 位宽 × CQI 来源的首传 BLER、吞吐、MCS、CQI。

用法：
    python cqi_compression/analyse_cqi_bler.py results/cqi_compression_20261029/main

输出（写入运行目录）：report.txt（表格）、fig_bler_tput_vs_width.png。
BLER 按 TB 汇总（所有撒点、UE、周期的出错 TB 数 / TB 总数）；"失败 UE" 为 BLER ≥ 0.5 的 UE-撒点比例。
UE 静止时同一撒点的 BLER 近似阶跃（MCS 低于实际 SINR 的门限时几乎不出错，高于时几乎全错），
所以汇总 BLER 主要反映 MCS 选高了的 UE 比例。
理想前传下用精灵 CQI 也失败（BLER ≥ 0.5）的 UE-撒点不参与汇总：它们的 SINR 低于最低可用 MCS
（oran/README 设计第 13 条），在任何位宽、任何 CQI 下都全错，与压缩无关。
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dynamic_split"))

from runio import read_rows  # noqa: E402

SOURCES = ("genie", "csi-rs", "aware", "oracle")
SOURCE_LABELS = {"genie": "不考虑压缩（精灵）", "csi-rs": "UE 实测（CSI-RS）",
                 "aware": "计入数据失真（解析）", "oracle": "计入全部失真（PDSCH 实测）"}
# 参考调色板的分类色前 4 个槽位（浅色），按固定顺序分配；标记形状作第二重编码
COLORS = dict(zip(SOURCES, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")))
MARKERS = dict(zip(SOURCES, ("o", "s", "^", "D")))
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def summarize(rows):
    """(Cat, 位宽, 来源) -> 汇总量。"""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["category"], r["width"], r["source"])].append(r)
    out = {}
    for key, rs in groups.items():
        errors = sum(sum(r["errors"]) for r in rs)
        blocks = sum(len(r["errors"]) * r["blocks_per_period"] for r in rs)
        tput = np.array([r["throughput_mbps"] for r in rs])
        sqnr = [r["uplane_sqnr_db"] for r in rs if r["uplane_sqnr_db"] is not None]
        out[key] = dict(
            bler=errors / blocks, n_ue=len(rs),
            fail=np.mean([r["bler"] >= 0.5 for r in rs]),
            tput=tput.mean(), tput_p5=np.percentile(tput, 5),
            mcs=np.mean([np.mean(r["mcs"]) for r in rs]),
            cqi=np.mean([np.mean(r["cqi"]) for r in rs]),
            sqnr=np.mean(sqnr) if sqnr else None,
            fronthaul=rs[0]["fronthaul_gbps"])
    return out


def width_order(widths):
    return sorted(widths, key=lambda w: (0, 0) if w == "ideal" else (1, -int(w)))


def label_of(w):
    return "理想" if w == "ideal" else f"{w} bit"


def report(stats, sqnr_table, meta):
    a = meta["args"]
    lines = [f"块缩放 U-plane（Cat B 权值 {a['bfw']}），{a['num_ue']} UE × {a['layers_per_ue']} 层，"
             f"MCS / CQI 表 {a['mcs_table']}，{a['num_prb']} PRB，OLLA 关闭（CQI 查表），"
             f"{a['drops']} 个撒点 × {a['periods']} 个周期 × {a['batch_size']} 个数据时隙", ""]
    if sqnr_table:
        lines.append("实际 U-plane 信号上的 SQNR [dB]（第一个撒点）：")
        for cat, tab in sqnr_table.items():
            widths = width_order({k.split(":")[1] for k in tab})
            lines.append(f"  Cat {cat}  " + "  ".join(f"{label_of(w):>7}" for w in widths))
            for method in ("bfp", "block_scaling", "mu_law"):
                vals = [tab.get(f"{method}:{w}") for w in widths]
                lines.append(f"  {method:<14}" + "  ".join(f"{v:7.1f}" if v is not None else f"{'-':>7}"
                                                          for v in vals))
        lines.append("")
    cats = sorted({k[0] for k in stats})
    for cat in cats:
        widths = width_order({k[1] for k in stats if k[0] == cat})
        lines.append(f"Cat {cat}：BLER / 失败 UE 比例 / 平均吞吐 Mbps / 平均 MCS")
        lines.append(f"{'位宽':<8}{'前传Gbps':>9}{'SQNR':>7}" + "".join(f"{s:>28}" for s in SOURCES))
        for w in widths:
            s0 = stats[(cat, w, SOURCES[0])]
            sq = f"{s0['sqnr']:7.1f}" if s0["sqnr"] is not None else f"{'inf':>7}"
            cells = "".join(
                f"{stats[(cat, w, s)]['bler']:9.3f}{stats[(cat, w, s)]['fail']:6.2f}"
                f"{stats[(cat, w, s)]['tput']:7.1f}{stats[(cat, w, s)]['mcs']:6.1f}" for s in SOURCES)
            lines.append(f"{label_of(w):<8}{s0['fronthaul']:9.2f}{sq}{cells}")
        lines.append("")
    lines.append("（前传 Gbps 为本配置 51 PRB 下单个 O-RU 的 U-plane + C-plane 比特率；SQNR 为该位宽全部时隙的 "
                 "U-plane SQNR 平均；失败 UE 为 BLER ≥ 0.5 的 UE-撒点比例）")
    return "\n".join(lines)


def plot(stats, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback", "Noto Sans CJK SC"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    cats = sorted({k[0] for k in stats})
    panels = [("bler", "首传 BLER"), ("tput", "平均 UE 吞吐 [Mbps]")]
    fig, axes = plt.subplots(len(panels), len(cats), figsize=(5.2 * len(cats), 7.4), squeeze=False,
                             sharex=True, sharey="row", facecolor=SURFACE)
    for col, cat in enumerate(cats):
        widths = width_order({k[1] for k in stats if k[0] == cat})
        xs = np.arange(len(widths))
        for row, (key, ylabel) in enumerate(panels):
            ax = axes[row, col]
            ax.set_facecolor(SURFACE)
            if key == "bler":
                ax.axhline(0.1, color=MUTED, lw=1, ls=":", zorder=1)
                ax.text(xs[-1], 0.1, "目标 0.1", color=MUTED, fontsize=8, ha="right", va="bottom")
            for s in SOURCES:
                ys = [stats[(cat, w, s)][key] for w in widths]
                ax.plot(xs, ys, color=COLORS[s], lw=2, marker=MARKERS[s], ms=6,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, label=SOURCE_LABELS[s], zorder=3)
            ax.set_xticks(xs)
            ax.set_xticklabels([label_of(w) for w in widths], color=MUTED, fontsize=9)
            ax.tick_params(colors=MUTED, labelsize=9, length=0)
            ax.grid(axis="y", color=GRID, lw=0.8)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            if col == 0:
                ax.set_ylabel(ylabel, color=INK, fontsize=10)
            if row == 0:
                ax.set_title(f"Cat {cat}", color=INK, fontsize=11)
            if row == len(panels) - 1:
                ax.set_xlabel("块缩放位宽（U-plane；Cat B 权值同位宽）", color=MUTED, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    ncol = min(len(SOURCES), 2 * len(cats))            # 单栏时图例分两行，避免截断
    fig.legend(handles, labels, loc="upper center", ncol=ncol, frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95 if ncol == len(SOURCES) else 0.92), h_pad=2.5)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    args = p.parse_args()
    d = Path(args.run_dir)
    meta = json.loads((d / "metadata.json").read_text())
    sqnr_path = d / "codec_sqnr.json"
    sqnr_table = json.loads(sqnr_path.read_text()) if sqnr_path.exists() else {}
    rows = read_rows(d)
    ue_key = lambda r: (r["category"], r["drop"], r["ue"])
    base_fail = {ue_key(r) for r in rows if r["width"] == "ideal" and r["source"] == "genie" and r["bler"] >= 0.5}
    stats = summarize([r for r in rows if ue_key(r) not in base_fail])
    text = report(stats, sqnr_table, meta)
    total = len({ue_key(r) for r in rows})
    text = (f"排除理想前传下精灵 CQI 也失败的 UE-撒点 {len(base_fail)}/{total} 个（链路本身不可用，与压缩无关）\n"
            + text)
    bad = {(r["category"], r["drop"], r["width"]) for r in rows if not r.get("recheck_ok", True)}
    if bad:
        text = (f"警告：{len(bad)} 个（Cat, 撒点, 位宽）的复核结果不一致（同一种子重跑 PDSCH 结果不同），"
                "设备不可靠，本次结果不能使用。\n\n" + text)
    print(text)
    (d / "report.txt").write_text(text + "\n", encoding="utf-8")
    plot(stats, d / "fig_bler_tput_vs_width.png")
    print(f"\n图与表已写入 {d.resolve()}")


if __name__ == "__main__":
    main()
