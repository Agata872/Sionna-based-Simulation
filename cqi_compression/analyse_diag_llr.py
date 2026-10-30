#!/usr/bin/env python3
"""汇总 diag_oracle_llr.py：各软解调方案的 BLER、LLR 过度自信量、数据时隙与探测时隙的 SINR 差。

排除口径与 analyse_cqi_bler.py 相同：理想前传下（thermal）BLER ≥ 0.5 的 UE-撒点不参与汇总
（这里没有精灵 CQI，用 oracle 代替）。

用法：python cqi_compression/analyse_diag_llr.py results/cqi_compression_20261030/diag_llr_A [diag_llr_B ...]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

VARIANTS = ("thermal", "scalar", "prb", "gauss-thermal", "gauss-prb")
CAT_ORDER = {"A": 0, "B": 1, "8": 2}


def load(dirs):
    rows = []
    for d in dirs:
        with open(Path(d) / "rows.jsonl", encoding="utf-8") as f:
            rows += [json.loads(line) for line in f]
    return rows


def main():
    dirs = sys.argv[1:]
    rows = load(dirs)
    ue_drop = lambda r: (r["category"], r["drop"], r["ue"])
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[ue_drop(r) + (r["width"],)]["rows"].append(r)
    bad = set()
    for key, a in agg.items():
        if key[3] == "ideal":
            rs = a["rows"]
            if sum(r["err_thermal"] for r in rs) / sum(r["blocks"] for r in rs) >= 0.5:
                bad.add(key[:3])

    cats = sorted({r["category"] for r in rows}, key=lambda c: CAT_ORDER.get(c, 9))
    widths = sorted({r["width"] for r in rows}, key=lambda w: (0, 0) if w == "ideal" else (1, -int(w)))
    lines = [f"排除理想前传下 BLER ≥ 0.5 的 UE-撒点 {len(bad)} 个（每个 Cat："
             + "、".join(f"{c} {sum(b[0] == c for b in bad)}" for c in cats) + "）", ""]
    head = f"{'Cat':4}{'位宽':>6}{'UE-撒点':>8}{'SQNR':>7}" + "".join(f"{v:>15}" for v in VARIANTS) \
        + f"{'过度自信 dB':>12}{'数据-探测 dB':>13}{'余量 dB':>9}"
    lines.append("首传 BLER（括号内为 BLER ≥ 0.5 的 UE 比例）；过度自信 = 实际均衡误差功率 / no_eff；"
                 "数据-探测 = 数据时隙实际有效 SINR - oracle 探测值；余量 = oracle 有效 SINR - 所选 MCS 门限")
    lines.append(head)
    for c in cats:
        for w in widths:
            keys = [k for k in agg if k[0] == c and k[3] == w and k[:3] not in bad]
            if not keys:
                continue
            rs = [r for k in keys for r in agg[k]["rows"]]
            blocks = sum(r["blocks"] for r in rs)
            cells = []
            for v in VARIANTS:
                bler = sum(r[f"err_{v}"] for r in rs) / blocks
                fail = np.mean([sum(r[f"err_{v}"] for r in agg[k]["rows"])
                                / sum(r["blocks"] for r in agg[k]["rows"]) >= 0.5 for k in keys])
                cells.append(f"{bler:7.3f} ({fail:4.2f})")
            sqnr = [r["uplane_sqnr_db"] for r in rs if r["uplane_sqnr_db"] is not None]
            over = np.mean([r["overconf_db"] for r in rs])
            gap = np.mean([r["eff_data_db"] - r["eff_oracle_db"] for r in rs])
            margin = np.mean([r["eff_oracle_db"] - r["mcs_threshold_db"] for r in rs])
            lines.append(f"{c:4}{w:>6}{len(keys):>8}{np.mean(sqnr) if sqnr else float('inf'):7.1f}"
                         + "".join(f"{x:>15}" for x in cells) + f"{over:12.2f}{gap:13.2f}{margin:9.2f}")
        lines.append("")

    # 按过度自信量分组：低位宽下 thermal 与 prb 的差别是否随过度自信量增大
    lines.append("按过度自信量分组（非理想位宽，全部 Cat）：UE-撒点数、thermal / prb / gauss-prb 的 BLER")
    edges = [-np.inf, 0, 3, 6, 10, np.inf]
    groups = defaultdict(list)
    for k, a in agg.items():
        if k[3] == "ideal" or k[:3] in bad:
            continue
        over = np.mean([r["overconf_db"] for r in a["rows"]])
        groups[int(np.searchsorted(edges, over)) - 1].append(a["rows"])
    for g in sorted(groups):
        rs = [r for x in groups[g] for r in x]
        blocks = sum(r["blocks"] for r in rs)
        b = {v: sum(r[f"err_{v}"] for r in rs) / blocks for v in ("thermal", "prb", "gauss-prb")}
        lines.append(f"  [{edges[g]:>5}, {edges[g + 1]:>5}) dB：{len(groups[g]):3d} 个，"
                     f"thermal {b['thermal']:.3f}，prb {b['prb']:.3f}，gauss-prb {b['gauss-prb']:.3f}")

    report = "\n".join(lines)
    print(report)
    out = Path(dirs[0]).parent / "diag_llr_report.txt"
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\n报告已写入 {out}")


if __name__ == "__main__":
    main()
