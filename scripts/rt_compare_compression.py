#!/usr/bin/env python3
"""闭环中比较 Cat A / Cat B 的前传压缩失真：CSI-RS 经前传发送，CQI 能看到多少失真，OLLA 又补了多少。

同一批撒点（配对种子）上依次运行各压缩配置：每个撒点只做一次射线追踪，比特、噪声在各配置间相同。

用法（先 source env.sh）：
    python scripts/rt_compare_compression.py --drops 4 --batches 8      # 快速试跑（2 UE，Cat A/B）
    # 完整对比：先生成撒点文件，两块 GPU 各跑一类（读同一个撒点文件），再合并画图。
    # 覆盖图是蒙特卡洛结果，各进程单独撒点时门限附近的个别撒点会不同，所以跨进程比较要共用撒点文件；
    # 用 CUDA_VISIBLE_DEVICES 让两个进程都在 "cuda:0" 上运行，两边的比特和噪声才相同
    python scripts/rt_compare_compression.py --save-drops results/drops_mu.json
    CUDA_VISIBLE_DEVICES=0 python scripts/rt_compare_compression.py --categories A --drops-file results/drops_mu.json --out results/csi_a &
    CUDA_VISIBLE_DEVICES=1 python scripts/rt_compare_compression.py --categories B --drops-file results/drops_mu.json --out results/csi_b &
    wait; python scripts/rt_compare_compression.py --merge results/csi_a results/csi_b --out results/csi_ab

--widths 的取值与 compare_cat_a_b.py 相同：ideal、none16、整数 w（BFP）或 "方法:w"（bfp、bs、mu）。
Cat B 的权值压缩默认与 U-plane 相同（--bfw-width 可固定为某一配置）。

输出（--out 目录，默认 results/csi_ab）：
    results.json   各配置、各撒点的结果
    summary.png    各压缩配置下的平均吞吐、5% 分位吞吐、首传 BLER、OLLA 偏移、CQI 相对精灵 CQI 的变化、
                   NZP-IMR 测得的干扰噪声比
    cdf.png        UE 吞吐量 CDF
    gap.png        CQI 高估量（CSI-RS 有效 SINR − 数据时隙实测 SINR，相对理想前传的增量）与宽带 SNR 的关系
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sionna.phy  # noqa: E402

from compare_cat_a_b import compression_of, label_of  # noqa: E402
from oran import ClosedLoopDownlink, LinkAdaptationConfig, ORANConfig  # noqa: E402
from oran.fronthaul import fronthaul_rate  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-ue", type=int, default=2)
    p.add_argument("--layers-per-ue", type=int, default=2)
    p.add_argument("--categories", nargs="+", default=["A", "B"], choices=["A", "B"])
    p.add_argument("--widths", nargs="+", default=["ideal", "9", "7", "5"])
    p.add_argument("--bfw-width", default=None, help="Cat B 权值压缩，默认与 U-plane 相同")
    p.add_argument("--cqi-source", default="csi-rs", choices=["csi-rs", "genie"])
    p.add_argument("--drops", type=int, default=30)
    p.add_argument("--batches", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=7)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--mcs-table", type=int, default=2, choices=[1, 2])
    p.add_argument("--num-prb", type=int, default=51)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--gpu", default="cuda:0")
    p.add_argument("--out", default="results/csi_ab")
    p.add_argument("--merge", nargs="+", metavar="DIR", help="合并多个目录的 results.json 并画图")
    p.add_argument("--save-drops", metavar="FILE", help="只撒点（--drops 个，种子 --seed 起）并保存到 FILE")
    p.add_argument("--drops-file", metavar="FILE", help="使用 FILE 中的撒点（覆盖 --drops 和 --seed）")
    return p.parse_args()


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def ue_values(run, key):
    """把一个配置所有撒点、所有 UE 的某个量摊平成一维数组（按撒点、UE 顺序）。"""
    return np.array([v for d in run["drops"] for v in d[key]], dtype=float)


def stats_of(run):
    tput = ue_values(run, "throughput_mbps")
    out = {
        "mean_tput": tput.mean(), "p5_tput": np.percentile(tput, 5),
        "bler": ue_values(run, "bler").mean(),
        "olla": ue_values(run, "mean_olla_offset_db").mean(),
        "dcqi": (ue_values(run, "cqi") - ue_values(run, "cqi_genie")).mean(),
        "mcs": ue_values(run, "mean_mcs").mean(),
    }
    if all("csi_inr_db" in d for d in run["drops"]):
        out["inr"] = 10 * np.log10(np.mean(10 ** (ue_values(run, "csi_inr_db") / 10)))
    return out


def print_table(out):
    print(f"\n{'配置':<16}{'前传(100MHz)':>12}{'平均吞吐':>10}{'5%吞吐':>9}{'BLER':>7}{'OLLA':>8}"
          f"{'ΔCQI':>7}{'INR':>7}{'MCS':>6}")
    for run in out["runs"]:
        s = stats_of(run)
        inr = f"{s['inr']:7.1f}" if "inr" in s else f"{'-':>7}"
        print(f"{'Cat ' + run['category'] + ' ' + run['label']:<16}{run['fronthaul_gbps']:10.2f} G"
              f"{s['mean_tput']:10.1f}{s['p5_tput']:9.1f}{s['bler']:7.3f}{s['olla']:+8.2f}"
              f"{s['dcqi']:+7.2f}{inr}{s['mcs']:6.1f}")
    print("（吞吐 Mbps；OLLA 为收敛期后的平均偏移 dB；ΔCQI = CSI-RS CQI − 精灵 CQI；"
          "INR 为 NZP-IMR 测得的干扰噪声比 dB）")


def plot(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    a = out["args"]
    runs = out["runs"]
    widths = list(dict.fromkeys(r["width"] for r in runs))
    colors = {w: plt.cm.tab10(i) for i, w in enumerate(widths)}
    style = {"A": dict(ls="-", marker="o"), "B": dict(ls="--", marker="s")}
    title = (f"{a['num_ue']} UE × {a['layers_per_ue']} 层，CQI 来源 {a['cqi_source']}，MCS 表 {a['mcs_table']}，"
             f"{a['num_prb']} PRB，{len(runs[0]['drops'])} 个撒点（实线 Cat A，虚线 Cat B）")

    # ---- 汇总 ----
    panels = [("mean_tput", "平均 UE 吞吐 [Mbps]"), ("p5_tput", "5% 分位 UE 吞吐 [Mbps]"),
              ("bler", "首传 BLER"), ("olla", "OLLA 平均偏移 [dB]"),
              ("dcqi", "CQI(CSI-RS) − CQI(精灵)"), ("inr", "NZP-IMR 测得的 INR [dB]")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    labels = [label_of(w) for w in widths]
    for ax, (key, ylabel) in zip(axes.flat, panels):
        for cat in ("A", "B"):
            xs, ys = [], []
            for i, w in enumerate(widths):
                run = next((r for r in runs if r["category"] == cat and r["width"] == w), None)
                if run is None:
                    continue
                s = stats_of(run)
                if key in s:
                    xs.append(i); ys.append(s[key])
            if xs:
                ax.plot(xs, ys, color="k", label=f"Cat {cat}", **style[cat])
        ax.set_xticks(range(len(widths))); ax.set_xticklabels(labels, fontsize=8, rotation=15)
        ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(path, "summary.png"), dpi=120); plt.close(fig)

    # ---- 吞吐量 CDF ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in runs:
        t = np.sort(ue_values(r, "throughput_mbps"))
        ax.step(t, np.arange(1, len(t) + 1) / len(t), where="post", color=colors[r["width"]],
                ls=style[r["category"]]["ls"], label=f"Cat {r['category']} {r['label']}")
    ax.set_xlabel("UE 吞吐量 [Mbps]"); ax.set_ylabel("CDF"); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2); ax.set_title(title, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(path, "cdf.png"), dpi=120); plt.close(fig)

    # ---- CQI 高估量（相对理想前传）----
    cats = [c for c in ("A", "B") if any(r["category"] == c for r in runs)]
    fig, axes = plt.subplots(1, len(cats), figsize=(6.5 * len(cats), 4.5), squeeze=False)
    for ax, cat in zip(axes[0], cats):
        base = next((r for r in runs if r["category"] == cat and r["width"] == "ideal"), None)
        if base is None:
            continue
        gap0 = ue_values(base, "sinr_eff_db") - ue_values(base, "data_sinr_db")
        for r in runs:
            if r["category"] != cat or r["width"] == "ideal":
                continue
            gap = ue_values(r, "sinr_eff_db") - ue_values(r, "data_sinr_db") - gap0
            ax.scatter(ue_values(r, "snr_db"), gap, s=12, color=colors[r["width"]], label=r["label"])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("宽带 SNR [dB]"); ax.set_ylabel("CQI 高估量相对理想前传的增量 [dB]")
        ax.set_title(f"Cat {cat}：(CSI 有效 SINR − 数据时隙实测 SINR) − 理想前传时的同一量", fontsize=8)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(path, "gap.png"), dpi=120); plt.close(fig)


def merge(dirs, out_dir):
    merged = None
    for d in dirs:
        with open(os.path.join(d, "results.json"), encoding="utf-8") as f:
            o = json.load(f)
        if merged is None:
            merged = o
        else:
            merged["runs"] += o["runs"]
    merged["args"]["categories"] = sorted({r["category"] for r in merged["runs"]})
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)
    return merged


def main():
    args = parse_args()
    if args.merge:
        out = merge(args.merge, args.out)
        print_table(out)
        plot(out, args.out)
        print(f"结果已保存到 {os.path.abspath(args.out)}")
        return
    sionna.phy.config.device = args.gpu
    if args.save_drops:
        cfg = ORANConfig(channel_model="rt", num_ue=args.num_ue, num_layers_per_ue=args.layers_per_ue,
                         num_prb=args.num_prb, mcs_table=args.mcs_table, device=args.gpu)
        sim = ClosedLoopDownlink(cfg)
        drops = [sim.drop(args.seed + i).tolist() for i in range(args.drops)]
        os.makedirs(os.path.dirname(os.path.abspath(args.save_drops)), exist_ok=True)
        with open(args.save_drops, "w", encoding="utf-8") as f:
            json.dump({"num_ue": args.num_ue, "seed": args.seed, "positions": drops}, f)
        print(f"{len(drops)} 个撒点已保存到 {os.path.abspath(args.save_drops)}")
        return
    positions_list = None
    if args.drops_file:
        with open(args.drops_file, encoding="utf-8") as f:
            saved = json.load(f)
        if saved["num_ue"] != args.num_ue:
            raise ValueError(f"撒点文件是 {saved['num_ue']} 个 UE，与 --num-ue 不符")
        positions_list = [np.asarray(p) for p in saved["positions"]]
        args.drops, args.seed = len(positions_list), saved["seed"]
    os.makedirs(args.out, exist_ok=True)
    la_cfg = LinkAdaptationConfig()
    out = {"args": vars(args), "link_adaptation": asdict(la_cfg), "runs": []}
    path = os.path.join(args.out, "results.json")

    for cat in args.categories:
        bfw_of = lambda w: compression_of(args.bfw_width or w)
        cfg = ORANConfig(channel_model="rt", category=cat, num_ue=args.num_ue,
                         num_layers_per_ue=args.layers_per_ue, num_prb=args.num_prb,
                         mcs_table=args.mcs_table, cqi_source=args.cqi_source, device=args.gpu,
                         uplane=compression_of(args.widths[0]), bfw=bfw_of(args.widths[0]))
        t0 = time.time()
        sim = ClosedLoopDownlink(cfg, la_cfg)
        print(f"Cat {cat}：初始化 {time.time() - t0:.1f} s", flush=True)
        runs = {}
        for w in args.widths:
            sim.set_compression(compression_of(w), bfw_of(w))
            run = {"category": cat, "width": w, "label": label_of(w),
                   "fronthaul_gbps": fronthaul_rate(cfg, num_prb=273).total / 1e9,
                   "config": jsonable(asdict(cfg)), "drops": []}
            runs[w] = run
            out["runs"].append(run)
        for i in range(args.drops):
            t = time.time()
            positions = sim.drop(args.seed + i) if positions_list is None else positions_list[i]
            line = []
            for w in args.widths:
                sim.set_compression(compression_of(w), bfw_of(w))
                sionna.phy.config.seed = args.seed + i
                s = sim.run_drop(positions, args.batches, args.batch_size, args.warmup) \
                    .summary(cfg.slots_per_second)
                runs[w]["drops"].append(jsonable(s))
                line.append(f"{label_of(w)} {np.mean(s['throughput_mbps']):6.1f}")
            snr = " ".join(f"{v:5.1f}" for v in runs[args.widths[0]]["drops"][-1]["snr_db"])
            print(f"Cat {cat} 撒点 {i:3d} | SNR {snr} dB | 平均吞吐 Mbps：{' | '.join(line)} | "
                  f"{time.time() - t:.1f} s", flush=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False)

    print_table(out)
    plot(out, args.out)
    print(f"结果已保存到 {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
