#!/usr/bin/env python3
"""O-RAN 7.2x 下行 Category A 与 Category B 对比：IQ 压缩位宽 × SNR 扫描。

用法（先 source env.sh）：
    python scripts/compare_cat_a_b.py                     # 默认：256QAM（MCS 表 2，MCS 20），压缩 {理想, BFP 9/7/6/5/4}
    python scripts/compare_cat_a_b.py --widths ideal 9 5 --snr 16 30 2 --max-batches 5   # 快速试跑
    # 两块 GPU 并行：分别跑 Cat A、Cat B，再合并。Sionna 的随机种子按设备序号偏移，
    # 用 CUDA_VISIBLE_DEVICES 让两个进程都在 "cuda:0" 上运行，两边的随机实现才相同
    CUDA_VISIBLE_DEVICES=0 python scripts/compare_cat_a_b.py --categories A --out results/cat_a &
    CUDA_VISIBLE_DEVICES=1 python scripts/compare_cat_a_b.py --categories B --out results/cat_b &
    wait; python scripts/compare_cat_a_b.py --merge results/cat_a results/cat_b --out results/cat_ab

--widths 的取值：ideal（理想前传，不量化）、none16（16 bit 定点不压缩）、整数 w（BFP，尾数 w bit）。
同一 --seed 下，各配置在同一 SNR 点上的比特、信道和噪声完全相同（配对比较）。

输出（--out 目录，默认 results/cat_ab）：
    results.json   全部数值结果（每跑完一个配置就写一次）
    bler.png       BLER-SNR 曲线（实线 Cat A，虚线 Cat B，颜色区分压缩位宽）
    evm.png        接收端 EVM-SNR 曲线
    tradeoff.png   前传比特率（100 MHz）与所需 SNR、发射端 EVM 的折中
"""

import argparse
import json
import os
import sys
import time
import unicodedata
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sionna.phy  # noqa: E402

from oran import CompressionConfig, ORANConfig, ORANDownlink  # noqa: E402
from oran.fronthaul import fronthaul_rate  # noqa: E402


def compression_of(width: str) -> CompressionConfig:
    if width == "ideal":
        return CompressionConfig("ideal", 16)
    if width == "none16":
        return CompressionConfig("none", 16)
    return CompressionConfig("bfp", int(width))


def label_of(width: str) -> str:
    return {"ideal": "理想前传", "none16": "16 bit 不压缩"}.get(width, f"BFP {width} bit")


def _display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _row(cells, widths):
    """按显示宽度对齐（中文字符占两列）：前两列左对齐，其余右对齐。"""
    out = []
    for i, (c, w) in enumerate(zip(cells, widths)):
        pad = " " * max(w - _display_width(c), 0)
        out.append(c + pad if i < 2 else pad + c)
    return "".join(out)


def required_snr(snr, bler, target):
    """BLER 首次降到 target 以下时的 SNR（对 log10(BLER) 线性插值）。"""
    for i in range(1, len(snr)):
        if bler[i - 1] >= target > bler[i]:
            lo, hi = np.log10(max(bler[i - 1], 1e-9)), np.log10(max(bler[i], 1e-9))
            t = (np.log10(target) - lo) / (hi - lo)
            return snr[i - 1] + t * (snr[i] - snr[i - 1])
    return None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--categories", nargs="+", default=["A", "B"], choices=["A", "B"])
    p.add_argument("--widths", nargs="+", default=["ideal", "9", "7", "6", "5", "4"])
    p.add_argument("--bfw-width", default="9", help="Cat B 权值的 BFP 位宽；same 表示与 U-plane 相同")
    p.add_argument("--snr", nargs=3, type=float, default=[14, 30, 2], metavar=("START", "STOP", "STEP"))
    p.add_argument("--mcs", type=int, default=20)
    p.add_argument("--mcs-table", type=int, default=2)
    p.add_argument("--num-prb", type=int, default=51)
    p.add_argument("--layers-per-ue", type=int, default=2)
    p.add_argument("--precoder", default="bd", choices=["bd", "ezf", "rzf"])
    p.add_argument("--prg-size", type=int, default=2)
    p.add_argument("--domain", default="freq", choices=["freq", "time"])
    p.add_argument("--channel-estimation", default="ls", choices=["ls", "perfect"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-batches", type=int, default=20)
    p.add_argument("--target-errors", type=int, default=100)
    p.add_argument("--bler-target", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", default="cuda:0")
    p.add_argument("--out", default="results/cat_ab")
    p.add_argument("--merge", nargs="+", metavar="DIR",
                   help="不仿真，合并这些目录下的 results.json 后汇总、画图（如两块 GPU 分别跑 Cat A 和 Cat B）")
    return p.parse_args()


def run(args):
    sionna.phy.config.device = args.gpu
    snr = np.arange(args.snr[0], args.snr[1] + 1e-9, args.snr[2]).tolist()
    os.makedirs(args.out, exist_ok=True)
    out = {"args": vars(args), "runs": []}

    for width in args.widths:
        uplane = compression_of(width)
        if width == "ideal":
            bfw = uplane
        else:
            bfw = compression_of(width if args.bfw_width == "same" else args.bfw_width)
        for cat in args.categories:
            cfg = ORANConfig(category=cat, mcs_index=args.mcs, mcs_table=args.mcs_table,
                             num_prb=args.num_prb, num_layers_per_ue=args.layers_per_ue,
                             precoder=args.precoder, prg_size=args.prg_size, domain=args.domain,
                             channel_estimation=args.channel_estimation, uplane=uplane, bfw=bfw,
                             device=args.gpu)
            print(f"== Cat {cat} | {label_of(width)} ==", flush=True)
            sim = ORANDownlink(cfg)
            t0 = time.time()
            res = sim.simulate(snr, args.batch_size, args.max_batches, args.target_errors,
                               seed=args.seed)
            out["runs"].append({
                "category": cat, "width": width, "label": label_of(width),
                "bfw": asdict(bfw) if cat == "B" else None,
                "tb_sizes": sim.tb_sizes, "time_s": time.time() - t0, "results": res,
                "fh_rate_sim": asdict(fronthaul_rate(cfg)),
                "fh_rate_100mhz": asdict(fronthaul_rate(cfg, num_prb=273)),
            })
            with open(os.path.join(args.out, "results.json"), "w") as f:
                json.dump(out, f, indent=1, ensure_ascii=False)
    return out


def summarize(out, bler_target):
    rows = []
    ref = {}
    for r in out["runs"]:
        snr = [x["snr_db"] for x in r["results"]]
        bler = [float(np.mean(x["bler"])) for x in r["results"]]
        req = required_snr(snr, bler, bler_target)
        if r["width"] == "ideal":
            ref[r["category"]] = req
        rows.append((r, req))
    ideal = [v for v in ref.values() if v is not None]
    ideal = ideal[0] if ideal else None

    widths = [4, 14, 12, 11, 9, 9, 9, 7]
    print()
    print(_row(["Cat", "压缩", "FH@仿真带宽", "FH@100MHz", "U面SQNR", "发射EVM", "所需SNR", "损失"],
               widths))
    print(_row(["", "", "(Gbps)", "(Gbps)", "(dB)", "(%)", "(dB)", "(dB)"], widths))
    for r, req in rows:
        res = r["results"]
        sqnr = np.mean([x["uplane_sqnr_db"] for x in res])
        tx_evm = np.mean([x["tx_evm_percent"] for x in res])
        rate_sim = r["fh_rate_sim"]["uplane_total"] + r["fh_rate_sim"]["cplane_total"]
        rate_100 = r["fh_rate_100mhz"]["uplane_total"] + r["fh_rate_100mhz"]["cplane_total"]
        print(_row([r["category"], r["label"], f"{rate_sim / 1e9:.2f}", f"{rate_100 / 1e9:.2f}",
                    f"{sqnr:.1f}" if np.isfinite(sqnr) else "∞", f"{tx_evm:.3f}",
                    f"{req:.2f}" if req is not None else "—",
                    f"{req - ideal:.2f}" if (req is not None and ideal is not None) else "—"],
                   widths))
    print(f"（所需 SNR：两个 UE 平均 BLER 降到 {bler_target:g} 时的 SNR；损失相对理想前传；"
          "FH 为单个 O-RU 的 U-plane + C-plane 总比特率，含 eCPRI/以太网开销）")
    return rows


def plot(out, rows, path, bler_target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 按字形回退：西文用 DejaVu Sans，中文用 Droid Sans Fallback
    matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    widths = list(dict.fromkeys(r["width"] for r in out["runs"]))
    colors = {w: plt.cm.tab10(i) for i, w in enumerate(widths)}
    style = {"A": dict(ls="-", marker="o"), "B": dict(ls="--", marker="s")}
    a = out["args"]
    title = (f"MCS 表 {a['mcs_table']} 序号 {a['mcs']}，{a['num_prb']} PRB，2 UE × {a['layers_per_ue']} 层，"
             f"8T，{a['precoder'].upper()}，{a['domain']} 域")

    for name, key, ylabel, logy in (("bler.png", "bler", "BLER（两 UE 平均）", True),
                                    ("evm.png", "evm_percent", "接收 EVM (%)", False)):
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for r in out["runs"]:
            snr = np.array([x["snr_db"] for x in r["results"]])
            y = np.array([np.mean(x[key]) for x in r["results"]])
            if logy:
                keep = y > 0
                snr, y = snr[keep], y[keep]
            ax.plot(snr, y, color=colors[r["width"]], **style[r["category"]],
                    label=f"Cat {r['category']}，{r['label']}")
        if logy:
            ax.set_yscale("log")
            ax.axhline(bler_target, color="gray", lw=0.8, ls=":")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(path, name), dpi=150)
        plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    for r, req in rows:
        rate = (r["fh_rate_100mhz"]["uplane_total"] + r["fh_rate_100mhz"]["cplane_total"]) / 1e9
        tx_evm = np.mean([x["tx_evm_percent"] for x in r["results"]])
        kw = dict(color=colors[r["width"]], marker=style[r["category"]]["marker"], s=50)
        if req is not None:
            ax1.scatter(rate, req, **kw)
            ax1.annotate(f"{r['category']}:{r['width']}", (rate, req), fontsize=7,
                         xytext=(3, 3), textcoords="offset points")
        if tx_evm > 0:
            ax2.scatter(rate, tx_evm, **kw)
            ax2.annotate(f"{r['category']}:{r['width']}", (rate, tx_evm), fontsize=7,
                         xytext=(3, 3), textcoords="offset points")
    ax1.set_xlabel("单个 O-RU 前传比特率 @100 MHz (Gbps)")
    ax1.set_ylabel(f"BLER={bler_target:g} 所需 SNR (dB)")
    ax2.set_xlabel("单个 O-RU 前传比特率 @100 MHz (Gbps)")
    ax2.set_ylabel("发射端 EVM (%)，天线口相对理想前传")
    ax2.set_yscale("log")
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.3)
    ax1.set_title("圆点 Cat A，方块 Cat B", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(path, "tradeoff.png"), dpi=150)
    plt.close(fig)


def merge(dirs):
    out = None
    for d in dirs:
        with open(os.path.join(d, "results.json")) as f:
            part = json.load(f)
        if out is None:
            out = part
        else:
            out["runs"] += part["runs"]
    # 按压缩位宽、再按 Cat 排序，便于对照
    order = {w: i for i, w in enumerate(dict.fromkeys(r["width"] for r in out["runs"]))}
    out["runs"].sort(key=lambda r: (order[r["width"]], r["category"]))
    return out


def main():
    args = parse_args()
    if args.merge:
        os.makedirs(args.out, exist_ok=True)
        out = merge(args.merge)
        with open(os.path.join(args.out, "results.json"), "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
    else:
        out = run(args)
    rows = summarize(out, args.bler_target)
    plot(out, rows, args.out, args.bler_target)
    print(f"\n结果已保存到 {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
