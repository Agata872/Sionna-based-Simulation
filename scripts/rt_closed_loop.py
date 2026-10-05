#!/usr/bin/env python3
"""射线追踪场景中的闭环下行仿真：随机撒点 → CQI → MCS（+ OLLA）→ PDSCH → CRC。

默认 munich 场景、1 个 O-RU（楼顶，扇区朝北），UE 在覆盖区内随机撒点（室外、静止）。

用法（先 source env.sh）：
    python scripts/rt_closed_loop.py                          # 单 UE（SU-MIMO 2 层），30 个撒点
    python scripts/rt_closed_loop.py --num-ue 2 --out results/rt_mu    # 2 UE MU-MIMO（BD）
    python scripts/rt_closed_loop.py --drops 5 --batches 8   # 快速试跑
    python scripts/rt_closed_loop.py --category B --uplane bfp:7 --bfw bfp:7   # 前传压缩
    python scripts/rt_closed_loop.py --replot results/rt_su  # 只按已有结果重新画图

--uplane / --bfw 的取值与 compare_cat_a_b.py 的 --widths 相同：ideal、none16、整数 w（BFP）或 "方法:w"。
同一 --seed 下撒点位置、比特和噪声相同，可配对比较不同配置。

输出（--out 目录，默认 results/rt_su 或 results/rt_mu）：
    results.json   配置与每个撒点的结果（含逐批次的 CQI、MCS、OLLA 偏移、误块数）
    map.png        覆盖图（单阵元路径增益）、O-RU 与各 UE 位置，颜色为吞吐量
    cdf.png        UE 吞吐量与宽带 SNR 的 CDF
    snr.png        宽带 SNR 与 CQI、平均 MCS、吞吐量的关系
    olla.png       前几个撒点的 MCS 与 OLLA 偏移随批次的变化
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-ue", type=int, default=1)
    p.add_argument("--layers-per-ue", type=int, default=2)
    p.add_argument("--drops", type=int, default=30)
    p.add_argument("--batches", type=int, default=16, help="每个撒点的批次数")
    p.add_argument("--batch-size", type=int, default=8, help="每批时隙数（同一 MCS）")
    p.add_argument("--warmup", type=int, default=4, help="不计入统计的前若干批次（OLLA 收敛）")
    p.add_argument("--category", default="A", choices=["A", "B"])
    p.add_argument("--uplane", default="9", help="U-plane 压缩")
    p.add_argument("--bfw", default="9", help="Cat B 权值压缩")
    p.add_argument("--mcs-table", type=int, default=2, choices=[1, 2])
    p.add_argument("--num-prb", type=int, default=51)
    p.add_argument("--tx-power", type=float, default=None, help="O-RU 总发射功率 [dBm]，默认见 RTConfig")
    p.add_argument("--channel-estimation", default="ls", choices=["ls", "perfect"])
    p.add_argument("--olla-step", type=float, default=0.5)
    p.add_argument("--no-olla", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--gpu", default="cuda:0")
    p.add_argument("--out", default=None)
    p.add_argument("--replot", metavar="DIR", help="只按 DIR/results.json 重新画图")
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


def plot(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    drops = out["drops"]
    a = out["args"]
    num_ue = a["num_ue"]
    ue = lambda key: np.array([d[key][k] for d in drops for k in range(num_ue)], dtype=float)
    tput, snr, cqi, mcs, bler = ue("throughput_mbps"), ue("snr_db"), ue("cqi"), ue("mean_mcs"), ue("bler")
    pos = np.array([d["positions"][k] for d in drops for k in range(num_ue)])
    title = (f"{out['config']['rt']['scene']}，{a['num_ue']} UE × {a['layers_per_ue']} 层，Cat {a['category']}，"
             f"U-plane {label_of(a['uplane'])}，MCS 表 {a['mcs_table']}，{a['num_prb']} PRB")

    # ---- 覆盖图与 UE 位置 ----
    cov = out["coverage"]
    fig, ax = plt.subplots(figsize=(9, 8))
    pg = np.array(cov["path_gain_db"])
    ext = cov["extent"]
    im = ax.imshow(pg, origin="lower", extent=ext, cmap="Greys_r", vmin=-150, vmax=-60)
    fig.colorbar(im, ax=ax, label="单阵元路径增益 [dB]", shrink=0.8)
    for o, (p, la) in enumerate(zip(out["config"]["rt"]["oru_positions"], out["config"]["rt"]["oru_look_at"])):
        ax.plot(p[0], p[1], "r^", ms=12)
        d = np.array(la[:2]) - np.array(p[:2])
        d = 30 * d / np.linalg.norm(d)
        ax.annotate("", xy=(p[0] + d[0], p[1] + d[1]), xytext=(p[0], p[1]),
                    arrowprops=dict(arrowstyle="->", color="r"))
    if num_ue > 1:
        for dd in drops:
            q = np.array(dd["positions"])
            ax.plot(q[:, 0], q[:, 1], "-", color="tab:blue", lw=0.5, alpha=0.5)
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=tput, cmap="viridis", s=30, edgecolors="w", linewidths=0.5)
    fig.colorbar(sc, ax=ax, label="UE 吞吐量 [Mbps]", shrink=0.8)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(path, "map.png"), dpi=120); plt.close(fig)

    # ---- CDF ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, x, xl in ((axes[0], tput, "UE 吞吐量 [Mbps]"), (axes[1], snr, "宽带 SNR [dB]（每接收天线，不含波束赋形增益）")):
        xs = np.sort(x)
        ax.step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post")
        ax.set_xlabel(xl); ax.set_ylabel("CDF"); ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(path, "cdf.png"), dpi=120); plt.close(fig)

    # ---- SNR 与 CQI / MCS / 吞吐量 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(snr, cqi, s=14, label="上报 CQI")
    axes[0].scatter(snr, mcs, s=14, marker="x", label="平均 MCS")
    axes[0].set_ylabel("序号"); axes[0].legend(fontsize=8)
    s2 = axes[1].scatter(snr, tput, c=bler, cmap="coolwarm", vmin=0, vmax=0.3, s=16)
    fig.colorbar(s2, ax=axes[1], label="首传 BLER")
    axes[1].set_ylabel("吞吐量 [Mbps]")
    for ax in axes:
        ax.set_xlabel("宽带 SNR [dB]"); ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(path, "snr.png"), dpi=120); plt.close(fig)

    # ---- OLLA ----
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for i, d in enumerate(drops[:6]):
        h = d["history"]
        for k in range(num_ue):
            lab = f"撒点 {i} UE{k}"
            axes[0].step(range(len(h["mcs"])), np.array(h["mcs"])[:, k], where="post", label=lab)
            axes[1].step(range(len(h["olla_offset_db"])), np.array(h["olla_offset_db"])[:, k], where="post")
    axes[0].set_ylabel("MCS"); axes[1].set_ylabel("OLLA 偏移 [dB]")
    axes[1].set_xlabel(f"批次（每批 {a['batch_size']} 个时隙）")
    for ax in axes:
        ax.axvline(a["warmup"] - 0.5, color="k", ls=":", lw=0.8); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=3)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(path, "olla.png"), dpi=120); plt.close(fig)


def coverage_for_plot(sim):
    rm = sim.scene.coverage_map()
    cc = rm.cell_centers.numpy()
    cell = sim.cfg.rt.coverage_cell_size
    pg = 10 * np.log10(rm.path_gain.numpy().max(axis=0) + 1e-30)
    extent = [float(cc[0, 0, 0] - cell / 2), float(cc[-1, -1, 0] + cell / 2),
              float(cc[0, 0, 1] - cell / 2), float(cc[-1, -1, 1] + cell / 2)]
    return {"path_gain_db": np.round(pg, 1).tolist(), "extent": extent}


def main():
    args = parse_args()
    if args.replot:
        with open(os.path.join(args.replot, "results.json"), encoding="utf-8") as f:
            plot(json.load(f), args.replot)
        print(f"图已更新：{os.path.abspath(args.replot)}")
        return
    out_dir = args.out or ("results/rt_su" if args.num_ue == 1 else "results/rt_mu")
    os.makedirs(out_dir, exist_ok=True)
    sionna.phy.config.device = args.gpu

    cfg = ORANConfig(channel_model="rt", category=args.category, num_ue=args.num_ue,
                     num_layers_per_ue=args.layers_per_ue, num_prb=args.num_prb,
                     mcs_table=args.mcs_table, uplane=compression_of(args.uplane),
                     bfw=compression_of(args.bfw), channel_estimation=args.channel_estimation,
                     device=args.gpu)
    if args.tx_power is not None:
        cfg.rt.tx_power_dbm = args.tx_power
    la_cfg = LinkAdaptationConfig(olla=not args.no_olla, olla_step_db=args.olla_step)

    t0 = time.time()
    sim = ClosedLoopDownlink(cfg, la_cfg)
    print(f"初始化 {time.time() - t0:.1f} s；可用 MCS {sim.la.allowed_mcs.tolist()}；"
          f"CQI 门限 [dB] {np.round(sim.reporter.threshold_db, 1).tolist()}", flush=True)
    out = {"args": vars(args), "config": jsonable(asdict(cfg)), "link_adaptation": asdict(la_cfg),
           "allowed_mcs": sim.la.allowed_mcs.tolist(),
           "cqi_threshold_db": sim.reporter.threshold_db.tolist(),
           "mcs_threshold_db": sim.la.mcs_threshold_db.tolist(),
           "coverage": coverage_for_plot(sim), "drops": []}
    path = os.path.join(out_dir, "results.json")
    for i in range(args.drops):
        t = time.time()
        res = sim.simulate(1, args.batches, args.batch_size, args.warmup, seed=args.seed + i,
                           verbose=False)[0]
        s = res.summary(cfg.slots_per_second)
        out["drops"].append(jsonable(s))
        ue = " | ".join(
            f"UE{k} ({s['positions'][k][0]:6.1f},{s['positions'][k][1]:6.1f}) SNR {s['snr_db'][k]:5.1f} dB "
            f"CQI {s['cqi'][k]:2d} MCS {s['mean_mcs'][k]:4.1f} BLER {s['bler'][k]:.2f} "
            f"{s['throughput_mbps'][k]:6.1f} Mbps" for k in range(cfg.num_ue))
        print(f"撒点 {i:3d} | {ue} | {time.time() - t:.1f} s", flush=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)

    tput = np.array([d["throughput_mbps"] for d in out["drops"]]).flatten()
    bler = np.array([d["bler"] for d in out["drops"]]).flatten()
    print(f"\nUE 吞吐量：平均 {tput.mean():.1f} Mbps，5% 分位 {np.percentile(tput, 5):.1f} Mbps，"
          f"中位 {np.median(tput):.1f} Mbps；平均首传 BLER {bler.mean():.3f}")
    plot(out, out_dir)
    print(f"结果已保存到 {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
