#!/usr/bin/env python3
"""CQI 查表选 MCS 时，是否考虑前传压缩对首传 BLER 的影响。

闭环（Sionna RT munich，默认 2 UE × 2 层 BD，MCS / CQI 表 2）中关闭 OLLA，MAC 只按上报的 CQI 查表选 MCS。
前传用块缩放（O-RAN CUS 附录 A.2，同位宽下三种算法中失真最小，见 codec_sqnr.json）扫描位宽，
每个位宽比较四种 CQI：
    genie   不考虑压缩：真实等效信道、理想权值，不经前传（ClosedLoopDownlink.measure_cqi_genie）
    csi-rs  UE 实测：预编码 CSI-RS / CSI-IM 与 PDSCH 一起经压缩前传（ClosedLoopDownlink.measure_cqi_csirs）
    aware   考虑数据 RE 上的压缩失真：精灵 CQI 中计入前传实际造成的失真（aware_cqi.measure_cqi_aware，解析）
    oracle  考虑全部压缩影响：在一个经压缩前传的 PDSCH 时隙上实测均衡后 SINR（aware_cqi.measure_cqi_oracle），
            另含 DMRS 失真带来的信道估计误差，是"完全考虑压缩"的参照
数据时隙的 PDSCH 都经过同一个压缩前传。UE 静止，每个撒点的信道只算一次；同一撒点、同一周期内，
各位宽、各 CQI 来源的比特和噪声相同（配对比较），选出相同 MCS 的 CQI 来源共用同一次 PDSCH 仿真。

默认只用 CPU（射线追踪走 Mitsuba 的 LLVM 后端）：WSL 笔记本的 RTX 4060 在持续负载下会读错显存
（LDPC 译码结果不确定、下标越界断言，纯 PyTorch 的类 BP 循环也能复现，见 README）。
--device cuda:0 才使用 GPU；GPU 的 OptiX 不可用时再加 --rt-llvm，射线追踪改用 CPU。

用法（先 source env.sh）：
    python cqi_compression/run_cqi_bler.py --drops 2 --periods 1 --widths ideal 5 --out results/cqi_compression_20261029/smoke
    python cqi_compression/run_cqi_bler.py --out results/cqi_compression_20261029/main
    python cqi_compression/analyse_cqi_bler.py results/cqi_compression_20261029/main

输出（--out 目录）：metadata.json、源码快照、rows.jsonl（每行一个 Cat × 撒点 × 位宽 × CQI 来源 × UE）、
codec_sqnr.json（各压缩算法、各位宽在实际 U-plane 信号上的 SQNR）、complete.json
（recheck_mismatches 为复核不一致的（撒点, 位宽）个数，不为 0 时结果作废）。
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dynamic_split"))

# 除非显式要求 CUDA 设备，在导入 torch / Dr.Jit 之前隐藏 GPU，射线追踪随之使用 LLVM 后端
if not any(a.startswith("cuda") for a in sys.argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
# --rt-llvm：射线追踪强制用 LLVM（CPU），GPU 只用于链路仿真；须在 sionna.rt 导入前设置变体
if "--rt-llvm" in sys.argv:
    import mitsuba as mi
    mi.set_variant("llvm_ad_mono_polarized")

import sionna.phy  # noqa: E402
import torch  # noqa: E402

from aware_cqi import measure_cqi_aware, measure_cqi_oracle  # noqa: E402
from oran import ClosedLoopDownlink, CompressionConfig, LinkAdaptationConfig, ORANConfig  # noqa: E402
from oran.channel import ChannelState  # noqa: E402
from oran.fronthaul import IQCodec, fronthaul_rate  # noqa: E402
from runio import Run  # noqa: E402

SOURCES = ("genie", "csi-rs", "aware", "oracle")
SQNR_METHODS = ("bfp", "block_scaling", "mu_law")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--categories", nargs="+", default=["A", "B"], choices=["A", "B"])
    p.add_argument("--method", default="block_scaling")
    p.add_argument("--widths", nargs="+", default=["ideal", "9", "8", "7", "6", "5", "4", "3"])
    p.add_argument("--bfw", default="same", help="Cat B 权值压缩：same（与 U-plane 相同）或 ideal / 整数位宽")
    p.add_argument("--drops", type=int, default=40)
    p.add_argument("--periods", type=int, default=4, help="每个撒点的 CSI-RS 周期数（不需要 OLLA 收敛期）")
    p.add_argument("--batch-size", type=int, default=7, help="每周期的数据时隙数")
    p.add_argument("--num-ue", type=int, default=2)
    p.add_argument("--layers-per-ue", type=int, default=2)
    p.add_argument("--mcs-table", type=int, default=2, choices=[1, 2])
    p.add_argument("--num-prb", type=int, default=51)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cpu", help="cpu（默认）或 cuda:0")
    p.add_argument("--rt-llvm", action="store_true", help="射线追踪用 CPU（GPU 的 OptiX 不可用时）")
    p.add_argument("--out", required=True)
    return p.parse_args()


def compression(method, width):
    return CompressionConfig("ideal", 16) if width == "ideal" else CompressionConfig(method, int(width))


def finite(x):
    """理想前传的 SQNR 为 inf，JSON 中记为 null。"""
    return float(x) if x is not None and np.isfinite(x) else None


def seed_for(args, drop, period, tag):
    """同一撒点、同一周期、同一用途（tag）的随机种子在各位宽、各 CQI 来源间相同。"""
    return ((args.seed * 100_003 + drop) * 1009 + period) * 4 + tag


@torch.no_grad()
def codec_sqnr(cl, state, widths):
    """实际 U-plane 信号上各压缩算法、各位宽的 SQNR [dB]（μ 律位宽须 ≥ 3）。"""
    sim, cfg = cl.sim, cl.cfg
    one = ChannelState(state.h_freq[:1])
    tx = sim.odu(1, sim.channel.csi(one, sim.csi_symbol))
    power = 1.0 / cfg.num_tx_ant if cfg.category == "A" else 1.0
    out = {}
    for method in SQNR_METHODS:
        for w in widths:
            if w == "ideal" or (method == "mu_law" and int(w) < 3):
                continue
            codec = IQCodec(CompressionConfig(method, int(w)), power, block_len=12)
            codec(tx.uplane[0])
            out[f"{method}:{w}"] = codec.stats.sqnr_db
    return out


def pdsch(cl, args, drop, period, mcs, state):
    """按给定 MCS 发送一个批次的 PDSCH，返回 (TBS, 出错 TB 数, 均衡误差能量, 参考能量)，均为 [K]。"""
    cl.sim.set_mcs(mcs)
    sionna.phy.config.seed = seed_for(args, drop, period, 2)
    stats = cl.sim.run_batch(args.batch_size, 0.0, state=state)    # 信道已含链路预算，噪声方差为 1
    return (np.array(cl.sim.tb_sizes), stats.block_errors.astype(int),
            stats.evm_err.copy(), stats.evm_ref.copy())


def run_width(cl, args, drop, state, cqi_genie, eff_genie):
    """一个位宽：每个周期测 CSI-RS、aware、oracle 三种 CQI，四种 CQI 各自查表选 MCS 并发送 PDSCH。

    第一个周期结束后，用同一种子把第一个 PDSCH 批次重跑一次，结果须逐位相同（recheck_ok）。
    WSL 笔记本的 GPU 会间歇性读错显存、自检也可能碰巧通过（见 gpu_selfcheck.py），这一步让每次运行自证可信。
    """
    num_ue = cl.cfg.num_ue
    rec = {s: {k: [] for k in ("cqi", "eff_sinr_db", "mcs", "tbs", "errors")} for s in SOURCES}
    evm = {s: np.zeros((2, num_ue)) for s in SOURCES}
    recheck_ok = True
    for p in range(args.periods):
        sionna.phy.config.seed = seed_for(args, drop, p, 0)
        cqi_c, eff_c, _ = cl.measure_cqi_csirs(state)
        sionna.phy.config.seed = seed_for(args, drop, p, 1)
        cqi_a, eff_a = measure_cqi_aware(cl, state)
        sionna.phy.config.seed = seed_for(args, drop, p, 3)
        cqi_o, eff_o = measure_cqi_oracle(cl, state)
        reports = {"genie": (cqi_genie, eff_genie), "csi-rs": (cqi_c, eff_c), "aware": (cqi_a, eff_a),
                   "oracle": (cqi_o, eff_o)}
        cache = {}
        for src, (cqi, eff) in reports.items():
            mcs = cl.la.select_mcs(cqi)
            key = tuple(int(m) for m in mcs)
            if key not in cache:
                cache[key] = pdsch(cl, args, drop, p, list(key), state)
            tbs, errors, err, ref = cache[key]
            r = rec[src]
            r["cqi"].append(cqi); r["eff_sinr_db"].append(eff); r["mcs"].append(mcs)
            r["tbs"].append(tbs); r["errors"].append(errors)
            evm[src] += np.stack([err, ref])
        if p == 0:
            key = next(iter(cache))
            again = pdsch(cl, args, drop, p, list(key), state)
            recheck_ok = all(np.array_equal(a, b) for a, b in zip(cache[key], again))
    return rec, evm, recheck_ok


def main():
    args = parse_args()
    sionna.phy.config.device = args.device
    torch.set_grad_enabled(False)
    la_cfg = LinkAdaptationConfig(olla=False)
    here = Path(__file__).resolve().parent
    run = Run(args.out, [here / "run_cqi_bler.py", here / "aware_cqi.py"],
              dict(args=vars(args), link_adaptation=asdict(la_cfg), sources=list(SOURCES)))
    positions_list, sqnr, mismatches = [], {}, 0
    duration = args.periods * args.batch_size

    for cat in args.categories:
        bfw_of = lambda w: compression(args.method, w if args.bfw == "same" else args.bfw)
        cfg = ORANConfig(channel_model="rt", category=cat, num_ue=args.num_ue,
                         num_layers_per_ue=args.layers_per_ue, num_prb=args.num_prb,
                         mcs_table=args.mcs_table, cqi_source="csi-rs", device=args.device,
                         uplane=compression(args.method, args.widths[0]), bfw=bfw_of(args.widths[0]))
        t0 = time.time()
        cl = ClosedLoopDownlink(cfg, la_cfg)
        import mitsuba as mi
        print(f"Cat {cat}：初始化 {time.time() - t0:.1f} s（链路 {args.device}，射线追踪 {mi.variant()}）", flush=True)
        rates = {}
        for w in args.widths:
            cl.set_compression(compression(args.method, w), bfw_of(w))
            rates[w] = fronthaul_rate(cfg, num_prb=args.num_prb).total / 1e9
        for i in range(args.drops):
            t = time.time()
            if len(positions_list) <= i:                    # 第一个 Cat 撒点（drop 已放置 UE 并做射线追踪）
                positions_list.append(cl.drop(args.seed + i))
            else:                                           # 其余 Cat 用同一批位置
                cl.sim.channel.set_ue_positions(positions_list[i])
            positions = positions_list[i]
            rt = cl.rt_channel
            path_gain, snr = rt.path_gain_db(), rt.wideband_snr_db()
            state = cl.sim.channel.sample(args.batch_size)  # 静止 UE：各周期信道相同
            cqi_g, eff_g = cl.measure_cqi_genie(state)
            if i == 0:
                sqnr[cat] = codec_sqnr(cl, state, args.widths)
            line = []
            for w in args.widths:
                cl.set_compression(compression(args.method, w), bfw_of(w))
                rec, evm, recheck_ok = run_width(cl, args, i, state, cqi_g, eff_g)
                mismatches += not recheck_ok
                fh = cl.sim.fronthaul_stats()
                for src in SOURCES:
                    r = {k: np.array(v) for k, v in rec[src].items()}   # [periods, K]
                    for k in range(cfg.num_ue):
                        err, ref = evm[src][:, k]
                        delivered = float(((args.batch_size - r["errors"][:, k]) * r["tbs"][:, k]).sum())
                        run.add(dict(
                            category=cat, drop=i, width=w, source=src, ue=k,
                            position=np.asarray(positions[k]).tolist(),
                            path_gain_db=float(path_gain[k]), snr_db=float(snr[k]),
                            cqi=r["cqi"][:, k].tolist(), eff_sinr_db=r["eff_sinr_db"][:, k].tolist(),
                            mcs=r["mcs"][:, k].tolist(), tbs=r["tbs"][:, k].tolist(),
                            errors=r["errors"][:, k].tolist(), blocks_per_period=args.batch_size,
                            bler=float(r["errors"][:, k].sum() / duration),
                            throughput_mbps=delivered / (duration / cfg.slots_per_second) / 1e6,
                            data_sinr_db=float(10 * np.log10(ref / max(err, 1e-30))),
                            uplane_sqnr_db=finite(fh["uplane"].sqnr_db),
                            bfw_sqnr_db=finite(fh["bfw"].sqnr_db) if "bfw" in fh else None,
                            fronthaul_gbps=rates[w], recheck_ok=recheck_ok))
                bler = {s: np.array(rec[s]["errors"]).sum() / (duration * cfg.num_ue) for s in SOURCES}
                line.append(f"{w}: " + "/".join(f"{bler[s]:.2f}" for s in SOURCES)
                            + ("" if recheck_ok else " 复核不一致！"))
            snr_txt = " ".join(f"{v:5.1f}" for v in snr)
            print(f"Cat {cat} 撒点 {i:3d} | SNR {snr_txt} dB | CQI(精灵) {cqi_g.tolist()} | "
                  f"BLER genie/csi-rs/aware/oracle：{' | '.join(line)} | {time.time() - t:.1f} s", flush=True)
            run.f.flush()
        del cl
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (Path(args.out) / "codec_sqnr.json").write_text(json.dumps(sqnr, indent=2))
    import mitsuba as mi
    run.finish(positions=[np.asarray(p).tolist() for p in positions_list], recheck_mismatches=mismatches,
               mitsuba_variant=mi.variant(), torch_version=torch.__version__,
               gpu=torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None)
    if mismatches:
        print(f"警告：{mismatches} 个（撒点, 位宽）的复核结果不一致，设备不可靠，本次结果不能使用")
    print(f"结果已保存到 {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
