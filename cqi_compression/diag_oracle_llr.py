#!/usr/bin/env python3
"""查明 Q5 结论 6：用 oracle CQI 时，低位宽下首传 BLER 仍超过目标 0.1 的原因。

oracle CQI 在一个经压缩前传的 PDSCH 探测时隙上实测均衡后 SINR（已含全部压缩失真和信道估计误差），
按 AWGN 门限选出的 MCS 本应满足 BLER ≤ 0.1。候选原因：
  1. LLR 失配：UE 的 LMMSE 只按热噪声 N0 算 no_eff（oran/README 设计第 5 条），压缩失真占主导时 LLR 过于自信。
     Sionna 的 LDPC 译码是 sum-product（boxplus-phi），LLR 限幅 ±20，对 LLR 的标度敏感。
  2. 失真不是高斯的，或与数据相关（如 Cat B 直接量化 QAM 层符号），AWGN 门限不再适用。
  3. 探测时隙与数据时隙的 SINR 不同。

做法：oracle CQI 选出 MCS 后，同一批数据时隙（同一接收信号、同一均衡输出）用不同的软解调噪声方差各译码一次：
    thermal  原接收机：LMMSE 的 no_eff（热噪声、同调度干扰、信道估计误差方差，不含压缩失真）
    scalar   no_eff 整体乘一个系数，使每 UE 每时隙的平均值等于实际均衡误差功率（只校正标度）
    prb      每 UE、每层、每 PRB 用实际均衡误差功率（已知发送数据，精灵）
再用同样的比特和热噪声发一次，把 U-plane 压缩误差换成同功率的复高斯噪声（每个天线 / 层、符号、PRB 的功率
与实际误差相同），用 thermal 与 prb 译码（gauss-thermal、gauss-prb）。
    prb 使 BLER 回到 0.1 附近                  -> 原因 1
    prb 仍超标、gauss-prb 回到 0.1 附近         -> 原因 2
另记录数据时隙实际的有效 SINR（每层每 PRB 的 SINR 经 EESM）与 oracle 探测值之差，检验原因 3。

每个（Cat, 位宽）在第一个撒点上检查 thermal 与 ORANDownlink.run_batch 的误块数逐个相同（复用同一接收链）。

用法（先 source env.sh）：
    python cqi_compression/diag_oracle_llr.py --drops 1 --periods 1 --widths ideal 5 --out results/cqi_compression_20261030/diag_smoke
    python cqi_compression/diag_oracle_llr.py --out results/cqi_compression_20261030/diag_llr
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_cqi_bler import cat_label, compression, finite, seed_for  # noqa: E402  （同时隐藏 GPU、设置路径）

import sionna.phy  # noqa: E402
import torch  # noqa: E402

from aware_cqi import measure_cqi_oracle  # noqa: E402
from oran import ClosedLoopDownlink, LinkAdaptationConfig, ORANConfig  # noqa: E402
from runio import Run  # noqa: E402

VARIANTS = ("thermal", "scalar", "prb", "gauss-thermal", "gauss-prb")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--categories", nargs="+", default=["A", "B", "8"], choices=["A", "B", "8"])
    p.add_argument("--method", default="block_scaling", help="Cat A / B 的 U-plane 压缩算法")
    p.add_argument("--o8-method", default="fixed", help="option 8 的时域量化")
    p.add_argument("--widths", nargs="+", default=["ideal", "6", "5", "4"])
    p.add_argument("--drops", type=int, default=20)
    p.add_argument("--periods", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=7)
    p.add_argument("--num-ue", type=int, default=2)
    p.add_argument("--layers-per-ue", type=int, default=2)
    p.add_argument("--num-prb", type=int, default=51)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--rt-llvm", action="store_true")
    p.add_argument("--out", required=True)
    return p.parse_args()


def prb_of_data(sim) -> torch.Tensor:
    """每个数据 RE（ResourceGridMapper 的输入顺序）所在的 PRB [num_data_symbols]。"""
    cfg = sim.cfg
    n = sim.rg.num_data_symbols
    idx = torch.arange(1, n + 1, dtype=torch.float32, device=sim.device)
    x = torch.complex(idx, torch.zeros_like(idx)).expand(1, cfg.num_ue, cfg.num_layers_per_ue, n)
    grid = sim.odu.rg_mapper(x)[0, 0, 0].real                      # [num_symbols, num_sc]
    pos = torch.nonzero(grid > 0)
    prb = torch.empty(n, dtype=torch.long, device=sim.device)
    prb[grid[pos[:, 0], pos[:, 1]].long() - 1] = pos[:, 1] // 12
    return prb


def transmit_gauss(sim, batch_size, state, gen):
    """ORANDownlink._transmit（噪声方差 1）的副本，U-plane 压缩误差换成同功率的复高斯噪声。

    功率按每个天线 / 层、每个符号、每个 PRB（12 个 RE）取实际误差的平均。高斯噪声用独立的随机数发生器，
    比特和热噪声与 _transmit 相同。Cat B 的权值仍用实际压缩后的值。
    """
    cfg = sim.cfg
    tx = sim.odu(batch_size, sim.channel.csi(state, sim.csi_symbol))
    grids = []
    for o in range(cfg.num_oru):
        u = tx.uplane[o]
        u_hat, w_hat = sim.fronthaul[o].transmit(u, tx.weights[o])
        p = ((u_hat - u).abs() ** 2).unflatten(-1, (-1, 12)).mean(-1, keepdim=True)
        z = torch.complex(torch.randn(p.shape[:-1] + (12,), generator=gen, device=u.device),
                          torch.randn(p.shape[:-1] + (12,), generator=gen, device=u.device))
        u_hat = u + (z * (p / 2).sqrt()).flatten(-2).to(u.dtype)
        grids.append(sim.orus[o].to_air(sim.orus[o].frequency_domain(u_hat, w_hat)))
    return tx, sim.channel(torch.stack(grids, dim=1), state, 1.0)


@torch.no_grad()
def equalize(sim, y):
    """UEReceivers 的信道估计与 LMMSE 均衡，返回 (x_hat, no_eff) [B, K, ν, N]。"""
    ue = sim.ue
    y = ue.frequency_grid(y)
    no = torch.tensor(1.0, dtype=y.real.dtype, device=y.device)
    h_hat, err_var = ue.estimator(y, no)
    return ue.equalizer(y, h_hat, err_var, no)


@torch.no_grad()
def decode(sim, x_hat, no_var) -> np.ndarray:
    """按给定的软解调噪声方差译码，返回各 UE 的出错 TB 数 [K]。"""
    ue = sim.ue
    errors = []
    for k in range(sim.cfg.num_ue):
        llr = ue.demappers[k](x_hat[:, k:k + 1], no_var[:, k:k + 1])
        _, crc = ue.tb_decoders[k](ue.layer_demappers[k](llr))
        errors.append(int((~crc.reshape(-1)).sum()))
    return np.array(errors)


def per_prb(v, prb, num_prb):
    """[B, K, ν, N] 按 PRB 求平均 -> [B, K, ν, num_prb]。"""
    out = torch.zeros(*v.shape[:-1], num_prb, dtype=v.dtype, device=v.device).index_add_(-1, prb, v)
    count = torch.bincount(prb, minlength=num_prb).to(v.dtype)
    return out / count


@torch.no_grad()
def data_slots(cl, tx, x_hat, no_eff, prb):
    """数据时隙的实际误差：返回 (各方案的噪声方差 dict, 每 UE 的有效 SINR dB, 每 UE 的 LLR 过度自信量 dB)。"""
    num_prb = cl.cfg.num_prb
    e2 = (x_hat - tx.x_data.to(x_hat.dtype)).abs() ** 2                       # [B, K, ν, N]
    e2 = e2.to(no_eff.dtype)
    var_prb = per_prb(e2, prb, num_prb).clamp_min(1e-12)                     # [B, K, ν, P]
    scale = e2.sum((2, 3)) / no_eff.sum((2, 3))                               # [B, K]
    sig_prb = per_prb((tx.x_data.abs() ** 2).to(no_eff.dtype), prb, num_prb)
    sinr = (sig_prb / var_prb).permute(1, 2, 0, 3)                           # [K, ν, B, P]
    _, eff = cl.reporter(sinr)
    overconf = 10 * np.log10((e2.sum((0, 2, 3)) / no_eff.sum((0, 2, 3))).cpu().numpy())
    variants = {"thermal": no_eff, "scalar": no_eff * scale[:, :, None, None], "prb": var_prb[..., prb]}
    return variants, eff, overconf


def run_period(cl, args, drop, period, state, prb, check):
    sim = cl.sim
    sionna.phy.config.seed = seed_for(args, drop, period, 3)
    cqi, eff_oracle = measure_cqi_oracle(cl, state)
    mcs = cl.la.select_mcs(cqi)
    sim.set_mcs(mcs)
    seed = seed_for(args, drop, period, 2)

    sionna.phy.config.seed = seed
    tx, y, _ = sim._transmit(args.batch_size, 1.0, state)
    x_hat, no_eff = equalize(sim, y)
    var, eff_data, overconf = data_slots(cl, tx, x_hat, no_eff, prb)
    errors = {v: decode(sim, x_hat, var[v]) for v in ("thermal", "scalar", "prb")}

    sionna.phy.config.seed = seed
    gen = torch.Generator(device=sim.device).manual_seed(seed)
    tx_g, y_g = transmit_gauss(sim, args.batch_size, state, gen)
    x_hat_g, no_eff_g = equalize(sim, y_g)
    var_g, eff_gauss, _ = data_slots(cl, tx_g, x_hat_g, no_eff_g, prb)
    errors["gauss-thermal"] = decode(sim, x_hat_g, var_g["thermal"])
    errors["gauss-prb"] = decode(sim, x_hat_g, var_g["prb"])

    check_ok = None
    if check:                                   # thermal 必须与原接收机逐个相同
        sionna.phy.config.seed = seed
        stats = sim.run_batch(args.batch_size, 0.0, state=state)
        check_ok = bool(np.array_equal(stats.block_errors.astype(int), errors["thermal"]))
    return dict(cqi=cqi, eff_oracle=eff_oracle, mcs=mcs, eff_data=eff_data, eff_gauss=eff_gauss,
                overconf=overconf, errors=errors, check_ok=check_ok)


def main():
    args = parse_args()
    sionna.phy.config.device = args.device
    torch.set_grad_enabled(False)
    la_cfg = LinkAdaptationConfig(olla=False)
    here = Path(__file__).resolve().parent
    run = Run(args.out, [here / "diag_oracle_llr.py", here / "run_cqi_bler.py", here / "aware_cqi.py"],
              dict(args=vars(args), link_adaptation={"olla": False}, variants=list(VARIANTS)))
    positions_list, checks = [], []
    for cat in args.categories:
        method = args.o8_method if cat == "8" else args.method
        uplane_of = lambda w: compression(method, w)
        bfw_of = lambda w: compression(args.method, w) if cat == "B" else None
        cfg = ORANConfig(channel_model="rt", category=cat, num_ue=args.num_ue,
                         num_layers_per_ue=args.layers_per_ue, num_prb=args.num_prb, mcs_table=2,
                         device=args.device, uplane=uplane_of(args.widths[0]),
                         bfw=bfw_of(args.widths[0]) or compression(args.method, "ideal"))
        cl = ClosedLoopDownlink(cfg, la_cfg)
        prb = prb_of_data(cl.sim)
        for i in range(args.drops):
            t = time.time()
            if len(positions_list) <= i:
                positions_list.append(cl.drop(args.seed + i))
            else:
                cl.sim.channel.set_ue_positions(positions_list[i])
            snr = cl.rt_channel.wideband_snr_db()
            state = cl.sim.channel.sample(args.batch_size)
            line = []
            for w in args.widths:
                cl.set_compression(uplane_of(w), bfw_of(w))
                errs = {v: 0 for v in VARIANTS}
                for p in range(args.periods):
                    r = run_period(cl, args, i, p, state, prb, check=(i == 0 and p == 0))
                    if r["check_ok"] is not None:
                        checks.append((cat, w, r["check_ok"]))
                    sqnr = finite(cl.sim.fronthaul_stats()["uplane"].sqnr_db)
                    for k in range(cfg.num_ue):
                        run.add(dict(
                            category=cat, drop=i, width=w, period=p, ue=k, snr_db=float(snr[k]),
                            cqi=int(r["cqi"][k]), eff_oracle_db=float(r["eff_oracle"][k]),
                            cqi_threshold_db=float(cl.reporter.threshold_db[max(int(r["cqi"][k]), 1) - 1]),
                            mcs=int(r["mcs"][k]), mcs_threshold_db=float(cl.la.mcs_threshold_db[r["mcs"][k]]),
                            eff_data_db=float(r["eff_data"][k]), eff_gauss_db=float(r["eff_gauss"][k]),
                            overconf_db=float(r["overconf"][k]), uplane_sqnr_db=sqnr,
                            blocks=args.batch_size, **{f"err_{v}": int(r["errors"][v][k]) for v in VARIANTS}))
                    for v in VARIANTS:
                        errs[v] += int(r["errors"][v].sum())
                n = args.periods * args.batch_size * cfg.num_ue
                line.append(f"{w}: " + "/".join(f"{errs[v] / n:.2f}" for v in VARIANTS))
            print(f"{cat_label(cat)} 撒点 {i:3d} | BLER {'/'.join(VARIANTS)}：{' | '.join(line)} | "
                  f"{time.time() - t:.1f} s", flush=True)
            run.f.flush()
        del cl
    bad = [c for c in checks if not c[2]]
    run.finish(positions=[np.asarray(p).tolist() for p in positions_list],
               thermal_check=[list(c) for c in checks], thermal_check_failures=len(bad))
    if bad:
        print(f"警告：thermal 与原接收机不一致：{bad}")
    print(f"结果已保存到 {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
