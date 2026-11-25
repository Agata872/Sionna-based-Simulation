#!/usr/bin/env python
"""第二阶段：7-1（DU 预编码后的天线 IQ）下逐 PRB 的 RU 掩码 + 压缩方法/位宽联合选择。

运行：../../../../.venv/bin/python -B phase2.py --workers 6
输出：results/phase2/{paired.npz, candidates.npz, summary.json, run.log, phase2_source.py}

模型（每项都写进 summary.json["setup"]）：
  * M = 16 个单天线 RU（4×4 方格），K = 4 个 UE，B = 4 个相邻 PRB，D = 32 个布局 d = 0..31。
    “掩码”是活跃 RU 的子集；逐天线功率限制就是逐 RU 限制（每 RU 每 PRB 10 mW）。
  * 信道：四径、完美 CSI，每个 PRB 内平坦（取 PRB 中心频点）；噪声 −70 dBm（每 RE，与每 RE 平均发射功率同口径）。
    scheduler.py / phase1.py 在本脚本落地时都不存在，信道由本脚本生成（见 make_channel）；
    可用 --channels X.npz（键 H，形状 [D, B, K, M]，单位 sqrt(mW/mW)）换成第一阶段的信道。
  * 预编码：W = pinv(H_S)，共同归一化 c = sqrt(P / max_m ||W_m||²)（逐天线限制）。
  * 压缩：oran.fronthaul.IQCodec，BFP / Block Scaling，q ∈ {5, 7, 9}，块 = 12 RE（一个 PRB 一个符号）。
    净荷 14·(24q+8) bit / RU-PRB（两种方法同一公式，运行时由压缩器的 bits_per_block 断言）。
  * 解压后再查功率：p̂_m = mean_RE |x̂_m|²，若 max p̂_m > P，整个候选统一乘 g = sqrt(P / max p̂_m)（记录 g）。
  * SE：r_k = h_k^T (g x̂)；a_k = <r_k, s_k>/<s_k, s_k>（相关增益）；e_k = r_k − a_k s_k
    （剩余多用户干扰 + 压缩失真）；SINR_k = |a_k|² mean|s_k|² / (mean|e_k|² + σ²)；逐 PRB sum SE = Σ_k log2(1 + SINR_k)。
    TWC 口径（Girycki Eq.6-8，bit/s/Hz，已含 PRB 平均，不乘 B）：twc_se = Σ_k log2(1 + Σ_b S_kb / Σ_b (I_kb + σ²))。
    调度器用的是逐 PRB SE 的原始和 sum_se = Σ_b Σ_k log2(1 + SINR_kb)（单位：bit/s/Hz 的 PRB 累加，与 twc_se 不同量纲，不可直接比较）。
  * 容量：每 RU 每时隙 k × 14·(24·9+8) bit，k ∈ {2, 3} → 6272 / 9408 bit。主口径 = 实际压缩净荷；
    旁证口径 = 固定按 PRB 节段打包 + eCPRI/以太网报头（见 ecpri_bits）。
"""

import argparse
import itertools
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT_DIR = HERE / "results" / "phase2"

SEED = 20261125
CHANNEL_TAG = 0xC4A1           # 信道种子：SeedSequence([SEED, CHANNEL_TAG, d])；符号种子：SeedSequence([SEED, d, b])
NUM_DROPS = 32
B = 4                          # 相邻 PRB
K = 4                          # UE
M = 16                         # 单天线 RU
NSYM = 14
NSC = 12
NRE = NSYM * NSC
P_MW = 10.0                    # 每 RU 每 PRB（每 RE 平均）功率上限
NOISE_DBM = -70.0
NOISE_MW = 10 ** (NOISE_DBM / 10)
CAPS = {2: 2 * 14 * (24 * 9 + 8), 3: 3 * 14 * (24 * 9 + 8)}   # 6272, 9408
assert CAPS == {2: 6272, 3: 9408}

CODECS = [("bfp", 5), ("bfp", 7), ("bfp", 9),
          ("block_scaling", 5), ("block_scaling", 7), ("block_scaling", 9)]
C = len(CODECS)
FIX = CODECS.index(("bfp", 9))
ALGOS = ["fallback", "greedy_fixed", "greedy_adaptive", "joint_fixed", "joint_adaptive"]

# 信道参数（本脚本自带的信道；--channels 时不用）
CH = dict(side_m=100.0, h_ru_m=10.0, h_ue_m=1.5, fc_ghz=3.5, scs_hz=30e3, min_dist_m=5.0,
          shadow_db=4.0, num_paths=4, max_excess_delay_s=300e-9, decay_s=100e-9,
          pathloss="TR 38.901 UMi LOS PL1: 32.4 + 21 log10(d3D) + 20 log10(fc/GHz) + N(0, 4 dB)")

# eCPRI 旁证口径（字节，与 oran/fronthaul.py 的常数一致）
ETH_OVERHEAD = 42              # 前导码+SFD 8 + MAC 14 + VLAN 4 + FCS 4 + IFG 12
ECPRI_HEADER = 8
APP_HEADER = 4                 # U-plane 应用层公共头（每条消息）
SECTION_HEADER = 6             # 段头 4 + udCompHdr 1 + reserved 1（每个 PRB 一个段）

QAM_LEVELS = np.arange(-7, 8, 2) / np.sqrt(42.0)


def payload_bits(q):
    return NSYM * (24 * q + 8)


# ---------------------------------------------------------------------------
# 信道与符号
# ---------------------------------------------------------------------------

def ru_positions():
    """真实 E8 RU 位置：same_total.ru_sites(1)（需可 import；与 audit_joint 同目录体系）。"""
    import audit_joint  # sets up the shared E8 simulation import path
    import same_total
    ru = np.asarray(same_total.ru_sites(1), float)
    assert ru.shape[0] == M, ru.shape
    return ru[:, :2]


def _local_grid_positions():
    g = int(round(np.sqrt(M)))
    coords = (np.arange(g) + 0.5) * CH["side_m"] / g
    return np.array([(x, y) for y in coords for x in coords])


def make_channel(d):
    """[B, K, M] 复信道（幅度，|h|² 为线性功率增益）及布局信息。"""
    rng = np.random.default_rng(np.random.SeedSequence([SEED, CHANNEL_TAG, d]))
    ru = _local_grid_positions()                                     # 旧自带信道（已不再使用）
    ue = []
    while len(ue) < K:
        p = rng.uniform(0, CH["side_m"], 2)
        if np.min(np.linalg.norm(ru - p, axis=1)) >= CH["min_dist_m"]:
            ue.append(p)
    ue = np.array(ue)
    dh = np.linalg.norm(ue[:, None, :] - ru[None, :, :], axis=-1)
    d3 = np.sqrt(dh ** 2 + (CH["h_ru_m"] - CH["h_ue_m"]) ** 2)
    pl_db = 32.4 + 21 * np.log10(d3) + 20 * np.log10(CH["fc_ghz"]) + rng.normal(0, CH["shadow_db"], (K, M))
    L = CH["num_paths"]
    tau_ex = np.concatenate([np.zeros((K, M, 1)),
                             np.sort(rng.uniform(0, CH["max_excess_delay_s"], (K, M, L - 1)), -1)], -1)
    pw = np.exp(-tau_ex / CH["decay_s"])
    pw /= pw.sum(-1, keepdims=True)
    a = np.sqrt(pw / 2) * (rng.standard_normal((K, M, L)) + 1j * rng.standard_normal((K, M, L)))
    tau = tau_ex + d3[..., None] / 299792458.0
    f = (np.arange(B) - (B - 1) / 2) * NSC * CH["scs_hz"]
    H = 10 ** (-pl_db / 20)[None] * np.sum(a[None] * np.exp(-2j * np.pi * f[:, None, None, None] * tau[None]), -1)
    return H, ue, pl_db


def get_channel(d, channels_path):
    if channels_path:
        with np.load(channels_path) as z:
            H = np.asarray(z["H"][d], dtype=np.complex128)
        assert H.shape == (B, K, M), H.shape
        return H, np.full((K, 2), np.nan), np.full((K, M), np.nan)
    import audit_joint
    H = np.asarray(audit_joint.channel(d, "taps", 0.)[0], dtype=np.complex128)   # E8 共享信道
    assert H.shape == (B, K, M), H.shape
    return H, np.full((K, 2), np.nan), np.full((K, M), np.nan)       # 可选元数据缺失 -> NaN


def symbols_for(d, b):
    """同一布局同一 PRB 的全部候选共用的 64-QAM 符号（自然映射 I = idx // 8, Q = idx % 8，单位平均能量）。"""
    rng = np.random.default_rng(np.random.SeedSequence([SEED, d, b]))
    idx = rng.integers(0, 64, size=(K, NSYM, NSC))
    s = QAM_LEVELS[idx // 8] + 1j * QAM_LEVELS[idx % 8]
    return idx.astype(np.int8), s


# ---------------------------------------------------------------------------
# 候选掩码
# ---------------------------------------------------------------------------

def zf_c2(Hb, masks):
    """理想 ZF（无压缩）的共同功率系数 c² = P / max_m ||W_m||²；masks: [n_masks, n] RU 索引。"""
    Hs = np.transpose(Hb[:, masks], (1, 0, 2))                       # [n_masks, K, n]
    W = np.linalg.pinv(Hs)                                            # [n_masks, n, K]
    return P_MW / (np.abs(W) ** 2).sum(-1).max(-1)


def make_score(H, b=None):
    """理想 ZF sum SE 评分：b 为 None 时取 B 个 PRB 的平均。"""
    def score(masks):
        masks = np.asarray(masks)
        prbs = range(B) if b is None else [b]
        return np.mean([K * np.log2(1 + zf_c2(H[bb], masks) / NOISE_MW) for bb in prbs], axis=0)
    return score


def beam_search(score, allowed, width, keep):
    allowed = sorted(allowed)
    out = {}
    cand = list(itertools.combinations(allowed, K))
    for n in range(K, len(allowed) + 1):
        sc = score(np.array(cand))
        order = np.argsort(-sc, kind="stable")
        beam = [cand[i] for i in order[:width]]
        out[n] = [cand[i] for i in order[:keep]]
        cand = sorted({tuple(sorted(S + (m,))) for S in beam for m in allowed if m not in S})
        if not cand:
            break
    return out


def disjoint_partition(score, allowed):
    remaining = sorted(allowed)
    parts = []
    while len(remaining) >= K:
        cand = list(itertools.combinations(remaining, K))
        best = cand[int(np.argmax(score(np.array(cand))))]
        parts.append(best)
        remaining = [m for m in remaining if m not in best]
    return parts


def fallback_masks():
    """确定的可行备用组合：4×4 方格上的 4 条循环对角线，互不相交、各 4 个 RU，PRB b 用第 b 条。"""
    g = int(round(np.sqrt(M)))
    assert g == K == B, "备用组合按 4×4 方格、K = B = 4 定义"
    return [tuple(sorted(r * g + (r + j) % g for r in range(g))) for j in range(B)]


def build_pool(H, width, keep, max_masks):
    pool, tags = [], []

    def add(S, tag):
        S = tuple(sorted(int(m) for m in S))
        if S in pool:
            tags[pool.index(S)] += "|" + tag
        elif len(pool) < max_masks:
            pool.append(S)
            tags.append(tag)

    fb = fallback_masks()
    for j, S in enumerate(fb):
        add(S, f"fallback{j}")
    scorers = [("avg", make_score(H))] + [(f"prb{b}", make_score(H, b)) for b in range(B)]
    beams = {name: beam_search(sc, range(M), width, keep) for name, sc in scorers}
    for n in range(K, M + 1):                              # 每个天线数的高得分掩码
        for name, _ in scorers:
            for r, S in enumerate(beams[name].get(n, [])):
                add(S, f"top_{name}_n{n}_r{r}")
    for name, sc in scorers:                               # 不同活跃 RU：互不相交的 size-K 划分
        for j, S in enumerate(disjoint_partition(sc, range(M))):
            add(S, f"disjoint_{name}_{j}")
    counts = np.zeros(M)
    for S in pool:
        counts[list(S)] += 1
    hot = [int(m) for m in np.argsort(-counts, kind="stable")[:K]]
    cool = [m for m in range(M) if m not in hot]
    for name, sc in scorers:                               # 避开热点 RU
        bs = beam_search(sc, cool, width, 1)
        for n in range(K, min(2 * K, len(cool)) + 1):
            for S in bs.get(n, []):
                add(S, f"nohot_{name}_n{n}")
    return pool, tags, fb, hot


# ---------------------------------------------------------------------------
# 前传与链路
# ---------------------------------------------------------------------------

def make_codec(method, q, nominal):
    import torch
    from oran.config import CompressionConfig
    from oran.fronthaul import IQCodec
    codec = IQCodec(CompressionConfig(method=method, iq_width=q), 1.0, block_len=NSC)
    assert codec.bits_per_block * NSYM == payload_bits(q), (method, q, codec.bits_per_block)
    nominal = np.full_like(np.asarray(nominal, float), P_MW)          # 固定 10 mW 标称，每天线相同，无额外校准
    codec.set_nominal_power(torch.as_tensor(nominal, dtype=torch.float64).reshape(-1, 1, 1))
    return codec


def precode(HS, s2):
    W = np.linalg.pinv(HS)                                            # [n, K]
    r = (np.abs(W) ** 2).sum(1)
    c = np.sqrt(P_MW / r.max())
    return W, c, c * (W @ s2), c * c * r                              # x: [n, NRE]，设计功率 c²r


def link_metrics(HS, xh, s2):
    p = np.mean(np.abs(xh) ** 2, axis=1)
    g = min(1.0, float(np.sqrt(P_MW / p.max())))
    xt = g * xh
    r = HS @ xt                                                       # [K, NRE] 无噪接收
    es = np.sum(np.abs(s2) ** 2, 1)
    a = np.sum(r * np.conj(s2), 1) / es
    e = r - a[:, None] * s2
    S = np.abs(a) ** 2 * es / NRE
    I = np.mean(np.abs(e) ** 2, 1)
    sinr = S / (I + NOISE_MW)
    return dict(se=float(np.sum(np.log2(1 + sinr))), g=g, p_pre=p, p_tx=g * g * p, x_tx=xt,
                gain=a, S=S, I=I, sinr=sinr)


def evaluate_prb(Hb, s, pool):
    """批量评估一个 PRB 上全部候选（掩码 × 码法）。返回 se [L, C], g [L, C], se_ideal [L], c [L]。"""
    import torch
    s2 = s.reshape(K, NRE)
    L = len(pool)
    xs, noms, cs, offs = [], [], np.zeros(L), [0]
    for j, S in enumerate(pool):
        _, c, x, nom = precode(Hb[:, list(S)], s2)
        xs.append(x)
        noms.append(nom)
        cs[j] = c
        offs.append(offs[-1] + len(S))
    X = np.concatenate(xs)
    Xt = torch.from_numpy(X.reshape(-1, NSYM, NSC).copy())
    nom = np.concatenate(noms)
    se = np.zeros((L, C))
    g = np.zeros((L, C))
    se_ideal = np.zeros(L)
    for j, S in enumerate(pool):
        se_ideal[j] = link_metrics(Hb[:, list(S)], xs[j], s2)["se"]
    for ci, (meth, q) in enumerate(CODECS):
        Xh = make_codec(meth, q, nom)(Xt).numpy().reshape(-1, NRE)
        for j, S in enumerate(pool):
            m = link_metrics(Hb[:, list(S)], Xh[offs[j]:offs[j + 1]], s2)
            se[j, ci], g[j, ci] = m["se"], m["g"]
    return se, g, se_ideal, cs


def evaluate_single(Hb, s, S, method, q):
    """单个候选的独立路径：重建波束、压缩包（定点整数 + udCompParam）与链路指标。"""
    import torch
    from oran.compression import from_blocks, to_blocks
    S = list(S)
    HS = Hb[:, S]
    s2 = s.reshape(K, NRE)
    W, c, x, nom = precode(HS, s2)
    codec = make_codec(method, q, nom)
    Xt = torch.from_numpy(x.reshape(-1, NSYM, NSC).copy())
    xh = codec(Xt).numpy().reshape(-1, NRE)
    x_int, nclip = codec.fixed_point.quantize(Xt)
    pkt = codec.compressor.compress(to_blocks(x_int, NSC))
    xh2 = codec.fixed_point.dequantize(from_blocks(codec.compressor.decompress(pkt))).numpy().reshape(-1, NRE)
    assert np.array_equal(xh, xh2), "IQCodec 与显式压缩包解压不一致"
    m = link_metrics(HS, xh, s2)
    ideal = link_metrics(HS, x, s2)
    param = pkt.param.reshape(len(S), NSYM).numpy() if pkt.param is not None else np.zeros((len(S), NSYM))
    return dict(W=W, c=c, x=x, m=m, se_ideal=ideal["se"], g_ideal=ideal["g"], nclip=int(nclip),
                pkt_i=pkt.i.reshape(len(S), NSYM, NSC).numpy(), pkt_q=pkt.q.reshape(len(S), NSYM, NSC).numpy(),
                pkt_param=param, fp_scale=codec.fixed_point.scale.reshape(-1).numpy())


# ---------------------------------------------------------------------------
# 选择算法（scheduler.py 落地后自动使用其实现）
# ---------------------------------------------------------------------------

def _feasible(costs, idx, budget):
    used = sum(costs[b][i] for b, i in enumerate(idx))
    return bool(np.all(used <= budget)), used


def local_joint_select(values, costs, budget, fallback=None, time_limit=120.0, **_):
    """候选集上的精确联合选择：每 PRB 恰选一个候选，逐 RU 净荷 ≤ budget（scipy HiGHS MILP）。"""
    from scipy.optimize import Bounds, LinearConstraint, milp
    t0 = time.perf_counter()
    nb = len(values)
    sizes = [len(v) for v in values]
    offs = np.concatenate([[0], np.cumsum(sizes)])
    A_eq = np.zeros((nb, offs[-1]))
    for b in range(nb):
        A_eq[b, offs[b]:offs[b + 1]] = 1
    A_cap = np.concatenate([np.asarray(c, float).T for c in costs], axis=1)
    res = milp(-np.concatenate(values), integrality=np.ones(offs[-1]), bounds=Bounds(0, 1),
               constraints=[LinearConstraint(A_eq, 1, 1), LinearConstraint(A_cap, -np.inf, np.asarray(budget, float))],
               options={"time_limit": time_limit, "mip_rel_gap": 1e-6, "disp": False})
    names = {0: "optimal", 1: "limit_reached", 2: "infeasible", 3: "unbounded", 4: "other"}
    status = names.get(res.status, str(res.status))
    if res.x is None:
        idx = list(fallback) if fallback is not None else None
        status += "->fallback" if idx is not None else ""
        gap = None
    else:
        idx = [int(np.argmax(res.x[offs[b]:offs[b + 1]])) for b in range(nb)]
        gap = float(getattr(res, "mip_gap", np.nan)) if getattr(res, "mip_gap", None) is not None else None
    value = float(sum(values[b][i] for b, i in enumerate(idx))) if idx is not None else float("nan")
    return dict(indices=idx, value=value, status=status, gap=gap, runtime_s=time.perf_counter() - t0)


def local_greedy_select(values, costs, budget, fallback=None, order="regret", **_):
    """逐块贪心。order="regret"：按（最佳候选值 − 备用候选值）降序处理 PRB；每块取剩余容量下值最大的候选，
    且要求其余未处理 PRB 仍能用各自的备用候选完成（可行性保护）。"""
    t0 = time.perf_counter()
    nb = len(values)
    remaining = np.asarray(budget, np.int64).copy()
    if order == "regret" and fallback is not None:
        prio = [float(np.max(values[b]) - values[b][fallback[b]]) for b in range(nb)]
        seq = list(np.argsort(-np.array(prio), kind="stable"))
    else:
        seq = list(range(nb))
    idx = [None] * nb
    status = "ok"
    for pos, b in enumerate(seq):
        rest = seq[pos + 1:]
        need = sum((costs[r][fallback[r]] for r in rest), np.zeros_like(remaining)) if fallback is not None \
            else np.zeros_like(remaining)
        for i in np.argsort(-np.asarray(values[b]), kind="stable"):
            if np.all(costs[b][i] + need <= remaining):
                idx[b] = int(i)
                break
        if idx[b] is None:
            status = "infeasible"
            break
        remaining = remaining - costs[b][idx[b]]
    value = float(sum(values[b][i] for b, i in enumerate(idx))) if status == "ok" else float("nan")
    return dict(indices=idx, value=value, status=status, gap=None, runtime_s=time.perf_counter() - t0,
                order=[int(b) for b in seq])


def load_scheduler():
    if (HERE / "scheduler.py").exists():
        try:
            import scheduler
            return scheduler.joint_select, scheduler.greedy_select, "scheduler.py"
        except Exception as exc:                                       # 接口未完成时退回内置实现
            return local_joint_select, local_greedy_select, f"phase2.py built-in (scheduler.py import failed: {exc!r})"
    return local_joint_select, local_greedy_select, "phase2.py built-in (scheduler.py absent)"


def call_selector(fn, values, costs, budget, fallback, **opts):
    t0 = time.perf_counter()
    res = None
    for kw in (dict(fallback=fallback, **opts), dict(fallback=fallback), {}):
        try:
            res = fn(values, costs, budget, **kw)
            break
        except TypeError:
            continue
    if res is None:
        raise RuntimeError("selector rejected all call signatures")
    res = dict(res)
    res.setdefault("runtime_s", time.perf_counter() - t0)
    res.setdefault("gap", None)
    ok, _ = _feasible(costs, res["indices"], budget) if res.get("indices") is not None else (False, None)
    if not ok:
        res["status"] = f"{res.get('status')}->infeasible_result_replaced_by_fallback"
        res["indices"] = list(fallback)
        res["value"] = float(sum(values[b][i] for b, i in enumerate(fallback)))
    return res


# ---------------------------------------------------------------------------
# 单个布局（worker）
# ---------------------------------------------------------------------------

def _worker_init():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)


def ecpri_bits(mask_rows, width_rows):
    """旁证：每 RU 每时隙，每个活跃 PRB 一个段，每符号一条消息（以太网 + eCPRI + 应用层头）。"""
    active = mask_rows.any(0)
    sec = sum(mask_rows[b] * (8 * SECTION_HEADER + 24 * width_rows[b] + 8) for b in range(len(width_rows)))
    return (NSYM * (8 * (ETH_OVERHEAD + ECPRI_HEADER + APP_HEADER) * active + sec)).astype(np.int64)


def process_drop(task):
    d, opts = task
    t_all = time.perf_counter()
    joint_fn, greedy_fn, sched_src = load_scheduler()
    H, ue_xy, pl_db = get_channel(d, opts["channels"])
    syms = [symbols_for(d, b) for b in range(B)]

    t0 = time.perf_counter()
    pool, tags, fb, hot = build_pool(H, opts["beam_width"], opts["keep"], opts["max_masks"])
    t_pool = time.perf_counter() - t0
    L = len(pool)
    mask_bool = np.zeros((L, M), bool)
    for j, S in enumerate(pool):
        mask_bool[j, list(S)] = True
    fb_j = [pool.index(S) for S in fb]

    t0 = time.perf_counter()
    cand = [evaluate_prb(H[b], syms[b][1], pool) for b in range(B)]
    t_eval = time.perf_counter() - t0
    cand_se = np.stack([c[0] for c in cand])                            # [B, L, C]
    cand_g = np.stack([c[1] for c in cand])
    cand_se_ideal = np.stack([c[2] for c in cand])
    cand_c = np.stack([c[3] for c in cand])

    bits = np.array([payload_bits(q) for _, q in CODECS], np.int64)
    cost_adapt = (mask_bool[:, None, :] * bits[None, :, None]).reshape(L * C, M).astype(np.int64)
    cost_fixed = (mask_bool * bits[FIX]).astype(np.int64)
    problems = {
        "adaptive": ([cand_se[b].reshape(-1) for b in range(B)], [cost_adapt] * B,
                     [fb_j[b] * C + FIX for b in range(B)], lambda i: divmod(int(i), C)),
        "fixed": ([cand_se[b][:, FIX].copy() for b in range(B)], [cost_fixed] * B,
                  list(fb_j), lambda i: (int(i), FIX)),
    }

    sels = {}
    for k, cap in CAPS.items():
        budget = np.full(M, cap, np.int64)
        for kind, (vals, costs, fbi, _) in problems.items():
            assert _feasible(costs, fbi, budget)[0], "备用组合不可行"
        for algo in ALGOS:
            if algo == "fallback":
                vals, costs, fbi, dec = problems["fixed"]
                res = dict(indices=list(fbi), value=float(sum(vals[b][i] for b, i in enumerate(fbi))),
                           status="fallback", gap=0.0, runtime_s=0.0)
            else:
                kind_sel, kind = algo.split("_")
                vals, costs, fbi, dec = problems[kind]
                if kind_sel == "joint":
                    if kind == "fixed":                            # 备用 = 同预算的 greedy_fixed
                        fbi = list(sels[(k, "greedy_fixed")]["indices"])
                    else:                                          # 备用 = max(greedy_adaptive, joint_fixed 映射)
                        ga = list(sels[(k, "greedy_adaptive")]["indices"])
                        jf = [int(i) * C + FIX for i in sels[(k, "joint_fixed")]["indices"]]
                        va = lambda ix: sum(vals[b][i] for b, i in enumerate(ix))
                        fbi = jf if va(jf) > va(ga) else ga
                    res = call_selector(joint_fn, vals, costs, budget, fbi, time_limit=opts["time_limit"],
                                        mip_rel_gap=1e-6)
                else:
                    res = call_selector(greedy_fn, vals, costs, budget, fbi, order=opts["greedy_order"])
            bound = float(sum(np.max(v) for v in vals))
            res["choice"] = [dec(i) for i in res["indices"]]
            res["bound_relaxed_budget"] = bound
            res["used_bits"] = _feasible(costs, res["indices"], budget)[1].astype(np.int64)
            sels[(k, algo)] = res

    # 选中候选的独立重算 + 原始数据
    cache = {}
    max_se_diff = 0.0
    rows = []
    for (k, algo), res in sels.items():
        mrows = np.zeros((B, M), bool)
        wrows = np.zeros(B, np.int64)
        per = []
        for b, (j, ci) in enumerate(res["choice"]):
            key = (b, j, ci)
            if key not in cache:
                cache[key] = evaluate_single(H[b], syms[b][1], pool[j], *CODECS[ci])
                max_se_diff = max(max_se_diff, abs(cache[key]["m"]["se"] - cand_se[b, j, ci]))
            per.append(cache[key])
            mrows[b] = mask_bool[j]
            wrows[b] = CODECS[ci][1]
        S_kb = np.stack([e["m"]["S"] for e in per])                       # [B, K]
        I_kb = np.stack([e["m"]["I"] for e in per])
        twc = float(np.sum(np.log2(1 + S_kb.sum(0) / (I_kb + NOISE_MW).sum(0))))
        se_sum = float(sum(e["m"]["se"] for e in per))
        ecpri = ecpri_bits(mrows, wrows)
        for b, ((j, ci), e) in enumerate(zip(res["choice"], per)):
            S = list(pool[j])
            full = lambda a, shape, dt, fill=0: _scatter(a, S, shape, dt, fill)
            rows.append(dict(
                drop=d, k=k, capacity_bits=CAPS[k], algorithm=algo, prb=b, mask_index=j, cand_codec=ci,
                mask=mask_bool[j], mask_size=len(S), width=CODECS[ci][1], codec=CODECS[ci][0],
                link_bits=(mask_bool[j] * payload_bits(CODECS[ci][1])).astype(np.int64),
                power_mw=full(e["m"]["p_tx"], (M,), np.float64),
                power_pre_scale_mw=full(e["m"]["p_pre"], (M,), np.float64),
                power_scale=e["m"]["g"], precoder_c=e["c"],
                precoder=full(e["c"] * e["W"], (M, K), np.complex128),
                se_per_prb=e["m"]["se"], se_ideal_per_prb=e["se_ideal"], sinr=e["m"]["sinr"],
                signal_mw=e["m"]["S"], intdist_mw=e["m"]["I"], gain=e["m"]["gain"],
                twc_se=twc, sum_se=se_sum, value=res["value"], mip_gap=_f(res.get("gap")),
                status=str(res.get("status")), runtime_s=float(res["runtime_s"]),
                bound=res["bound_relaxed_budget"], used_bits=res["used_bits"], ecpri_bits=ecpri,
                num_clipped=e["nclip"],
                x_pre=full(e["x"].reshape(-1, NSYM, NSC), (M, NSYM, NSC), np.complex64),
                x_tx=full(e["m"]["x_tx"].reshape(-1, NSYM, NSC), (M, NSYM, NSC), np.complex64),
                packet_i=full(e["pkt_i"], (M, NSYM, NSC), np.int16),
                packet_q=full(e["pkt_q"], (M, NSYM, NSC), np.int16),
                packet_param=full(e["pkt_param"], (M, NSYM), np.int16, fill=-1),
                fp_scale=full(e["fp_scale"], (M,), np.float64),
                greedy_order=np.array(res.get("order", [-1] * B), np.int64),
            ))
    return dict(drop=d, H=H, ue_xy=ue_xy, pl_db=pl_db, qam_idx=np.stack([s[0] for s in syms]),
                pool_masks=mask_bool, pool_tags=tags, hot=hot, fallback_mask_index=fb_j,
                cand_se=cand_se, cand_g=cand_g, cand_se_ideal=cand_se_ideal, cand_c=cand_c,
                rows=rows, max_batch_single_se_diff=max_se_diff, scheduler=sched_src,
                timing=dict(pool_s=t_pool, eval_s=t_eval, total_s=time.perf_counter() - t_all))


def _scatter(a, S, shape, dt, fill=0):
    out = np.full(shape, fill, dtype=dt)
    out[S] = a
    return out


def _f(x):
    return float("nan") if x is None else float(x)


# ---------------------------------------------------------------------------
# 统计、验证、主程序
# ---------------------------------------------------------------------------

def ci95(x):
    from scipy import stats
    x = np.asarray(x, float)
    n = len(x)
    m = float(x.mean())
    s = float(x.std(ddof=1)) if n > 1 else 0.0
    h = float(stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)) if n > 1 else float("nan")
    return dict(mean=m, std=s, ci95=[m - h, m + h], min=float(x.min()), max=float(x.max()), n=n)


def reconstruct_check(z, r, channels_path):
    """只用 paired.npz 的原始记录（+ 种子）重建一行并复算 SE。"""
    import torch
    from oran.compression import CompressedIQ, make_compressor
    d, b = int(z["drop"][r]), int(z["prb"][r])
    H = get_channel(d, channels_path)[0]
    idx, s = symbols_for(d, b)
    out = dict(row=int(r), drop=d, prb=b, algorithm=str(z["algorithm"][r]), k=int(z["k"][r]),
               channel_equal=bool(np.allclose(H, z["H"][d])), symbols_equal=bool(np.array_equal(idx, z["qam_idx"][d, b])))
    S = np.flatnonzero(z["mask"][r])
    meth, q = str(z["codec"][r]), int(z["width"][r])
    comp = make_compressor(meth, iq_width=q)
    pkt = CompressedIQ(meth, q, comp.param_bits,
                       torch.from_numpy(z["packet_i"][r][S].astype(np.int32)).reshape(len(S), NSYM, 1, NSC),
                       torch.from_numpy(z["packet_q"][r][S].astype(np.int32)).reshape(len(S), NSYM, 1, NSC),
                       torch.from_numpy(z["packet_param"][r][S].astype(np.int64)).reshape(len(S), NSYM, 1)
                       if comp.param_bits else None)
    xh = comp.decompress(pkt).numpy().reshape(len(S), NRE) / z["fp_scale"][r][S][:, None]
    xt = z["power_scale"][r] * xh
    out["packet_vs_x_tx_max_abs"] = float(np.max(np.abs(xt - z["x_tx"][r][S].reshape(len(S), NRE))))
    out["packet_bits_match"] = bool(pkt.total_bits == z["link_bits"][r].sum())
    m = link_metrics(H[b][:, S], xh, s.reshape(K, NRE))
    out["se_from_packet"] = m["se"]
    out["se_stored"] = float(z["se_per_prb"][r])
    out["se_abs_diff"] = abs(m["se"] - out["se_stored"])
    beam = z["precoder"][r][S]
    out["precoder_zf_residual"] = float(np.max(np.abs(H[b][:, S] @ beam - z["precoder_c"][r] * np.eye(K))))
    e = evaluate_single(H[b], s, S, meth, q)
    out["independent_rerun_se_diff"] = abs(e["m"]["se"] - out["se_stored"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--drops", type=int, default=NUM_DROPS)
    ap.add_argument("--beam-width", type=int, default=8)
    ap.add_argument("--keep", type=int, default=2, help="每个天线数、每个评分口径保留的高得分掩码数")
    ap.add_argument("--max-masks", type=int, default=200)
    ap.add_argument("--time-limit", type=float, default=4.0)
    ap.add_argument("--output", default=str(OUT_DIR), help="输出目录（默认沿用 results/phase2）")
    ap.add_argument("--greedy-order", default="regret", choices=["regret", "index"])
    ap.add_argument("--channels", default=None, help="可选：第一阶段信道 npz（键 H [D, B, K, M]）")
    ap.add_argument("--num-check", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    t_start = time.perf_counter()
    opts = dict(beam_width=args.beam_width, keep=args.keep, max_masks=args.max_masks, time_limit=args.time_limit,
                greedy_order=args.greedy_order, channels=args.channels)
    tasks = [(d, opts) for d in range(args.drops)]
    log(f"phase2: {args.drops} drops, workers={args.workers}, opts={opts}")
    results = []
    if args.workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=_worker_init) as ex:
            for res in ex.map(process_drop, tasks):
                results.append(res)
                log(f"  drop {res['drop']:2d}: pool {len(res['pool_masks'])} masks, "
                    f"{res['timing']['total_s']:.1f} s, scheduler = {res['scheduler']}")
    else:
        _worker_init()
        for t in tasks:
            res = process_drop(t)
            results.append(res)
            log(f"  drop {res['drop']:2d}: pool {len(res['pool_masks'])} masks, {res['timing']['total_s']:.1f} s")
    results.sort(key=lambda r: r["drop"])
    t_compute = time.perf_counter() - t_start

    # ---- paired.npz ----
    rows = [r for res in results for r in res["rows"]]
    keys = rows[0].keys()
    arr = {}
    for key in keys:
        vals = [r[key] for r in rows]
        if isinstance(vals[0], str):
            arr[key] = np.array(vals, dtype="U64")
        else:
            arr[key] = np.stack([np.asarray(v) for v in vals])
    arr["antenna_power_mw"] = arr["power_mw"]                    # 单天线 RU：天线功率 = RU 功率
    arr["H"] = np.stack([res["H"] for res in results])
    arr["qam_idx"] = np.stack([res["qam_idx"] for res in results])
    arr["ue_xy"] = np.stack([res["ue_xy"] for res in results])
    arr["ru_xy"] = ru_positions()
    arr["pl_db"] = np.stack([res["pl_db"] for res in results])
    arr["noise_mw"] = np.array(NOISE_MW)
    arr["p_max_mw"] = np.array(P_MW)
    arr["qam_levels"] = QAM_LEVELS
    np.savez(out_dir / "paired.npz", **arr)

    Lmax = max(len(res["pool_masks"]) for res in results)
    D = len(results)
    cm = np.zeros((D, Lmax, M), bool)
    cse = np.full((D, B, Lmax, C), np.nan)
    cg = np.full((D, B, Lmax, C), np.nan)
    cid = np.full((D, B, Lmax), np.nan)
    ctag = np.full((D, Lmax), "", dtype="U400")
    for i, res in enumerate(results):
        L = len(res["pool_masks"])
        cm[i, :L] = res["pool_masks"]
        cse[i, :, :L] = res["cand_se"]
        cg[i, :, :L] = res["cand_g"]
        cid[i, :, :L] = res["cand_se_ideal"]
        ctag[i, :L] = res["pool_tags"]
    np.savez(out_dir / "candidates.npz", masks=cm, se=cse, power_scale=cg, se_ideal=cid, tags=ctag,
             num_masks=np.array([len(r["pool_masks"]) for r in results]),
             fallback_mask_index=np.array([r["fallback_mask_index"] for r in results]),
             codecs=np.array([f"{m}{q}" for m, q in CODECS]))

    # ---- 验证 ----
    z = dict(np.load(out_dir / "paired.npz"))
    sel_keys = sorted({(int(a), int(b), str(c)) for a, b, c in zip(z["drop"], z["k"], z["algorithm"])})
    cap_viol = 0
    ecpri_over = {}
    util = {}
    for d, k, algo in sel_keys:
        rs = np.flatnonzero((z["drop"] == d) & (z["k"] == k) & (z["algorithm"] == algo))
        used = z["link_bits"][rs].sum(0)
        cap_viol += int(np.any(used > CAPS[k]))
        ecpri_over.setdefault((k, algo), []).append(bool(np.any(z["ecpri_bits"][rs[0]] > CAPS[k])))
        util.setdefault((k, algo), []).append(float(used.sum() / (M * CAPS[k])))
    pmax = float(z["power_mw"].max())
    all_cand_g = np.concatenate([r["cand_g"].ravel() for r in results])
    rng = np.random.default_rng(SEED)
    check_rows = sorted(rng.choice(len(z["drop"]), size=min(args.num_check, len(z["drop"])), replace=False).tolist())
    recon = [reconstruct_check(z, r, args.channels) for r in check_rows]
    twc_err = 0.0
    for d, k, algo in sel_keys:
        rs = np.flatnonzero((z["drop"] == d) & (z["k"] == k) & (z["algorithm"] == algo))
        Sk, Ik = z["signal_mw"][rs].sum(0), (z["intdist_mw"][rs] + NOISE_MW).sum(0)
        twc_err = max(twc_err, abs(np.sum(np.log2(1 + Sk / Ik)) - z["twc_se"][rs[0]]))

    # ---- 聚合 ----
    def per_sel(field, k, algo):
        return np.array([z[field][np.flatnonzero((z["drop"] == d) & (z["k"] == k) & (z["algorithm"] == algo))[0]]
                         for d in range(D)])

    metrics, paired, opt = {}, {}, {}
    order_viol = []
    for k in CAPS:
        mk = {}
        for algo in ALGOS:
            rsel = (z["k"] == k) & (z["algorithm"] == algo)
            mk[algo] = dict(
                sum_se=ci95(per_sel("sum_se", k, algo)), twc_se=ci95(per_sel("twc_se", k, algo)),
                se_ideal_same_selection=ci95([z["se_ideal_per_prb"][(z["drop"] == d) & rsel].sum() for d in range(D)]),
                mean_mask_size=float(z["mask_size"][rsel].mean()),
                codec_share={f"{m}{q}": float(np.mean((z["codec"][rsel] == m) & (z["width"][rsel] == q))) for m, q in CODECS},
                payload_utilization=ci95(util[(k, algo)]),
                ecpri_packing_exceeds_capacity_frac=float(np.mean(ecpri_over[(k, algo)])),
                power_scale_min=float(z["power_scale"][rsel].min()),
                power_scaled_frac=float(np.mean(z["power_scale"][rsel] < 1)),
                runtime_s=ci95(per_sel("runtime_s", k, algo)),
                status_counts={s: int(np.sum(per_sel("status", k, algo) == s)) for s in set(per_sel("status", k, algo))},
            )
        metrics[f"k={k}"] = mk
        pk = {}
        for a, b_ in [("greedy_adaptive", "greedy_fixed"), ("joint_adaptive", "joint_fixed"),
                      ("joint_fixed", "greedy_fixed"), ("joint_adaptive", "greedy_adaptive"),
                      ("joint_fixed", "fallback"), ("joint_adaptive", "fallback")]:
            for f in ("sum_se", "twc_se"):
                diff = per_sel(f, k, a) - per_sel(f, k, b_)
                pk[f"{a} - {b_} [{f}]"] = dict(**ci95(diff), wins=int(np.sum(diff > 1e-9)), losses=int(np.sum(diff < -1e-9)))
        paired[f"k={k}"] = pk
        ok = {}
        for kind in ("fixed", "adaptive"):
            jv, gv, bd = per_sel("value", k, f"joint_{kind}"), per_sel("value", k, f"greedy_{kind}"), per_sel("bound", k, f"joint_{kind}")
            gaps = per_sel("mip_gap", k, f"joint_{kind}")
            ok[kind] = dict(mip_gap_max=float(np.nanmax(gaps)) if np.any(np.isfinite(gaps)) else None,
                            mip_gap_nan_count=int(np.sum(~np.isfinite(gaps))),
                            all_optimal=bool(np.all(per_sel("status", k, f"joint_{kind}") == "optimal")),
                            greedy_relative_gap_to_joint=ci95((jv - gv) / jv),
                            greedy_equals_joint_frac=float(np.mean(np.abs(jv - gv) < 1e-9)),
                            joint_gap_to_budget_relaxation_bound=ci95((bd - jv) / bd))
            order_viol += [f"k={k} {kind} d={d}: greedy > joint" for d in range(D) if gv[d] > jv[d] + 1e-7]
        order_viol += [f"k={k} d={d}: joint_fixed > joint_adaptive" for d in range(D)
                       if per_sel("value", k, "joint_fixed")[d] > per_sel("value", k, "joint_adaptive")[d] + 1e-7]
        opt[f"k={k}"] = ok

    n_masks = np.array([len(r["pool_masks"]) for r in results])
    size_hist = np.bincount(np.concatenate([r["pool_masks"].sum(1) for r in results]), minlength=M + 1)
    summary = dict(
        script="phase2.py", date=time.strftime("%Y-%m-%d %H:%M:%S"), python=sys.version.split()[0],
        platform=platform.platform(), argv=sys.argv, scheduler_impl=sorted({r["scheduler"] for r in results}),
        setup=dict(
            drops=D, num_prb=B, num_ue=K, num_ru=M, antennas_per_ru=1,
            interpretation=("RU = 单天线 AP（4 个互不相交的 size-4 备用掩码、每 RU 仅用 1 个 BFP9 PRB = 3136 bit ≤ 6272，"
                            "与委托一致）；掩码 = 活跃 RU 子集；逐天线限制 = 逐 RU 限制。paired.npz 的 k = 容量档"
                            "（BFP9 PRB 等价数 2/3），PRB 索引在 prb 字段。"),
            channel=("--channels " + args.channels) if args.channels else
            dict(source='audit_joint.channel(d, "taps", 0.)[0]（E8 共享信道）', ru_positions="same_total.ru_sites(1)",
                 note="ue_xy / pl_db 不可得时为 NaN"),
            csi="perfect", noise_dbm=NOISE_DBM, noise_mw=NOISE_MW, p_max_mw_per_ru_per_prb=P_MW,
            power_unit="每 RE 平均功率（168 RE 实测平均），噪声同口径",
            precoder="W = pinv(H_S); c = sqrt(P / max_m ||W_m||^2); x = c W s",
            symbols="64-QAM 单位能量，自然映射 idx -> (I = idx//8, Q = idx%8)；全部候选共用",
            seeds=dict(base=SEED, channel="SeedSequence([20261125, 0xC4A1, d])", symbols="SeedSequence([20261125, d, b])",
                       check_rows=f"default_rng(20261125)"),
            codecs=[f"{m}{q}" for m, q in CODECS], fixed_codec="bfp9", block_len_re=12, fixed_point="16 bit, backoff 15 dB 相对固定标称功率：每天线 10 mW（np.full_like(nominal, P_MW)，无额外校准）",
            payload_formula="14*(24q+8) bit per active RU per PRB（BFP 与 Block Scaling 相同，运行时断言）",
            capacities_bits={f"k={k}": v for k, v in CAPS.items()},
            capacity_main_metric="实际压缩净荷 link_bits（容量约束按此执行）",
            ecpri_side_metric=dict(note="旁证，不参与约束：每活跃 RU-PRB 一个段，每 RU 每符号一条消息",
                                   eth_bytes=ETH_OVERHEAD, ecpri_bytes=ECPRI_HEADER, app_header_bytes=APP_HEADER,
                                   section_bytes=SECTION_HEADER, formula="14*(8*(42+8+4)*1{RU active} + sum_b 1{active_b}*(48 + 24q_b + 8))"),
            power_check="解压后 p̂_m = mean|x̂_m|^2；若 max > P，整个候选乘 g = sqrt(P / max p̂_m)（所有方案同一规则，含理想参考）",
            se="a_k = <r_k,s_k>/<s_k,s_k>; I_k = mean|r_k - a_k s_k|^2（剩余多用户干扰 + 失真）; SINR = |a_k|^2 mean|s_k|^2 / (I_k + σ²)",
            twc_se="Girycki Eq.6-8：sum_k log2(1 + sum_b S_kb / sum_b (I_kb + σ²))，无 B 因子，同一选择复算（bit/s/Hz，PRB 平均口径）",
            sum_se_units="sum_se = 逐 PRB SE 的原始和（调度器目标，PRB 累加口径）；twc_se 为 TWC 口径，二者单位/量纲不同，不可直接比较",
            fallback_chain="joint_fixed 备用 = greedy_fixed（同预算）；joint_adaptive 备用 = greedy_adaptive 与 joint_fixed 映射（j*C+FIX）中较优者",
            algorithms=dict(fallback="对角线备用组合 + BFP9", greedy_fixed="逐块贪心，仅 BFP9", greedy_adaptive="逐块贪心，6 种码法",
                            joint_fixed="候选集上精确 MILP，仅 BFP9", joint_adaptive="候选集上精确 MILP，6 种码法"),
            greedy_ordering=("共享 scheduler 的 greedy_select 在全部 24 种 PRB 排列中取最优，result['order'] 存入 greedy_order；"
                             f"仅在内置回退实现下才使用 --greedy-order={args.greedy_order}"),
            candidate_pool=dict(beam_width=args.beam_width, keep_per_size=args.keep, max_masks=args.max_masks,
                                sources="备用对角线；每天线数 n=4..16 的 beam search 高得分（平均 + 逐 PRB 评分，理想 ZF sum SE）；"
                                        "逐评分的互不相交 size-4 划分；避开 4 个最热 RU 的 n=4..8 掩码",
                                num_masks=ci95(n_masks), mask_size_hist={int(n): int(c) for n, c in enumerate(size_hist) if c},
                                candidates_per_prb_adaptive=[int(n_masks.min() * C), int(n_masks.max() * C)],
                                candidates_per_prb_fixed=[int(n_masks.min()), int(n_masks.max())],
                                total_masks_possible=int(sum(len(list(itertools.combinations(range(M), n))) for n in range(K, M + 1)))),
        ),
        optimality_statement=("joint_* 为候选集上的精确最优（HiGHS MILP，mip_rel_gap = 1e-6，见 optimality.*.all_optimal / mip_gap_max）；"
                              "不是对全部 RU 子集的最优。joint_gap_to_budget_relaxation_bound 为去掉前传约束（每块取最佳候选）的上界间隙。"),
        optimality=opt, metrics=metrics, paired_differences=paired,
        fixed_baseline={f"k={k}": dict(fallback=metrics[f"k={k}"]["fallback"]["sum_se"],
                                       greedy_fixed=metrics[f"k={k}"]["greedy_fixed"]["sum_se"],
                                       joint_fixed=metrics[f"k={k}"]["joint_fixed"]["sum_se"]) for k in CAPS},
        power_scaling=dict(all_candidates_min_g=float(all_cand_g.min()), all_candidates_scaled_frac=float(np.mean(all_cand_g < 1)),
                           selected_min_g=float(z["power_scale"].min()), selected_scaled_frac=float(np.mean(z["power_scale"] < 1))),
        validation=dict(
            capacity_violations=cap_viol, selections_checked=len(sel_keys),
            max_power_mw=pmax, power_violations=int(np.sum(z["power_mw"] > P_MW * (1 + 1e-9))),
            max_batch_vs_single_se_diff=float(max(r["max_batch_single_se_diff"] for r in results)),
            twc_recompute_max_abs_err=float(twc_err), ordering_violations=order_viol, reconstruction=recon,
            num_clipped_total=int(z["num_clipped"].sum())),
        timing=dict(total_s=time.perf_counter() - t_start, compute_s=t_compute,
                    per_drop_total_s=ci95([r["timing"]["total_s"] for r in results]),
                    per_drop_pool_s=ci95([r["timing"]["pool_s"] for r in results]),
                    per_drop_eval_s=ci95([r["timing"]["eval_s"] for r in results])),
        files=dict(paired="paired.npz: 每行 = (drop, k, algorithm, prb)；x_tx 为 RU 实际发射的频域 IQ（解压 × g），"
                          "packet_i/q/param + fp_scale 为压缩包（x_tx = g · decompress(packet) / fp_scale），precoder = c·W（未含 g）",
                   candidates="candidates.npz: 每布局的掩码池、逐 PRB 候选 SE / g / 理想 SE"),
    )
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    shutil.copy2(__file__, out_dir / "phase2_source.py")

    log("\n== results (sum SE over 4 PRBs, bit/s/Hz, mean [95% CI]) ==")
    for k in CAPS:
        for algo in ALGOS:
            m = metrics[f"k={k}"][algo]
            log(f"k={k} {algo:16s} sumSE {m['sum_se']['mean']:7.3f} [{m['sum_se']['ci95'][0]:.3f}, {m['sum_se']['ci95'][1]:.3f}]"
                f"  twc {m['twc_se']['mean']:7.3f}  |S| {m['mean_mask_size']:.2f}  util {m['payload_utilization']['mean']:.3f}"
                f"  eCPRI>cap {m['ecpri_packing_exceeds_capacity_frac']:.2f}")
        for kind, o in opt[f"k={k}"].items():
            log(f"k={k} joint_{kind}: all_optimal={o['all_optimal']} mip_gap_max={o['mip_gap_max']} "
                f"greedy gap {o['greedy_relative_gap_to_joint']['mean']:.4f} bound gap {o['joint_gap_to_budget_relaxation_bound']['mean']:.4f}")
    v = summary["validation"]
    log(f"\nvalidation: cap violations {v['capacity_violations']}/{v['selections_checked']}, max power {v['max_power_mw']:.6f} mW "
        f"(violations {v['power_violations']}), batch/single SE diff {v['max_batch_vs_single_se_diff']:.2e}, "
        f"twc err {v['twc_recompute_max_abs_err']:.2e}, ordering violations {len(order_viol)}")
    for r in recon:
        log(f"  recon row {r['row']}: d={r['drop']} b={r['prb']} {r['algorithm']} k={r['k']} se diff {r['se_abs_diff']:.2e} "
            f"rerun {r['independent_rerun_se_diff']:.2e} pkt {r['packet_vs_x_tx_max_abs']:.2e} bits {r['packet_bits_match']} "
            f"H {r['channel_equal']} sym {r['symbols_equal']}")
    log(f"total {summary['timing']['total_s']:.1f} s; outputs in {out_dir}")
    (out_dir / "run.log").write_text("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
