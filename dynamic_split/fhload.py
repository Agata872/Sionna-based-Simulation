"""前传负载：Girycki 等的公式（式 9、10、13）以及加入压缩、CI 和权值后的扩展。

符号与论文一致：7-1 = 波束赋形在 DU、前传传天线级 IQ（≈ 你的 Cat A）；
7-2 = 波束赋形在 RU、前传传 UE 层符号（≈ 你的 Cat B，ZF 时 DU 下发权值）。

论文模式（overhead='paper'）：
    R7-2 = Σ_r (Γ·Σ_{u∈r} ν_u + γ7-2)，R7-1 = Σ_r (Γ·Σ_{n∈r} μ_n + γ7-1)，
    Γ = 50 PRB × 12 × 28 符号/ms × q；q = 32 时 Γ = 537.6 Mbit/s；
    γ7-2 = 713.9、γ7-1 = 121 Mbit/s（TR 38.816 附录 A），对全部 RU 计入（反推论文数字得到）。
扩展模式（overhead='explicit'）：
    U-plane 与 C-plane 按 O-RAN 消息逐 eAxC 计费（oran/fronthaul.py 的 uplane_rate / cplane_rate，
    含 eCPRI 与以太网头），只有参与服务的 RU 有负载；
    7-1 的 eAxC = 服务天线，7-2 的 eAxC = RU 服务的 UE（每 UE 一层）。
压缩（每 RE 比特）：none = 32；bfp9 = (12·18 + 8)/12；modcomp = Qm（CUS v22 附录 A.5，只用于 7-2 的层符号）。
CI 与权值（ci='srs'）：
    上行 CI：7-1 由 RU 转发服务天线上 SRS 符号的天线级 IQ（BFP9）；
             7-2 + ZF 取"天线级 SRS IQ"与"每 UE 每 PRG 的信道估计（BFP9，类似 ST6）"中较小者；
             7-2 + MRT 由 RU 用本地 SRS 自算权值，不需 CI。
    下行权值：7-2 + ZF 用 SE-11 下发 BFP9 权值，每时隙一次（bfw_update='slot'）或每 SRS 时机一次（'srs'）。
"""

import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oran.fronthaul import cplane_rate as _cplane_rate  # noqa: E402
from oran.fronthaul import uplane_rate as _uplane_rate  # noqa: E402


@lru_cache(maxsize=None)
def uplane_rate(*args):
    return _uplane_rate(*args)


@lru_cache(maxsize=None)
def cplane_rate(*args):
    return _cplane_rate(*args)

NUM_ANT = 64
NUM_PRB = 50
SYMBOLS_PER_SLOT = 14
SLOTS_PER_S = 2000                    # 30 kHz
RE_PER_S = NUM_PRB * 12 * SYMBOLS_PER_SLOT * SLOTS_PER_S   # 16.8e6
GAMMA_72, GAMMA_71 = 713.9e6, 121e6   # bit/s，TR 38.816 V15.0.0 附录 A 的 G1


def bits_per_prb(comp, qm=None):
    if comp == 'none':
        return 12 * 32
    if comp.startswith('bfp'):
        return 12 * 2 * int(comp[3:]) + 8
    if comp == 'modcomp':
        return 12 * int(qm)
    raise ValueError(comp)


def bfp_bits(num_values, width=9):
    return num_values * 2 * width + math.ceil(num_values / 12) * 8


def bfw_block_bits(num_weights, width):
    """一个 SE-11 权值块（一个 eAxC 在一个 bundle 上的 N 个权值）的比特数：BFP 带 8 bit 参数，16 bit 为不压缩。"""
    return 32 * num_weights if width >= 16 else num_weights * 2 * width + 8


@dataclass
class Usage:
    """一个撒点、一个状态下的服务情况。"""

    ants: np.ndarray        # [M] 每根天线是否服务
    ue_of_ant: np.ndarray   # [M] MRT 时该天线服务的 UE，ZF 时为 -1（服务全部 UE）
    num_ue: int
    bf: str
    qm: np.ndarray          # [U] 每 UE 的调制阶数（调制压缩用）

    @classmethod
    def from_row(cls, row, n_consider, qm):
        order, incl, assign = (np.asarray(row[k]) for k in ('order', 'incl', 'assign'))
        ants = np.zeros(NUM_ANT, bool)
        ue_of_ant = np.full(NUM_ANT, -1)
        sel = np.arange(n_consider)[incl[:n_consider].astype(bool)]
        ants[order[sel]] = True
        ue_of_ant[order[sel]] = assign[sel]
        return cls(ants, ue_of_ant, row['num_ue'], row['bf'], qm)

    def per_ru(self, k):
        """每个 RU 的服务天线数和所服务 UE 的列表。"""
        n_ru = NUM_ANT // k
        ants = self.ants.reshape(n_ru, k)
        n_ant = ants.sum(1)
        if self.bf == 'zf':
            ues = [list(range(self.num_ue)) if n else [] for n in n_ant]
        else:
            uo = self.ue_of_ant.reshape(n_ru, k)
            ues = [sorted(set(uo[r][ants[r]].tolist())) for r in range(n_ru)]
        return n_ant, ues


def paper_load(usage, k, q=32, overhead_rus='all'):
    """论文式 (10)、(13)。返回 {split: 每 RU 的 bit/s 数组}，网络总负载是其和。"""
    n_ant, ues = usage.per_ru(k)
    gamma = RE_PER_S * q
    serving = n_ant > 0
    oh = np.ones(len(n_ant)) if overhead_rus == 'all' else serving.astype(float)
    return {'7-1': gamma * n_ant + GAMMA_71 * oh,
            '7-2': gamma * np.array([len(u) for u in ues]) + GAMMA_72 * oh}


@dataclass
class Profile:
    """一种扩展计费口径。"""

    name: str
    comp_71: str = 'none'          # 7-1 天线级 IQ 的压缩
    comp_72: str = 'none'          # 7-2 层符号的压缩（'modcomp' 按每 UE 的 Qm）
    ci: str = 'none'               # 'none' 或 'srs'
    t_srs_ms: float = 10.0
    bfw_update: str = 'slot'       # 'slot' 或 'srs'
    bfw_width: int = 9
    prg_prb: int = 2
    ue_per_srs_symbol: int = 16    # comb-4 × 4 个循环移位


def explicit_load(usage, k, prof):
    """逐 RU 按 O-RAN 消息计费。

    返回 {split: {分量: 每 RU 的 bit/s 数组}}，分量为 uplane、cplane、bfw（下行）和 ci（上行），
    另有 dl = uplane + cplane + bfw、ul = ci。
    """
    n_ant, ues = usage.per_ru(k)
    n_ru = len(n_ant)
    n_prg = -(-NUM_PRB // prof.prg_prb)
    srs_symbols = -(-usage.num_ue // prof.ue_per_srs_symbol)
    srs_rate = 1000.0 / prof.t_srs_ms
    out = {}
    for split in ('7-1', '7-2'):
        c = {name: np.zeros(n_ru) for name in ('uplane', 'cplane', 'bfw', 'ci')}
        for r in range(n_ru):
            if n_ant[r] == 0:
                continue
            na = int(n_ant[r])
            # 天线级 SRS IQ（BFP9）：7-1 总要，7-2 + ZF 作为候选之一
            srs_ant = uplane_rate(na, NUM_PRB, bits_per_prb('bfp9'), srs_symbols, 1)[1] * srs_rate
            if split == '7-1':
                c['uplane'][r] = uplane_rate(na, NUM_PRB, bits_per_prb(prof.comp_71), SYMBOLS_PER_SLOT,
                                             SLOTS_PER_S)[1]
                c['cplane'][r] = cplane_rate(na, SLOTS_PER_S)
                if prof.ci == 'srs':
                    c['ci'][r] = srs_ant
                continue
            nu = len(ues[r])
            c['uplane'][r] = sum(uplane_rate(1, NUM_PRB, bits_per_prb(prof.comp_72, usage.qm[u]),
                                             SYMBOLS_PER_SLOT, SLOTS_PER_S)[1] for u in ues[r])
            c['cplane'][r] = cplane_rate(nu, SLOTS_PER_S)
            if usage.bf == 'zf' and prof.ci == 'srs':
                blk = bfw_block_bits(na, prof.bfw_width)
                if prof.bfw_update == 'slot':
                    c['bfw'][r] = cplane_rate(nu, SLOTS_PER_S, n_prg, na, blk) - c['cplane'][r]
                else:
                    c['bfw'][r] = cplane_rate(nu, srs_rate, n_prg, na, blk)
                st6 = bfp_bits(na * usage.num_ue * n_prg) * srs_rate
                c['ci'][r] = min(srs_ant, st6)
        c['dl'] = c['uplane'] + c['cplane'] + c['bfw']
        c['ul'] = c['ci']
        out[split] = c
    return out
