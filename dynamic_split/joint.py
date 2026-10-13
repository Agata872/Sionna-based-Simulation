"""阶段 3（方向 ②+③）：在论文框架内联合选择服务集合、每个 RU 的切分和压缩。

场景与阶段 0 相同（64 根分布式单天线、K 根一组为一个 RU、ZF/MRT 按状态选，同论文）。
每个 RU 的选项：
  7-1：U-plane BFP4–12 或 16 bit；上行 CI = 服务天线的天线级 SRS IQ（BFP9）
  7-2：层符号调制压缩（无损）；ZF 时 DU 用 SE-11 下发权值，位宽 BFP4–12 或 16 bit，CI = min(天线级 SRS IQ, ST 6)；
       MRT 时 RU 用本地 SRS 自算权值，无权值、无 CI
优化器里的失真模型（由 oran/compression 的真实编解码器标定，见 calibrate）：
  7-1 RU r：每根服务天线叠加白噪声，方差 = 该天线每 PRB 功率 × r71(w) + 16 bit 定点底；
            UE u 收到 D = Σ_{m∈r} |h_um|² P_m r71(w)
  7-2 RU r（ZF）：权值按 BFP 规则解析计算误差（bfp_block_var：块内最大值定指数，误差方差 = 步长²/12），
            UE u 收到其他 UE 波束的泄漏 D = Σ_{u'≠u} var_{r,u'} Σ_{m∈r} |h_um|²
每个 RU 的选择用拉格朗日法：最小化 R_r(o) + λ Σ_{b,u} g_bu D_r,bu(o)，g 为 Shannon 和速率对失真的导数（线性化），
各 RU 独立；扫描 λ 得到"和速率–负载"的一族点。可行性用真实的频谱效率模型（MCS 截顶或 Shannon）检查。
服务集合两条路径：
  paper：论文的逐根考察（阶段 0 记录的顺序与取舍）
  la   ：负载感知贪心——每步在剩余天线（MRT 时还要选 UE）中选"理想和速率增量 / 参考负载增量"最大的一个，
         参考负载 = 每个 RU 在参考精度（7-1 BFP9、7-2 调制压缩 + 权值 BFP9）下两种切分中较小者
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import cfsim  # noqa: E402
import distortion as dz  # noqa: E402
import fhload  # noqa: E402

UP71 = [f'bfp{w}' for w in range(4, 13)] + ['none']
BFW = list(range(4, 13)) + [16]
KAPPA = 0.98
KAPPA_S = [0.98, 0.984, 0.988, 0.992, 0.996, 1.0]
LAMBDAS = np.logspace(-3, 3, 31)            # Gbit/s 每 bit/s/Hz
SRS_PERIOD_MS = 10.0
UE_PER_SRS_SYMBOL = 16
LOG2N = np.arange(7)                         # 权值块长 1..64


def calibrate(seed=0):
    """用真实编解码器测相对失真：r71[opt]（块长 12）、rw[b][log2 n]，以及 16 bit 定点底（相对标称功率）。"""
    rng = np.random.default_rng(seed)
    p = cfsim.P_MAX_PRB_MW
    r71 = {}
    for opt in UP71:
        x = np.sqrt(p / 2) * (rng.standard_normal((400, 168)) + 1j * rng.standard_normal((400, 168)))
        r71[opt] = float(np.mean(np.abs(dz.apply(opt, x, p, 12) - x) ** 2) / p)
    rw = {}
    for b in BFW:
        opt = 'none' if b >= 16 else f'bfp{b}'
        row = []
        for ln in LOG2N:
            n = 2 ** ln
            x = np.sqrt(p / 20) * (rng.standard_normal((2000, n)) + 1j * rng.standard_normal((2000, n)))
            row.append(float(np.mean(np.abs(dz.apply(opt, x, p / 10, n) - x) ** 2) / (p / 10)))
        rw[b] = np.array(row)
    return dict(r71=r71, rw=rw, floor=r71['none'])


def bfp_block_var(w_blk, width, nominal, backoff_db=15.0):
    """SE-11 权值块的 BFP 量化误差方差（每个复元素）。w_blk [B, n, U]：每个 (PRB, UE) 的 n 个权值为一块。
    与 oran/compression/bfp.py 的规则一致：16 bit 定点化（标称功率 + 回退）后，
    指数 e = max(floor(log2(maxV)) + 1 - width + 1, 0)，四舍五入误差在 ±2^e/2 内均匀分布；width ≥ 16 为不压缩。"""
    scale = 2.0 ** 15 * 10 ** (-backoff_db / 20) / np.sqrt(nominal / 2)
    x = w_blk * scale
    maxv = np.maximum(np.abs(x.real), np.abs(x.imag)).max(axis=1)              # [B, U]
    maxv = np.maximum(np.round(maxv), 1.0)
    if width >= 16:
        e = np.zeros_like(maxv)
    else:
        e = np.maximum(np.floor(np.log2(maxv)) + 1 - width + 1, 0)
    step2 = 4.0 ** e
    return 2 * (step2 / 12 + 1 / 12 * (e == 0)) / scale ** 2                 # 两个实分量


def rw_of(cal, b, n):
    return float(np.interp(np.log2(max(n, 1)), LOG2N, cal['rw'][b]))


# ---------------------------------------------------------------------------- 负载（每 RU，bit/s）

def _up(n_eaxc, bits_prb):
    return fhload.uplane_rate(int(n_eaxc), fhload.NUM_PRB, int(bits_prb), fhload.SYMBOLS_PER_SLOT,
                              fhload.SLOTS_PER_S)[1]


def _srs_raw(n_ant, num_ue):
    sym = -(-num_ue // UE_PER_SRS_SYMBOL)
    return fhload.uplane_rate(int(n_ant), fhload.NUM_PRB, fhload.bits_per_prb('bfp9'), sym, 1)[1] * 1000.0 / SRS_PERIOD_MS


def ru_load(split, opt, n_ant, ues, qm, bf, num_ue, uplane72='modcomp'):
    """一个 RU 的 (下行, 上行) 负载。opt：7-1 为 U-plane 选项，7-2 为权值位宽（MRT 时忽略）。"""
    if n_ant == 0:
        return 0.0, 0.0
    if split == '7-1':
        return (_up(n_ant, fhload.bits_per_prb(opt)) + fhload.cplane_rate(int(n_ant), fhload.SLOTS_PER_S),
                _srs_raw(n_ant, num_ue))
    nu = len(ues)
    if uplane72 == 'modcomp':
        dl = sum(_up(1, 12 * qm[u]) for u in ues)
    else:
        dl = nu * _up(1, fhload.bits_per_prb(uplane72))
    if bf == 'zf':
        dl += fhload.cplane_rate(nu, fhload.SLOTS_PER_S, fhload.NUM_PRB, int(n_ant), fhload.bfw_block_bits(int(n_ant), opt))
        st6 = fhload.bfp_bits(int(n_ant) * num_ue * fhload.NUM_PRB) * 1000.0 / SRS_PERIOD_MS
        return dl, min(_srs_raw(n_ant, num_ue), st6)
    return dl + fhload.cplane_rate(nu, fhload.SLOTS_PER_S), 0.0


# ---------------------------------------------------------------------------- 一个撒点

class DropModel:
    def __init__(self, h, bf, noise_mw, se_model, cal):
        self.h, self.bf, self.noise, self.model, self.cal = h, bf, noise_mw, se_model, cal
        self.h2 = np.abs(h) ** 2                                       # [B, U, M]
        self.num_ue = h.shape[1]

    # -- 理想状态
    def state(self, serving, assign_s):
        serving = np.asarray(serving)
        b_, u_, m_ = self.h.shape
        w = np.zeros((b_, m_, u_), complex)
        w[:, serving, :] = dz.ideal_weights(self.h[:, :, serving], self.bf, assign_s)
        g = self.h @ w
        pw = np.abs(g) ** 2
        sig = np.einsum('buu->bu', pw)
        st = dict(serving=serving, assign=np.asarray(assign_s), w=w, sig=sig, intf=pw.sum(-1) - sig,
                  p_ant=np.sum(np.abs(w) ** 2, axis=2))               # [B, M]
        se_ue = cfsim.se_from_sinr(sig / (st['intf'] + self.noise), self.model).mean(0)
        st['se_ideal'] = float(se_ue.sum())
        st['qm'] = cfsim.qm_from_se(se_ue, self.model)
        # Shannon 线性化的失真权重：-∂SE/∂D
        x = st['intf'] + self.noise
        st['grad'] = sig / (x * (x + sig)) / np.log(2) / b_
        return st

    def se(self, st, d):
        return float(cfsim.se_from_sinr(st['sig'] / (st['intf'] + d + self.noise), self.model).mean(0).sum())

    # -- 每个 RU 的选项：(split, opt) -> (下行+上行负载, 最忙方向负载, 失真 [B, U], Σ g·失真)
    def ru_options(self, st, k):
        cal = self.cal
        out = []
        ru = st['serving'] // k
        for r in np.unique(ru):
            idx = st['serving'][ru == r]
            n = len(idx)
            h2 = self.h2[:, :, idx]                                    # [B, U, n]
            if self.bf == 'zf':
                ues = list(range(self.num_ue))
            else:
                ues = sorted(set(st['assign'][ru == r].tolist()))
            a71 = np.einsum('bun,bn->bu', h2, st['p_ant'][:, idx])
            floor71 = np.einsum('bun->bu', h2) * cfsim.P_MAX_PRB_MW * cal['floor']
            opts = []
            for opt in UP71:
                dl, ul = ru_load('7-1', opt, n, ues, st['qm'], self.bf, self.num_ue)
                dist = a71 * cal['r71'][opt] + floor71
                opts.append((('7-1', opt), dl + ul, max(dl, ul), dist, float(np.sum(st['grad'] * dist))))
            if self.bf == 'zf':
                h2sum = np.einsum('bun->bu', h2)                           # [B, U]
                for b in BFW:
                    dl, ul = ru_load('7-2', b, n, ues, st['qm'], self.bf, self.num_ue)
                    var = bfp_block_var(st['w'][:, idx, :], b, cfsim.P_MAX_PRB_MW / self.num_ue)   # [B, U']
                    dist = (var.sum(1, keepdims=True) - var) * h2sum       # 只计其他 UE 波束的泄漏
                    opts.append((('7-2', b), dl + ul, max(dl, ul), dist, float(np.sum(st['grad'] * dist))))
            else:
                dl, ul = ru_load('7-2', None, n, ues, st['qm'], self.bf, self.num_ue)
                opts.append((('7-2', None), dl + ul, max(dl, ul), np.zeros_like(a71), 0.0))
            out.append((int(r), opts))
        return out

    def lagrange(self, st, ru_opts, lam):
        """每个 RU 选 argmin R + λ Σ g D。返回 (和速率, 总负载 Gbit/s, 最忙链路 Gbit/s, 7-2 的 RU 比例, 选择)。"""
        d = np.zeros_like(st['sig'])
        total, link, n72, choice = 0.0, 0.0, 0, {}
        for r, opts in ru_opts:
            j = int(np.argmin([o[1] / 1e9 + lam * o[4] for o in opts]))
            key, ld, lk, dist, _ = opts[j]
            d += dist
            total += ld
            link = max(link, lk)
            n72 += key[0] == '7-2'
            choice[r] = key
        return self.se(st, d), total / 1e9, link / 1e9, n72 / max(len(ru_opts), 1), choice

    def uniform(self, st, ru_opts, key):
        """所有 RU 用同一个选项（B1）。"""
        d = np.zeros_like(st['sig'])
        total, link = 0.0, 0.0
        for r, opts in ru_opts:
            o = next(o for o in opts if o[0] == key)
            d += o[3]
            total += o[1]
            link = max(link, o[2])
        return self.se(st, d), total / 1e9, link / 1e9


# ---------------------------------------------------------------------------- 服务集合路径

def paper_path(row):
    """阶段 0 的逐根考察：返回每个位置 n（考察 n 根）的 (服务天线, MRT 的 UE 分配)。"""
    order, incl, assign = (np.asarray(row[k]) for k in ('order', 'incl', 'assign'))
    out = []
    for n in range(1, len(order) + 1):
        sel = np.arange(n)[incl[:n].astype(bool)]
        out.append((order[sel], assign[sel]))
    return out


def _ref_ru_load(n_ant, ues, qm, bf, num_ue):
    """参考精度下（7-1 BFP9、7-2 调制压缩 + 权值 BFP9）一个 RU 两种切分中较小的总负载。"""
    if n_ant == 0:
        return 0.0
    a = sum(ru_load('7-1', 'bfp9', n_ant, ues, qm, bf, num_ue))
    b = sum(ru_load('7-2', 9, n_ant, ues, qm, bf, num_ue))
    return min(a, b)


def la_path(model, order, k, se_model, score_mode='load'):
    """负载感知贪心路径。返回 [(服务天线, UE 分配, 理想和速率)]，按加入顺序。
    每步的负载增量只来自新天线所在的 RU。score_mode='se' 时不看负载，每步取和速率增量最大者（消融用）。"""
    h, bf, noise, u_ = model.h, model.bf, model.noise, model.num_ue
    remaining = list(order)
    ru_ant = {}                       # RU -> 服务天线数
    ru_ues = {}                       # RU -> 所服务的 UE 集合（MRT）
    path = []
    if bf == 'zf':
        serving = remaining[:u_]
        remaining = remaining[u_:]
        for m in serving:
            ru_ant[m // k] = ru_ant.get(m // k, 0) + 1
        se_ue = cfsim.se_from_sinr(cfsim.zf_sinr(h[:, :, serving], noise), se_model).mean(0)
        cur, qm = se_ue.sum(), cfsim.qm_from_se(se_ue, se_model)
        allu = list(range(u_))
        path.append((np.array(serving), np.full(len(serving), -1), float(cur)))
        while remaining:
            best = None
            for m in remaining:
                trial = serving + [m]
                se_t = cfsim.se_from_sinr(cfsim.zf_sinr(h[:, :, trial], noise), se_model).mean(0).sum()
                dse = se_t - cur
                if dse <= 0:
                    continue
                n0 = ru_ant.get(m // k, 0)
                dr = _ref_ru_load(n0 + 1, allu, qm, bf, u_) - _ref_ru_load(n0, allu, qm, bf, u_)
                score = dse if score_mode == 'se' else (np.inf if dr <= 0 else dse / dr)
                if best is None or score > best[0] or (score == best[0] and dse > best[2]):
                    best = (score, m, dse, se_t)
            if best is None:
                break
            m = best[1]
            serving.append(m)
            remaining.remove(m)
            ru_ant[m // k] = ru_ant.get(m // k, 0) + 1
            cur = best[3]
            path.append((np.array(serving), np.full(len(serving), -1), float(cur)))
        return path
    # MRT：候选为 (天线, UE)；对一根天线的所有 UE 一次算完
    b_ = h.shape[0]
    amp = np.zeros((b_, u_, u_), complex)
    serving, assign = [], []
    cur, qm = 0.0, np.full(u_, 2)
    eye = np.eye(u_, dtype=bool)
    while remaining:
        best = None
        for m in remaining:
            hm = h[:, :, m]                                                  # [B, U]
            w = np.sqrt(cfsim.P_MAX_PRB_MW) * np.conj(hm) / np.abs(hm)       # [B, U'] 发给 U' 的权值
            # 候选 u：amp[:, :, u] += hm * w[:, u] → 对所有 u 并行 [U_cand, B, U, U]
            delta = hm[None, :, :] * w.T[:, :, None]                         # [U_cand, B, U_rx]
            a_t = np.broadcast_to(amp, (u_,) + amp.shape).copy()
            a_t[np.arange(u_), :, :, np.arange(u_)] += delta
            pw = np.abs(a_t) ** 2
            sig = pw[:, :, eye]                                              # [U_cand, B, U]
            sinr = sig / (pw.sum(-1) - sig + noise)
            se_t = cfsim.se_from_sinr(sinr, se_model).mean(1).sum(-1)        # [U_cand]
            r = m // k
            n0, ues0 = ru_ant.get(r, 0), ru_ues.get(r, set())
            base = _ref_ru_load(n0, sorted(ues0), qm, bf, u_)
            for u in range(u_):
                dse = se_t[u] - cur
                if dse <= 0:
                    continue
                dr = _ref_ru_load(n0 + 1, sorted(ues0 | {u}), qm, bf, u_) - base
                score = dse if score_mode == 'se' else (np.inf if dr <= 0 else dse / dr)
                if best is None or score > best[0] or (score == best[0] and dse > best[2]):
                    best = (score, m, dse, float(se_t[u]), u, a_t[u])
        if best is None:
            break
        _, m, _, se_t, u, amp = best
        amp = amp.copy()
        serving.append(m)
        assign.append(u)
        remaining.remove(m)
        r = m // k
        ru_ant[r] = ru_ant.get(r, 0) + 1
        ru_ues.setdefault(r, set()).add(u)
        cur = se_t
        qm = cfsim.qm_from_se(cfsim.se_from_sinr(cfsim.mrt_sinr(amp, noise), se_model).mean(0), se_model)
        path.append((np.array(serving), np.array(assign), float(cur)))
    return path
