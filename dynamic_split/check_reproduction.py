"""S9：MRT 在 −50 dBm 下复现偏低约 30% 的原因检验（Shannon，[10] 的启发式，论文图 5 为参照）。

变体（都用 [10] 的服务 AP 选择，按 κ = 0.98 在平均曲线上取 M̂，报告 κ 处的平均和速率）：
  base      阶段 0 的口径：固定噪声，和速率 = 各 PRB log₂(1 + SINR) 的平均
  wb        [10] 式 (6)：每个 UE 先把各 PRB 的信号、干扰加噪声分别求和再取比值（宽带 SINR）
  ln_u      噪声按 [10]：对数正态、标准差 4.8 dB、线性均值等于标称值；每个 UE 一个值（外部干扰随位置变）
  ln_ub     同上，但每个 (UE, PRB) 独立
  ln_u_wb   ln_u + wb
  ln_ub_wb  ln_ub + wb
  inv       ZF 按 [10] 式 (18) 归一化：不做列归一化（信道求逆，各 UE 接收功率相同），整体缩放到最强 AP 满功率
  ln_u_inv / ln_ub_inv  对数正态噪声 + inv

用法：
    .venv/bin/python dynamic_split/check_reproduction.py --out results/dynamic_split_20261012/stage9_reproduction
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import cfsim  # noqa: E402
import heuristics10 as H10  # noqa: E402
import summarize_stage0 as s0  # noqa: E402
from runio import Run  # noqa: E402

VARIANTS = ['base', 'wb', 'ln_u', 'ln_ub', 'ln_u_wb', 'ln_ub_wb', 'inv', 'ln_u_inv', 'ln_ub_inv']
SIGMA_DB = 4.8


def noise_draw(rng, mean_mw, shape, sigma_db=SIGMA_DB):
    s = sigma_db * np.log(10) / 10
    return mean_mw * np.exp(s * rng.standard_normal(shape) - s ** 2 / 2)


def se_sum(sig, intf, noise, wideband):
    if wideband:
        return float(np.log2(1 + sig.sum(0) / (intf + noise).sum(0)).sum())
    return float(np.log2(1 + sig / (intf + noise)).mean(0).sum())


def zf_terms(h_s, inversion=False):
    g = h_s @ np.conj(np.swapaxes(h_s, 1, 2))
    w = np.conj(np.swapaxes(h_s, 1, 2)) @ np.linalg.inv(g)
    if inversion:                                   # [10] 式 (18)：H W̃ = I，每个 UE 接收幅度相同
        p = cfsim.P_MAX_PRB_MW / np.max(np.sum(np.abs(w) ** 2, axis=2), axis=1)
        return np.repeat(p[:, None], h_s.shape[1], axis=1), np.zeros((h_s.shape[0], h_s.shape[1]))
    norm = np.linalg.norm(w, axis=1)
    w = w / norm[:, None, :]
    p = cfsim.P_MAX_PRB_MW / np.max(np.sum(np.abs(w) ** 2, axis=2), axis=1)
    return p[:, None] / norm ** 2, np.zeros_like(norm)


def mrt_terms(amp):
    pw = np.abs(amp) ** 2
    sig = np.einsum('buu->bu', pw)
    return sig, pw.sum(-1) - sig


def zf_curve(h, noise, wideband, inversion=False):
    order = H10.zf_order(h)
    u_ = h.shape[1]
    serving, cur, curve = [], 0.0, []
    for m in order:
        trial = serving + [m]
        if len(trial) < u_:
            serving = trial
        else:
            se = se_sum(*zf_terms(h[:, :, trial], inversion), noise, wideband)
            if len(serving) < u_ or se > cur:
                serving, cur = trial, se
        curve.append(cur)
    return np.array(curve)


def mrt_curve(h, noise, wideband):
    pw = np.mean(np.abs(h) ** 2, axis=0)
    u_, m_ = pw.shape
    ue_of = np.argmax(pw, axis=0)
    srt = np.sort(pw, axis=0)
    order = np.argsort(-(srt[-1] / np.maximum(srt[-2], 1e-300)), kind='stable')
    strongest = pw.max(axis=1)
    amp = np.zeros((h.shape[0], u_, u_), complex)

    def add(m, a):
        hm = h[:, :, m]
        w = np.sqrt(cfsim.P_MAX_PRB_MW) * np.conj(hm[:, ue_of[m]]) / np.abs(hm[:, ue_of[m]])
        a = a.copy()
        a[:, :, ue_of[m]] += hm * w[:, None]
        return a

    serving, curve, has = [], [], np.zeros(u_, bool)
    for m in order:
        if not has[ue_of[m]]:
            has[ue_of[m]] = True
            amp = add(m, amp)
            serving.append(m)
            curve.append(se_sum(*mrt_terms(amp), noise, wideband))
    cur = curve[-1] if curve else 0.0
    for m in order:
        if m in serving:
            continue
        if pw[ue_of[m], m] * H10.P_OFFSET >= strongest[ue_of[m]]:
            a_t = add(m, amp)
            se = se_sum(*mrt_terms(a_t), noise, wideband)
            if se > cur:
                amp, cur = a_t, se
                serving.append(m)
        curve.append(cur)
    return np.array(curve)


def task(item):
    u, d, noises = item
    h, _, _ = cfsim.drop(np.random.default_rng([2024, d, u]), u, 'm2135')
    rng = np.random.default_rng([2024, d, u, 99])
    b_ = h.shape[0]
    out = np.zeros((len(noises), len(VARIANTS), 2, cfsim.NUM_ANT))
    for ni, nd in enumerate(noises):
        mean = 10 ** (nd / 10)
        draws = {'u': noise_draw(rng, mean, (1, u)) * np.ones((b_, 1)), 'ub': noise_draw(rng, mean, (b_, u))}
        for vi, v in enumerate(VARIANTS):
            noise = draws['ub'] if 'ln_ub' in v else draws['u'] if 'ln_u' in v else np.full((b_, u), mean)
            wideband = v.endswith('wb')    # inv 变体用逐 PRB 平均
            out[ni, vi, 0] = zf_curve(h, noise, wideband, v.endswith('inv'))
            out[ni, vi, 1] = mrt_curve(h, noise, wideband)
    return (u, d), out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--drops', type=int, default=100)
    p.add_argument('--workers', type=int, default=48)
    args = p.parse_args()
    here = Path(__file__).parent
    noises = [-90.0, -70.0, -50.0]
    ues = [5, 10, 20]
    run = Run(args.out, [here / f for f in ('check_reproduction.py', 'heuristics10.py', 'cfsim.py')],
              dict(purpose='S9: why MRT at -50 dBm reproduces ~30% low; noise model and SINR definition of [10]',
                   args=vars(args), variants=VARIANTS, sigma_db=SIGMA_DB, se_model='shannon'))
    items = [(u, d, noises) for u in ues for d in range(args.drops)]
    curves = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for (u, d), out in ex.map(task, items, chunksize=2):
            curves.setdefault(u, []).append(out)
    print('state      | ' + ' | '.join(f'{v:>17s}' for v in VARIANTS) + ' | paper')
    for u in ues:
        c = np.mean(curves[u], axis=0)                        # [noise, variant, bf, M]
        for ni, nd in enumerate(noises):
            cells, row = [], dict(num_ue=u, noise_dbm=nd)
            for vi, v in enumerate(VARIANTS):
                res = {}
                for bi, bf in enumerate(('zf', 'mrt')):
                    mean = c[ni, vi, bi]
                    m_hat = int(np.argmax(mean >= s0.KAPPA * mean.max())) + 1
                    res[bf] = (m_hat, float(mean[m_hat - 1]))
                bf = max(res, key=lambda b: res[b][1])
                row[v] = dict(bf=bf, zf=res['zf'], mrt=res['mrt'])
                cells.append(f"{bf.upper():3s} {res[bf][1]:6.1f} ({res[bf][0]:2d})")
            paper = s0.PAPER_FIG5.get((u, nd))
            row['paper'] = None if paper is None else [paper[0], paper[1]]
            run.add(row)
            print(f'({u:2d},{int(nd)}) | ' + ' | '.join(f'{x:>17s}' for x in cells) +
                  f" | {'' if paper is None else (paper[0].upper(), paper[1])}")
    run.finish()


if __name__ == '__main__':
    main()
