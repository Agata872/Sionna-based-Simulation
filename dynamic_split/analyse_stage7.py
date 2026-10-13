"""S7 分析：动态调整的价值分解（MCS，κ = 0.98，P1 平均负载）。

策略（每个 K 分别计算；"状态"为 9 个 (UE 数, 噪声)，等权）：
  D        逐撒点：每个撒点任选 (κ_s, τ)，规则模式（命题 2 + 命题 3）
  S        逐状态：每个状态固定 (κ_s, τ)，规则模式（每个 RU 的切分与位宽仍由本地量按公式算出）
  G        全局固定名义位宽：所有状态共用一个 (κ_s, τ)，规则模式（只有切分随状态变）
  F        固定切分：全部 7-1 或全部 7-2（对所有状态同一种），每个状态固定 (κ_s, τ)（只有位宽随状态变）
  SU       逐状态、全网统一切分与位宽（阶段 1b 的动态）
  ST       完全静态：所有状态共用一个 (κ_s, 切分, 位宽)，全网统一（阶段 1b 的静态）
可行：撒点平均的和速率 ≥ κ × 撒点平均的 SE_max（逐撒点策略按每个撒点检查）。

用法：
    .venv/bin/python dynamic_split/analyse_stage7.py results/dynamic_split_20261012/stage7_dynamics
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from closed_form import TAU  # noqa: E402

KAPPA = 0.98


def per_state_best(ratio, load, se_max, fixed_axes_mask=None):
    """ratio/load: [S, D, P]（P 为参数组合）。返回每个状态的最小平均负载及所选参数 [S]。"""
    w = np.nan_to_num(se_max)[:, :, None]                          # 无效撒点（SE_max 为 NaN）权重为 0
    mean_ratio = np.nansum(ratio * w, 1) / w.sum(1)                  # [S, P]
    mean_load = np.nanmean(load, 1)
    ok = mean_ratio >= KAPPA - 1e-9
    masked = np.where(ok, mean_load, np.inf)
    j = masked.argmin(1)
    return masked[np.arange(len(j)), j], j


def global_best(ratio, load, se_max):
    """所有状态共用一个参数：每个状态都须可行；返回平均负载（对状态平均）与参数。"""
    w = np.nan_to_num(se_max)[:, :, None]
    mean_ratio = np.nansum(ratio * w, 1) / w.sum(1)
    mean_load = np.nanmean(load, 1)
    ok = (mean_ratio >= KAPPA - 1e-9).all(0)
    tot = np.where(ok, mean_load.mean(0), np.inf)
    j = int(tot.argmin())
    return float(tot[j]), j


def analyse(z):
    states = [tuple(s) for s in z['states']]
    bfs = list(z['bfs'])
    se_max = z['se_max']                                            # [S, D]
    out = []
    for ki, k in enumerate(z['ks']):
        g = z['grid'][:, :, ki]                                     # [S, D, KS, M, T, 3]
        u = z['uni'][:, :, ki]                                      # [S, D, KS, 2, W, 2]
        s_, d_ = se_max.shape
        rule_r, rule_l = g[:, :, :, 0, :, 0].reshape(s_, d_, -1), g[:, :, :, 0, :, 1].reshape(s_, d_, -1)
        rule_f72 = g[:, :, :, 0, :, 2].reshape(s_, d_, -1)
        # D
        okd = rule_r >= KAPPA - 1e-9
        d_load = np.where(okd, rule_l, np.inf).min(2)
        d_load[np.isnan(se_max)] = np.nan
        D = float(np.nanmean(d_load))
        # S
        s_load, s_j = per_state_best(rule_r, rule_l, se_max)
        S = float(s_load.mean())
        s_f72 = [float(np.nanmean(rule_f72[si, :, j])) for si, j in enumerate(s_j)]
        s_tau = [float(TAU[j % len(TAU)]) for j in s_j]                 # 每个状态所选的名义位宽 τ
        # G
        G, _ = global_best(rule_r, rule_l, se_max)
        # F：固定切分（对所有状态同一种），每个状态自选 (κ_s, τ)
        F_modes = {}
        for mi, mode in ((1, 'all71'), (2, 'all72')):
            r_, l_ = g[:, :, :, mi, :, 0].reshape(s_, d_, -1), g[:, :, :, mi, :, 1].reshape(s_, d_, -1)
            ls, _ = per_state_best(r_, l_, se_max)
            F_modes[mode] = float(ls.mean())
        F = min(F_modes.values())
        # SU / ST
        ur, ul = u[..., 0].reshape(s_, d_, -1), u[..., 1].reshape(s_, d_, -1)
        su_load, _ = per_state_best(ur, ul, se_max)
        SU = float(su_load.mean())
        ST, _ = global_best(ur, ul, se_max)
        out.append(dict(k=int(k), D=D, S=S, G=G, F=F, F_modes=F_modes, SU=SU, ST=ST,
                        per_state_S=s_load.tolist(), per_state_f72=s_f72, per_state_tau=s_tau, states=[list(x) for x in states], bfs=bfs))
    return out


def report(res):
    print('Mean P1 load over states [Gbit/s] (MCS, kappa 0.98); savings relative to fully static ST')
    print(' K |   ST   SU    F    G    S    D | ST→SU  SU→S  F→S  G→S  S→D | dynamic total')
    for e in res:
        st = e['ST']
        pct = lambda a, b: 100 * (1 - b / a)
        print(f"{e['k']:2d} | {e['ST']:4.2f} {e['SU']:4.2f} {e['F']:4.2f} {e['G']:4.2f} {e['S']:4.2f} {e['D']:4.2f} | "
              f"{pct(e['ST'], e['SU']):5.1f} {pct(e['SU'], e['S']):5.1f} {pct(e['F'], e['S']):5.1f} {pct(e['G'], e['S']):5.1f} "
              f"{pct(e['S'], e['D']):5.1f} | ST→S {pct(st, e['S']):5.1f}%  ST→D {pct(st, e['D']):5.1f}%")
    print('\nPer-state optimum (S): 7-2 RU share / nominal width tau, per state:')
    for e in res:
        print(f"  K={e['k']:2d}: " + '  '.join(f"({int(s[0])},{int(s[1])}){b[0].upper()} {f:.2f}/{t:.1f}" for s, b, f, t in
                                             zip(e['states'], e['bfs'], e['per_state_f72'], e['per_state_tau'])))


def main(run_dir):
    run_dir = Path(run_dir)
    z = dict(np.load(run_dir / 'stage7.npz'))
    res = analyse(z)
    (run_dir / 'stage7_summary.json').write_text(json.dumps(res, indent=1))
    report(res)


if __name__ == '__main__':
    main(sys.argv[1])
