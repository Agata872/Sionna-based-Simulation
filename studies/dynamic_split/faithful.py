"""模型口径开关。

  'fixed'    阶段 0–8 的口径：固定噪声，ZF 列归一化后统一缩放
  'paper10'  按 [10] / EuCNC 复现（S9 验证，9 个状态全部对上论文图 5）：
             每个 UE 一个对数正态噪声值（标准差 4.8 dB，线性均值 = 标称值，同一撒点在各噪声水平下共用同一组随机数），
             ZF 按 [10] 式 (18) 归一化（不做列归一化）
逐状态的波束赋形方式按论文规则选：[10] 服务 AP 选择的平均曲线上取 M̂（κ = 0.98），κ 处和速率较高者。
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np

import cfsim
import heuristics10 as H10

MODELS = ('fixed', 'paper10')
KAPPA = 0.98


def setup(model):
    cfsim.set_zf_norm('inversion' if model == 'paper10' else 'equal')


def noise(model, d, num_ue, noise_dbm):
    mean = 10 ** (noise_dbm / 10)
    if model == 'fixed':
        return mean
    return cfsim.noise_per_ue(np.random.default_rng([2024, d, num_ue, 77]), mean, num_ue)


def _curves(item):
    model, se_model, u, nd, d = item
    setup(model)
    h, _, _ = cfsim.drop(np.random.default_rng([2024, d, u]), u, 'm2135')
    n = noise(model, d, u, nd)
    return (u, nd), np.array([[p[2] for p in H10.path(h, bf, n, se_model)] for bf in ('zf', 'mrt')])


def select_bf(keys, drops, model, se_model, workers=48):
    """返回 {(UE 数, 噪声): 'zf' / 'mrt'} 以及每个状态两种 BF 的 (M̂, κ 处和速率)。"""
    items = [(model, se_model, int(u), float(n), d) for (u, n) in keys for d in range(drops)]
    acc = {}
    with ProcessPoolExecutor(workers) as ex:
        for key, c in ex.map(_curves, items, chunksize=4):
            acc.setdefault(key, []).append(c)
    bf, info = {}, {}
    for key, cs in acc.items():
        mean = np.mean(cs, axis=0)
        res = {}
        for bi, name in enumerate(('zf', 'mrt')):
            m_hat = int(np.argmax(mean[bi] >= KAPPA * mean[bi].max())) + 1
            res[name] = (m_hat, float(mean[bi][m_hat - 1]))
        bf[key] = max(res, key=lambda b: res[b][1])
        info[key] = res
    return bf, info
