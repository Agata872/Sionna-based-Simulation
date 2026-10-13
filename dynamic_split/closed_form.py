"""三步闭式方法（theory_notes 第 8 节）：服务集合（[10]）→ 逐 RU 位宽（命题 3）→ 逐 RU 切分（命题 2）。

位宽用"名义位宽" τ 参数化：w_r = round(τ + x71_r − med)，b_r = round(τ + x72_r − med)，限制在 4..12，
其中 x71_r = ½ log₂(G_r / (336 n_r))，x72_r = ½ log₂(G'_r / (2 n_r U))，med 为各 RU x71 的中位数，
所以 τ 就是"中位 RU 的 7-1 位宽"，不同撒点、状态之间可比。
切分模式：'rule'（命题 2 / 推论 2）、'all71'、'all72'（ZF 下全网固定一种切分，位宽仍按命题 3）。
"""

import numpy as np

import joint as J

TAU = np.arange(2.0, 14.01, 0.25)
UNIFORM_W = list(range(4, 13))


def payload71(n, w):
    return n * 168 * (2 * w + 2 / 3)


def payload72(qm_sum, n_ue, n, b):
    return 168 * qm_sum + n_ue * (2 * b * n + 24)


class CFState:
    """一个服务集合、一个 K：每个 RU 的选项表（负载、失真）和命题 3 的偏移量。"""

    def __init__(self, model, st, k, cal):
        self.model, self.st, self.k = model, st, k
        opts = model.ru_options(st, k)
        ru = st['serving'] // k
        self.rus = [r for r, _ in opts]
        self.n = np.array([int((ru == r).sum()) for r in self.rus])
        self.keys = [[o[0] for o in ro] for _, ro in opts]
        self.load = [np.array([o[1] for o in ro]) for _, ro in opts]
        self.dist = [np.stack([o[3] for o in ro]) for _, ro in opts]
        self.zf = model.bf == 'zf'
        self.u = model.num_ue
        self.qm_sum = float(st['qm'].sum())
        ref = cal['r71']['bfp4']
        gd = [{o[0]: o[4] for o in ro} for _, ro in opts]
        self.x71 = np.array([0.5 * np.log2(max(g[('7-1', 'bfp4')] / ref, 1e-300) / (336 * n))
                             for g, n in zip(gd, self.n)])
        self.x72 = (np.array([0.5 * np.log2(max(g[('7-2', 4)] / ref, 1e-300) / (2 * n * self.u))
                              for g, n in zip(gd, self.n)]) if self.zf else None)
        self.med = float(np.median(self.x71))

    def _index(self, j, key):
        return self.keys[j].index(key)

    def choose(self, tau, mode='rule'):
        """返回每个 RU 的选项下标与 (切分, 位宽)。"""
        idx, picks = [], []
        for j, n in enumerate(self.n):
            w = int(np.clip(np.round(tau + self.x71[j] - self.med), 4, 12))
            if not self.zf:
                key = ('7-1', f'bfp{w}') if mode == 'all71' else ('7-2', None)
            else:
                b = int(np.clip(np.round(tau + self.x72[j] - self.med), 4, 12))
                if mode == 'rule':
                    use72 = payload72(self.qm_sum, self.u, n, b) < payload71(n, w)
                else:
                    use72 = mode == 'all72'
                key = ('7-2', b) if use72 else ('7-1', f'bfp{w}')
            idx.append(self._index(j, key))
            picks.append(key)
        return idx, picks

    def uniform(self, split, w):
        """全网同一切分、同一位宽（阶段 1b 的口径）。"""
        if split == '7-1':
            key = ('7-1', f'bfp{w}')
        else:
            key = ('7-2', w) if self.zf else ('7-2', None)
        return [self._index(j, key) for j in range(len(self.n))]

    def evaluate(self, idx):
        d = sum(self.dist[j][i] for j, i in enumerate(idx))
        load = sum(self.load[j][i] for j, i in enumerate(idx)) / 1e9
        f72 = float(np.mean([self.keys[j][i][0] == '7-2' for j, i in enumerate(idx)]))
        return self.model.se(self.st, d), load, f72
