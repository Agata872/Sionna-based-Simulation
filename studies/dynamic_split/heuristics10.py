"""论文参考文献 [10]（Girycki 等，IEEE TWC 23(5), 2024）的启发式服务 AP 选择，按原文算法实现。

  MRT（算法 1，EuCNC 中的 MRT-H2）：
    1. 每个 AP 只服务"从它收到最强功率"的 UE（假设 2、3）；
    2. AP 按"最强 / 次强接收功率之比"降序排列；
    3. 循环 1：按此顺序，若该 AP 对应的 UE 还没有 AP，则选中；
    4. 循环 2：按此顺序遍历其余 AP，若该 AP 到其 UE 的功率不弱于该 UE 最强 AP 功率的 1/P_offset（P_offset = 200，
       即 23 dB），且加入后和速率提高，则选中。
  ZF（算法 2 给出顺序，EuCNC 第 III-C 节的加入规则）：
    第 i 轮取每个 UE 的第 i 强 AP，本轮内按接收功率降序排列，已在序列中的跳过；
    沿这个顺序逐个考察，前 U 个直接加入（ZF 可行），之后能提高和速率才加入。
与 [10] 的差别：[10] 每个 PRB 单独选 AP（假设 1）；这里与阶段 0 和论文的负载公式一致，服务集合对所有 PRB 相同，
"接收功率"取各 PRB 的平均 |h|²。ZF 的功率归一化沿用阶段 0（列归一化后统一缩放到最强 AP 满功率）。
返回的路径与 joint.paper_path 格式相同：第 n 个元素为考察了 n+1 个 AP 之后的 (服务天线, MRT 的 UE 分配)，另附理想和速率。
"""

import numpy as np

import cfsim

P_OFFSET = 200.0


def _rx_power(h):
    return np.mean(np.abs(h) ** 2, axis=0)                      # [U, M]


def zf_order(h):
    pw = _rx_power(h)
    u_, m_ = pw.shape
    rank = np.argsort(-pw, axis=1)                              # 每个 UE 的 AP 由强到弱
    order, seen = [], set()
    for i in range(m_):
        cand = [(pw[u, rank[u, i]], rank[u, i]) for u in range(u_)]
        for _, m in sorted(cand, key=lambda t: -t[0]):
            if m not in seen:
                seen.add(m)
                order.append(int(m))
    return np.array(order)


def zf_path(h, noise_mw, se_model):
    order = zf_order(h)
    u_ = h.shape[1]
    serving, cur, out = [], 0.0, []
    for m in order:
        trial = serving + [m]
        if len(trial) < u_:
            serving = trial
        else:
            se = cfsim.se_from_sinr(cfsim.zf_sinr(h[:, :, trial], noise_mw), se_model).mean(0).sum()
            if len(serving) < u_ or se > cur:
                serving, cur = trial, se
        out.append((np.array(serving), np.full(len(serving), -1), float(cur)))
    return out


def mrt_path(h, noise_mw, se_model):
    pw = _rx_power(h)                                           # [U, M]
    u_, m_ = pw.shape
    ue_of = np.argmax(pw, axis=0)                               # 每个 AP 服务的 UE
    srt = np.sort(pw, axis=0)
    ratio = srt[-1] / np.maximum(srt[-2], 1e-300) if u_ > 1 else np.ones(m_)
    order = np.argsort(-ratio, kind='stable')
    strongest = pw.max(axis=1)                                  # 每个 UE 最强 AP 的功率
    b_ = h.shape[0]
    amp = np.zeros((b_, u_, u_), complex)

    def add(m, a):
        hm = h[:, :, m]
        w = np.sqrt(cfsim.P_MAX_PRB_MW) * np.conj(hm[:, ue_of[m]]) / np.abs(hm[:, ue_of[m]])
        a = a.copy()
        a[:, :, ue_of[m]] += hm * w[:, None]
        return a

    def se_of(a):
        return float(cfsim.se_from_sinr(cfsim.mrt_sinr(a, noise_mw), se_model).mean(0).sum())

    serving, assign, out = [], [], []
    has = np.zeros(u_, bool)
    for m in order:                                             # 循环 1
        if not has[ue_of[m]]:
            has[ue_of[m]] = True
            amp = add(m, amp)
            serving.append(int(m))
            assign.append(int(ue_of[m]))
            out.append((np.array(serving), np.array(assign), se_of(amp)))
    cur = out[-1][2] if out else 0.0
    for m in order:                                             # 循环 2
        if m in serving:
            continue
        if pw[ue_of[m], m] * P_OFFSET >= strongest[ue_of[m]]:
            a_t = add(m, amp)
            se = se_of(a_t)
            if se > cur:
                amp, cur = a_t, se
                serving.append(int(m))
                assign.append(int(ue_of[m]))
        out.append((np.array(serving), np.array(assign), float(cur)))
    return out


def path(h, bf, noise_mw, se_model):
    return zf_path(h, noise_mw, se_model) if bf == 'zf' else mrt_path(h, noise_mw, se_model)
