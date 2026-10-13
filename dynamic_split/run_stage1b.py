"""阶段 1b：真实编解码器的失真计入和速率，在每个配置（切分、U-plane 压缩、权值位宽）和 K 上
记录和速率与前传负载。服务天线、波束赋形和理想 Qm 取自阶段 0（与它同种子重建信道）。

配置：7-1 × U-plane {BFP4..12, 16 bit 不压缩}；7-2 × U-plane {BFP4..12, 16 bit, 调制压缩} × 权值 {BFP4..12, 16 bit}。
MRT 状态下 7-2 的权值由 RU 本地计算，权值位宽不起作用（各位宽结果相同，便于统一枚举）。
负载（每 RU，bit/s）：U-plane 与 C-plane 按 O-RAN 消息计费（fhload），bundle = 1 PRB；
调制压缩的 Qm 取该配置下 UE 实际频谱效率对应的 MCS；上行 CI 同阶段 1a（SRS 每 10 ms，天线级 IQ BFP9）。
CI 本身按理想信道计（论文的完全 CSI），量化误差留到阶段 2。

用法：
    .venv/bin/python dynamic_split/run_stage1b.py results/dynamic_split_20261012/stage0_repro \
        --out results/dynamic_split_20261012/stage1b_distortion
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import cfsim  # noqa: E402
import distortion as dz  # noqa: E402
import fhload  # noqa: E402
import summarize_stage0 as s0  # noqa: E402
from runio import Run, read_rows  # noqa: E402

KS = [1, 2, 4, 8, 16, 32, 64]
CONFIGS = [('7-1', up, None) for up in dz.UP71] + [('7-2', up, b) for up in dz.UP72 for b in dz.BFW]
SRS_PERIOD_MS = 10.0
UE_PER_SRS_SYMBOL = 16


def ru_view(serving, assign_s, bf, num_ue, k):
    n_ru = cfsim.NUM_ANT // k
    ru = serving // k
    n_ant = np.bincount(ru, minlength=n_ru)
    if bf == 'zf':
        ues = [list(range(num_ue)) if n else [] for n in n_ant]
    else:
        ues = [sorted(set(assign_s[ru == r].tolist())) for r in range(n_ru)]
    return n_ant, ues


def rate_up(n_eaxc, bits):
    return fhload.uplane_rate(int(n_eaxc), fhload.NUM_PRB, int(bits), fhload.SYMBOLS_PER_SLOT, fhload.SLOTS_PER_S)[1]


def loads(cfg, n_ant, ues, bf, qm_final, num_ue):
    """一个配置在一个 K 下的每 RU 下行负载与上行 CI（bit/s）。"""
    split, up, b = cfg
    srs_sym = -(-num_ue // UE_PER_SRS_SYMBOL)
    srs_rate = 1000.0 / SRS_PERIOD_MS
    dl = np.zeros(len(n_ant))
    ul = np.zeros(len(n_ant))
    for r, na in enumerate(n_ant):
        if na == 0:
            continue
        srs_ant = fhload.uplane_rate(int(na), fhload.NUM_PRB, fhload.bits_per_prb('bfp9'), srs_sym, 1)[1] * srs_rate
        if split == '7-1':
            dl[r] = rate_up(na, fhload.bits_per_prb(up)) + fhload.cplane_rate(int(na), fhload.SLOTS_PER_S)
            ul[r] = srs_ant
            continue
        nu = len(ues[r])
        if up == 'modcomp':
            dl[r] = sum(rate_up(1, 12 * qm_final[u]) for u in ues[r])
        else:
            dl[r] = nu * rate_up(1, fhload.bits_per_prb(up))
        if bf == 'zf':
            dl[r] += fhload.cplane_rate(nu, fhload.SLOTS_PER_S, fhload.NUM_PRB, int(na),
                                        fhload.bfw_block_bits(int(na), b))
            st6 = fhload.bfp_bits(int(na) * num_ue * fhload.NUM_PRB) * srs_rate
            ul[r] = min(srs_ant, st6)
        else:
            dl[r] += fhload.cplane_rate(nu, fhload.SLOTS_PER_S)
    return dl, ul


def task(item):
    (model_i, state_i, d), (num_ue, noise_dbm, se_model, bf, m_hat, row, se_ref) = item
    rng = np.random.default_rng([2024, d, num_ue])
    h, beta, _ = cfsim.drop(rng, num_ue, 'm2135')
    order, incl, assign = (np.asarray(row[k]) for k in ('order', 'incl', 'assign'))
    assert (cfsim.antenna_order(beta) == order).all(), 'stage-0 drop not reproduced'
    sel = np.arange(m_hat)[incl[:m_hat].astype(bool)]
    serving, assign_s = order[sel], assign[sel]
    noise = 10 ** (noise_dbm / 10)
    ev = dz.DropEval(h, serving, assign_s, bf, noise, se_model, cfsim.qm_from_se(se_ref),
                     np.random.default_rng([2024, d, num_ue, model_i, state_i, 7]))
    n_cfg, n_k = len(CONFIGS), len(KS)
    se = np.zeros((n_cfg, n_k))
    dl_tot = np.zeros((n_cfg, n_k))
    dl_link = np.zeros((n_cfg, n_k))
    ul_tot = np.zeros((n_cfg, n_k))
    se71 = {up: ev.se_71(up) for up in dz.UP71}
    views = {k: ru_view(serving, assign_s, bf, num_ue, k) for k in KS}
    for ki, k in enumerate(KS):
        w_hat = {b: ev.weights_72(k, b) for b in dz.BFW} if bf == 'zf' else {b: ev.w for b in dz.BFW}
        n_ant, ues = views[k]
        for ci, cfg in enumerate(CONFIGS):
            split, up, b = cfg
            se_ue = se71[up] if split == '7-1' else ev.se_72(up, w_hat[b])
            dl, ul = loads(cfg, n_ant, ues, bf, cfsim.qm_from_se(se_ue), num_ue)
            se[ci, ki] = se_ue.sum()
            dl_tot[ci, ki], dl_link[ci, ki], ul_tot[ci, ki] = dl.sum(), dl.max(), ul.sum()
    return (model_i, state_i, d), float(ev.se_ideal().sum()), se, dl_tot, dl_link, ul_tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage0')
    p.add_argument('--out', required=True)
    p.add_argument('--models', nargs='+', default=['mcs', 'shannon'])
    p.add_argument('--drops', type=int, default=100)
    p.add_argument('--workers', type=int, default=48)
    args = p.parse_args()
    here = Path(__file__).parent
    stage0 = Path(args.stage0)
    run = Run(args.out, [here / f for f in ('run_stage1b.py', 'distortion.py', 'fhload.py', 'cfsim.py',
                                            'summarize_stage0.py', 'runio.py')],
              dict(purpose='Stage 1b: compression distortion through real O-RAN codecs coupled to SE; '
                           'per-config SE and fronthaul load for joint split / U-plane width / weight width choice',
                   stage0_run=str(stage0), args=vars(args), ks=KS, configs=CONFIGS, kappa_paper=s0.KAPPA,
                   srs_period_ms=SRS_PERIOD_MS, ue_per_srs_symbol=UE_PER_SRS_SYMBOL,
                   assumptions=['Serving set, BF type and ideal per-UE Qm from stage 0 at M-hat (paper method).',
                                'One slot per drop: 14 symbols x 12 REs per PRB, QAM of the UE ideal MCS order.',
                                'Codecs from oran/compression via IQCodec (16-bit fixed point, 15 dB backoff); '
                                '7-1 nominal power Pmax per antenna, 7-2 symbols 1, weights Pmax/U.',
                                'Weights: one SE-11 block = K_r weights of one UE on one PRB (bundle = 1 PRB).',
                                'UE: per-PRB LMMSE effective gain from 168 REs; SE from MCS table 2 or Shannon.',
                                'Modulation compression lossless, bits = Qm of the MCS the UE actually gets.',
                                'CI: full CSI (paper), cost as stage 1a (antenna-level SRS IQ BFP9 every 10 ms).',
                                'MRT + 7-2: weights computed at the RU, no weight quantization or C-plane weights.']))
    rows0 = read_rows(stage0)
    per_ue = np.load(stage0 / 'per_ue_se.npz')
    states = s0.select_states(rows0)
    state_keys = sorted({(u, n) for (u, n, _) in states})
    items = []
    for mi, sm in enumerate(args.models):
        for si, (u, n) in enumerate(state_keys):
            st = states[(u, n, sm)]
            bf, m_hat = st['bf'], st[st['bf']]['m_hat']
            for r in rows0:
                if (r['num_ue'], r['noise_dbm'], r['se_model'], r['bf']) == (u, n, sm, bf) and r['drop'] < args.drops:
                    items.append(((mi, si, r['drop']), (u, n, sm, bf, m_hat, r, per_ue[r['per_ue_key']][m_hat - 1])))
    shape = (len(args.models), len(state_keys), args.drops, len(CONFIGS), len(KS))
    arr = {k: np.zeros(shape, np.float32) for k in ('se', 'dl', 'link', 'ul')}
    se_ideal = np.zeros(shape[:3])
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (idx, ideal, se, dl, link, ul) in enumerate(ex.map(task, items, chunksize=2)):
            se_ideal[idx] = ideal
            for k, v in zip(('se', 'dl', 'link', 'ul'), (se, dl, link, ul)):
                arr[k][idx] = v
            if (i + 1) % 100 == 0:
                print(f'{i + 1}/{len(items)} drops', flush=True)
    np.savez_compressed(run.path / 'per_drop.npz', se_ideal=se_ideal, **arr)
    for mi, sm in enumerate(args.models):
        for si, (u, n) in enumerate(state_keys):
            st = states[(u, n, sm)]
            ideal = float(se_ideal[mi, si].mean())
            for ci, (split, up, b) in enumerate(CONFIGS):
                for ki, k in enumerate(KS):
                    run.add(dict(se_model=sm, num_ue=u, noise_dbm=n, bf=st['bf'], k=k, split=split, uplane=up,
                                 bfw=b, se=float(arr['se'][mi, si, :, ci, ki].mean()), se_ideal=ideal,
                                 se_ratio=float(arr['se'][mi, si, :, ci, ki].mean()) / ideal,
                                 dl_gbps=float(arr['dl'][mi, si, :, ci, ki].mean()) / 1e9,
                                 link_gbps=float(arr['link'][mi, si, :, ci, ki].mean()) / 1e9,
                                 ul_gbps=float(arr['ul'][mi, si, :, ci, ki].mean()) / 1e9))
    run.finish(state_keys=[list(s) for s in state_keys], models=args.models)
    print('done', run.path)


if __name__ == '__main__':
    main()
