#!/usr/bin/env python3
"""Link-level check of the stage-4 PHY abstraction with O-RU-side aged ZF weights.

Uses the project's NR chain (oran/: PDSCH, DMRS LS estimation, LMMSE, LDPC, CRC)
with Cat-B layer U-plane BFP9 and continuous CDL Doppler traces (stage-3 trace
generator). For each sample the O-RU weights are eigen-RZF from the channel
`age` slots earlier; per-RE genie LMMSE SINR at the data slot gives the
abstraction's EESM effective SINR and TBLER for MCS m*, m*+1, m*+2 (m* = the
10%-threshold choice). Each MCS is transmitted and decoded; the reliability of
predicted TBLER vs actual CRC is the output. No ICI (CDL links, AWGN only).
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import sionna.phy
from oran import ORANConfig, ORANDownlink, CompressionConfig
from oran.channel import ChannelState
from oran.csi import lmmse_sinr, reference_data_symbols
from oran.fronthaul import IQCodec
from oran.precoding import apply_precoder
from sionna.phy.channel import cir_to_ofdm_channel
from explore_csirs_precoding import Recorder
from srsbf_core import DropConfig, MCSTable, layer_channels, rzf_weights


def continuous_trace(sim, slots, speed, seed):
    """Stage-3 trace generator generalised to K UEs (one CDL call per trace)."""
    sionna.phy.config.seed = seed
    rng = np.random.default_rng(seed)
    for k, row in enumerate(sim.channel.model.links):
        angle = rng.uniform(0, 2*math.pi)
        v = torch.tensor([speed*math.cos(angle), speed*math.sin(angle), 0.], device=sim.device)
        for cdl in row:
            cdl._ut_velocity = v
    ns = sim.rg.num_ofdm_symbols
    a, tau = sim.channel.model(1, slots*ns, 1/sim.rg.ofdm_symbol_duration)
    hs, csis = [], []
    for t in range(slots):
        h = cir_to_ofdm_channel(sim.channel.frequencies, a[..., t*ns:(t+1)*ns], tau)
        hs.append(h)
        csis.append(sim.channel.csi(ChannelState(h), 0))
    return hs, csis


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--drops', type=int, default=8)
    p.add_argument('--seed-start', type=int, default=20267000)
    p.add_argument('--slots', type=int, default=40)
    p.add_argument('--speed', type=float, default=30., help='km/h')
    p.add_argument('--ages', type=int, nargs='+', default=[0, 3, 10])
    p.add_argument('--snr', type=float, default=20.)
    p.add_argument('--num-prb', type=int, default=24)
    p.add_argument('--antennas', type=int, default=32)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    sionna.phy.config.device = args.device
    cfg = ORANConfig(category='B', num_oru=1, num_oru_ant=args.antennas, num_prb=args.num_prb, num_ue=4,
                     num_ue_ant=2, num_layers_per_ue=1, mcs_table=2, precoder='ezf',
                     uplane=CompressionConfig('bfp', 9), device=args.device)
    sim = ORANDownlink(cfg)
    allowed = set(sim.supported_mcs())
    tab = MCSTable(DropConfig(num_prb=args.num_prb), args.device)
    ucodec = IQCodec(cfg.uplane, 1., 12)
    no = 10**(-args.snr/10)
    data = [l for l in range(14) if l not in set(int(x) for x in sim.pdsch_configs[0].dmrs_symbol_indices)]
    rec = Recorder(args.out)
    (rec.path/'experiment_source.py').write_bytes(Path(__file__).read_bytes())
    (rec.path/'metadata.json').write_text(json.dumps(dict(args=vars(args), config=asdict(cfg),
        allowed_mcs=sorted(allowed), started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), indent=2))
    for drop in range(args.drops):
        seed = args.seed_start+drop
        hs, hc = continuous_trace(sim, args.slots, args.speed/3.6, seed)   # trace generator takes m/s
        for t in range(max(args.ages), args.slots):
            one = ChannelState(hs[t])
            for age in args.ages:
                g = layer_channels(hc[t-age][0], 1)[..., 0, :]          # P,K,M (PRG centres)
                w = rzf_weights(g, 4*no)[None]                          # 1,P,M,K
                sinr = lmmse_sinr(one.stacked()[..., data, :], w, no, sim.odu.prg_of_sc, 1)[0, :, 0]  # K,S,F
                genie = sinr.flatten(1).cpu().numpy()

                def send(mcs, salt):
                    sim.set_mcs(mcs)
                    sionna.phy.config.seed = seed*1000+t*10+salt
                    tx = sim.odu(1, hc[t])
                    grid = apply_precoder(ucodec(tx.x_layers), w, sim.odu.prg_of_sc)
                    sionna.phy.config.seed = seed*1000+t*10+salt+500
                    y = sim.channel(grid.unflatten(1, (1, args.antennas)), one, no)
                    rx = sim.ue(y, no)
                    rx_sinr = (1/rx.no_eff[0, :, 0].clamp_min(1e-12)).cpu().numpy()      # K, data REs
                    return rx.crc_ok[0].cpu().numpy(), rx_sinr

                lo = min(allowed)
                clamp = lambda m: int(m) if m in allowed else (max(a for a in allowed if a <= m) if m >= lo else lo)
                _, probe = send([lo]*4, 9)                                   # post-EQ SINR of the real receiver
                base = tab.select_from_prg(probe, 0.)
                for step in (-1, 0, 1):
                    mcs = [clamp(min(26, max(0, b+step))) for b in base]
                    ok, rx_sinr = send(mcs, step+2)
                    for k in range(4):
                        eff_rx = float(tab.eesm_db(rx_sinr[k], mcs[k]))
                        eff_g = float(tab.eesm_db(genie[k], mcs[k]))
                        rec.add(dict(drop=drop, slot=t, age=age, step=step, ue=k, mcs=mcs[k],
                                     eff_rx_db=eff_rx, eff_genie_db=eff_g,
                                     pred_tbler=float(tab.tbler(mcs[k], eff_rx)),
                                     pred_tbler_genie=float(tab.tbler(mcs[k], eff_g)), crc_ok=bool(ok[k])))
        print(f'drop={drop} rows={len(rec.rows)}', flush=True)
    rows = rec.rows
    rec.finish()
    bins = [0, .01, .05, .1, .2, .4, .7, 1.0001]
    report = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = [r for r in rows if lo <= r['pred_tbler'] < hi]
        if s:
            report.append(dict(pred_lo=lo, pred_hi=hi, n=len(s), mean_pred=float(np.mean([r['pred_tbler'] for r in s])),
                               actual=float(1-np.mean([r['crc_ok'] for r in s]))))
    for age in args.ages:
        s = [r for r in rows if r['age'] == age and r['step'] == 0]
        report.append(dict(age=age, step0_n=len(s), step0_pred=float(np.mean([r['pred_tbler'] for r in s])),
                           step0_actual=float(1-np.mean([r['crc_ok'] for r in s])),
                           genie_minus_rx_db=float(np.mean([r['eff_genie_db']-r['eff_rx_db'] for r in s]))))
    (Path(args.out)/'reliability.json').write_text(json.dumps(report, indent=1))
    for r in report:
        print(r)


if __name__ == '__main__':
    main()
