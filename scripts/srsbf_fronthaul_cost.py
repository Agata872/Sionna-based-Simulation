#!/usr/bin/env python3
"""O-RU -> O-DU information cost of the SRS-BF-DL options (CUS v22.00 field sizes).

Rough wire-rate accounting (bits/s) per option; message overhead = Ethernet
(preamble+SFD 8, header 14, VLAN 4, FCS 4, IFG 12) + eCPRI 8 + C-plane common
header 8 + one 8-byte section header per UE/request; MTU 1500 fragmentation.
Field sizes: ST 6 SRS CI BFP (ciIqWidth I/Q per element per PRB group, 8-bit
exponent per group), T-matrix (tmtxLayerNorm 16 b/layer, tmtxOrthRe/Im 16 b),
normalized beam power (beamMask ceil(Kb/8) B + 16 b/beam per port),
MEAS_DL_POST_EQ_SINR (request: 32 b command header + 32 b per layer for
ueId+channelPathLoss; response: 16 b dlPostEqSinr per layer).
"""
import json
import math
import sys

ETH = 8+14+4+4+12
OVH = ETH+8+8          # per message, before section headers
MTU = 1500


def wire_bits(payload_bytes, sections):
    body = payload_bytes+8*sections
    msgs = max(1, math.ceil(body/(MTU-16)))
    return 8*(body+msgs*OVH)


def options(cfg):
    m, nap, k, tsrs, grp, w = cfg['M'], cfg['N_ap'], cfg['K'], cfg['T_srs'], cfg['groups'], cfg['ci_width']
    slots = 2000.
    occ = k*slots/tsrs                                                # SRS occasions per second
    out = {}
    ci = nap*grp*(m*2*w+8)/8
    out['ST6_SRS_CI_BFP%d' % w] = occ*wire_bits(ci, 1)
    tmatrix = (nap*16+nap*nap*32)/8
    beams = nap*(math.ceil(m/8)+cfg['beams']*2)
    out['ST13_Tmatrix+beam_power'] = occ*wire_bits(tmatrix+beams, 1)
    out['ST13_CI_coherence_(proposed)'] = occ*wire_bits(2, 1)
    lay = cfg['layers']
    req = 4+4*lay
    resp = 2*lay
    per_period = cfg['budget']*(wire_bits(req, 1)+wire_bits(resp, 1))
    out['ST11+13_queries_B%d_per_TDD' % cfg['budget']] = per_period*slots/cfg['tdd_period']
    uplane = lay*cfg['prb']*12*14*2*9*slots                           # DL layer U-plane payload, BFP9
    out['reference_DL_Uplane_payload_%d_layers' % lay] = uplane
    return out


def main():
    cases = {
        'simulated (M=32, 2 ports, 52 PRB, K=8)': dict(M=32, N_ap=2, K=8, T_srs=10, groups=26, ci_width=9,
                                                        beams=8, layers=8, budget=4, tdd_period=5, prb=52),
        '100 MHz 64T (M=64, 4 ports, 273 PRB, K=32)': dict(M=64, N_ap=4, K=32, T_srs=10, groups=137, ci_width=9,
                                                            beams=16, layers=16, budget=16, tdd_period=5, prb=273),
    }
    res = {}
    for name, cfg in cases.items():
        res[name] = {k: v/1e6 for k, v in options(cfg).items()}
        print(f'\n{name}  (Mbit/s, O-RU->O-DU unless noted)')
        for k, v in res[name].items():
            print(f'  {k:45s} {v:10.3f}')
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'w') as f:
            json.dump(dict(cases=res, note=__doc__), f, indent=1)


if __name__ == '__main__':
    main()
