#!/usr/bin/env python3
"""Information counterexamples and directional wire-rate audit, not novelty claims."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from oran import ORANConfig, CompressionConfig
from oran.fronthaul import fronthaul_rate,cplane_rate
from explore_coordination_budget import wire_bits


def equation_validation():
    """Independent SINR identity and a static SOPIP fixed-point check on CPU."""
    import torch
    from explore_dynamic_measurement import reported_sinr, ORAN_precoder, ru_power, sopip_pipeline
    generator = torch.Generator(device='cpu').manual_seed(314159)
    def channel(shape):
        return torch.complex(torch.randn(shape, generator=generator, dtype=torch.float64),
                             torch.randn(shape, generator=generator, dtype=torch.float64)) * .25
    g = channel((1, 3, 2, 4, 4))
    noise = .01
    direct = torch.empty((1, 2, 2, 3), dtype=torch.float64)
    eye = torch.eye(4, dtype=torch.complex128)
    for user in range(2):
        for layer in range(2):
            index = user * 2 + layer
            desired = g[..., user, :, index]
            interference = g[..., user, :, [j for j in range(4) if j != index]]
            covariance = interference @ interference.mH + noise * eye
            gamma = (desired.conj() * torch.linalg.solve(covariance, desired)).sum(-1).real
            direct[:, user, layer, :] = gamma
    relative = float(((reported_sinr(g, noise) - direct).abs() / direct).max())
    assert relative < 1e-10
    h = channel((1, 3, 2, 4, 8))
    target = ru_power(ORAN_precoder(h), 4)
    result = sopip_pipeline([h] * 12, {0: target}, 4, 1, alpha=512., jcc=20, jap=20, width=0)
    fixed_error = max(float(torch.linalg.vector_norm(w - target)) for w in result)
    assert fixed_error < 1e-10
    return dict(reported_sinr_relative_error_vs_independent_interference_covariance=relative,
                static_unquantized_sopip_fixed_point_error=fixed_error,
                dtype='complex128', device='cpu', seed=314159, status='passed')


def raw_float_bits(values):
    payload=values*8  # complex64 = float32 real + float32 imag
    return 8*(payload+math.ceil(payload/1476)*66)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);a=p.parse_args()
    dest=Path(a.out)
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir(parents=True)
    (dest/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    no=.01;u=np.array([.1,0.]);v=np.array([0.,math.sqrt(.99)])
    # Single transmit antenna per RU; W_l=sqrt(0.5), channels unit norm.
    h1a=u+v;h1b=u-v;h2=u+v;w=math.sqrt(.5);projection=np.array([[1.,0.]])
    assert np.allclose(projection@(h1a*w),projection@(h1b*w))
    assert np.allclose(np.linalg.norm(h1a),np.linalg.norm(h1b))
    sinra=np.linalg.norm((h1a+h2)*w)**2/no
    sinrb=np.linalg.norm((h1b+h2)*w)**2/no
    counterexample=dict(projection=projection.tolist(),ru1_world_a=h1a.tolist(),ru1_world_b=h1b.tolist(),
        ru2_same_both_worlds=h2.tolist(),power_per_ru=.5,noise_variance=no,
        identical_projected_ru1_messages=True,sinr_linear_a=sinra,sinr_linear_b=sinrb,
        sinr_db_a=float(10*np.log10(sinra)),sinr_db_b=float(10*np.log10(sinrb)),
        exact_difference_db=float(10*np.log10(sinra/sinrb)),
        scope='Fixed rank-deficient linear receiver projection cannot certify adaptive-receiver quality uniformly over admissible remote states. This is a diagnostic construction, not a new communication-complexity theorem.')
    (dest/'counterexample.json').write_text(json.dumps(counterexample,indent=2))
    inventory=[]
    for n in [4,8]:
        cfg=ORANConfig(category='B',num_oru=2,num_oru_ant=n,num_prb=6,
                       uplane=CompressionConfig('bfp',9),bfw=CompressionConfig('bfp',9))
        rate=fronthaul_rate(cfg);base=cplane_rate(cfg.num_layers,cfg.slots_per_second)
        down_per_ru=rate.uplane_total+base+(rate.cplane_total-base)/16
        # Existing main runs use perfect delayed global CSI as an abstraction.
        # Charge lossless raw float32/complex64 transport, not fictitious H9.
        old_h=raw_float_bits(cfg.num_ue*cfg.num_ue_ant*n*cfg.num_prg)*cfg.slots_per_second/16
        for width,period in [(6,1),(9,1),(9,4),(9,8)]:
            report=wire_bits(cfg.num_ue*cfg.num_ue_ant,cfg.num_layers*cfg.num_prg,width)
            extra=report*cfg.slots_per_second/period
            inventory.append(dict(n=n,width=width,period_slots=period,
                per_ru_g_report_bits=report,per_ru_extra_ul_mbps=extra/1e6,
                baseline_per_ru_ul_old_csi_mbps=old_h/1e6,
                total_per_ru_ul_mbps=(old_h+extra)/1e6,
                total_two_ru_downlink_mbps=2*down_per_ru/1e6,
                extra_two_ru_ul_mbps=2*extra/1e6,
                accounting='Full-grid U-plane and per-slot scheduling C-plane, SE11 BFW once/16 slots; lossless delayed CSI float32 upload once/16 slots; hypothetical packed G messages with 66-byte overhead. No physical SRS, air CQI feedback, ACK bytes, transport queues, or retransmission traffic modeled.'))
    (dest/'directional_fronthaul.json').write_text(json.dumps(inventory,indent=2))
    (dest/'equation_validation.json').write_text(json.dumps(equation_validation(), indent=2))
    (dest/'complete.json').write_text(json.dumps(dict(status='passed',sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    print('Same projection, same powers, adaptive-receiver SINR difference:',counterexample['exact_difference_db'])
    for r in inventory:print(r['n'],r['width'],r['period_slots'],r['per_ru_extra_ul_mbps'],r['total_two_ru_downlink_mbps'])


if __name__=='__main__':main()
