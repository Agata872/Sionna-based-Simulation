#!/usr/bin/env python3
"""Screen receiver-projected RU messages against full effective-channel reports.

These are ordinary projection/robust-bound baselines, NOT a novelty claim.
Projection filters use only each DU anchor's old global CSI and old W. Current
RU messages use own current CSI/W; delayed reports are explicitly not certified
for the current time unless an independently justified aging bound is supplied.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import sionna.phy
from oran import ORANConfig, ORANDownlink, CompressionConfig
from explore_dynamic_measurement import continuous_trace, quantize, tracking
from explore_coordination_budget import wire_bits, fitting_width, quantize_matrix
from explore_csirs_precoding import Recorder


def reference_filters(g, no, nu):
    """Normalized per-stream linear MMSE vectors: [...,K,R,L] -> [...,L,R]."""
    out = []
    eye = torch.eye(g.shape[-2], dtype=g.dtype, device=g.device)
    for k in range(g.shape[-3]):
        a = g[..., k, :, :]
        v = torch.linalg.solve(a@a.mH+no*eye, a[..., k*nu:(k+1)*nu])
        v = v/torch.linalg.vector_norm(v,dim=-2,keepdim=True).clamp_min(1e-12)
        out.append(v.mH)
    return torch.cat(out,-2)


def projected(g, filters, nu):
    return torch.cat([filters[...,k*nu:(k+1)*nu,:]@g[...,k,:,:] for k in range(g.shape[-3])],-2)


def fixed_sinr(z, no):
    signal = z.diagonal(dim1=-2,dim2=-1).abs().square()
    return signal/((z.abs().square().sum(-1)-signal).clamp_min(0)+no)


def optimal_sinr(g, no, nu):
    return fixed_sinr(projected(g,reference_filters(g,no,nu),nu),no)


def quantized_with_radius(x,width):
    """Pure column BFP with shared exponent, no preliminary fixed16 clipping.

    Per column step is the next power of two accommodating real/imag maxima;
    8-bit exponent charged by wire_bits, exponent overflow rejected. Rectangular
    real/imag quantization gives sqrt(2)*step/2 complex-error bound. This is a
    hypothetical coordination codec, not asserted to be an O-RAN message.
    """
    lim = 2**(width-1)-1
    maximum = torch.maximum(x.real.abs(),x.imag.abs()).amax(-2,keepdim=True)
    exponent = torch.ceil(torch.log2((maximum/lim).clamp_min(2**-100)))
    if not ((exponent >= -128)&(exponent <= 127)).all():raise ValueError('exponent outside int8')
    step = torch.exp2(exponent)
    q = torch.complex(torch.round(x.real/step),torch.round(x.imag/step))*step
    radius = math.sqrt(2)*step.expand_as(x.real)/2
    assert (q-x).abs().le(radius+1e-6).all()
    return q, radius


def certified_sinr(z,radius,no):
    desired=(z.diagonal(dim1=-2,dim2=-1).abs()-radius.diagonal(dim1=-2,dim2=-1)).clamp_min(0).square()
    upper=(z.abs()+radius).square()
    interference=(upper.sum(-1)-upper.diagonal(dim1=-2,dim2=-1)).clamp_min(0)
    return desired/(interference+no)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    p.add_argument('--drops',type=int,default=16)
    p.add_argument('--slots',type=int,default=96)
    p.add_argument('--warmup',type=int,default=32)
    p.add_argument('--speed',type=float,default=3.)
    p.add_argument('--antennas',type=int,default=8)
    p.add_argument('--seed-start',type=int,default=20264000)
    p.add_argument('--subspace',action='store_true',help='also reoptimize receiver inside the old MMSE subspace')
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    torch.set_num_threads(2); sionna.phy.config.device=args.device
    cfg=ORANConfig(category='B',num_oru=2,num_oru_ant=args.antennas,num_prb=6,device=args.device)
    sim=ORANDownlink(cfg);no=.01;nu=2;k=2;r=4;l=k*nu
    rec=Recorder(args.out)
    (rec.path/'experiment_source.py').write_bytes(Path(__file__).read_bytes())
    for name in ['explore_dynamic_measurement.py','explore_coordination_budget.py','explore_csirs_precoding.py']:
        (rec.path/f'dependency_{name}').write_bytes(Path(__file__).with_name(name).read_bytes())
    (rec.path/'metadata.json').write_text(json.dumps(dict(args=vars(args),config=asdict(cfg),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        assumptions=['Continuous fixed-angle CDL Doppler, perfect RU CSI. No SRS/air CQI/CRC in this screen.',
                     'DU sends reference filters along with W every 16 slots, arrives after 4 slots.',
                     'Both message directions include 66-byte custom headers and byte alignment; per-PRG numbers below.',
                     'Reference filters normalized before transmission; actual reconstructed filters consistently used by both ends.',
                     'Fresh quantization bound is deterministic and validated; aged report is NOT a present-time bound.',
                     'Projection and triangle bounds are strong conventional baselines, not new algorithms.']),indent=2))
    max_violation=-float('inf')
    for drop in range(args.drops):
        hs,hc=continuous_trace(sim,args.slots,args.speed,args.seed_start+drop)
        anchors={t:quantize(sim.odu.precoder(hc[t]),args.antennas,9) for t in range(0,args.slots,16)}
        filters={t:reference_filters(hc[t]@anchors[t].unsqueeze(-3),no,nu) for t in anchors}
        for width in [6,9]:
            # Downlink filters encoded in R x L layout then independently renormalized.
            qfilters={}
            for ts,v in filters.items():
                q,_=quantized_with_radius(v.mH,width)
                q=q.mH/torch.linalg.vector_norm(q.mH,dim=-1,keepdim=True).clamp_min(1e-12)
                qfilters[ts]=q
            history=[]
            for t in range(args.slots):
                ts=max(0,((t-4)//16)*16)
                w=tracking(hc[ts],hc[t],anchors[ts],args.antennas,1.)
                g=hc[t]@w.unsqueeze(-3)
                true=optimal_sinr(g,no,nu)
                v=qfilters[ts]
                parts=[hc[t][...,j*args.antennas:(j+1)*args.antennas]@w[...,j*args.antennas:(j+1)*args.antennas,:].unsqueeze(-3)
                       for j in range(2)]
                zs=[projected(a,v,nu) for a in parts]
                messages=[quantized_with_radius(z,width) for z in zs]
                zhat=sum(x[0] for x in messages);radius=sum(x[1] for x in messages)
                fresh_bound=certified_sinr(zhat,radius,no)
                max_violation=max(max_violation,float((fresh_bound-true).max()))
                assert (fresh_bound <= true+2e-3).all()
                cap=wire_bits(l,l,width)
                full_width=fitting_width(k*r,l,cap)
                full=sum(quantized_with_radius(a.flatten(-3,-2),full_width)[0] for a in parts).unflatten(-2,(k,r))
                estimates={'projected_point':fixed_sinr(zhat,no),
                           'projected_bound':fresh_bound,'fixed_receiver_unquantized':fixed_sinr(sum(zs),no),
                           'full_g_same_budget':optimal_sinr(full,no,nu)}
                if args.subspace:
                    # Orthonormal old receiver span: adaptation within the span
                    # is a necessary strong baseline before proposing a new codec.
                    rows=[]
                    for user in range(k):
                        q=torch.linalg.qr(v[...,user*nu:(user+1)*nu,:].mH).Q
                        rows.append(q.mH)
                    basis=torch.cat(rows,-2)
                    sm=[quantized_with_radius(projected(a,basis,nu),width) for a in parts]
                    shat=sum(x[0] for x in sm).unflatten(-2,(k,nu))
                    sradius=sum(x[1] for x in sm)
                    sb=reference_filters(shat,no,nu)
                    processed=projected(shat,sb,nu)
                    processed_radius=torch.cat([sb[...,user*nu:(user+1)*nu,:].abs()@
                                                sradius[...,user*nu:(user+1)*nu,:] for user in range(k)],-2)
                    sbound=certified_sinr(processed,processed_radius,no)
                    assert (sbound <= true+2e-3).all()
                    estimates['subspace_point']=optimal_sinr(shat,no,nu)
                    estimates['subspace_bound']=sbound
                history.append(estimates)
                if t<args.warmup:continue
                for age in [0,1,4]:
                    if t<age:continue
                    for name,value in history[t-age].items():
                        # Aged fixed filters may differ from current filters, but still no future information.
                        estimated_db=10*torch.log10(value.clamp_min(1e-12))
                        true_db=10*torch.log10(true.clamp_min(1e-12))
                        diff=(estimated_db-true_db).flatten().cpu().tolist()
                        rec.add(dict(drop=drop,slot=t,width=width,age=age,method=name,
                                     mean_overprediction_db=sum(diff)/len(diff),max_overprediction_db=max(diff),
                                     mean_absolute_error_db=sum(abs(x) for x in diff)/len(diff),
                                     estimate_se=float(torch.log2(1+value).sum(-1).mean()),
                                     actual_se=float(torch.log2(1+true).sum(-1).mean()),
                                     uplink_bits_per_ru_prg_update=cap,
                                     downlink_filter_bits_per_ru_prg_update=wire_bits(r,l,width),
                                     full_g_matching_width=full_width,raw_csi9_bits=wire_bits(k*r,args.antennas,9)))
            print(f'drop={drop} width={width} rows={len(rec.rows)}',flush=True)
    (rec.path/'validation.json').write_text(json.dumps(dict(max_fresh_bound_minus_optimal_sinr=max_violation,
        status='passed',note='Fresh reports only; no deterministic current-time guarantee claimed for delayed reports.'),indent=2))
    rec.finish()


if __name__=='__main__':main()
