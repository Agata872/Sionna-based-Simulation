#!/usr/bin/env python3
"""Stage-2 diagnostic: information age versus weight precision in two RUs.

Uses existing compute_precoder/IQCodec with an explicitly abstract correlated
Rayleigh/Gauss-Markov channel, not a CDL mobility trajectory or packet model.
Local RZF and an aligned blend are screening baselines, not Team-MMSE/SOPIP.
All weights satisfy the same per-RU power constraints after reconstruction.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import sionna.phy
from oran import ORANConfig, CompressionConfig
from oran.fronthaul import IQCodec
from oran.precoding import compute_precoder
from explore_csirs_precoding import Recorder


def equal_ru_power(w, n):
    q=w.clone()
    for o in range(2):
        block=q[..., o*n:(o+1)*n, :]
        q[..., o*n:(o+1)*n, :]=block / torch.linalg.vector_norm(block, dim=(-2,-1), keepdim=True).clamp_min(1e-12) / math.sqrt(2)
    return q


def reconstruct(w,cfg,width):
    parts=[]
    for o in range(2):
        c=IQCodec(CompressionConfig("bfp",width),1/(cfg.num_tx_ant*cfg.num_layers),cfg.num_oru_ant)
        parts.append(c(w[...,o*cfg.num_oru_ant:(o+1)*cfg.num_oru_ant,:].transpose(-1,-2)).transpose(-1,-2))
    return equal_ru_power(torch.cat(parts,-2),cfg.num_oru_ant)


def metrics(h,w,no,nu):
    g=h@w[:,None]
    sinrs=[]
    for k in range(2):
        d=g[:,k,:,k*nu:(k+1)*nu]
        i=torch.cat([g[:,k,:,:k*nu],g[:,k,:,(k+1)*nu:]],-1)
        r=i@i.mH + no*torch.eye(h.shape[-2],device=h.device)
        err=torch.linalg.inv(torch.eye(nu,device=h.device)+d.mH@torch.linalg.solve(r,d))
        sinrs.append((1/torch.diagonal(err,dim1=-2,dim2=-1).real-1).clamp_min(1e-9))
    s=torch.stack(sinrs,1)
    return 10*torch.log10(s).mean(-1),torch.log2(1+s).sum(-1)


def sample_channel(batch, n, correlation, generator, device):
    z=torch.complex(torch.randn(batch,2,4,2*n,device=device,generator=generator),
                    torch.randn(batch,2,4,2*n,device=device,generator=generator))/math.sqrt(2)
    common=torch.complex(torch.randn(batch,1,4,2*n,device=device,generator=generator),
                         torch.randn(batch,1,4,2*n,device=device,generator=generator))/math.sqrt(2)
    return math.sqrt(1-correlation)*z+math.sqrt(correlation)*common


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--seed',type=int,default=20261006)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    torch.set_num_threads(2)
    sionna.phy.config.device=args.device
    cfg=ORANConfig(category='B',num_oru=2,num_oru_ant=4,device=args.device,num_prb=2)
    rec=Recorder(args.out)
    meta=dict(args=vars(args),config=asdict(cfg),gpu=torch.cuda.get_device_name(),torch=torch.__version__,
              cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              model='Two RU x4 Tx, two UE x4 Rx, two streams/UE; user-correlated Rayleigh; H_new=rho H_old+sqrt(1-rho^2)E; per-RU power=0.5.',
              caveats=['Perfect fresh local and old global CSI; acquisition cost/errors not modeled.',
                       'No NR pilot samples, CQI, actual CRC or latency mapping; SINR prediction is a genie diagnostic.',
                       'Fresh-global reference is noncausal and not a proven upper bound.',
                       'Aligned local/global blend is a screening baseline; Team-MMSE and SOPIP still required.',
                       'Extra beamforming payload per RU/update counted, headers and shared data excluded.'])
    (rec.path/'metadata.json').write_text(json.dumps(meta,indent=2))
    (rec.path/'experiment_source.py').write_text(Path(__file__).read_text())
    gen=torch.Generator(device=args.device).manual_seed(args.seed)
    for corr in (0.,.7):
        old=sample_channel(args.batch_size,4,corr,gen,args.device)
        innovation=sample_channel(args.batch_size,4,corr,gen,args.device)
        old_w=equal_ru_power(compute_precoder(old,2,'bd'),4)
        for rho in (1.,.99,.95,.85,.7):
            fresh=rho*old+math.sqrt(1-rho*rho)*innovation
            for snr in (10.,20.,30.):
                no=10**(-snr/10)
                fresh_w=equal_ru_power(compute_precoder(fresh,2,'bd'),4)
                local_parts=[]
                for o in range(2):
                    # Regularized local precoding fits all streams, unlike local BD (infeasible at 4 Tx).
                    local=compute_precoder(fresh[...,o*4:(o+1)*4],2,'rzf',4*no)
                    ref=old_w[...,o*4:(o+1)*4,:]
                    phase=(local.conj()*ref).sum(-2,keepdim=True)
                    local=local*phase/phase.abs().clamp_min(1e-12)
                    local_parts.append(local)
                local_w=equal_ru_power(torch.cat(local_parts,-2),4)
                cases=[('fresh_global_reference',16,fresh_w),('stale_global_ideal',16,old_w)]
                for width in (3,6,9):
                    cases.append(('stale_global_bfp',width,reconstruct(old_w,cfg,width)))
                # This local solution uses the DU's old W for phase alignment;
                # its coordination message is explicitly counted below.
                cases.append(('local_rzf_with_du_phase',16,local_w))
                cached9=reconstruct(old_w,cfg,9)
                aligned9=local_w.clone()
                for o in range(2):
                    block=aligned9[...,o*4:(o+1)*4,:]
                    phase=(block.conj()*cached9[...,o*4:(o+1)*4,:]).sum(-2,keepdim=True)
                    aligned9[...,o*4:(o+1)*4,:]=block*phase/phase.abs().clamp_min(1e-12)
                for mixing in (.25,.5,.75):
                    # Local floating-point computation after receipt of BFP9 W;
                    # no new precoder or hidden ideal W crosses the fronthaul.
                    mix=equal_ru_power((1-mixing)*cached9+mixing*aligned9,4)
                    cases.append((f'blend_{mixing:g}',9,mix))
                # Stronger local baseline: preserve each RU's old effective
                # contribution, using its own cached old H and received BFP9 W.
                # The local ridge solve has zero update when rho=1. No fresh
                # global CSI, ideal DU W, or additional target message is used.
                for reg in (.01,.1,1.,10.):
                    parts=[]
                    for o in range(2):
                        h_old=old[...,o*4:(o+1)*4].flatten(1,2)
                        h_new=fresh[...,o*4:(o+1)*4].flatten(1,2)
                        prior=cached9[...,o*4:(o+1)*4,:]
                        residual=(h_old-h_new)@prior
                        delta=torch.linalg.solve(h_new.mH@h_new+reg*torch.eye(4,device=args.device),h_new.mH@residual)
                        parts.append(prior+delta)
                    tracked=equal_ru_power(torch.cat(parts,-2),4)
                    cases.append((f'local_tracking_{reg:g}',9,tracked))
                prediction,_=metrics(old,old_w,no,2)
                for name,width,w in cases:
                    db,se=metrics(fresh,w,no,2)
                    assert torch.allclose(w[...,:4,:].abs().square().sum((-2,-1)),torch.full((args.batch_size,),.5,device=args.device),atol=2e-6)
                    assert torch.allclose(w[...,4:,:].abs().square().sum((-2,-1)),torch.full((args.batch_size,),.5,device=args.device),atol=2e-6)
                    # Weight-message payload is unchanged in the blend: local update at RU uses cached DU W.
                    bits=4*(8+2*4*width) if width!=16 else 4*2*4*16
                    for b in range(args.batch_size):
                        for k in range(2):
                            rec.add(dict(seed=args.seed,drop=b,ue=k,user_correlation=corr,rho=rho,snr_db=snr,method=name,width=width,
                                         true_se=float(se[b,k]),true_sinr_db=float(db[b,k]),prediction_error_db=float(prediction[b,k]-db[b,k]),
                                         bfw_payload_bits_per_ru_update=bits))
                print(f'corr={corr:g} rho={rho:g} snr={snr:g} rows={len(rec.rows)}',flush=True)
    rec.finish()


if __name__=='__main__':
    main()
