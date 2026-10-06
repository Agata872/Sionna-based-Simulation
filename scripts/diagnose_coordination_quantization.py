#!/usr/bin/env python3
"""Separate statistics quantization from coefficient quantization failures."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from explore_coordination_budget import (cn,inverse_stage,expected_b,coordination,
                                        per_ru_power,quantize_matrix,se_metrics)
from explore_csirs_precoding import Recorder


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--drops',type=int,default=256);p.add_argument('--mc-samples',type=int,default=2048)
    args=p.parse_args()
    torch.set_num_threads(2)
    rec=Recorder(args.out)
    (rec.path/'experiment_source.py').write_bytes(Path(__file__).read_bytes())
    (rec.path/'dependency_explore_coordination_budget.py').write_bytes(Path(__file__).with_name('explore_coordination_budget.py').read_bytes())
    meta=dict(args=vars(args),seed=20261007,scope='rho=.95, eta=0, SNR20; same original load indices and evaluated drops',
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              caveat='Ideal downlink coefficient branches isolate impairments and are not finite-budget deployable schemes.')
    (rec.path/'metadata.json').write_text(json.dumps(meta,indent=2))
    for load_id,n,k in [(0,4,2),(2,4,8),(4,16,8),(6,32,8)]:
        with torch.no_grad():
            gen=torch.Generator(device=args.device).manual_seed(20261007+load_id*100003)
            old=cn((args.drops,2,k,n),gen,args.device)
            innovation=cn(old.shape,gen,args.device)
            fresh=.95*old+math.sqrt(1-.95**2)*innovation
            sigma=k*.01
            local=inverse_stage(fresh,sigma)
            mcgen=torch.Generator(device=args.device).manual_seed(20261007+900000+load_id*100003)
            b=expected_b(old,.95,0,sigma,args.mc_samples,mcgen)
            identity=torch.eye(k,device=args.device,dtype=b.dtype)
            qb=quantize_matrix(b,9,1/k)
            qd=quantize_matrix(identity-b,9,1/k)
            bd=identity-qd
            ideal_c=coordination(b)
            cases=[('ideal_stats_ideal_coeff',b,ideal_c),
                   ('ideal_stats_coeff9',b,quantize_matrix(ideal_c,9,1.)),
                   ('stats9_ideal_coeff',qb,coordination(qb)),
                   ('stats9_coeff9',qb,quantize_matrix(coordination(qb),9,1.)),
                   ('residual9_ideal_coeff',bd,coordination(bd)),
                   ('residual9_coeff9',bd,quantize_matrix(coordination(bd),9,1.))]
            # A conventional matrix-validity repair; no extra bits.
            evals,evecs=torch.linalg.eigh((qb+qb.mH)/2)
            for margin in [.0001,.001,.01]:
                projected=evecs@torch.diag_embed(evals.clamp(0,1-margin).to(b.dtype))@evecs.mH
                cp=quantize_matrix(coordination(projected),9,1.)
                cases.append((f'psd_projection_margin{margin:g}_coeff9',projected,cp))
            for name,stats,c in cases:
                se,_=se_metrics(fresh,per_ru_power(local@c),.01)
                symmetric=(stats+stats.mH)/2
                eigenvalues=torch.linalg.eigvalsh(symmetric)
                system=torch.cat([torch.cat([identity.expand(args.drops,k,k),stats[:,1]],-1),
                                  torch.cat([stats[:,0],identity.expand(args.drops,k,k)],-1)],-2)
                condition=torch.linalg.cond(system)
                mismatch=torch.linalg.vector_norm(stats-b,dim=(-2,-1)).mean(-1)
                for d in range(args.drops):
                    rec.add(dict(drop=d,n=n,k=k,method=name,sum_se=float(se[d]),
                                 system_condition=float(condition[d]),stats_error_frobenius=float(mismatch[d]),
                                 stats_max_eigenvalue=float(eigenvalues[d].max()),
                                 stats_min_eigenvalue=float(eigenvalues[d].min()),
                                 coefficient_norm=float(torch.linalg.vector_norm(c[d]))))
            print('diagnosed',n,k,flush=True)
    rec.finish()
    import numpy as np
    from summarize_csirs_exploration import ci_mean
    rng=np.random.default_rng(20261008)
    groups=[]
    baseline={(r['n'],r['k'],r['drop']):r for r in rec.rows if r['method']=='ideal_stats_ideal_coeff'}
    for n,k in [(4,2),(4,8),(16,8),(32,8)]:
        for method in sorted({r['method'] for r in rec.rows}):
            rs=[r for r in rec.rows if (r['n'],r['k'],r['method'])==(n,k,method)]
            delta=[r['sum_se']-baseline[n,k,r['drop']]['sum_se'] for r in rs]
            groups.append(dict(n=n,k=k,method=method,sum_se_mean=float(np.mean([r['sum_se'] for r in rs])),
                               delta_vs_reference=float(np.mean(delta)),delta_ci95=ci_mean(delta,rng),
                               eigenvalue_above_one_fraction=float(np.mean([r['stats_max_eigenvalue']>=1 for r in rs])),
                               median_system_condition=float(np.median([r['system_condition'] for r in rs])),
                               p95_system_condition=float(np.quantile([r['system_condition'] for r in rs],.95))))
    (rec.path/'summary.json').write_text(json.dumps(dict(groups=groups,scope=meta['scope']),indent=2))
    (rec.path/'analysis_dependency.py').write_bytes(Path(__file__).with_name('summarize_csirs_exploration.py').read_bytes())


if __name__=='__main__':main()
