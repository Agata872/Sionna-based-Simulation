#!/usr/bin/env python3
"""Drop-cluster summaries for coordination-budget or actual-LA experiments."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from summarize_csirs_exploration import ci_mean


def write(directory, groups, kind):
    (directory/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    dependency=Path(__file__).with_name('summarize_csirs_exploration.py')
    (directory/'analysis_dependency.py').write_bytes(dependency.read_bytes())
    result=dict(kind=kind,confidence='3000 paired bootstrap draws, independent channel drops clustered across UEs and temporal slots',groups=groups)
    (directory/'summary.json').write_text(json.dumps(result,indent=2))
    with (directory/'summary.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,list(groups[0]));wr.writeheader();wr.writerows(groups)
    files=['rows.jsonl','metadata.json','experiment_source.py','analysis_source.py','analysis_dependency.py']
    (directory/'manifest.json').write_text(json.dumps({f:hashlib.sha256((directory/f).read_bytes()).hexdigest() for f in files},indent=2))


def coordination(directory, rows):
    groups={}
    for r in rows:groups.setdefault((r['n'],r['k'],r['rho'],r['eta'],r['method']),[]).append(r)
    rng=np.random.default_rng(20261008)
    key=lambda r:(r['n'],r['k'],r['rho'],r['eta'],r['drop'])
    team={key(r):r for r in rows if r['method']=='team_reference'}
    local={key(r):r for r in rows if r['method']=='local_posterior_mmse'}
    output=[]
    for (n,k,rho,eta,method),rs in sorted(groups.items()):
        se=[r['sum_se'] for r in rs]
        delta=[r['paired_delta_vs_full_w9'] for r in rs]
        dt=[r['sum_se']-team[key(r)]['sum_se'] for r in rs]
        dl=[r['sum_se']-local[key(r)]['sum_se'] for r in rs]
        output.append(dict(n=n,k=k,rho=rho,eta=eta,method=method,drops=len(rs),width=rs[0]['width'],
                           sum_se_mean=float(np.mean(se)),sum_se_ci95=ci_mean(se,rng),
                           delta_vs_full_w9=float(np.mean(delta)),delta_vs_full_w9_ci95=ci_mean(delta,rng),
                           delta_vs_team_reference=float(np.mean(dt)),delta_vs_team_reference_ci95=ci_mean(dt,rng),
                           delta_vs_local=float(np.mean(dl)),delta_vs_local_ci95=ci_mean(dl,rng),
                           ul_wire_bits=rs[0]['ul_wire_bits_per_ru_update'],dl_wire_bits=rs[0]['dl_wire_bits_per_ru_update'],
                           clipped_fraction=rs[0]['batch_quantized_clipped_fraction']))
    write(directory,output,'coordination')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    loads=sorted({(r['n'],r['k']) for r in output})
    fig,axes=plt.subplots(2,4,figsize=(15,7),squeeze=False)
    methods=['full_w_bfp9','proximal_tracking_bfp9','local_posterior_mmse','team_reference','team_summary_bfp9','team_residual_bfp9','team_residual_matched_budget']
    for ax,(n,k) in zip(axes.flat,loads):
        for m in methods:
            rs=sorted([r for r in output if (r['n'],r['k'])==(n,k) and r['eta']==0 and r['method']==m],key=lambda r:r['rho'])
            ax.plot([r['rho'] for r in rs],[r['sum_se_mean'] for r in rs],'.-',label=m)
        ax.set_title(f'N/RU={n}, K={k}, fresh CSI perfect')
        ax.set_xlabel('Endpoint correlation rho');ax.set_ylabel('Instantaneous sum SE');ax.grid(alpha=.2)
    if len(loads)<8:axes.flat[-1].axis('off');axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(),loc='center',fontsize=8)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(directory/f'coordination_budget.{ext}',dpi=160)
    print('coordination groups',len(output))


def closed_loop(directory,rows):
    meta=json.loads((directory/'metadata.json').read_text())['args']
    groups={}
    for r in rows:
        if r['is_warmup']:continue
        groups.setdefault((r['snr_db'],r['method'],r['width'],r['mode']),{}).setdefault(r['drop'],[]).append(r)
    per_drop={}
    for key,drops in groups.items():
        for d,rs in drops.items():
            transmissions=len(rs)/2
            measured=len({r['batch'] for r in rs if r['measurement_slot_before_batch']})
            wall_slots=transmissions+measured
            mbps=sum(r['successful_bits'] for r in rs)/wall_slots*2000/1e6
            per_drop[(*key,d)]=dict(mbps=mbps,bler=np.mean([not r['crc_ok'] for r in rs]),
                                  mean_mcs=np.mean([r['mcs'] for r in rs]),
                                  mean_offset=np.mean([r['olla_before_db'] for r in rs]),
                                  cqi_discrepancy=np.mean([r['rs_cqi']-r['genie_cqi'] for r in rs]),
                                  blocks=len(rs),slots=wall_slots)
    rng=np.random.default_rng(20261008)
    output=[]
    for (snr,method,width,mode),drops in sorted(groups.items()):
        vals=[per_drop[(snr,method,width,mode,d)] for d in drops]
        delta=[v['mbps']-per_drop[(snr,'bfp_norm',3,mode,d)]['mbps'] for d,v in zip(drops,vals)]
        output.append(dict(snr_db=snr,method=method,width=width,mode=mode,independent_drops=len(drops),
                           sum_goodput_mbps=float(np.mean([v['mbps'] for v in vals])),
                           sum_goodput_ci95=ci_mean([v['mbps'] for v in vals],rng),
                           paired_goodput_delta_vs_bfp3=float(np.mean(delta)),paired_delta_ci95=ci_mean(delta,rng),
                           bler=float(np.mean([v['bler'] for v in vals])),bler_ci95=ci_mean([v['bler'] for v in vals],rng),
                           mean_mcs=float(np.mean([v['mean_mcs'] for v in vals])),mean_olla_offset_db=float(np.mean([v['mean_offset'] for v in vals])),
                           mean_rs_minus_genie_cqi=float(np.mean([v['cqi_discrepancy'] for v in vals])),
                           evaluated_crc_blocks=sum(v['blocks'] for v in vals),wall_slots_per_drop=vals[0]['slots']))
    write(directory,output,'closed_loop')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    modes=meta['modes'];fig,axes=plt.subplots(1,len(modes),figsize=(5*len(modes),4),squeeze=False)
    for ax,mode in zip(axes.flat,modes):
        rs=[r for r in output if r['mode']==mode]
        for r in rs:
            label=f"{r['method']} / {r['width']}"
            lo,hi=r['sum_goodput_ci95'];mean=r['sum_goodput_mbps']
            ax.errorbar(label,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',capsize=4)
        ax.set_title(mode);ax.set_ylabel('Sum first-transmission goodput (Mbit/s)');ax.tick_params(axis='x',labelrotation=25);ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(directory/f'closed_loop_goodput.{ext}',dpi=160)
    print('closed-loop groups',len(output))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('kind',choices=['coordination','closed_loop']);p.add_argument('directory',type=Path)
    args=p.parse_args()
    if not (args.directory/'complete.json').exists():raise RuntimeError('Incomplete run')
    rows=[json.loads(s) for s in (args.directory/'rows.jsonl').read_text().splitlines()]
    (coordination if args.kind=='coordination' else closed_loop)(args.directory,rows)


if __name__=='__main__':main()
