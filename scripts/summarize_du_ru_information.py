#!/usr/bin/env python3
"""Summarize stage-2 diagnostic; no actual NR or novelty conclusions."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from summarize_csirs_exploration import ci_mean


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    args=p.parse_args()
    directory=args.directory
    if not (directory/'complete.json').exists(): raise RuntimeError('Experiment incomplete')
    rows=[json.loads(s) for s in (directory/'rows.jsonl').read_text().splitlines()]
    keys=lambda r:(r['user_correlation'],r['rho'],r['snr_db'],r['drop'],r['ue'])
    baseline={keys(r):r for r in rows if r['method']=='stale_global_bfp' and r['width']==9}
    fresh={keys(r):r for r in rows if r['method']=='fresh_global_reference'}
    groups={}
    for r in rows:
        groups.setdefault((r['user_correlation'],r['rho'],r['snr_db'],r['method'],r['width']),[]).append(r)
    rng=np.random.default_rng(20261006)
    output=[]
    for (corr,rho,snr,method,width),rs in sorted(groups.items()):
        drops={}
        for r in rs: drops.setdefault(r['drop'],[]).append(r)
        se=[sum(r['true_se'] for r in d) for d in drops.values()]
        delta=[sum(r['true_se']-baseline[keys(r)]['true_se'] for r in d) for d in drops.values()]
        risk=[np.mean([r['prediction_error_db']>3 for r in d]) for d in drops.values()]
        ref=np.mean([sum(fresh[keys(r)]['true_se'] for r in d) for d in drops.values()])
        output.append(dict(user_correlation=corr,rho=rho,snr_db=snr,method=method,width=width,independent_drops=len(drops),
                           sum_se_mean=float(np.mean(se)),sum_se_loss_vs_fresh_reference_percent=float(100*(1-np.mean(se)/ref)),
                           paired_sum_se_delta_vs_stale9_mean=float(np.mean(delta)),paired_sum_se_delta_vs_stale9_ci95=ci_mean(delta,rng),
                           genie_overprediction_gt3db_percent=float(100*np.mean(risk)),
                           genie_overprediction_gt3db_percent_ci95=[100*x for x in ci_mean(risk,rng)],
                           bfw_payload_bits_per_ru_update=rs[0]['bfw_payload_bits_per_ru_update']))
    (directory/'summary.json').write_text(json.dumps(dict(confidence_method='paired drop-cluster bootstrap, 3000 draws',
                                                         caveat='Genie prediction risk; not NR CQI error or BLER. Baselines are screening only.',groups=output),indent=2))
    with (directory/'summary.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,list(output[0]));wr.writeheader();wr.writerows(output)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,corr in zip(axes,[0.,.7]):
        methods=[('fresh_global_reference',16),('stale_global_ideal',16),('stale_global_bfp',9),('local_rzf_with_du_phase',16),('blend_0.5',9)]
        if any(r['method']=='local_tracking_1' for r in output):methods.append(('local_tracking_1',9))
        for method,width in methods:
            rs=sorted([r for r in output if r['user_correlation']==corr and r['snr_db']==20 and r['method']==method and r['width']==width],key=lambda r:r['rho'])
            ax.plot([r['rho'] for r in rs],[r['sum_se_mean'] for r in rs],'o-',label=method)
        ax.set_title(f'User correlation {corr:g}, SNR 20 dB')
        ax.set_xlabel('Channel correlation rho (not milliseconds)');ax.set_ylabel('Genie sum SE (bit/s/Hz)')
        ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(directory/f'information_age_screen.{ext}',dpi=160)
    print('complete',directory,len(output),'groups')


if __name__=='__main__':main()
