#!/usr/bin/env python3
"""Drop-clustered summaries, paired intervals, plots, and reproducibility audit."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def bootstrap(x,seed=20261007):
    x=np.asarray(x,dtype=float)
    rng=np.random.default_rng(seed)
    ci=np.quantile(x[rng.integers(len(x),size=(10000,len(x)))].mean(1),[.025,.975])
    return dict(mean=float(x.mean()),ci95=ci.tolist(),drops=len(x))


def load_rows(p):
    if not (p/'complete.json').exists():raise ValueError(f'Incomplete run: {p}')
    return [json.loads(x) for x in (p/'rows.jsonl').read_text().splitlines()]


def metrics(rows,slot_s):
    rows=[r for r in rows if not r.get('is_warmup',False)]
    out={}
    slots=len({r['slot'] for r in rows})
    num_ue=len({r['ue'] for r in rows})
    data=[r for r in rows if not r['measurement']]
    out['sum_se_data']=float(np.mean([r['sum_stream_se'] for r in data])*num_ue)
    out['net_sum_se']=float(sum(r['sum_stream_se'] for r in data)/slots)
    if 'successful_bits' in rows[0]:
        out['goodput_mbps']=sum(r['successful_bits'] for r in rows)/(slots*slot_s)/1e6
        out['bler']=float(np.mean([not r['crc_ok'] for r in data]))
        out['pilot_fraction']=len([r for r in rows if r['measurement']])/(slots*num_ue)
        runs=[];window_rates=[]
        for ue in range(num_ue):
            ue_rows=sorted((r for r in data if r['ue']==ue),key=lambda r:r['slot'])
            errors=np.array([not r['crc_ok'] for r in ue_rows],dtype=int)
            run=maximum=0
            for e in errors:
                run=run+1 if e else 0; maximum=max(run,maximum)
            runs.append(maximum)
            window_rates.extend(np.convolve(errors,np.ones(8)/8,'valid').tolist())
        out['max_consecutive_failed_tbs']=max(runs)
        out['fraction_8tb_windows_over_25pct']=float(np.mean(np.asarray(window_rates)>.25))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--baseline',default='track16_ageolla')
    p.add_argument('--wire-audit',help='directional_fronthaul.json from audit_dynamic_information.py')
    args=p.parse_args()
    dest=Path(args.out)
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir(parents=True)
    (dest/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    groups=collections.defaultdict(dict);inputs=[]
    for name in args.runs:
        path=Path(name);meta=json.loads((path/'metadata.json').read_text()); rows=load_rows(path)
        inputs.append(dict(path=str(path),metadata=meta))
        g=collections.defaultdict(list)
        for r in rows:g[(r['speed'],r['policy'],r['drop'])].append(r)
        for (speed,method,drop),rs in g.items():
            key=(meta['args']['antennas'],speed,meta['args'].get('local_error',0.),method)
            if drop in groups[key]:raise ValueError(f'Duplicate drop {key}/{drop}')
            groups[key][drop]=metrics(rs,meta['slot_duration_s'])
    summaries=[];paired=[];drop_rows=[]
    for (n,speed,local_error,method),ds in groups.items():
        summary=dict(n=n,speed=speed,local_error=local_error,method=method,drops=len(ds))
        for metric in next(iter(ds.values())):
            stat=bootstrap([v[metric] for v in ds.values()])
            summary[metric]=stat['mean'];summary[metric+'_ci95']=stat['ci95']
        summaries.append(summary)
        for drop,v in ds.items():drop_rows.append(dict(n=n,speed=speed,local_error=local_error,method=method,drop=drop,**v))
        baseline=groups.get((n,speed,local_error,args.baseline),{})
        common=sorted(set(ds)&set(baseline))
        if common and method!=args.baseline:
            for metric in next(iter(ds.values())):
                stat=bootstrap([ds[d][metric]-baseline[d][metric] for d in common])
                paired.append(dict(n=n,speed=speed,local_error=local_error,method=method,baseline=args.baseline,metric=metric,**stat))
    for file,rs in [('summary.csv',summaries),('paired.csv',paired),('per_drop.csv',drop_rows)]:
        fields=sorted({k for row in rs for k in row})
        with (dest/file).open('w') as f:
            w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rs)
    (dest/'summary.json').write_text(json.dumps(summaries,indent=2))
    (dest/'paired.json').write_text(json.dumps(paired,indent=2))
    (dest/'inputs.json').write_text(json.dumps(inputs,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    crc=[r for r in summaries if 'goodput_mbps' in r]
    if crc:
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        order=sorted(crc,key=lambda r:(r['n'],r['method']))
        labels=[str(r['n'])+' / '+r['method']+(f" / error={r['local_error']}" if r['local_error'] else '') for r in order]
        for ax,metric,title in zip(axes,['goodput_mbps','bler'],['Actual first-transmission goodput (Mbit/s)','Actual first-transmission BLER']):
            y=np.array([r[metric] for r in order]);ci=np.array([r[metric+'_ci95'] for r in order])
            ax.bar(np.arange(len(order)),y,color=['#2166ac' if '_g' in r['method'] else '#888888' for r in order])
            ax.errorbar(np.arange(len(order)),y,yerr=np.stack([y-ci[:,0],ci[:,1]-y]),fmt='none',color='black',capsize=3)
            ax.set_xticks(np.arange(len(order)),labels,rotation=70,ha='right');ax.set_title(title)
            if metric=='bler':ax.axhline(.1,color='firebrick',ls='--')
        fig.tight_layout();fig.savefig(dest/'crc_comparison.png',dpi=180);plt.close(fig)
        if args.wire_audit:
            wire=json.loads(Path(args.wire_audit).read_text())
            methods={'track16_g6d1':(6,1),'track16_g9d1':(9,1),'track16_g9d4':(9,1),
                     'track16_g9e4d1':(9,4),'track16_g9e8d1':(9,8)}
            fig,ax=plt.subplots(figsize=(8,4.8))
            for r in crc:
                if r['n']!=4 or r['local_error']!=0 or r['method'] not in methods:continue
                width,period=methods[r['method']]
                x=next(w['per_ru_extra_ul_mbps'] for w in wire if w['n']==4 and w['width']==width and w['period_slots']==period)
                y=r['goodput_mbps'];ci=r['goodput_mbps_ci95']
                late=r['method']=='track16_g9d4'
                ax.errorbar(x,y,yerr=[[y-ci[0]],[ci[1]-y]],fmt='s' if late else 'o',color='#b2182b' if late else '#2166ac',capsize=4)
                label=f'{width}-bit, every {period} slot'+('s' if period>1 else '')+(', delay=4' if late else ', delay=1')
                ax.annotate(label,(x,y),xytext=(5,-15 if r['method']=='track16_g9d1' else 8),textcoords='offset points',fontsize=8)
            for method,color,label in [('track16_ageolla','#666666','Age-grouped OLLA'),('track16_genie','#444444','Fresh genie CQI reference')]:
                r=next(r for r in crc if r['n']==4 and r['local_error']==0 and r['method']==method)
                ax.axhline(r['goodput_mbps'],color=color,ls='--',lw=1,label=label)
            ax.set_xlabel('Additional RU-to-DU wire rate per RU (Mbit/s)')
            ax.set_ylabel('Two-UE first-transmission goodput (Mbit/s)')
            ax.set_xlim(0,6.7);ax.grid(alpha=.2);ax.legend(loc='lower right',fontsize=8)
            ax.set_title('Actual CRC: information timeliness versus reporting cost')
            fig.tight_layout();fig.savefig(dest/'quality_information_frontier.png',dpi=200);plt.close(fig)
            (dest/'wire_audit.json').write_bytes(Path(args.wire_audit).read_bytes())
    manifest={}
    for item in inputs:
        for f in sorted(Path(item['path']).glob('*')):
            if f.is_file():manifest[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (dest/'complete.json').write_text(json.dumps(dict(inputs=args.runs,summary_rows=len(summaries),
        paired_rows=len(paired),ci_method='Paired percentile bootstrap over independent channel drops, not over slots or UEs.'),indent=2))
    for r in summaries:print(r['n'],r['speed'],r['method'],round(r.get('goodput_mbps',r['sum_se_data']),3),flush=True)


if __name__=='__main__':main()
