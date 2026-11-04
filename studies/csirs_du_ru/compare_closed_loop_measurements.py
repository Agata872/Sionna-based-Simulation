#!/usr/bin/env python3
"""Paired matched/mismatched CSI-RS representations; includes limit checks."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from summarize_csirs_exploration import ci_mean


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('matched',type=Path);p.add_argument('mismatched',type=Path)
    args=p.parse_args()
    def read(d):
        if not (d/'complete.json').exists():raise RuntimeError('Incomplete batch')
        return [json.loads(s) for s in (d/'rows.jsonl').read_text().splitlines()]
    a,b=read(args.matched),read(args.mismatched)
    key=lambda r:(r['seed'],r['method'],r['width'],r['mode'],r['batch'],r['slot_in_batch'],r['ue'])
    match={key(r):r for r in a}
    counts=0
    for r in b:
        x=match[key(r)]
        if r['width']==9:
            for f in ['crc_ok','mcs','cqi','successful_bits','olla_before_db']:
                assert r[f]==x[f],(key(r),f,r[f],x[f])
            counts+=1
    groups=defaultdict(lambda:defaultdict(list))
    for r in b:
        if not r['is_warmup']:groups[r['method'],r['width'],r['mode']][r['seed']].append(r)
    rng=np.random.default_rng(20261008);out=[]
    for (method,width,mode),drops in groups.items():
        deltas=[];bl=[];bias=[];offset=[]
        for seed,rs in drops.items():
            wall=len(rs)/2+len({r['batch'] for r in rs if r['measurement_slot_before_batch']})
            deltas.append(sum(r['successful_bits']-match[key(r)]['successful_bits'] for r in rs)/wall*.002)
            bl.append(np.mean([not r['crc_ok'] for r in rs]))
            bias.append(np.mean([r['rs_cqi']-r['genie_cqi'] for r in rs]))
            offset.append(np.mean([r['olla_before_db'] for r in rs]))
        out.append(dict(method=method,width=width,mode=mode,independent_drops=len(drops),
                        mismatch_minus_matched_goodput_mbps=float(np.mean(deltas)),paired_ci95=ci_mean(deltas,rng),
                        mismatch_bler=float(np.mean(bl)),mismatch_rs_minus_genie_cqi=float(np.mean(bias)),
                        mismatch_mean_olla_offset_db=float(np.mean(offset))))
    temporal=[]
    tg=defaultdict(list)
    for r in b:tg[r['method'],r['width'],r['mode'],r['batch']].append(r)
    for (method,width,mode,batch),rs in sorted(tg.items()):
        temporal.append(dict(method=method,width=width,mode=mode,batch=batch,
                             bler=float(np.mean([not r['crc_ok'] for r in rs])),
                             offset_db=float(np.mean([r['olla_before_db'] for r in rs])),
                             success_bits_mean=float(np.mean([r['successful_bits'] for r in rs]))))
    dest=args.mismatched
    (dest/'measurement_comparison.json').write_text(json.dumps(dict(control_check='BFP9 data/pilot rows identical',
                                                                  checked_control_rows=counts,groups=out,temporal=temporal),indent=2))
    (dest/'comparison_source.py').write_bytes(Path(__file__).read_bytes())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,field,label in zip(axes,['bler','offset_db'],['First-transmission BLER','Mean OLLA offset (dB)']):
        for method,width in [('bfp_norm',3),('phase_channel_norm',3),('bfp_norm',9)]:
            rs=[r for r in temporal if (r['method'],r['width'],r['mode'])==(method,width,'rs_olla')]
            ax.plot([4*r['batch'] for r in rs],[r[field] for r in rs],label=f'{method}/{width}')
        ax.axvline(64,color='grey',linestyle=':',label='End of warmup');ax.set_xlabel('Data transmissions before batch');ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(dest/f'measurement_mismatch_recovery.{ext}',dpi=160)
    print('paired groups',len(out),'control rows',counts)


if __name__=='__main__':main()
