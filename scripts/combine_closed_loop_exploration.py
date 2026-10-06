#!/usr/bin/env python3
"""Combine independent closed-loop batches after checking compatible settings."""
import argparse
import json
from pathlib import Path
from explore_csirs_precoding import Recorder
from summarize_next_exploration import closed_loop


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directories',nargs='+',type=Path);p.add_argument('--out',required=True)
    args=p.parse_args()
    recorder=Recorder(args.out)
    first=None;seeds=set();metadata=[]
    for directory in args.directories:
        if not (directory/'complete.json').exists():raise RuntimeError(f'Incomplete {directory}')
        meta=json.loads((directory/'metadata.json').read_text());a=meta['args'].copy()
        # Older run has implicit matched policy; later scripts write it explicitly.
        a.setdefault('measurement_policy','matched')
        for field in ['out','seed_start','drops']:a.pop(field,None)
        if first is not None and first!=a:raise ValueError('Different experiment configurations')
        first=a
        rs=[json.loads(s) for s in (directory/'rows.jsonl').read_text().splitlines()]
        newseeds={r['seed'] for r in rs}
        if seeds&newseeds:raise ValueError('Overlapping drops between batches')
        seeds|=newseeds
        for r in rs:
            r['drop']=r['seed'];r['source_directory']=str(directory)
            recorder.add(r)
        metadata.append(meta)
    meta=dict(args=dict(first,drops=len(seeds)),source_directories=list(map(str,args.directories)),
              source_metadata=metadata,scope='Combined independent drops, matched policy and controller settings verified')
    (recorder.path/'metadata.json').write_text(json.dumps(meta,indent=2))
    (recorder.path/'experiment_source.py').write_bytes(Path(__file__).read_bytes())
    recorder.finish()
    closed_loop(recorder.path,recorder.rows)


if __name__=='__main__':main()
