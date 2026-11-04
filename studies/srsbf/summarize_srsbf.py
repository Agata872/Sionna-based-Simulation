#!/usr/bin/env python3
"""Summarize stage-4 SRS-BF scheduling runs with drop-clustered paired bootstrap."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np


def load(run):
    run = Path(run)
    meta = json.loads((run/'metadata.json').read_text())
    rows = [json.loads(x) for x in (run/'rows.jsonl').open()]
    return meta, rows


def per_drop(meta, rows):
    """policy -> drop -> metrics dict."""
    a = meta['args']
    span = (a['slots']-a['warmup'])*0.5e-3
    k = a['num_ue']
    out = defaultdict(dict)
    groups = defaultdict(list)
    for r in rows:
        groups[(r['policy'], r['drop'])].append(r)
    for (pol, drop), rs in groups.items():
        s = [r for r in rs if r['ue'] >= 0]
        ue_bits = np.zeros(k)
        for r in s:
            ue_bits[r['ue']] += r['bits']
        ue_rate = ue_bits/span/1e6
        err = np.array([r['pred_eff_db']-r['true_eff_db'] for r in s]) if s else np.zeros(1)
        layers = {}
        for r in rs:
            layers[r['slot']] = r['layers']
        out[pol][drop] = dict(
            thr=float(ue_rate.sum()),
            util=float(np.sum(np.log(np.maximum(ue_rate, 1e-3)))),
            p10=float(np.percentile(ue_rate, 10)),
            bler=float(1-np.mean([r['ok'] for r in s])) if s else float('nan'),
            layers=float(np.mean(list(layers.values()))),
            err_mean=float(err.mean()), err_std=float(err.std()), err_mae=float(np.abs(err).mean()),
            olla=float(np.mean([r['olla'] for r in s])) if s else 0.)
    return out


def boot(x, n=10000, seed=1):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+')
    p.add_argument('--ref', default='oracle')
    p.add_argument('--out')
    a = p.parse_args()
    report = {}
    for run in a.runs:
        meta, rows = load(run)
        d = per_drop(meta, rows)
        drops = sorted(set.intersection(*[set(v) for v in d.values()]))
        print(f'\n== {run}  speed={meta["args"].get("speed")}  drops={len(drops)}')
        print(f'{"policy":10s} {"thr":>7s} {"vs_ref%":>8s} {"CI95":>17s} {"p10":>6s} {"util":>7s} {"BLER":>6s} {"lay":>5s} {"errM":>6s} {"errS":>6s} {"olla":>6s}')
        ref = d[a.ref]
        res = {}
        for pol, v in d.items():
            m = {key: float(np.mean([v[x][key] for x in drops])) for key in v[drops[0]]}
            rel = [100*(v[x]['thr']/ref[x]['thr']-1) for x in drops]
            lo, hi = boot(rel)
            res[pol] = dict(mean=m, rel_thr_pct=float(np.mean(rel)), rel_ci95=[lo, hi])
            print(f'{pol:10s} {m["thr"]:7.1f} {np.mean(rel):+8.1f} [{lo:+7.1f},{hi:+7.1f}] {m["p10"]:6.2f} {m["util"]:7.2f} '
                  f'{m["bler"]:6.3f} {m["layers"]:5.2f} {m["err_mean"]:+6.2f} {m["err_std"]:6.2f} {m["olla"]:+6.2f}')
        report[str(run)] = dict(speed=meta['args'].get('speed'), drops=drops, ref=a.ref, policies=res)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
