#!/usr/bin/env python3
"""Paired (same seeds/drops) differences between stage-4 policies, possibly across runs."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from summarize_srsbf import load, per_drop, boot

R = Path('results/stage4_srsbf_20261007')
_c = {}


def get(run, pol):
    if run not in _c:
        _c[run] = per_drop(*load(R/run))
    return _c[run][pol]


def diff(a, b, key):
    da, db = get(*a), get(*b)
    drops = sorted(set(da) & set(db))
    x = np.array([da[d][key]-db[d][key] for d in drops])
    rel = np.array([100*(da[d][key]/db[d][key]-1) for d in drops]) if key == 'thr' else None
    out = dict(a=f'{a[1]}@{a[0]}', b=f'{b[1]}@{b[0]}', key=key, mean=float(x.mean()), ci95=list(boot(x)), drops=len(drops))
    if rel is not None:
        out.update(rel_pct=float(rel.mean()), rel_ci95=list(boot(rel)))
    return out


PAIRS = [
    (('main_v3_8', 'qr_cqi'), ('main_v3_8', 'su_olla')),
    (('main_v3_8', 'beam2r_olla'), ('main_v3_8', 'su_olla')),
    (('main_v3_8', 'set_qr'), ('main_v3_8', 'beam2r_olla')),
    (('main_v3_8', 'qr_cqi'), ('main_v3_8', 'set_qr')),
    (('main_v3_8', 'qr_cqi'), ('main_v3_8', 'ref_dec')),
    (('lat_v3_8', 'qr_cqi+A'), ('lat_v3_8', 'su_olla+A')),
    (('lat_v3_8', 'qd_cqi'), ('lat_v3_8', 'q_cqi')),
    (('aging_v30_srs10_8', 'qr_cqi+A'), ('aging_v30_srs10_8', 'su_olla+A')),
    (('aging_v30_srs10_8', 'su_olla+A'), ('aging_v30_srs10_8', 'su_olla')),
    (('aging_v30_srs10_8', 'qs_cqi'), ('aging_v30_srs10_8', 'qr_cqi+A')),
    (('main_v30_8', 'qa_cqi'), ('aging_v30_srs10_8', 'qs_cqi')),
    (('learn20_v30_srs10_8', 'qg_cqi'), ('aging_v30_srs10_8', 'qr_cqi+A')),
    (('aging_v30_srs20_8', 'qr_cqi+A'), ('aging_v30_srs20_8', 'su_olla+A')),
    (('aging_v30_srs20_8', 'qs_cqi'), ('aging_v30_srs20_8', 'qr_cqi+A')),
    (('learn20_v30_srs20_8', 'qg_cqi'), ('aging_v30_srs20_8', 'qr_cqi+A')),
    (('learn20_v30_srs20_8', 'qg_cqi'), ('aging_v30_srs20_8', 'qs_cqi')),
    (('lat_v3_8', 'qr_cqi+A'), ('main_v3_8', 'qr_cqi')),
    (('lat_v3_8', 'su_olla+A'), ('main_v3_8', 'su_olla')),
    (('srs5plus_v30_8', 'qr_cqi+A'), ('srs5plus_v30_8', 'su_olla+A')),
    (('srs5plus_v30_8', 'qs_cqi'), ('srs5plus_v30_8', 'qr_cqi+A')),
    (('learn20_v30_srs5_8', 'qg_cqi'), ('srs5plus_v30_8', 'qr_cqi+A')),
]

if __name__ == '__main__':
    res = []
    for a, b in PAIRS:
        for key in ['thr', 'util', 'p10']:
            try:
                d = diff(a, b, key)
            except (FileNotFoundError, KeyError):
                continue
            res.append(d)
            tail = f"  ({d['rel_pct']:+.1f}% [{d['rel_ci95'][0]:+.1f},{d['rel_ci95'][1]:+.1f}])" if key == 'thr' else ''
            print(f"{d['a']:32s} - {d['b']:32s} {key:5s} {d['mean']:+7.2f} [{d['ci95'][0]:+.2f},{d['ci95'][1]:+.2f}]{tail}")
    (R/'paired_key_comparisons.json').write_text(json.dumps(res, indent=1))
