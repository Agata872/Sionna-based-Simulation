"""S14 后处理：命题 3 的图改为"预测位宽 vs 优化器所选位宽"（含 4..12 的截断）。

每个 (状态, 撒点, K, 切分) 组内只有一个标量 t（命题 3）。用组内未截断 RU（4 < w < 12）的 median(w − x) 估计 t̂，
预测（未取整、未截断）位宽为 t̂ + x_r；按它分箱，统计 OPT 实际选择的位宽，并与 clip(round(t̂ + x_r), 4, 12) 对照。

用法：
    .venv/bin/python dynamic_split/fig_prop3_post.py results/dynamic_split_20261012/stage14_figdata \
        --copy-to /home/user/Dynamic-Compression-With-Fronthaul-Load-Estimation/figures/data
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from runio import read_rows  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_dir')
    p.add_argument('--copy-to', default='')
    args = p.parse_args()
    run = Path(args.run_dir)
    rows = [r for r in read_rows(run) if r.get('bf') == 'zf' and 'width' in r]
    groups = {}
    for r in rows:
        groups.setdefault((tuple(r['state']), r['drop'], r['k'], r['split']), []).append(r)
    pts = {'7-1': [], '7-2': []}
    for (_, _, _, split), rs in groups.items():
        x = np.array([q['x71'] if split == '7-1' else q['x72'] for q in rs])
        w = np.array([q['width'] for q in rs], float)
        inner = (w > 4) & (w < 12)
        if inner.sum() == 0:
            continue
        t_hat = float(np.median(w[inner] - x[inner]))
        for xi, wi in zip(x, w):
            if wi <= 12:
                pts[split].append((t_hat + xi, wi))
    edges = np.arange(1.0, 13.01, 0.5)
    out, stats = [], {}
    for split in ('7-1', '7-2'):
        a = np.array(pts[split])
        pred = np.clip(np.round(a[:, 0]), 4, 12)
        stats[split] = dict(points=int(len(a)), exact=float(np.mean(pred == a[:, 1])),
                            within_one=float(np.mean(np.abs(pred - a[:, 1]) <= 1)))
    for lo, hi in zip(edges[:-1], edges[1:]):
        row = [0.5 * (lo + hi)]
        for split in ('7-1', '7-2'):
            a = np.array(pts[split])
            m = (a[:, 0] >= lo) & (a[:, 0] < hi)
            row += [float(a[m, 1].mean()) if m.sum() >= 5 else float('nan'),
                    float(a[m, 1].std()) if m.sum() >= 5 else float('nan'), int(m.sum())]
        out.append(row)
    path = run / 'fig_prop3.dat'
    with open(path, 'w') as f:
        f.write('pred w71 sd71 n71 w72 sd72 n72\n')
        for r in out:
            f.write(' '.join('nan' if (isinstance(v, float) and not np.isfinite(v)) else
                             (f'{v:.6g}' if isinstance(v, float) else str(v)) for v in r) + '\n')
    summ = json.loads((run / 'summary.json').read_text())
    summ['prop3_abs'] = stats
    (run / 'summary.json').write_text(json.dumps(summ, indent=1))
    print(stats)
    if args.copy_to:
        shutil.copy(path, Path(args.copy_to) / path.name)


if __name__ == '__main__':
    main()
