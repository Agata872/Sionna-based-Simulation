#!/usr/bin/env python3
"""Paper-1 scene figure: the Sionna RT Munich twin used in all experiments.

(a) top view: building heights from downward ray casting, the four O-RUs of the K = 8 runs (seed
20269300, positions from results/paper1_20261008/main_k8_nominal_16/metadata.json) with their
boresights, the UE drop area, an example UE with 64 posterior samples (sigma = 2 m, s = sqrt(5) m)
and its twin paths to every O-RU (same solver settings as the experiments).
(b) perspective render of the same twin with the example UE's paths (Sionna RT render_to_file).
Writes paper1/manuscript/figures/fig_scene.pdf/.png and the render as fig_scene_render.png.
"""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mitsuba as mi
from sionna.rt import Camera, Receiver
from explore_twin_cellfree import MultiSite

OUT = Path('paper1/manuscript/figures')
RUN = Path('results/paper1_20261008/main_k8_nominal_16')
SLOT = ['#2a78d6', '#eb6834', '#1baf7a', '#e87ba4']          # validated categorical slots 1, 2, 3, 5 (one per O-RU)


def site_args(meta):
    a = argparse.Namespace(**meta['args'])
    a.groups = a.num_prb//a.grp_prb
    a.reality, a.real_geom, a.real_material, a.real_cars, a.twin_scattering = 'nominal', 0., 0., 0, 0.
    return a


def height_map(scene, x0, x1, y0, y1, res):
    xs, ys = np.arange(x0, x1, res), np.arange(y0, y1, res)
    X, Y = np.meshgrid(xs, ys)
    z = 300.
    o = mi.Point3f(mi.Float(X.ravel().astype(np.float32)), mi.Float(Y.ravel().astype(np.float32)),
                   mi.Float(np.full(X.size, z, np.float32)))
    si = scene.mi_scene.ray_intersect(mi.Ray3f(o, mi.Vector3f(0, 0, -1)))
    t = np.where(si.is_valid().numpy(), si.t.numpy(), z)
    return xs, ys, (z-t).reshape(X.shape)


def trace_paths(site, ue):
    sc = site.scene
    for name in list(sc.receivers):
        sc.remove(name)
    sc.add(Receiver('ue', position=[float(v) for v in ue]))
    paths = site.solver(sc, max_depth=site.args.max_depth, samples_per_src=site.args.samples, diffraction=True,
                        synthetic_array=True)
    a, _ = paths.cir(normalize_delays=False, out_type='numpy')
    pw = (np.abs(a)**2).sum(axis=(1, 3, 5))[0]                        # [num_tx, num_paths]
    verts = np.array(paths.vertices)                                  # [depth, rx, tx, paths, 3]
    inter = np.array(paths.interactions)                               # [depth, rx, tx, paths]
    valid = np.array(paths.valid)[0]                                   # [tx, paths]
    out = []
    for l in range(verts.shape[2]):
        tx = np.array(sc.transmitters[f'ru{l}'].position).reshape(-1)
        lines = []
        for p in np.flatnonzero(valid[l]):
            pts = [tx]+[verts[d, 0, l, p] for d in range(verts.shape[0]) if inter[d, 0, l, p] != 0]+[np.asarray(ue)]
            lines.append((np.array(pts), pw[l, p]))
        out.append(lines)
    return paths, out


def main():
    meta = json.loads((RUN/'metadata.json').read_text())
    args = site_args(meta)
    ru = np.array(meta['ru_positions'])
    site = MultiSite(args, ru, np.random.default_rng(0))
    lo, hi = np.array(args.area_lo), np.array(args.area_hi)
    xs, ys, H = height_map(site.scene, lo[0]-40, hi[0]+40, lo[1]-40, hi[1]+40, .5)

    def outdoor(p):
        o = mi.Point3f(float(p[0]), float(p[1]), 1.5)
        return not bool(site.scene.mi_scene.ray_intersect(mi.Ray3f(o, mi.Vector3f(0, 0, 1))).is_valid().numpy()[0])
    # example UE: among outdoor grid points of the drop area with paths to every O-RU, the one farthest from
    # its nearest O-RU (a typical cell-free UE rather than one next to a pole)
    cand = [np.array([x, y, 1.5]) for x in np.arange(lo[0]+10, hi[0]-9, 10.) for y in np.arange(lo[1]+10, hi[1]-9, 10.)]
    cand = [q for q in cand if outdoor(q)]
    traced = site.trace(cand)
    best, ue, far = -1, None, -1.
    for q, t in zip(cand, traced):
        n = sum(1 for x in t if len(x['tau']))
        dmin = np.linalg.norm(ru[:, :2]-q[:2], axis=1).min()
        if (n, dmin) > (best, far):
            best, ue, far = n, q, dmin
    paths, lines = trace_paths(site, ue)
    rng = np.random.default_rng(1)
    s = math.sqrt(2.**2+1.)
    samples = ue[:2]+rng.normal(size=(64, 2))*s

    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 7.5, 'axes.edgecolor': '#c3c2b7',
                         'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e'})
    fig, ax = plt.subplots(figsize=(3.5, 3.55))
    im = ax.imshow(np.clip(H, 0, 60), origin='lower', extent=(xs[0], xs[-1]+.5, ys[0], ys[-1]+.5), cmap='Greys',
                   vmin=-8, vmax=60, interpolation='nearest')
    cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.02)
    cb.set_label('Building height (m)', color='#52514e')
    cb.ax.tick_params(labelsize=6.5)
    ax.add_patch(plt.Rectangle(lo, *(hi-lo), fill=False, linestyle=(0, (4, 2)), edgecolor='#0b0b0b', linewidth=.8))
    ax.text(lo[0]+2, hi[1]-7, 'UE drop area', fontsize=6.5, color='#0b0b0b')
    for l, (pos, ls) in enumerate(zip(ru, lines)):
        pmax = max((p for _, p in ls), default=1.)
        for pts, p in sorted(ls, key=lambda x: x[1])[-12:]:                # 12 strongest paths per O-RU
            w = .25+1.1*max(0., 1+10*np.log10(max(p, 1e-30)/pmax)/30)   # line width by power (30 dB range)
            ax.plot(pts[:, 0], pts[:, 1], color=SLOT[l], linewidth=w, alpha=.85, solid_capstyle='round', zorder=3)
        d = np.array(args.area_centre)-pos[:2]
        d = 14*d/np.linalg.norm(d)
        ax.annotate('', xy=pos[:2]+d, xytext=pos[:2], arrowprops=dict(arrowstyle='-|>', color=SLOT[l], lw=1.1), zorder=5)
        ax.plot(*pos[:2], marker='^', markersize=7.5, color=SLOT[l], markeredgecolor='white', markeredgewidth=.8, zorder=6)
        off = [(5, -3), (5, -9), (-5, 6), (5, -3)][l]                     # label offsets chosen to avoid the paths
        ax.text(pos[0]+off[0], pos[1]+off[1], f'O-RU {l+1}', fontsize=7, color=SLOT[l], fontweight='bold', zorder=6,
                ha='left' if off[0] > 0 else 'right', bbox=dict(boxstyle='square,pad=.1', fc='white', ec='none', alpha=.75))

    ax.plot(*ue[:2], marker='*', markersize=9, color='#eda100', markeredgecolor='#0b0b0b', markeredgewidth=.5, zorder=7)
    ax.text(ue[0]-3, ue[1]+6, 'UE', fontsize=7, color='#0b0b0b', ha='right', fontweight='bold', zorder=7)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT/'fig_scene.pdf', bbox_inches='tight', pad_inches=.02)
    fig.savefig(OUT/'fig_scene.png', dpi=220, bbox_inches='tight', pad_inches=.02)
    plt.close(fig)
    info = dict(ue=ue.tolist(), o_rus_with_paths=int(best), nearest_o_ru_m=float(far), paths_per_ru=[len(x) for x in lines], posterior_std_m=s)
    try:
        for name in list(site.scene.receivers):
            site.scene.remove(name)
        site.scene.add(Receiver('ue', position=[float(v) for v in ue], display_radius=3., color=(.93, .63, 0.)))
        for l in range(len(ru)):
            site.scene.transmitters[f'ru{l}'].display_radius = 3.
        c = np.array(args.area_centre+[0.])
        cam = Camera(position=mi.Point3f(float(c[0]-60), float(c[1]-260), 230.), look_at=mi.Point3f(float(c[0]), float(c[1]+5), 0.))
        site.scene.render_to_file(camera=cam, filename=str(OUT/'fig_scene_render.png'), resolution=(1400, 1000),
                                  num_samples=128, show_devices=True, show_orientations=False)
        info['render'] = 'ok'
    except Exception as e:                                             # rendering is optional
        info['render'] = f'failed: {e}'
    (OUT/'fig_scene_info.json').write_text(json.dumps(info, indent=1))
    print(info)


if __name__ == '__main__':
    main()
