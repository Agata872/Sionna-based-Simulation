#!/usr/bin/env python3
"""Decompose O-DU MU-SINR prediction error under SRS-BF-DL (stage-4 diagnostic).

Random co-scheduled sets on random slots; the RU applies eigen-RZF with the
CI it would actually use at the data slot.  Each predictor removes one
information deficit, so the error ladder isolates its contribution:

  ru_lin      RU report: CI at decision slot as channel truth, scalar genie
              noise at decision slot, linear PRB average (clause 9.2.2.22)
  ru_eesm     same, but per-PRG values kept so the O-DU can apply EESM
  ru_cov      + mean colored ICI covariance instead of a white scalar
  true_dec    + true channel at decision slot and the RU's weights for that slot
  true_data   + true channel at data slot, RU's data-slot weights, mean ICI
  oracle      + the realized ICI at the data slot (= truth)

All errors are EESM(beta of the true-SINR MCS) in dB, predicted minus true.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import sionna.phy
from srsbf_core import ASSUMPTIONS, DropConfig, MCSTable, set_sinr
from explore_srsbf_scheduling import World
from explore_csirs_precoding import Recorder


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--drops', type=int, default=4)
    p.add_argument('--seed-start', type=int, default=20266000)
    p.add_argument('--speeds', type=float, nargs='+', default=[3., 30.])
    p.add_argument('--slots', type=int, default=200)
    p.add_argument('--samples', type=int, default=300)
    p.add_argument('--srs-period', type=int, default=10)
    p.add_argument('--csi-period', type=int, default=10)
    p.add_argument('--csi-delay', type=int, default=4)
    p.add_argument('--decision-lead', type=int, default=2)
    p.add_argument('--ci-snr-db', type=float, default=30.)
    p.add_argument('--ru-alpha-db', type=float, default=0.)
    p.add_argument('--reported-beams', type=int, default=8)
    p.add_argument('--ici-window', type=int, default=20)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    sionna.phy.config.device = args.device
    cfg = DropConfig(slots=args.slots)
    tab = MCSTable(cfg, args.device)
    rec = Recorder(args.out)
    (rec.path/'experiment_source.py').write_bytes(Path(__file__).read_bytes())
    (rec.path/'metadata.json').write_text(json.dumps(dict(args=vars(args), assumptions=ASSUMPTIONS), indent=2))
    eye = torch.eye(cfg.num_ut_ant, dtype=torch.complex64, device=args.device)
    for drop in range(args.drops):
        for speed in args.speeds:
            seed = args.seed_start+drop
            w = World(cfg, args, seed, speed, args.device)
            rng = np.random.default_rng(seed+3)
            # mean ICI covariance over a causal window ending at the decision slot
            for n in range(args.samples):
                size = int(rng.integers(1, 9))
                t = int(rng.integers(args.srs_period+args.decision_lead+args.ici_window, args.slots))
                td = t-args.decision_lead
                users = rng.choice(cfg.num_ue, size, replace=False)
                st = torch.as_tensor(users[None], device=args.device)
                true = w.true_sinr(t, st)[0]
                mcs = tab.select_from_prg(true.cpu().numpy(), 0.)
                mean_cov = w.cov[:, td-args.ici_window+1:td+1].mean(1)            # K,F,Nr,Nr
                scalar = w.cov[:, td].diagonal(dim1=-2, dim2=-1).real.mean((-1, -2))
                preds = {}
                # RU report at decision slot (white scalar noise)
                g_dec, ci_dec = w.ru_state(td)
                wd = w.weights(g_dec[st])
                cov_w = (scalar[st][..., None, None]*eye)[:, None].expand(-1, cfg.num_prg, -1, -1, -1)
                ru = set_sinr(ci_dec[st].transpose(1, 2), wd, cov_w).transpose(1, 2)[0]
                preds['ru_lin'] = None
                preds['ru_eesm'] = ru
                cov_c = mean_cov[st].transpose(1, 2)
                preds['ru_cov'] = set_sinr(ci_dec[st].transpose(1, 2), wd, cov_c).transpose(1, 2)[0]
                preds['true_dec'] = set_sinr(w.h_t[td][st].transpose(1, 2), wd, cov_c).transpose(1, 2)[0]
                g_dat, _ = w.ru_state(t)
                wt = w.weights(g_dat[st])
                preds['true_data'] = set_sinr(w.h_t[t][st].transpose(1, 2), wt, cov_c).transpose(1, 2)[0]
                true_np = true.cpu().numpy()
                eff_true = tab.eesm_db(true_np, mcs)
                row = dict(drop=drop, speed=speed, slot=t, size=size,
                           ci_age=float(np.mean(t-w.ru_src[t, users])), dec_ci_age=float(np.mean(td-w.ru_src[td, users])))
                for name, val in preds.items():
                    if name == 'ru_lin':
                        x = ru.cpu().numpy().mean(-1)
                        e = 10*np.log10(np.maximum(x, 1e-12))-eff_true
                    else:
                        e = tab.eesm_db(val.cpu().numpy(), mcs)-eff_true
                    row[name+'_mean'] = float(np.mean(e))
                    row[name+'_abs'] = float(np.mean(np.abs(e)))
                    row[name+'_max'] = float(np.max(e))
                row['true_eff_mean'] = float(np.mean(eff_true))
                rec.add(row)
            print(f'drop={drop} speed={speed} rows={len(rec.rows)}', flush=True)
    rec.finish()


if __name__ == '__main__':
    main()
