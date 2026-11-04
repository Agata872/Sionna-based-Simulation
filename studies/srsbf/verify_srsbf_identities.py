#!/usr/bin/env python3
"""Numerical checks of the two analytic building blocks (stage 4).

1. channelPathLoss folding: an RU report computed with nominal white noise N0
   and a path-loss scaling c^2 equals the white-noise SINR at noise level
   N0 * Delta when the O-DU supplies c^2 / Delta.  Exact for white noise and
   PL-independent weights; we also measure the error w.r.t. colored ICI.
2. First-order aging law for ZF-type RU weights with Gauss-Markov innovation:
   E[gamma_k] ~= rho^2 g / (1 + (1-rho^2) snr_el (L-1)/L) with isotropic
   innovation; compared with Monte Carlo on i.i.d. and correlated channels.
"""
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from srsbf_core import layer_channels, rzf_weights, set_sinr

torch.manual_seed(0)
dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
out = {}


def cn(*shape, scale=1.):
    return torch.complex(torch.randn(*shape, device=dev), torch.randn(*shape, device=dev))*math.sqrt(scale/2)


# ---------------------------------------------------------------- identity 1
M, Nr, U, F = 32, 2, 4, 200
h = cn(F, U, Nr, M, scale=1e3)                       # high-SNR channel, noise = 1
g = layer_channels(h, 1)[..., 0, :]
w = rzf_weights(g, 1e-6)                             # PL-independent ZF-type weights
eye = torch.eye(Nr, dtype=torch.complex64, device=dev)
delta = torch.tensor([3., 30., 300., 3000.], device=dev)       # per-UE noise rise
true_white = set_sinr(h, w, (delta[:, None, None]*eye)[None].expand(F, -1, -1, -1))
folded = set_sinr(h/torch.sqrt(delta)[None, :, None, None], w, eye[None, None].expand(F, U, -1, -1))
out['pl_folding_max_rel_err_white'] = float(((folded-true_white).abs()/true_white).max())
# colored ICI with the same trace as the white equivalent
q = cn(F, U, Nr, Nr)
cov = q@q.mH
cov = cov/cov.diagonal(dim1=-2, dim2=-1).real.mean(-1)[..., None, None]*delta[None, :, None, None]
colored = set_sinr(h, w, cov)
ratio_db = 10*torch.log10(colored/true_white)
out['colored_minus_white_db_mean'] = float(ratio_db.mean())
out['colored_minus_white_db_p95'] = float(torch.quantile(ratio_db.flatten(), .95))

# ---------------------------------------------------------------- aging law
rows = []
for corr in ['iid', 'cluster']:
    for L in [2, 4, 8]:
        for rho in [.99, .95, .9, .8, .6]:
            for snr_db in [10., 20., 30.]:
                trials, F2 = 400, 1
                beta = 10**(snr_db/10)                    # per-element SNR incl. ICI, noise 1
                if corr == 'iid':
                    h0 = cn(trials, L, Nr, M, scale=beta)
                    e = cn(trials, L, Nr, M, scale=beta)
                else:
                    # each UE: 6 random paths of a ULA-like array response
                    ang = torch.rand(trials, L, 6, device=dev)*math.pi-math.pi/2
                    a = torch.exp(1j*math.pi*torch.arange(M, device=dev)*torch.sin(ang)[..., None])  # T,L,P,M
                    coef0 = cn(trials, L, Nr, 6, scale=beta/6)
                    coef1 = cn(trials, L, Nr, 6, scale=beta/6)
                    h0 = coef0@a.to(torch.complex64)
                    e = coef1@a.to(torch.complex64)
                ht = rho*h0+math.sqrt(1-rho**2)*e
                g0 = layer_channels(h0, 1)[..., 0, :]
                w = rzf_weights(g0, 1e-6)
                cov = eye.expand(trials, L, -1, -1)
                est = set_sinr(h0, w, cov)               # RU-style report: CI taken as truth
                act = set_sinr(ht, w, cov)
                snr_el = (h0.abs()**2).mean((-1, -2))    # per-element SNR seen by the formula
                pred = rho**2*est/(1+(1-rho**2)*snr_el*(L-1)/L)
                rows.append(dict(corr=corr, L=L, rho=rho, snr_db=snr_db,
                                 report_bias_db=float((10*torch.log10(est/act)).mean()),
                                 law_err_db=float((10*torch.log10(pred/act)).mean()),
                                 law_abs_err_db=float((10*torch.log10(pred/act)).abs().mean())))
out['aging_law'] = rows
summary = {}
for corr in ['iid', 'cluster']:
    r = [x for x in rows if x['corr'] == corr]
    summary[corr] = dict(mean_report_bias_db=float(np.mean([x['report_bias_db'] for x in r])),
                         mean_law_err_db=float(np.mean([x['law_err_db'] for x in r])),
                         mean_law_abs_err_db=float(np.mean([x['law_abs_err_db'] for x in r])),
                         max_report_bias_db=float(np.max([x['report_bias_db'] for x in r])))
out['aging_summary'] = summary
print(json.dumps({k: v for k, v in out.items() if k != 'aging_law'}, indent=1))
for x in rows:
    if x['snr_db'] == 20. and x['corr'] == 'iid':
        print(x)
if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
