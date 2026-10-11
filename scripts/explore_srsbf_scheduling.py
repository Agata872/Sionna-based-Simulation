#!/usr/bin/env python3
"""Stage-4 screen: O-DU PF scheduling + link adaptation under O-RU SRS-BF-DL.

Independent exploration entry point (see srsbf_core.ASSUMPTIONS). Compares
O-DU information policies on identical drops, channels, ICI and ACK seeds:

  oracle         true data-time SINR of every candidate set (not deployable)
  su_olla        SU CQI minus power split + per-UE OLLA (O-DU blind to MU)
  beam_olla      su_olla + MU penalty from DFT beam-power overlap (JSDM-like)
  q_nom          unlimited, zero-latency post-EQ SINR queries, nominal noise
  q_cal          same, noise rise calibrated through channelPathLoss (genie scalar)
  q_cqi          same, noise rise estimated from CQI and T-matrix norms
  ci_zf          SRS CI at O-DU (BFP), O-DU assumes ZF, CQI-calibrated noise

Every policy uses the same PF greedy grouping and per-UE OLLA (except oracle).
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import sionna.phy
from srsbf_core import (ASSUMPTIONS, DropConfig, MCSTable, layer_channels, make_drop,
                        rzf_weights, set_sinr)
from explore_csirs_precoding import Recorder

CQI2 = [(2, 78), (2, 193), (2, 449), (4, 378), (4, 490), (4, 616), (6, 466), (6, 567), (6, 666),
        (6, 772), (6, 873), (8, 711), (8, 797), (8, 885), (8, 948)]
SE2 = [2*120/1024, 2*193/1024, 2*308/1024, 2*449/1024, 2*602/1024, 4*378/1024, 4*434/1024, 4*490/1024,
       4*553/1024, 4*616/1024, 4*658/1024, 6*466/1024, 6*517/1024, 6*567/1024, 6*616/1024, 6*666/1024,
       6*719/1024, 6*772/1024, 6*822/1024, 6*873/1024, 8*682.5/1024, 8*711/1024, 8*754/1024,
       8*797/1024, 8*841/1024, 8*885/1024, 8*916.5/1024, 8*948/1024]


def float_bfp(x, width, dim=-1):
    lim = 2**(width-1)-1
    peak = torch.maximum(x.real.abs(), x.imag.abs()).amax(dim, keepdim=True)
    step = torch.exp2(torch.ceil(torch.log2((peak/lim).clamp_min(2**-100))))
    return torch.complex(torch.round(x.real/step), torch.round(x.imag/step))*step


def dft_basis(cfg, device):
    """Beam basis I_2 (x) D_cols (x) D_rows matching Sionna PanelArray ordering."""
    n, m = cfg.bs_cols, cfg.bs_rows
    dn = torch.exp(2j*math.pi*torch.outer(torch.arange(n), torch.arange(n))/n)/math.sqrt(n)
    dm = torch.exp(2j*math.pi*torch.outer(torch.arange(m), torch.arange(m))/m)/math.sqrt(m)
    return torch.kron(torch.eye(2), torch.kron(dn, dm)).to(torch.complex64).to(device)


class CQIMap:
    """CQI table 2 entries mapped to the nearest MCS-table-2 10% thresholds."""

    def __init__(self, tab):
        self.thr = []
        for qm, r in CQI2:
            m = int(np.argmin([abs(qm*r/1024-x) for x in SE2]))
            self.thr.append((tab.threshold_db[m], tab.beta[m]))

    def report(self, sinr_lin):
        cqi = 0
        for i, (thr, beta) in enumerate(self.thr):
            eff = -beta*np.log(np.mean(np.exp(-np.asarray(sinr_lin, dtype=float)/beta)))
            if np.isfinite(thr) and 10*np.log10(max(eff, 1e-12)) >= thr:
                cqi = i+1
        return cqi

    @property
    def floor_db(self):
        return min(t for t, _ in self.thr if np.isfinite(t))

    def sinr_db(self, cqi):
        return self.thr[cqi-1][0] if cqi > 0 else self.floor_db-3.


class World:
    """Channel, CI, CQI and RRM state for one drop (shared by every policy)."""

    def __init__(self, cfg, args, seed, speed, device):
        self.cfg, self.args, self.device = cfg, args, device
        self.h, self.ici, self.geo = make_drop(cfg, seed, speed, device)
        k, t, period = cfg.num_ue, cfg.slots, args.srs_period
        eye = torch.eye(cfg.num_ut_ant, dtype=torch.complex64, device=device)
        self.cov = self.ici+eye                                   # K,T,F,Nr,Nr true I+N
        self.h_t = self.h.permute(3, 0, 4, 1, 2).contiguous()     # T,K,F,Nr,M
        gen = torch.Generator(device=device).manual_seed(seed+5151)
        self.ci_store = {}
        for u in range(k):
            power = float((self.h[u].abs()**2).mean())
            sig = math.sqrt(power/10**(args.ci_snr_db/10)/2)
            for s in range(u % period, t, period):
                e = torch.complex(torch.randn(self.h_t[s, u].shape, generator=gen, device=device),
                                  torch.randn(self.h_t[s, u].shape, generator=gen, device=device))*sig
                self.ci_store[(u, s)] = self.h_t[s, u]+e
        # latest SRS slot whose CI the RU can use at slot s (processing: srs <= s-1)
        self.ru_src = np.full((t, k), -1)
        for s in range(1, t):
            for u in range(k):
                off = u % period
                if s-1 >= off:
                    self.ru_src[s, u] = off+((s-1-off)//period)*period
        self.g_cache = {key: layer_channels(v, 1)[..., 0, :] for key, v in self.ci_store.items()}   # F,M
        basis = dft_basis(cfg, device)
        self.beam_power = {}
        for key, v in self.ci_store.items():
            p = (v@basis).abs().square().sum((0, 1))
            keep = torch.topk(p, args.reported_beams).indices
            q = torch.zeros_like(p)
            q[keep] = p[keep]
            self.beam_power[key] = (q/q.sum()).cpu().numpy()
        self.su_gain = {key: float((g.abs()**2).sum(-1).mean()) for key, g in self.g_cache.items()}
        # Candidate RU report: directional coherence of consecutive retained CI of a UE
        self.ci_coherence = {}
        for (u, s_) in self.ci_store:
            prev = (u, s_-period)
            if prev in self.g_cache:
                a = self.g_cache[(u, s_)]
                b = self.g_cache[prev]
                c = (a.conj()*b).sum(-1).abs().square()/((a.abs()**2).sum(-1)*(b.abs()**2).sum(-1)).clamp_min(1e-30)
                self.ci_coherence[(u, s_)] = float(c.mean())
        self.cqi_events = {}
        for u in range(k):
            for s in range(u % args.csi_period, t, args.csi_period):
                hk = self.h_t[s, u]
                g = layer_channels(hk, 1)[..., 0, :]
                w = (g/torch.linalg.vector_norm(g, dim=-1, keepdim=True)).conj()
                d = (hk@w[..., None])[..., 0]
                x = torch.linalg.solve(self.cov[u, s], d[..., None])[..., 0]
                sinr = (d.conj()*x).sum(-1).real.cpu().numpy()
                self.cqi_events.setdefault(s+args.csi_delay, []).append((u, s, sinr))
        self.alpha_ru = 10**(args.ru_alpha_db/10)
        self._slot_cache = {}
        self._coh_cache = {}

    def direction_rho2(self, s):
        """[K,F] |<v_ci, v_true>|^2 of dominant directions: RU CI used at slot s vs truth at s."""
        if ('rho', s) not in self._slot_cache:
            g, _ = self.ru_state(s)
            v_ci = g/torch.linalg.vector_norm(g, dim=-1, keepdim=True).clamp_min(1e-30)
            from srsbf_core import layer_channels as _lc
            gt = _lc(self.h_t[s], 1)[..., 0, :]
            v_t = gt/torch.linalg.vector_norm(gt, dim=-1, keepdim=True).clamp_min(1e-30)
            self._slot_cache[('rho', s)] = (v_ci.conj()*v_t).sum(-1).abs().square().cpu().numpy()
        return self._slot_cache[('rho', s)]

    def ru_covariance(self, s):
        """[K,M,M] RU-side transmit covariance from the last `cov_snapshots` retained CI."""
        key = ('cov', s)
        if key not in self._slot_cache:
            period, n = self.args.srs_period, self.args.cov_snapshots
            mats = []
            for u in range(self.cfg.num_ue):
                src = int(self.ru_src[s, u])
                hs = [self.ci_store[(u, src-i*period)] for i in range(n) if (u, src-i*period) in self.ci_store]
                if not hs:                                               # no CI retained yet
                    m = self.cfg.num_bs_ant
                    mats.append(torch.zeros(m, m, dtype=torch.complex64, device=self.device))
                    continue
                h = torch.stack(hs)                                      # n,F,Nr,M
                r = torch.einsum('nfrm,nfrq->mq', h.conj(), h)/(h.shape[0]*h.shape[1]*h.shape[2])
                mats.append(r)
            self._slot_cache[key] = torch.stack(mats)
        return self._slot_cache[key]

    def ru_estimate_fold(self, s_query, sets, noise, rho2):
        """Aging folded into the noise (lower-bound deterministic equivalent):
        RU report at N'_k = (N_k + (1-rho_k^2) l_k)/rho_k^2, l_k = tr(W^H R_k W) per PRG,
        R_k the RU's CI covariance estimate, rho2[u] the aging coherence for the data slot."""
        g, ci = self.ru_state(s_query)
        w = self.weights(g[sets])                                           # N,F,M,U
        h = ci[sets].transpose(1, 2)                                        # N,F,U,Nr,M
        r = self.ru_covariance(s_query)[sets]                               # N,U,M,M
        leak = torch.einsum('nfmj,nuqm,nfqj->nfu', w.conj(), r, w).real.clamp_min(0)   # tr(W^H R_k W)
        nz = torch.as_tensor(noise, dtype=torch.float32, device=self.device)[sets][:, None, :]
        r2 = torch.as_tensor(np.clip(rho2, 1e-3, 1.), dtype=torch.float32, device=self.device)[sets][:, None, :]
        n_eff = (nz+(1-r2)*leak)/r2                                         # N,F,U
        eye = torch.eye(self.cfg.num_ut_ant, dtype=torch.complex64, device=self.device)
        return set_sinr(h, w, n_eff[..., None, None]*eye).transpose(1, 2)

    def ru_estimate_aged(self, s_query, sets, noise, t_data, samples=8):
        """Proposed RU report for an O-DU-indicated data slot: E[post-EQ SINR] over a
        Gauss-Markov innovation with RU-estimated coherence and transmit covariance."""
        g, ci = self.ru_state(s_query)
        w = self.weights(g[sets])                                       # N,F,M,U
        h = ci[sets].transpose(1, 2)                                    # N,F,U,Nr,M
        period = self.args.srs_period
        rho2 = np.ones(self.cfg.num_ue)
        for u in range(self.cfg.num_ue):
            coh = self.ci_coherence.get((u, int(self.ru_src[s_query, u])))
            if coh is not None:
                rho2[u] = min(1., coh)**((t_data-int(self.ru_src[s_query, u]))/period)
        rho = torch.as_tensor(np.sqrt(rho2), dtype=torch.float32, device=self.device)[sets]   # N,U
        cov = self.ru_covariance(s_query).to(torch.complex128)
        load = 1e-6*cov.diagonal(dim1=-2, dim2=-1).real.mean(-1).clamp_min(1e-30)
        cov = cov+load[:, None, None]*torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device)
        evals, evecs = torch.linalg.eigh(cov)
        root = (evecs*evals.clamp_min(0).sqrt()[..., None, :]).to(torch.complex64)   # K,M,M (R = root root^H)
        root = root[sets]                                               # N,U,M,M
        eye = torch.eye(self.cfg.num_ut_ant, dtype=torch.complex64, device=self.device)
        nz = torch.as_tensor(noise, dtype=torch.float32, device=self.device)[sets]
        cv = (nz[..., None, None]*eye)[:, None].expand(-1, h.shape[1], -1, -1, -1)
        gen = torch.Generator(device=self.device).manual_seed(int(s_query)*7919+int(t_data))
        acc = 0.
        for _ in range(samples):
            z = torch.complex(torch.randn(h.shape, generator=gen, device=self.device),
                              torch.randn(h.shape, generator=gen, device=self.device))/math.sqrt(2)
            e = torch.einsum('nfurm,nuqm->nfurq', z, root.conj())        # rows ~ CN(0, R^T)
            hp = rho[:, None, :, None, None]*h+torch.sqrt(1-rho**2)[:, None, :, None, None]*e
            acc = acc+set_sinr(hp, w, cv)
        return (acc/samples).transpose(1, 2)

    def stat_rho2(self, t_data):
        """[K] statistical-genie aging: trace-average directional correlation at the CI age
        of each UE at the data slot (exact second-order statistics, no realization)."""
        if not hasattr(self, '_stat'):
            from srsbf_core import layer_channels as _lc
            g = _lc(self.h_t, 1)[..., 0, :]                                    # T,K,F,M
            v = g/torch.linalg.vector_norm(g, dim=-1, keepdim=True).clamp_min(1e-30)
            tmax = 3*self.args.srs_period+self.args.decision_lead+2
            self._stat = np.ones((self.cfg.num_ue, tmax+1))
            for lag in range(1, tmax+1):
                c = (v[lag:].conj()*v[:-lag]).sum(-1).abs().square().mean((0, 2))
                self._stat[:, lag] = c.cpu().numpy()
        tau = t_data-self.ru_src[t_data]
        tau = np.clip(tau, 0, self._stat.shape[1]-1)
        return self._stat[np.arange(self.cfg.num_ue), tau]

    def model_rho2(self, t_dec, t_data):
        """[K] proposed report: RU-averaged directional coherence at lags T and 2T over the
        last `coh_window` SRS occasions; O-DU fits rho2(tau) = a + (1-a) exp(-tau/c)."""
        period, win = self.args.srs_period, self.args.coh_window
        out = np.ones(self.cfg.num_ue)
        for u in range(self.cfg.num_ue):
            src = int(self.ru_src[t_dec, u])
            if src < 0:
                continue
            r1, r2 = [], []
            for i in range(win):
                s_ = src-i*period
                a, b, c = (u, s_), (u, s_-period), (u, s_-2*period)
                if a in self.g_cache and b in self.g_cache:
                    r1.append(self._coh(a, b))
                if a in self.g_cache and c in self.g_cache:
                    r2.append(self._coh(a, c))
            if not r1:
                continue
            x1 = float(np.mean(r1))
            x2 = float(np.mean(r2)) if r2 else x1**2
            x2 = min(x2, x1)
            den = 2*x1-1-x2
            a_ = (x1**2-x2)/den if abs(den) > 1e-6 else 0.
            a_ = float(np.clip(a_, 0., x2))
            ratio = (x1-a_)/max(1-a_, 1e-6)
            tau = t_data-int(self.ru_src[t_data, u])
            out[u] = a_+(1-a_)*max(ratio, 1e-6)**(tau/period)
        return np.clip(out, 1e-3, 1.)

    def _coh(self, a, b):
        key = ('coh', a, b)
        if key not in self._coh_cache:
            x, y = self.g_cache[a], self.g_cache[b]
            c = (x.conj()*y).sum(-1).abs().square()/((x.abs()**2).sum(-1)*(y.abs()**2).sum(-1)).clamp_min(1e-30)
            self._coh_cache[key] = float(c.mean())
        return self._coh_cache[key]

    def reported_rho2(self, t_dec, t_data):
        """[K] causal aging estimate: rho2(tau) = coh^(tau/T_srs), coh = latest reported
        consecutive-CI coherence known at t_dec, tau = data-slot age of the CI the RU will use."""
        period = self.args.srs_period
        out = np.ones(self.cfg.num_ue)
        for u in range(self.cfg.num_ue):
            src = int(self.ru_src[t_dec, u])
            coh = self.ci_coherence.get((u, src))
            if coh is None:
                continue
            tau = t_data-int(self.ru_src[t_data, u])
            out[u] = min(1., coh)**(tau/period)
        return out

    def ru_state(self, s):
        """RU layer channels [K,F,M] and CI [K,F,Nr,M] in use at slot s."""
        if s not in self._slot_cache:
            src = self.ru_src[s]
            g = torch.stack([self.g_cache[(u, int(src[u]))] if src[u] >= 0 else
                             torch.zeros_like(self.g_cache[(u, u % self.args.srs_period)])
                             for u in range(self.cfg.num_ue)])
            h = torch.stack([self.ci_store[(u, int(src[u]))] if src[u] >= 0 else
                             torch.zeros_like(self.ci_store[(u, u % self.args.srs_period)])
                             for u in range(self.cfg.num_ue)])
            if len(self._slot_cache) > 256:
                self._slot_cache.clear()
            self._slot_cache[s] = (g, h)
        return self._slot_cache[s]

    def weights(self, g_sets):
        """g_sets [N,U,F,M] -> RU eigen-RZF weights [N,F,M,U]."""
        return rzf_weights(g_sets.transpose(1, 2), self.alpha_ru*g_sets.shape[1])

    def true_sinr(self, s, sets):
        """sets LongTensor [N,U] -> true data SINR [N,U,F] with the RU's actual weights."""
        g, _ = self.ru_state(s)
        w = self.weights(g[sets])
        h = self.h_t[s][sets].transpose(1, 2)                      # N,F,U,Nr,M
        cov = self.cov[:, s][sets].transpose(1, 2)
        return set_sinr(h, w, cov).transpose(1, 2)

    def ru_estimate(self, s_query, sets, noise):
        """RU post-EQ SINR at processing slot s_query, white noise noise[u]; [N,U,F]."""
        g, ci = self.ru_state(s_query)
        w = self.weights(g[sets])
        h = ci[sets].transpose(1, 2)
        eye = torch.eye(self.cfg.num_ut_ant, dtype=torch.complex64, device=self.device)
        nz = torch.as_tensor(noise, dtype=torch.float32, device=self.device)[sets]  # N,U
        cov = (nz[..., None, None]*eye)[:, None].expand(-1, h.shape[1], -1, -1, -1)
        return set_sinr(h, w, cov).transpose(1, 2)


class Policy:
    gpu = False

    def __init__(self, name, world, tab, cqimap, args):
        self.name, self.w, self.tab, self.cqimap, self.args = name, world, tab, cqimap, args
        k = world.cfg.num_ue
        self.offset = np.zeros(k)
        self.layer_olla = name.endswith('+L')
        self.hier_olla = name.endswith('+H')
        self.offset_h = np.zeros(args.max_layers+1)
        self.age_olla = name.endswith('+A')
        self.offset_a = np.zeros((k, 3))
        self.offset_l = np.zeros((k, args.max_layers+1))
        self.avg = np.full(k, 1e5)
        self.cqi = np.zeros(k, dtype=int)
        self.noise_cqi = np.ones(k)

    def update_cqi(self, t):
        for u, s, sinr in self.w.cqi_events.get(t, []):
            self.cqi[u] = self.cqimap.report(sinr)
            srs = int(self.w.ru_src[min(s+1, self.w.cfg.slots-1), u])
            if srs >= 0:
                # CQI 0 only says SINR < CQI-1 threshold: assume 3 dB below it.
                gamma_db = self.cqimap.sinr_db(self.cqi[u]) if self.cqi[u] > 0 else self.cqimap.floor_db-3.
                gain = self.w.su_gain[(u, srs)]
                self.noise_cqi[u] = max(1., gain/10**(gamma_db/10))

    def predict(self, t_dec, t_data, sets):
        """sets ndarray [N,U] -> wideband dB [N,U] or per-PRG linear [N,U,F]."""
        raise NotImplementedError

    def age_bucket(self, t_data, users):
        tau = t_data-self.w.ru_src[t_data][users]
        period = self.args.srs_period
        return np.clip((3*(tau-1))//max(period, 1), 0, 2)

    def aging_feedback(self, u, ok, layers, tau):
        pass

    def olla_offset(self, sets, t_data=None):
        if self.age_olla and t_data is not None:
            return self.offset_a[sets, self.age_bucket(t_data, sets)]
        if self.hier_olla:
            return self.offset[sets]+self.offset_h[sets.shape[1]]
        return self.offset_l[sets, sets.shape[1]] if self.layer_olla else self.offset[sets]

    def olla_update(self, u, ok, layers, step, bucket=None):
        delta = step*.1/.9 if ok else -step
        if self.age_olla and bucket is not None:
            self.offset_a[u, bucket] = float(np.clip(self.offset_a[u, bucket]+delta, -20., 10.))
        elif self.hier_olla:
            self.offset[u] = float(np.clip(self.offset[u]+delta/2, -20., 10.))
            self.offset_h[layers] = float(np.clip(self.offset_h[layers]+delta/2, -20., 10.))
        elif self.layer_olla:
            self.offset_l[u, layers] = float(np.clip(self.offset_l[u, layers]+delta, -20., 10.))
        else:
            self.offset[u] = float(np.clip(self.offset[u]+delta, -20., 10.))

    def mcs_and_value(self, t_dec, t_data, sets):
        pred = self.predict(t_dec, t_data, sets)
        off = self.olla_offset(sets, t_data)
        mcs = self.tab.select_from_prg(pred, off) if pred.ndim == 3 else self.tab.select_from_eff(pred+off)
        value = (self.tab.expected_bits(mcs)/self.avg[sets]).sum(-1)
        return mcs, value

    def schedule(self, t_dec, t_data):
        k = self.w.cfg.num_ue
        eligible = [j for j in range(k) if self.w.ru_src[t_data, j] >= 0 and self.w.ru_src[t_dec, j] >= 0]
        chosen, best, mcs_best = [], 0., None
        while len(chosen) < self.args.max_layers:
            cand = [j for j in eligible if j not in chosen]
            if not cand:
                break
            sets = np.array([chosen+[j] for j in cand])
            mcs, value = self.mcs_and_value(t_dec, t_data, sets)
            i = int(np.argmax(value))
            if value[i] <= best*(1+1e-9):
                break
            chosen, best, mcs_best = list(sets[i]), float(value[i]), mcs[i]
        return chosen, mcs_best


class Oracle(Policy):
    def predict(self, t_dec, t_data, sets):
        return self.w.true_sinr(t_data, torch.as_tensor(sets, device=self.w.device)).cpu().numpy()

    def mcs_and_value(self, t_dec, t_data, sets):
        sinr = self.predict(t_dec, t_data, sets)
        mcs = self.tab.select_from_prg(sinr, 0.)
        eff = self.tab.eesm_db(sinr, mcs)
        value = (self.tab.expected_bits(mcs, eff)/self.avg[sets]).sum(-1)
        return mcs, value


class SUOLLA(Policy):
    def predict(self, t_dec, t_data, sets):
        base = np.array([self.cqimap.sinr_db(c) for c in self.cqi])
        return base[sets]-10*math.log10(sets.shape[1])


class BeamOLLA(Policy):
    def predict(self, t_dec, t_data, sets):
        base = np.array([self.cqimap.sinr_db(c) for c in self.cqi])[sets]-10*math.log10(sets.shape[1])
        src = self.w.ru_src[t_dec]
        bp = np.stack([self.w.beam_power[(u, int(src[u]))] if src[u] >= 0 else np.zeros(self.w.cfg.num_bs_ant)
                       for u in range(self.w.cfg.num_ue)])
        bc = np.sqrt(bp)@np.sqrt(bp).T                                       # Bhattacharyya overlap
        np.fill_diagonal(bc, 0.)
        loss = (bc**2)[sets[..., :, None], sets[..., None, :]].sum(-1)
        return base+10*np.log10(np.maximum(.05, 1-loss))


class Query(Policy):
    """Unlimited zero-latency queries at decision time (upper bound of the tool)."""

    def __init__(self, name, world, tab, cqimap, args, calibration):
        super().__init__(name, world, tab, cqimap, args)
        self.calibration = calibration

    def noise(self, t_dec):
        if self.calibration == 'nominal':
            return np.ones(self.w.cfg.num_ue)
        if self.calibration == 'genie':
            return self.w.cov[:, t_dec].diagonal(dim1=-2, dim2=-1).real.mean((-1, -2)).cpu().numpy()
        return self.noise_cqi

    def predict(self, t_dec, t_data, sets):
        # 'qd_': unlimited queries answered with the RU state of t_dec - query_delay
        sq = max(1, t_dec-self.args.query_delay) if self.name.startswith('qd_') else t_dec
        s = self.w.ru_estimate(sq, torch.as_tensor(sets, device=self.w.device), self.noise(t_dec))
        return 10*np.log10(np.maximum(s.mean(-1).cpu().numpy(), 1e-12))   # spec: linear average


class QueryLearnedAging(Query):
    """Calibrated query + aging structure learned from HARQ (no new RU report).

    rho2_k(tau) = max(.02, exp(-(tau/c_k)^2)); per-PRG correction
    rho2 g / (1 + (1-rho2) snr_el (L-1)/L). c_k follows a stochastic-approximation
    update driven by ACK/NACK weighted by the age sensitivity (tau/T)^2, alongside
    the ordinary per-UE OLLA offset (age-independent bias).
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.c = np.full(self.w.cfg.num_ue, 2.*self.args.srs_period)

    def aging_feedback(self, u, ok, layers, tau):
        if layers < 1:
            return
        x = (tau/self.args.srs_period)**2
        step = self.args.aging_lr*(0.1/0.9 if ok else -1.)*x
        self.c[u] = float(np.clip(self.c[u]*math.exp(step), .5, 50.*self.args.srs_period))

    def predict(self, t_dec, t_data, sets):
        st = torch.as_tensor(sets, device=self.w.device)
        noise = self.noise(t_dec)
        g = self.w.ru_estimate(t_dec, st, noise).cpu().numpy()            # N,U,F
        tau = (t_data-self.w.ru_src[t_data])[sets]
        rho2 = np.maximum(.02, np.exp(-(tau/self.c[sets])**2))[..., None]
        src = self.w.ru_src[t_dec]
        gain = np.array([self.w.su_gain.get((u, int(src[u])), 1.) for u in range(self.w.cfg.num_ue)])
        snr_el = (gain/(self.w.cfg.num_bs_ant*np.asarray(noise)))[sets][..., None]
        l = sets.shape[1]
        return rho2*g/(1+(1-rho2)*snr_el*(l-1)/l)


class CIZF(Policy):
    """SRS CI conveyed to O-DU (BFP), O-DU assumes ZF and white CQI-calibrated noise."""

    def __init__(self, *a):
        super().__init__(*a)
        self.cache = {}

    def predict(self, t_dec, t_data, sets):
        src = tuple(self.w.ru_src[t_dec])       # ST 6 CI of SRS <= t_dec-1 (one-slot latency)
        if src not in self.cache:
            self.cache = {src: torch.stack([float_bfp(self.w.ci_store[(u, int(src[u]))], self.args.ci_width)
                                            if src[u] >= 0 else torch.zeros_like(self.w.ci_store[(u, u % self.args.srs_period)])
                                            for u in range(self.w.cfg.num_ue)])}
        ci = self.cache[src]
        st = torch.as_tensor(sets, device=self.w.device)
        h = ci[st].transpose(1, 2)                                # N,F,U,Nr,M
        g = layer_channels(h, 1)[..., 0, :]                       # N,F,U,M
        w = rzf_weights(g, 1e-6)                                  # ZF (tiny loading for rank safety)
        eye = torch.eye(self.w.cfg.num_ut_ant, dtype=torch.complex64, device=self.w.device)
        nz = torch.as_tensor(self.noise_cqi, dtype=torch.float32, device=self.w.device)[st]
        cov = (nz[..., None, None]*eye)[:, None].expand(-1, h.shape[1], -1, -1, -1)
        return set_sinr(h, w, cov).transpose(1, 2).cpu().numpy()


class RefDecision(Policy):
    """Causal-ideal O-DU: true channel at the decision slot, the RU's weights for
    that slot, mean colored ICI over a causal window. Not deployable; separates
    information deficits from unpredictable future fading/ICI."""

    def predict(self, t_dec, t_data, sets):
        st = torch.as_tensor(sets, device=self.w.device)
        g, _ = self.w.ru_state(t_dec)
        wd = self.w.weights(g[st])
        lo = max(0, t_dec-self.args.ici_window+1)
        cov = self.w.cov[:, lo:t_dec+1].mean(1)[st].transpose(1, 2)
        h = self.w.h_t[t_dec][st].transpose(1, 2)
        return set_sinr(h, wd, cov).transpose(1, 2).cpu().numpy()


class QueryAged(Query):
    """Calibrated query + first-order aging correction for ZF-type RU weights.

    SINR_k <- rho_k^2 g_k / (1 + (1-rho_k^2) snr_el_k (L-1)/L), rho_k the
    directional correlation between the CI the RU uses at the data slot and
    the data-slot channel (genie here; the practical estimate is a later step),
    snr_el_k = SU gain / (M N_k).  Per PRG, then EESM.
    """

    def predict(self, t_dec, t_data, sets):
        st = torch.as_tensor(sets, device=self.w.device)
        noise = self.noise(t_dec)
        if self.name.startswith('qx_'):
            return self.w.ru_estimate_aged(t_dec, st, noise, t_data).cpu().numpy()
        if self.name.startswith('qf'):
            rho2 = {'qfs_': self.w.stat_rho2(t_data), 'qfm_': self.w.model_rho2(t_dec, t_data),
                    'qfa_': self.w.direction_rho2(t_data).mean(-1),
                    'qfr_': self.w.reported_rho2(t_dec, t_data)}[self.name[:4]]
            return self.w.ru_estimate_fold(t_dec, st, noise, rho2).cpu().numpy()
        g = self.w.ru_estimate(t_dec, st, noise).cpu().numpy()            # N,U,F
        if self.name.startswith('qr_'):
            rho2 = self.w.reported_rho2(t_dec, t_data)[sets][..., None]
        elif self.name.startswith('qs_'):
            rho2 = self.w.stat_rho2(t_data)[sets][..., None]
        elif self.name.startswith('qm_'):
            rho2 = self.w.model_rho2(t_dec, t_data)[sets][..., None]
        else:
            rho2 = self.w.direction_rho2(t_data)[sets]                     # N,U,F
        src = self.w.ru_src[t_dec]
        gain = np.array([self.w.su_gain.get((u, int(src[u])), 1.) for u in range(self.w.cfg.num_ue)])
        snr_el = (gain/(self.w.cfg.num_bs_ant*np.asarray(noise)))[sets][..., None]
        l = sets.shape[1]
        return rho2*g/(1+(1-rho2)*snr_el*(l-1)/l)


class Surrogate(Policy):
    """Structured O-DU model of the opaque RU MU-SINR (stage-4 candidate).

    gamma_k(S) ~= gamma_CQI,k / |S| * eta_k(S) [* aging],
    eta_k(S) = max(eps, 1 - sum over j in S, j != k, of c_kj).
    c_kj: prior from reported DFT beam powers (random-phase model sum_i p_ki p_ji),
    replaced by pair-query measurements c = 1 - 2 N_k r_k / G_k when available.
    Budgeted queries (`query_budget` per `tdd_period`, latency `query_delay`):
      mode 'pairs' -> refresh the oldest pairs among the PF-relevant UEs,
      mode 'sets'  -> query the set the surrogate plans to schedule; a fresh
                      exact-set answer replaces the surrogate for that set.
    """

    def __init__(self, name, world, tab, cqimap, args, mode, aging):
        super().__init__(name, world, tab, cqimap, args)
        k = world.cfg.num_ue
        self.mode, self.aging = mode, aging
        self.corrected = name.startswith(('topc_', 'setc_'))
        self.sur_bias = np.zeros(k)
        self.c_meas = np.full((k, k), np.nan)
        self.c_time = np.full((k, k), -10**9)
        self.set_cache = {}
        self.inflight = []
        self.budget_used = {}

    def prior(self, t_dec):
        src = self.w.ru_src[t_dec]
        bp = np.stack([self.w.beam_power[(u, int(src[u]))] if src[u] >= 0 else np.zeros(self.w.cfg.num_bs_ant)
                       for u in range(self.w.cfg.num_ue)])
        c = bp@bp.T
        np.fill_diagonal(c, 0.)
        return c

    def pair_matrix(self, t_dec):
        c = self.prior(t_dec)
        fresh = (t_dec-self.c_time) <= self.args.pair_valid
        return np.where(fresh & np.isfinite(self.c_meas), self.c_meas, c)

    def surrogate_db(self, t_dec, t_data, sets, raw=False):
        base = np.array([self.cqimap.sinr_db(c) for c in self.cqi])[sets]-10*math.log10(sets.shape[1])
        if self.corrected and not raw:
            base = base+self.sur_bias[sets]
        c = self.pair_matrix(t_dec)
        loss = c[sets[..., :, None], sets[..., None, :]].sum(-1)
        out = base+10*np.log10(np.maximum(.03, 1-loss))
        if self.aging and not raw:
            out = out+self.aging_db(t_dec, t_data, sets)
        elif self.aging and raw:
            out = out+self.aging_db(t_dec, t_data, sets)
        return out

    def aging_mode(self):
        return 'reported' if self.aging == 'ru' else self.aging

    def aging_db(self, t_dec, t_data, sets):
        mode = self.aging_mode()
        if mode == 'genie':
            rho2 = self.w.direction_rho2(t_data).mean(-1)[sets]
        elif mode == 'model':
            rho2 = self.w.model_rho2(t_dec, t_data)[sets]
        elif mode == 'stat':
            rho2 = self.w.stat_rho2(t_data)[sets]
        else:
            rho2 = self.w.reported_rho2(t_dec, t_data)[sets]
        gain = np.array([self.w.su_gain.get((u, int(self.w.ru_src[t_dec][u])), 1.) for u in range(self.w.cfg.num_ue)])
        snr_el = (gain/(self.w.cfg.num_bs_ant*self.noise_cqi))[sets]
        l = sets.shape[1]
        return 10*np.log10(rho2/(1+(1-rho2)*snr_el*(l-1)/l))

    def receive(self, t_dec):
        keep = []
        for arrive, kind, users, r, tq in self.inflight:
            if arrive > t_dec:
                keep.append((arrive, kind, users, r, tq))
                continue
            if kind == 'pair':
                a, b = users
                ga = self.w.su_gain[(a, int(self.w.ru_src[tq][a]))]
                gb = self.w.su_gain[(b, int(self.w.ru_src[tq][b]))]
                ca = 1-2*self.noise_used[(tq, a)]*r[0]/ga
                cb = 1-2*self.noise_used[(tq, b)]*r[1]/gb
                self.c_meas[a, b] = self.c_meas[b, a] = float(np.clip((ca+cb)/2, 0., .99))
                self.c_time[a, b] = self.c_time[b, a] = tq
            else:
                self.set_cache[tuple(sorted(users))] = (dict(zip(users, r)), tq)
                if self.corrected:
                    # surrogate (without its own bias term) for the same set at query time
                    row = np.array([users])
                    sur = self.surrogate_db(tq, tq+self.args.decision_lead, row, raw=True)[0]
                    if self.aging in ('genie', 'reported', 'model', 'stat'):
                        sur = sur-self.aging_db(tq, tq+self.args.decision_lead, row)[0]
                    for u, x, y in zip(users, r, sur):
                        self.sur_bias[u] += .2*(10*math.log10(max(float(x), 1e-12))-float(y)-self.sur_bias[u])
        self.inflight = keep

    def query(self, t_dec, users):
        if any(self.w.ru_src[t_dec][u] < 0 for u in users):
            return True                                   # RU holds no CI: request not allowed, skip
        period = t_dec//self.args.tdd_period
        if self.budget_used.get(period, 0) >= self.args.query_budget:
            return False
        self.budget_used[period] = self.budget_used.get(period, 0)+1
        st = torch.as_tensor(np.array([users]), device=self.w.device)
        noise = self.noise_cqi.copy()
        if not hasattr(self, 'noise_used'):
            self.noise_used = {}
        for u in users:
            self.noise_used[(t_dec, u)] = float(noise[u])
        if self.aging == 'ru':
            r = self.w.ru_estimate_aged(t_dec, st, noise, t_dec+self.args.decision_lead)[0].mean(-1).cpu().numpy()
        else:
            r = self.w.ru_estimate(t_dec, st, noise)[0].mean(-1).cpu().numpy()   # linear PRB average
        kind = 'pair' if (self.mode == 'pairs' and len(users) == 2) else 'set'
        users = list(users)
        self.inflight.append((t_dec+self.args.query_delay, kind, list(users), r, t_dec))
        return True

    def predict(self, t_dec, t_data, sets):
        out = self.surrogate_db(t_dec, t_data, sets)
        if self.mode in ('sets', 'top'):
            for i, row in enumerate(sets):
                hit = self.set_cache.get(tuple(sorted(int(x) for x in row)))
                if hit is not None and t_dec-hit[1] <= self.args.set_valid:
                    out[i] = [10*math.log10(max(hit[0][int(u)], 1e-12)) for u in row]
                    if self.aging in ('genie', 'reported', 'model', 'stat'):
                        out[i] += self.aging_db(t_dec, t_data, row[None])[0]
        return out

    def mcs_and_value(self, t_dec, t_data, sets):
        mcs, value = super().mcs_and_value(t_dec, t_data, sets)
        if self.mode == 'top':
            for row, v in zip(sets, value):
                self.evaluated.append((float(v), tuple(sorted(int(x) for x in row))))
        return mcs, value

    def schedule(self, t_dec, t_data):
        self.receive(t_dec)
        self.evaluated = []
        users, mcs = super().schedule(t_dec, t_data)
        if self.mode == 'top':
            # query the chosen set first, then the most promising other evaluated sets
            # that have no fresh exact answer (budget enforced per TDD period)
            seen, order = set(), [tuple(sorted(int(x) for x in users))] if len(users) > 1 else []
            for v, key in sorted(self.evaluated, reverse=True):
                if len(key) > 1:
                    order.append(key)
            for key in order:
                if key in seen:
                    continue
                seen.add(key)
                hit = self.set_cache.get(key)
                if hit is not None and t_dec-hit[1] <= self.args.set_valid//2:
                    continue
                if not self.query(t_dec, list(key)):
                    break
                if len(seen) >= self.args.query_budget:
                    break
        elif self.mode == 'sets' and len(users) > 1:
            self.query(t_dec, users)
        elif self.mode == 'pairs':
            # refresh oldest pairs among the top-PF UEs (scheduled set first)
            k = self.w.cfg.num_ue
            order = list(users)+[u for u in np.argsort(-1/self.avg) if u not in users]
            top = order[:max(len(users)+2, 4)]
            pairs = sorted(((self.c_time[a, b], a, b) for i, a in enumerate(top) for b in top[i+1:]))
            for _, a, b in pairs:
                if not self.query(t_dec, [int(a), int(b)]):
                    break
        return users, mcs


def build(full, world, tab, cqimap, args):
    pol = _build(full.removesuffix('+L').removesuffix('+H').removesuffix('+A'), world, tab, cqimap, args)
    pol.name = full
    pol.layer_olla, pol.hier_olla, pol.age_olla = full.endswith('+L'), full.endswith('+H'), full.endswith('+A')
    return pol


def _build(name, world, tab, cqimap, args):
    table = {'oracle': Oracle, 'su_olla': SUOLLA, 'beam_olla': BeamOLLA, 'ci_zf': CIZF, 'ref_dec': RefDecision}
    if name in table:
        return table[name](name, world, tab, cqimap, args)
    sur = {'beam2_olla': ('none', False), 'beam2a_olla': ('none', 'genie'), 'beam2r_olla': ('none', 'reported'),
           'pair_q': ('pairs', False), 'pair_qa': ('pairs', 'genie'), 'pair_qr': ('pairs', 'reported'),
           'set_q': ('sets', False), 'set_qa': ('sets', 'genie'), 'set_qr': ('sets', 'reported'),
           'set_qx': ('sets', 'ru'), 'top_qr': ('top', 'reported'), 'top_q': ('top', False),
           'topc_qr': ('top', 'reported'), 'setc_qr': ('sets', 'reported'),
           'beam2m_olla': ('none', 'model'), 'topc_qm': ('top', 'model'), 'setc_qm': ('sets', 'model'),
           'beam2s_olla': ('none', 'stat')}
    if name in sur:
        return Surrogate(name, world, tab, cqimap, args, *sur[name])
    if name == 'qg_cqi':
        return QueryLearnedAging(name, world, tab, cqimap, args, 'cqi')
    cal = {'q_nom': 'nominal', 'q_cal': 'genie', 'q_cqi': 'cqi', 'qa_cal': 'genie', 'qa_cqi': 'cqi', 'qr_cqi': 'cqi',
           'qx_cqi': 'cqi', 'qs_cqi': 'cqi', 'qm_cqi': 'cqi', 'qd_cqi': 'cqi',
           'qfs_cqi': 'cqi', 'qfm_cqi': 'cqi', 'qfa_cqi': 'cqi', 'qfr_cqi': 'cqi'}
    if name in cal:
        aged = name[:3] in ('qa_', 'qr_', 'qx_', 'qs_', 'qm_') or name.startswith('qf')
        return (QueryAged if aged else Query)(name, world, tab, cqimap, args, cal[name])
    raise ValueError(name)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--drops', type=int, default=4)
    p.add_argument('--seed-start', type=int, default=20265000)
    p.add_argument('--speed', type=float, default=30.)
    p.add_argument('--slots', type=int, default=400)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--num-ue', type=int, default=8)
    p.add_argument('--max-layers', type=int, default=8)
    p.add_argument('--srs-period', type=int, default=10)
    p.add_argument('--csi-period', type=int, default=10)
    p.add_argument('--csi-delay', type=int, default=4)
    p.add_argument('--decision-lead', type=int, default=2)
    p.add_argument('--ack-delay', type=int, default=4)
    p.add_argument('--ci-snr-db', type=float, default=30.)
    p.add_argument('--ci-width', type=int, default=9)
    p.add_argument('--ru-alpha-db', type=float, default=0.)
    p.add_argument('--reported-beams', type=int, default=8)
    p.add_argument('--olla-step', type=float, default=.5)
    p.add_argument('--pf-tc', type=float, default=100.)
    p.add_argument('--ici-window', type=int, default=20)
    p.add_argument('--tdd-period', type=int, default=5)
    p.add_argument('--query-budget', type=int, default=4, help='MEAS_DL_POST_EQ_SINR requests per TDD period')
    p.add_argument('--query-delay', type=int, default=2, help='non-timing-controlled ST 13 latency in slots')
    p.add_argument('--pair-valid', type=int, default=40)
    p.add_argument('--set-valid', type=int, default=10)
    p.add_argument('--cov-snapshots', type=int, default=4)
    p.add_argument('--coh-window', type=int, default=6, help='SRS occasions averaged in the coherence report')
    p.add_argument('--aging-lr', type=float, default=.05)
    p.add_argument('--policies', nargs='+', default=['oracle', 'ref_dec', 'su_olla', 'beam_olla', 'q_nom', 'q_cal', 'q_cqi',
                                                     'qa_cal', 'qa_cqi', 'ci_zf'])
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    torch.set_num_threads(2)
    sionna.phy.config.device = args.device
    cfg = DropConfig(num_ue=args.num_ue, slots=args.slots)
    rec = Recorder(args.out)
    (rec.path/'experiment_source.py').write_bytes(Path(__file__).read_bytes())
    (rec.path/'dependency_srsbf_core.py').write_bytes(Path(__file__).with_name('srsbf_core.py').read_bytes())
    meta = dict(args=vars(args), drop_config=asdict(cfg), assumptions=ASSUMPTIONS,
                torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (rec.path/'metadata.json').write_text(json.dumps(meta, indent=2))
    tab = MCSTable(cfg, args.device)
    cqimap = CQIMap(tab)
    geos = []
    for drop in range(args.drops):
        seed = args.seed_start+drop
        world = World(cfg, args, seed, args.speed, args.device)
        geos.append(dict(drop=drop, seed=seed, **world.geo))
        for name in args.policies:
            t0 = time.time()
            pol = build(name, world, tab, cqimap, args)
            rng = np.random.default_rng(seed+77)
            acks = {}
            beta_pf = 1/args.pf_tc
            for t in range(cfg.slots):
                # O-DU state is frozen at the decision slot: only CQI/ACK that
                # arrived by t_dec are visible when scheduling data slot t.
                t_dec = t-args.decision_lead
                pol.update_cqi(t_dec)
                for u, ok, nl, bucket, tau in acks.pop(t_dec, []):
                    pol.olla_update(u, ok, nl, args.olla_step, bucket)
                    pol.aging_feedback(u, ok, nl, tau)
                served = np.zeros(cfg.num_ue)
                draws = rng.uniform(size=cfg.num_ue)
                if t_dec >= 1:
                    users, mcs = pol.schedule(t_dec, t)
                    if users:
                        st = torch.as_tensor(np.array([users]), device=args.device)
                        sinr = world.true_sinr(t, st)[0].cpu().numpy()
                        eff = tab.eesm_db(sinr, mcs)
                        ok = draws[users] >= tab.tbler(mcs, eff)
                        pred = pol.predict(t_dec, t, np.array([users]))[0]
                        pred_eff = tab.eesm_db(pred, mcs) if pred.ndim == 2 else pred
                        for i, u in enumerate(users):
                            bits = int(ok[i])*int(tab.tbs[mcs[i]])
                            served[u] = bits
                            if name != 'oracle':
                                acks.setdefault(t+args.ack_delay, []).append(
                                    (u, bool(ok[i]), len(users), int(pol.age_bucket(t, np.array([u]))[0]),
                                     int(t-world.ru_src[t, u])))
                            if t >= args.warmup:
                                rec.add(dict(drop=drop, policy=name, slot=t, ue=int(u), layers=len(users),
                                             mcs=int(mcs[i]), ok=bool(ok[i]), bits=bits,
                                             true_eff_db=float(eff[i]), pred_eff_db=float(pred_eff[i]),
                                             olla=float(pol.olla_offset(np.array([users]), t)[0][i]),
                                             ci_age=int(t-world.ru_src[t, u]),
                                             cqi=int(pol.cqi[u])))
                pol.avg = (1-beta_pf)*pol.avg+beta_pf*served
                if t >= args.warmup and not np.any(served):
                    rec.add(dict(drop=drop, policy=name, slot=t, ue=-1, layers=0, bits=0))
            print(f'drop={drop} policy={name} rows={len(rec.rows)} sec={time.time()-t0:.1f}', flush=True)
    (rec.path/'geometry.json').write_text(json.dumps(geos, indent=1))
    rec.finish()


if __name__ == '__main__':
    main()
