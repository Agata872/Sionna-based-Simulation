"""Abstraction-level model of O-RAN SRS-BF-DL MU-MIMO for stage-4 exploration.

Independent research module; it does not change ``oran/``.  It models the
information split of O-RAN.WG4.TS.CUS.0-R005-v22.00 clause 12.7 (SRS-BF-DL):

* the O-RU estimates SRS channel information (CI) per UE antenna port and
  computes DL beamforming weights itself (vendor algorithm, unknown to O-DU);
* the O-DU schedules UEs and selects MCS from UE CQI, HARQ, SRS RRM
  measurements (T-matrix layer norms, normalized DFT beam power) and, on
  request, the O-RU's DL post-equalization SINR (clause 9.2.2.22, ST 11/13).

Physical layer is abstracted: 3GPP TR 38.901 UMi channels from Sionna, PRG
centre frequency responses, UE MMSE-IRC post-equalization SINR, EESM and the
Sionna PDSCH BLER tables.  No SRS waveform, no CSI-RS waveform, no LDPC.
All assumptions are listed in ``ASSUMPTIONS`` and copied into run metadata.
"""
from dataclasses import dataclass
import json
import math
import os

import numpy as np
import torch
import sionna.phy
import sionna.sys
from sionna.phy.channel import cir_to_ofdm_channel
from sionna.phy.channel.tr38901 import PanelArray, UMi
from sionna.sys import PHYAbstraction

ASSUMPTIONS = [
    'TR 38.901 UMi (Sionna 2.2), 7 sites x 3 sectors, ISD 200 m, serving sector = site 0 sector 0; UEs outdoor, dropped in the serving sector/Voronoi region; no wraparound.',
    'Serving link: one continuous CIR trace per drop sampled once per slot (Doppler evolves, fixed clusters; no spatial consistency over the trace).',
    'Inter-cell interference: 20 co-channel sectors, full buffer, independent isotropic random precoders per slot and PRG (flashlight); interferer CIR sampled every `coarse` slots.',
    'Frequency domain at PRG centres only; per-PRG SINR is used for EESM; no intra-PRG selectivity, no ICI from Doppler.',
    'SRS CI = true serving channel at the SRS slot plus i.i.d. CN error at a configurable per-element SNR; no SRS waveform, comb, TA or power-control model.',
    'O-RU weights: per-PRG eigen-RZF on SVD layer channels of the retained CI, equal power per layer, total power 1 per RE.',
    'UE receiver: MMSE-IRC with perfect knowledge of its own effective channel and interference covariance.',
    'CQI: SU rank-1 eigen-beamforming SINR with true interference at the CSI-RS slot, EESM, TS 38.214 CQI table 2; no codebook quantization loss.',
    'PHY abstraction: Sionna PDSCH MCS table 2 BLER tables via PHYAbstraction with EESM betas; no HARQ retransmissions.',
]


@dataclass
class DropConfig:
    num_ue: int = 8
    bs_rows: int = 4
    bs_cols: int = 4
    ut_cols: int = 1
    fc: float = 3.5e9
    scs: float = 30e3
    num_prb: int = 52
    prg: int = 2
    isd: float = 200.
    bs_height: float = 10.
    downtilt_deg: float = 10.
    tx_power_dbm: float = 44.
    ue_nf_db: float = 9.
    min_dist: float = 10.
    slots: int = 600
    slot_s: float = 0.5e-3
    coarse: int = 5
    int_layers: int = 4
    data_symbols: int = 12

    @property
    def num_bs_ant(self):
        return 2*self.bs_rows*self.bs_cols

    @property
    def num_ut_ant(self):
        return 2*self.ut_cols

    @property
    def num_prg(self):
        return -(-self.num_prb//self.prg)


def _arrays(cfg, device):
    bs = PanelArray(num_rows_per_panel=cfg.bs_rows, num_cols_per_panel=cfg.bs_cols,
                    polarization='dual', polarization_type='cross', antenna_pattern='38.901',
                    carrier_frequency=cfg.fc, device=device)
    ut = PanelArray(num_rows_per_panel=1, num_cols_per_panel=cfg.ut_cols,
                    polarization='dual', polarization_type='cross', antenna_pattern='omni',
                    carrier_frequency=cfg.fc, device=device)
    return bs, ut


def _sites(cfg):
    xy = [(0., 0.)]+[(cfg.isd*math.cos(math.radians(60*i)), cfg.isd*math.sin(math.radians(60*i)))
                     for i in range(6)]
    yaws = [30., 150., 270.]
    loc, orient = [], []
    for x, y in xy:
        for yaw in yaws:
            loc.append([x, y, cfg.bs_height])
            orient.append([math.radians(yaw), math.radians(cfg.downtilt_deg), 0.])
    return np.array(xy), np.array(loc), np.array(orient)


def drop_ues(cfg, rng):
    """Uniform outdoor drop in the serving sector (boresight 30 deg) of site 0."""
    sites, _, _ = _sites(cfg)
    out = []
    while len(out) < cfg.num_ue:
        r = cfg.isd*math.sqrt(rng.uniform())
        a = math.radians(30.)+rng.uniform(-math.pi/3, math.pi/3)
        p = np.array([r*math.cos(a), r*math.sin(a)])
        if r < cfg.min_dist:
            continue
        if any(np.linalg.norm(p-s) < r for s in sites[1:]):
            continue
        out.append([p[0], p[1], 1.5])
    return np.array(out)


def prg_frequencies(cfg):
    centres = np.minimum(np.arange(0, cfg.num_prb, cfg.prg)*12+cfg.prg*6, cfg.num_prb*12-1)
    return (centres-cfg.num_prb*6)*cfg.scs


@torch.no_grad()
def make_drop(cfg, seed, speed_kmh, device):
    """Return channels scaled so that UE thermal noise per RE is 1.

    h:   [K, Nr, M, T, F] serving sector, complex64
    ici: [K, T, F, Nr, Nr] inter-cell interference covariance (flashlight)
    """
    sionna.phy.config.seed = seed
    rng = np.random.default_rng(seed)
    bs_arr, ut_arr = _arrays(cfg, device)
    _, bs_loc, bs_or = _sites(cfg)
    ue = drop_ues(cfg, rng)
    k = cfg.num_ue
    speed = speed_kmh/3.6
    heading = rng.uniform(0, 2*math.pi, k)
    vel = np.stack([speed*np.cos(heading), speed*np.sin(heading), np.zeros(k)], -1)
    ut_or = np.stack([rng.uniform(0, 2*math.pi, k), np.zeros(k), np.zeros(k)], -1)
    t = lambda x: torch.tensor(x, dtype=torch.float32, device=device)[None]
    freqs = torch.tensor(prg_frequencies(cfg), dtype=torch.float32, device=device)
    noise_w = 10**((-174+10*math.log10(cfg.scs)+cfg.ue_nf_db)/10)/1e3
    p_re = 10**(cfg.tx_power_dbm/10)/1e3/(cfg.num_prb*12)
    amp = math.sqrt(p_re/noise_w)
    indoor = torch.zeros(1, k, dtype=torch.bool, device=device)
    serving = UMi(carrier_frequency=cfg.fc, o2i_model='low', ut_array=ut_arr, bs_array=bs_arr,
                  direction='downlink', device=device)
    serving.set_topology(t(ue), t(bs_loc[:1]), t(ut_or), t(bs_or[:1]), t(vel), indoor)
    a, tau = serving(cfg.slots, 1/cfg.slot_s)
    h = cir_to_ofdm_channel(freqs, a, tau, normalize=False)[0, :, :, 0]*amp  # K,Nr,M,T,F
    del a, tau
    inter = UMi(carrier_frequency=cfg.fc, o2i_model='low', ut_array=ut_arr, bs_array=bs_arr,
                direction='downlink', device=device)
    inter.set_topology(t(ue), t(bs_loc[1:]), t(ut_or), t(bs_or[1:]), t(vel), indoor)
    tc = -(-cfg.slots//cfg.coarse)
    a, tau = inter(tc, 1/(cfg.slot_s*cfg.coarse))
    hi = cir_to_ofdm_channel(freqs, a, tau, normalize=False)[0]*amp  # K,Nr,B,M,Tc,F
    del a, tau
    gen = torch.Generator(device=device).manual_seed(seed+911)
    nb, m = hi.shape[2], hi.shape[3]
    ici = torch.zeros(k, cfg.slots, cfg.num_prg, cfg.num_ut_ant, cfg.num_ut_ant,
                      dtype=torch.complex64, device=device)
    for s in range(cfg.slots):
        g = torch.complex(torch.randn(nb, cfg.num_prg, m, cfg.int_layers, generator=gen, device=device),
                          torch.randn(nb, cfg.num_prg, m, cfg.int_layers, generator=gen, device=device))
        q, _ = torch.linalg.qr(g)
        v = q/math.sqrt(cfg.int_layers)                       # isotropic, unit total power
        x = torch.einsum('krbmf,bfml->kfrbl', hi[..., s//cfg.coarse, :], v)
        ici[:, s] = torch.einsum('kfrbl,kfsbl->kfrs', x, x.conj())
    geometry = dict(ue_xy=ue[:, :2].tolist(), heading=heading.tolist(), speed_kmh=speed_kmh,
                    serving_gain_db=(10*torch.log10((h.abs()**2).mean((1, 2, 3, 4)))).tolist(),
                    ici_db=(10*torch.log10(ici.diagonal(dim1=-2, dim2=-1).real.mean((1, 2, 3)))).tolist())
    return h.contiguous(), ici, geometry


# ---------------------------------------------------------------- beamforming
def layer_channels(ci, rank=1):
    """[..., Nr, M] -> [..., rank, M] = sigma_i v_i^H (T-matrix layer channels)."""
    _, s, vh = torch.linalg.svd(ci, full_matrices=False)
    return s[..., :rank, None].to(vh.dtype)*vh[..., :rank, :]


def rzf_weights(g, alpha):
    """g: [..., L, M] layer channels -> W [..., M, L], each column power 1/L."""
    eye = torch.eye(g.shape[-2], dtype=g.dtype, device=g.device)
    w = g.mH@torch.linalg.inv(g@g.mH+alpha*eye)
    w = w/torch.linalg.vector_norm(w, dim=-2, keepdim=True).clamp_min(1e-30)
    return w/math.sqrt(g.shape[-2])


def mmse_irc_sinr(heff, cov):
    """Rank-1-per-UE post-equalization SINR.

    heff: [..., U, Nr, L] effective channel of scheduled UE u (layer u is its own)
    cov:  [..., U, Nr, Nr] interference-plus-noise covariance outside the cell
    returns [..., U]
    """
    u = heff.shape[-3]
    idx = torch.arange(u, device=heff.device)
    d = heff[..., idx, :, idx]                                # [U, ..., Nr]
    d = torch.movedim(d, 0, -2)                               # [..., U, Nr]
    # interference covariance built from the other columns only (no total - d d^H
    # cancellation, which loses definiteness in float32 at 60-70 dB SNR)
    mask = 1-torch.eye(u, heff.shape[-1], dtype=heff.dtype, device=heff.device)[:, None, :]   # [U,1,L]
    other = heff*mask
    r = other@other.mH+cov
    x = torch.linalg.solve(r, d[..., None])[..., 0]
    return (d.conj()*x).sum(-1).real.clamp_min(0)


def set_sinr(h_ue, w, cov):
    """h_ue: [..., U, Nr, M] channels of scheduled UEs, w: [..., M, U]."""
    return mmse_irc_sinr(h_ue@w[..., None, :, :], cov)


# ------------------------------------------------------------- link abstraction
# TS 38.214 Table 5.1.3.1-2 spectral efficiencies Qm*R
SE_TABLE2 = [2*120/1024, 2*193/1024, 2*308/1024, 2*449/1024, 2*602/1024, 4*378/1024, 4*434/1024,
             4*490/1024, 4*553/1024, 4*616/1024, 4*658/1024, 6*466/1024, 6*517/1024, 6*567/1024,
             6*616/1024, 6*666/1024, 6*719/1024, 6*772/1024, 6*822/1024, 6*873/1024, 8*682.5/1024,
             8*711/1024, 8*754/1024, 8*797/1024, 8*841/1024, 8*885/1024, 8*916.5/1024, 8*948/1024]


class MCSTable:
    """PDSCH MCS table 2 thresholds/betas/TBS for a fixed full-band allocation."""

    def __init__(self, cfg, device, bler_target=.1, rank=1):
        self.phy = PHYAbstraction(device=device)
        self.device = device
        path = os.path.join(os.path.dirname(sionna.sys.__file__), 'esm_params', 'eesm_beta_table.json')
        self.beta = np.array(json.load(open(path))['index']['2'], dtype=float)
        self.mcs = np.arange(len(self.beta))
        self.num_re = cfg.num_prb*12*cfg.data_symbols*rank
        grid = np.arange(-10, 40, .02)
        n = len(self.mcs)
        tb = self._tbler(np.repeat(self.mcs, len(grid)), np.tile(grid, n)).reshape(n, len(grid))
        # MCS rows without Sionna BLER-table support (NaN: table-2 MCS 0, 1, 27) are
        # filled by shifting the nearest supported curve by the Shannon gap, as
        # oran/bler_model.py does for thresholds.  TBS is scaled by SE.
        se = np.array(SE_TABLE2)
        valid = np.isfinite(tb).all(-1)
        self.extrapolated = [int(m) for m in self.mcs if not valid[m]]
        step = grid[1]-grid[0]
        for m in self.extrapolated:
            r = min((i for i in self.mcs if valid[i]), key=lambda i: abs(se[i]-se[m]))
            shift = 10*np.log10((2**se[m]-1)/(2**se[r]-1))
            tb[m] = np.interp(grid-shift, grid, tb[r], left=1., right=tb[r][-1])
        self.grid, self.tbler_grid = grid, tb
        thr = []
        for m in range(n):
            ok = np.flatnonzero(tb[m] <= bler_target)
            thr.append(grid[ok[0]] if len(ok) else np.inf)
        self.threshold_db = np.array(thr)
        _, _, _, tbs = self._raw(self.mcs, np.full(n, 40.))
        tbs = tbs.astype(float)
        for m in self.extrapolated:
            r = min((i for i in self.mcs if i not in self.extrapolated), key=lambda i: abs(se[i]-se[m]))
            tbs[m] = 8*round(tbs[r]*se[m]/se[r]/8)
        self.tbs = tbs.astype(int)

    def _raw(self, mcs, eff_db):
        mcs_t = torch.tensor(mcs, dtype=torch.int32, device=self.device)
        eff = torch.tensor(10**(np.asarray(eff_db)/10), dtype=torch.float32, device=self.device)
        nre = torch.full_like(mcs_t, self.num_re)
        bits, _, _, tbler, _ = self.phy(mcs_t, sinr_eff=eff, num_allocated_re=nre,
                                       mcs_table_index=2, mcs_category=1)
        return None, None, tbler.cpu().numpy(), bits.cpu().numpy()

    def _tbler(self, mcs, eff_db):
        return self._raw(mcs, eff_db)[2]

    def tbler(self, mcs, eff_db):
        mcs = np.asarray(mcs, dtype=int)
        eff = np.clip(np.asarray(eff_db, dtype=float), self.grid[0], self.grid[-1])
        pos = (eff-self.grid[0])/(self.grid[1]-self.grid[0])
        lo = np.floor(pos).astype(int).clip(0, len(self.grid)-2)
        frac = pos-lo
        return (1-frac)*self.tbler_grid[mcs, lo]+frac*self.tbler_grid[mcs, lo+1]

    def eesm_db(self, sinr_lin, mcs):
        """sinr_lin: [..., F] linear numpy; mcs broadcast to [...]."""
        b = self.beta[np.asarray(mcs, dtype=int)]
        x = np.asarray(sinr_lin, dtype=float)
        val = -b*(np.log(np.mean(np.exp(-(x/b[..., None])), axis=-1)+1e-300))
        return 10*np.log10(np.maximum(val, 1e-12))

    def eesm_all_db(self, sinr_lin):
        """[..., F] -> [..., n_mcs] effective SINR for every MCS beta."""
        x = np.asarray(sinr_lin, dtype=float)[..., None, :]
        b = self.beta[:, None]
        val = -self.beta*np.log(np.mean(np.exp(-x/b), axis=-1)+1e-300)
        return 10*np.log10(np.maximum(val, 1e-12))

    def select_from_prg(self, sinr_lin, offset_db=0.):
        """Largest MCS whose EESM(beta_m) + offset clears its 10% threshold."""
        eff = self.eesm_all_db(sinr_lin)+np.asarray(offset_db)[..., None]
        ok = eff >= self.threshold_db
        idx = np.where(ok.any(-1), ok.shape[-1]-1-np.argmax(ok[..., ::-1], -1), 0)
        return idx

    def select_from_eff(self, eff_db):
        ok = np.asarray(eff_db)[..., None] >= self.threshold_db
        return np.where(ok.any(-1), ok.shape[-1]-1-np.argmax(ok[..., ::-1], -1), 0)

    def expected_bits(self, mcs, eff_db=None, target=.1):
        bits = self.tbs[np.asarray(mcs, dtype=int)]
        if eff_db is None:
            return bits*(1-target)
        return bits*(1-self.tbler(mcs, eff_db))
