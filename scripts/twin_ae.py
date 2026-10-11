"""Data-driven baseline for paper 1: per-O-RU autoencoders trained on twin channels (in the spirit of
Luo, Jiang, Khosravirad, Alkhateeb, arXiv 2509.25793, where a site twin generates the training data).

One model per O-RU, trained once per run on twin CFRs sampled over the whole area (the same samples as
site_klt). Input: noisy CI of one link, normalized to unit norm (real/imag stacked); target: the clean
unit-norm CFR (so the model also denoises, as the KLT projection does). Latent: up to D_MAX complex
values with nested dropout (random prefix length per batch), so one model serves every budget; the
O-RU sends the first d latents (BFP9, blocks of 12, 4-bit exponent) and the norm of the noisy CI
(8 bits, 0.25 dB steps); the O-DU rescales the decoded shape by sqrt(max(norm^2 - N nu, 0)).

Variant 'pos' (position-conditioned, the learned counterpart of the region subspace): encoder and
decoder also see random-Fourier features of the estimated UE position, which both ends know; during
training that position is the true one plus N(0, s^2 I) noise, s ~ U(0, --ae-pos-noise).
"""
import math

import numpy as np
import torch
from torch import nn

D_MAX = 128


def _fourier(pos, centre, B):
    z = (torch.as_tensor(pos[..., :2], dtype=torch.float32)-torch.as_tensor(centre[:2], dtype=torch.float32))@B
    return torch.cat([torch.sin(z), torch.cos(z)], -1)


class _Net(nn.Module):
    def __init__(self, n_in, n_pos):
        super().__init__()
        self.n_pos = n_pos
        self.enc = nn.Sequential(nn.Linear(n_in+n_pos, 1024), nn.LeakyReLU(.1), nn.Linear(1024, 512), nn.LeakyReLU(.1),
                                 nn.Linear(512, 2*D_MAX))
        self.dec = nn.Sequential(nn.Linear(2*D_MAX+n_pos, 512), nn.LeakyReLU(.1), nn.Linear(512, 1024), nn.LeakyReLU(.1),
                                 nn.Linear(1024, n_in))

    def encode(self, x, pf):
        return self.enc(torch.cat([x, pf], -1) if self.n_pos else x)

    def decode(self, z, pf):
        return self.dec(torch.cat([z, pf], -1) if self.n_pos else z)


def _q_bfp9(c):
    """BFP9 on complex latents in send order: blocks of 12 share a power-of-two step. Returns (values, bits)."""
    out = np.empty_like(c)
    bits = 0
    for b0 in range(0, len(c), 12):
        x = c[b0:b0+12]
        mx = max(np.abs(x.real).max(), np.abs(x.imag).max(), 1e-30)
        step = 2.**math.ceil(math.log2(mx/255))
        out[b0:b0+12] = (np.clip(np.round(x.real/step), -256, 255)+1j*np.clip(np.round(x.imag/step), -256, 255))*step
        bits += 4+18*len(x)
    return out, bits


class TwinAE:
    """Per-O-RU autoencoder. fit() on twin CFRs [S, A, F] (UE-earliest-arrival reference, channel units
    where the UE thermal noise is 1) and sample positions [S, 3]; code() compresses one noisy link."""

    def __init__(self, A, F, nvar, use_pos, centre, seed, device=None):
        self.A, self.F, self.nvar, self.use_pos = A, F, nvar, use_pos
        self.centre = np.asarray(centre, float)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        g = torch.Generator().manual_seed(seed)
        # Fourier scales from 1 m to 100 m (channel subspaces change over metres, not wavelengths)
        self.B = (torch.randn(2, 32, generator=g)*torch.logspace(math.log10(1/100.), math.log10(1/1.), 32)[None, :]*2*math.pi)
        torch.manual_seed(seed)
        self.net = _Net(2*A*F, 64 if use_pos else 0).to(self.device)

    def _x(self, h):
        v = np.concatenate([h.real.reshape(len(h), -1), h.imag.reshape(len(h), -1)], 1)
        return torch.as_tensor(v, dtype=torch.float32, device=self.device)

    def _pf(self, pos):
        if not self.use_pos:
            return torch.zeros(len(pos), 0, device=self.device)
        return _fourier(np.asarray(pos), self.centre, self.B).to(self.device)

    def fit(self, H, pos, steps=4000, batch=512, lr=1e-3, pos_noise=5., seed=0):
        rng = np.random.default_rng(seed)
        H = np.asarray(H)
        nrm = np.linalg.norm(H.reshape(len(H), -1), axis=1)
        keep = nrm > 0
        H, pos, nrm = H[keep], np.asarray(pos)[keep], nrm[keep]
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        budgets = np.array([4, 8, 16, 32, 64, 128])
        last = []
        for it in range(steps):
            i = rng.integers(len(H), size=batch)
            h = H[i]
            n = (rng.normal(size=h.shape)+1j*rng.normal(size=h.shape))*math.sqrt(self.nvar/2)
            y = h+n
            xin = self._x(y/np.linalg.norm(y.reshape(batch, -1), axis=1)[:, None, None])
            tgt = self._x(h/nrm[i][:, None, None])
            p = pos[i].copy()
            if self.use_pos:
                s = rng.uniform(0, pos_noise, size=(batch, 1))
                p[:, :2] += rng.normal(size=(batch, 2))*s
            pf = self._pf(p)
            z = self.net.encode(xin, pf)
            d = int(rng.choice(budgets))                      # nested dropout over the budgets used in the paper
            mask = torch.zeros(1, 2*D_MAX, device=self.device)
            mask[:, :d] = 1.
            mask[:, D_MAX:D_MAX+d] = 1.
            out = self.net.decode(z*mask, pf)
            loss = ((out-tgt)**2).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            last.append(float(loss.detach()))
        return float(np.mean(last[-200:]))

    @torch.no_grad()
    def code(self, y, d, pos):
        """Compress one noisy link y [A, F] with d complex latents. Returns (reconstruction, coefficients, bits)."""
        d = int(min(d, D_MAX))
        if d <= 0:
            return np.zeros_like(y), 0, 0
        N = y.size
        e = float(np.linalg.norm(y))
        if e <= 0:
            return np.zeros_like(y), 0, 8
        e_q = 10**(np.round(20*math.log10(e)/.25)*.25/20)     # 8-bit norm, 0.25 dB steps (range not limited here)
        pf = self._pf(np.asarray(pos, float)[None])
        z = self.net.encode(self._x((y/e)[None]), pf)[0].cpu().numpy()
        c = z[:d]+1j*z[D_MAX:D_MAX+d]
        cq, bits = _q_bfp9(c)
        zq = np.zeros(2*D_MAX, np.float32)
        zq[:d], zq[D_MAX:D_MAX+d] = cq.real, cq.imag
        out = self.net.decode(torch.as_tensor(zq[None], device=self.device), pf)[0].cpu().numpy()
        shape = (out[:N]+1j*out[N:]).reshape(y.shape)
        shape /= max(np.linalg.norm(shape), 1e-30)
        return shape*math.sqrt(max(e_q**2-N*self.nvar, 0.)), d, bits+8
