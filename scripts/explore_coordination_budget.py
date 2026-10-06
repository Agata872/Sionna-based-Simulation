#!/usr/bin/env python3
"""Independent Team-MMSE equation reproduction and finite-message screening.

Two RUs, K single-antenna UEs, stationary iid Gaussian channels. This is not
the paper's network/drop simulation or an NR/SRS/packet-delay implementation.
The ideal equations are checked before per-realization power normalization;
reported SE uses identical per-RU power and is not an optimality claim.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import sionna.phy
from oran import CompressionConfig
from oran.fronthaul import IQCodec
from explore_csirs_precoding import Recorder


def cn(shape, gen, device, dtype=torch.complex64):
    real = torch.float64 if dtype == torch.complex128 else torch.float32
    return torch.complex(torch.randn(shape, generator=gen, device=device, dtype=real),
                         torch.randn(shape, generator=gen, device=device, dtype=real)) / math.sqrt(2)


def inverse_stage(a, regularization):
    """A:[...,K,N]; (A^H A + lambda I)^-1 A^H via K-dimensional solve."""
    eye = torch.eye(a.shape[-2], device=a.device, dtype=a.dtype)
    return torch.linalg.solve(a @ a.mH + regularization * eye, a).mH


def coordination(b):
    """C_l + sum_{j!=l} B_j C_j = I, for exactly two independent RUs."""
    k = b.shape[-1]
    eye = torch.eye(k, device=b.device, dtype=b.dtype).expand(*b.shape[:-3], k, k)
    system = torch.cat([torch.cat([eye, b[..., 1, :, :]], -1),
                        torch.cat([b[..., 0, :, :], eye], -1)], -2)
    solution = torch.linalg.solve(system, torch.cat([eye, eye], -2))
    return solution.unflatten(-2, (2, k))


def per_ru_power(w):
    """Normalize both blocks to 0.5. Explicitly changes the raw team solution."""
    return w / torch.linalg.vector_norm(w, dim=(-2, -1), keepdim=True).clamp_min(1e-12) / math.sqrt(2)


def wire_bits(rows, cols, width):
    # One custom message per RU, including timestamp/version/width/scale ID.
    # 16-byte application header + 8-byte eCPRI-like header + Ethernet42.
    # Packetization MTU=1500; custom B/C/G protocol is NOT an O-RAN extension.
    block_bytes = math.ceil((8 + 2 * rows * width) / 8)
    payload_bytes = cols * block_bytes
    packets = math.ceil(payload_bytes / (1500 - 24))
    return 8 * (payload_bytes + packets * 66)


def fitting_width(rows, cols, budget):
    allowed = [b for b in range(3, 17) if wire_bits(rows, cols, b) <= budget]
    return max(allowed) if allowed else 0


def quantize_matrix(x, width, nominal_power, diagnostics=None):
    """Column BFP, fixed scale set by dimensions/configuration, never test CSI."""
    c = IQCodec(CompressionConfig('bfp', width), nominal_power, x.shape[-2])
    q = c(x.transpose(-1, -2)).transpose(-1, -2)
    if diagnostics is not None:
        diagnostics.append(dict(clipped_real_components=c.stats.num_clipped,
                                total_real_components=2*x.numel(),
                                signal_energy=c.stats.signal_energy,error_energy=c.stats.error_energy))
    return q


def posterior(old, observation, rho, eta):
    """Exact Gaussian posterior from perfect own-old H and noisy current H."""
    v = 1 - rho * rho
    gain = 1. if eta == 0 else v / (v + eta)
    mean = rho * old + gain * (observation - rho * old)
    variance = 0. if eta == 0 else v * eta / (v + eta)
    return mean, variance


def expected_b(old, rho, eta, sigma, samples, generator):
    """MC conditional E[H_l F_l|H_l,old], independent of evaluated fresh CSI.

    Posterior-mean distribution given own old H has variance v^2/(v+eta).
    E[H F|local observations]=mu F; the posterior variance adds K*var to
    regularization. Conditional independence between RUs is essential here.
    """
    k = old.shape[-2]
    v = 1 - rho * rho
    postvar = 0. if eta == 0 else v * eta / (v + eta)
    meanvar = v if eta == 0 else v * v / (v + eta)
    if meanvar == 0:
        mu = rho * old
        return mu @ inverse_stage(mu, sigma + k * postvar)
    total = torch.zeros(*old.shape[:-1], k, device=old.device, dtype=old.dtype)
    for start in range(0, samples, 64):
        count = min(64, samples - start)
        mu = rho * old[None] + math.sqrt(meanvar) * cn((count, *old.shape), generator, old.device)
        total += (mu @ inverse_stage(mu, sigma + k * postvar)).sum(0)
    return total / samples


def se_metrics(h, w, no):
    g = torch.einsum('blkn,blnj->bkj', h, w)
    signal = torch.diagonal(g, dim1=-2, dim2=-1).abs().square()
    interference = (g.abs().square().sum(-1) - signal).clamp_min(0)
    sinr = signal / (interference + no)
    return torch.log2(1 + sinr).sum(-1), 10 * torch.log10(sinr.clamp_min(1e-12))


def validate(device, seed):
    """Meaningful identity check against a direct centralized MMSE solve."""
    gen = torch.Generator(device=device).manual_seed(seed + 700001)
    checks = []
    for n, k in [(4, 2), (4, 6), (16, 8)]:
        h = cn((16, 2, k, n), gen, device, torch.complex128)
        sigma = .1
        f = inverse_stage(h, sigma)
        c = coordination(h @ f)
        team = (f @ c).flatten(1, 2)
        full_h = h.transpose(1, 2).flatten(-2, -1)
        reference = inverse_stage(full_h, sigma)
        relative_error = float(torch.linalg.vector_norm(team-reference) / torch.linalg.vector_norm(reference))
        assert relative_error < 1e-9, relative_error
        checks.append(dict(n=n, k=k, raw_team_vs_central_relative_error=relative_error))
    return dict(checks=checks, status='passed',
                scope='P=I, two independent RUs, no aging/error, no quantization/power normalization')


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--loads', nargs='+', default=['4:2', '4:4', '4:8', '16:4', '16:8', '16:16', '32:8'])
    p.add_argument('--rhos', nargs='+', type=float, default=[1, .99, .95, .85, .7])
    p.add_argument('--errors', nargs='+', type=float, default=[0, .01, .1])
    p.add_argument('--snr', type=float, default=20)
    p.add_argument('--drops', type=int, default=256)
    p.add_argument('--mc-samples', type=int, default=512)
    p.add_argument('--seed', type=int, default=20261007)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    torch.set_num_threads(2)
    sionna.phy.config.device = args.device
    rec = Recorder(args.out)
    dependencies = ['explore_csirs_precoding.py']
    (rec.path/'experiment_source.py').write_bytes(Path(__file__).read_bytes())
    for dep in dependencies:
        (rec.path/f'dependency_{dep}').write_bytes(Path(__file__).with_name(dep).read_bytes())
    meta = dict(args=vars(args), started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                sources=['https://arxiv.org/html/2404.12786v1',
                         'https://www.comm.toronto.edu/~liang/publications/ToWC_HierarchicalSemiOnlineInformationParsing.pdf'],
                timing=dict(own_old_sample=0, upload_arrives_du=1, du_computation_done=2, downlink_arrives_ru=3,
                            rho='Endpoint correlation between t=0 and t=3, no physical-millisecond mapping'),
                assumptions=['Perfect cached own-old CSI; only fresh local CSI has additive CN error eta.',
                             'Known iid Gaussian law, rho and eta; conditional MC independent of evaluation innovations.',
                             'Original Team-MMSE equations independently implemented, not original network simulation.',
                             'Noisy Team-MMSE uses exact posterior under this particular information model.',
                             'SE comparisons normalize instantaneous per-RU power to 0.5 and are not the paper optimality claim.',
                             'B/C/G are hypothetical custom coordination messages, not validated O-RAN formats.',
                             'Both control directions include byte-aligned BFP and 66-byte/message overhead, MTU1500.',
                             'Shared U-plane, real SRS acquisition, physical time evolution and packet queues absent.',
                             'Ideal reference messages unquantized and have wire bits=-1, never a finite-budget scheme.',
                             'Target-only ridge and proximal tracking are baselines inspired by SOPIP, not full SOPIP.'])
    (rec.path/'metadata.json').write_text(json.dumps(meta, indent=2))
    (rec.path/'validation.json').write_text(json.dumps(validate(args.device,args.seed),indent=2))
    no = 10**(-args.snr/10)
    for load_id, load in enumerate(args.loads):
        n,k = map(int,load.split(':'))
        gen = torch.Generator(device=args.device).manual_seed(args.seed + load_id*100003)
        old = cn((args.drops,2,k,n),gen,args.device)
        innovation = cn(old.shape,gen,args.device)
        estimation_noise = cn(old.shape,gen,args.device)
        sigma = k * no
        full_h = old.transpose(1,2).flatten(-2,-1)
        old_w = per_ru_power(inverse_stage(full_h,sigma).unflatten(-2,(2,n)))
        # Finite UL: serialize K CSI rows as N-element blocks.
        uploaded = quantize_matrix(old.transpose(-1,-2),9,1.).transpose(-1,-2)
        full_uploaded = uploaded.transpose(1,2).flatten(-2,-1)
        old_w_upload9 = per_ru_power(inverse_stage(full_uploaded,sigma).unflatten(-2,(2,n)))
        received_w9 = per_ru_power(quantize_matrix(old_w_upload9,9,1/(2*n*k)))
        full_budget = wire_bits(n,k,9)
        small_width = fitting_width(k,k,full_budget)
        for rho in args.rhos:
            fresh = rho*old + math.sqrt(1-rho*rho)*innovation
            fresh_reference = per_ru_power(inverse_stage(fresh.transpose(1,2).flatten(-2,-1),sigma).unflatten(-2,(2,n)))
            for eta in args.errors:
                observation = fresh + math.sqrt(eta)*estimation_noise
                mu, postvar = posterior(old,observation,rho,eta)
                local_f = inverse_stage(mu,sigma+k*postvar)
                mcgen=torch.Generator(device=args.device).manual_seed(args.seed+900000+load_id*100003)
                b = expected_b(old,rho,eta,sigma,args.mc_samples,mcgen)
                c = coordination(b)
                team = per_ru_power(local_f @ c)
                cases = [('fresh_global_reference',fresh_reference,-1,-1,0),
                         ('stale_global_reference',old_w,-1,-1,0),
                         ('local_posterior_mmse',per_ru_power(local_f),0,0,0),
                         ('team_reference',team,-1,-1,0),
                         ('full_w_bfp9',received_w9,full_budget,full_budget,9)]
                # Strong simple baseline preserves each local old contribution.
                residual = (old-mu) @ received_w9
                delta = inverse_stage(mu,1.+k*postvar) @ residual
                tracked = per_ru_power(received_w9+delta)
                cases.append(('proximal_tracking_bfp9',tracked,full_budget,full_budget,9))
                # Target-only representation requires no full old W at RU.
                target = uploaded @ old_w_upload9
                quantization_diagnostics={}
                for width,label in [(9,'bfp9'),(small_width,'matched_budget')]:
                    if width:
                        diag=[]
                        qb = quantize_matrix(b,width,1/k,diag)
                        qc = quantize_matrix(coordination(qb),width,1.,diag)
                        team_q = per_ru_power(local_f @ qc)
                        small_bits=wire_bits(k,k,width)
                        cases.append((f'team_summary_{label}',team_q,small_bits,small_bits,width))
                        quantization_diagnostics[f'team_summary_{label}']=diag
                        # Same scalar codec and message dimensions; receiver
                        # knows the identity. Deterministic format ID in header.
                        # A conventional residual representation, no novelty claim.
                        diag=[]
                        identity=torch.eye(k,device=args.device,dtype=b.dtype)
                        qd=quantize_matrix(identity-b,width,1/k,diag)
                        residual_c=coordination(identity-qd)
                        residual_qc=quantize_matrix(residual_c,width,1.,diag)
                        cases.append((f'team_residual_{label}',per_ru_power(local_f@residual_qc),small_bits,small_bits,width))
                        quantization_diagnostics[f'team_residual_{label}']=diag
                        qtarget=quantize_matrix(target,width,n/k)
                        target_w=per_ru_power(inverse_stage(mu,sigma+k*postvar) @ qtarget)
                        cases.append((f'target_only_{label}',target_w,full_budget,small_bits,width))
                    else:
                        # No legal >=3-bit summary fits cap: zero coordination.
                        cases.append((f'team_summary_{label}_fallback',per_ru_power(local_f),0,0,0))
                # Original raw quadratic objective, before normalization.
                raw_team = local_f @ c
                g_raw = torch.einsum('blkn,blnj->bkj',fresh,raw_team)
                raw_objective = (g_raw-torch.eye(k,device=args.device)).abs().square().sum((-2,-1)) + sigma*raw_team.abs().square().sum((-3,-2,-1))
                reference_se,_ = se_metrics(fresh,fresh_reference,no)
                stale_se,_ = se_metrics(fresh,received_w9,no)
                for name,w,ul,dl,width in cases:
                    assert torch.allclose(w.abs().square().sum((-2,-1)),torch.full((args.drops,2),.5,device=args.device),atol=3e-6)
                    se,db=se_metrics(fresh,w,no)
                    diag=quantization_diagnostics.get(name,[])
                    clipped_fraction=sum(x['clipped_real_components'] for x in diag)/max(1,sum(x['total_real_components'] for x in diag))
                    for d in range(args.drops):
                        rec.add(dict(seed=args.seed,drop=d,n=n,k=k,rho=rho,eta=eta,snr_db=args.snr,method=name,width=width,
                                     sum_se=float(se[d]),paired_delta_vs_full_w9=float(se[d]-stale_se[d]),
                                     fresh_reference_se=float(reference_se[d]),worst_ue_sinr_db=float(db[d].min()),
                                     ul_wire_bits_per_ru_update=ul,dl_wire_bits_per_ru_update=dl,
                                     full_w9_budget_per_direction=full_budget,
                                     batch_quantized_clipped_fraction=clipped_fraction,
                                     raw_team_objective=float(raw_objective[d]) if name=='team_reference' else None))
                print(f'N={n} K={k} rho={rho:g} eta={eta:g} summary_width={small_width} rows={len(rec.rows)}',flush=True)
    rec.finish()


if __name__=='__main__':
    main()
