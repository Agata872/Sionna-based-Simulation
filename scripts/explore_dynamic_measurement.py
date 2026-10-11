#!/usr/bin/env python3
"""Causal two-RU continuous-CDL pilot/precoder/link-adaptation falsification.

Independent research entry point. A whole trace is generated in ONE CDL call;
no concatenation of independent drops. Fixed physical slots include pilot cost.
Perfect timestamped local CSI is a screening assumption, not implemented SRS.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import sionna.phy
from sionna.phy.channel import cir_to_ofdm_channel
from oran import ORANConfig, ORANDownlink, CompressionConfig
from oran.channel import ChannelState
from oran.csi import CQIReporter, lmmse_sinr, reference_num_re, reference_data_symbols
from oran.csi_rs import CSIRS, CSIRSReceiver
from oran.fronthaul import IQCodec, fronthaul_rate
from oran.link_adaptation import LinkAdaptation, LinkAdaptationConfig
from oran.bler_model import AWGNLinkModel
from oran.precoding import apply_precoder
from explore_csirs_precoding import Recorder


def ru_power(w, n):
    parts = w.unflatten(-2, (2, n))
    return (parts / torch.linalg.vector_norm(parts, dim=(-2, -1), keepdim=True).clamp_min(1e-12)
            / math.sqrt(2)).flatten(-3, -2)


def quantize(w, n, width):
    codec = IQCodec(CompressionConfig('bfp', width), 1/(2*n*w.shape[-1]), n)
    return ru_power(torch.cat([codec(w[..., j*n:(j+1)*n, :].transpose(-2, -1)).transpose(-2, -1)
                              for j in range(2)], -2), n)


def tracking(old_h, new_h, prior, n, ridge):
    out = []
    for j in range(2):
        a = old_h[..., j*n:(j+1)*n].flatten(-3, -2)
        b = new_h[..., j*n:(j+1)*n].flatten(-3, -2)
        w = prior[..., j*n:(j+1)*n, :]
        delta = torch.linalg.solve(b.mH@b + ridge*torch.eye(n, device=b.device), b.mH@((a-b)@w))
        out.append(w+delta)
    return ru_power(torch.cat(out, -2), n)


def effective_report(h, w, n, width, no):
    """Each RU independently reports its actual H_l W_l at PRG centers.

    Hypothetical float BFP with int8 exponent per column and 6/9-bit signed
    real/imag mantissas. No fixed16 clipping or hidden exact norm side channel.
    Returns the CU-reconstructed aggregate effective channel, not current truth.
    """
    parts=[]
    for j in range(2):
        g=h[...,j*n:(j+1)*n]@w[...,j*n:(j+1)*n,:].unsqueeze(-3)
        x=g.flatten(-3,-2)
        lim=2**(width-1)-1
        maximum=torch.maximum(x.real.abs(),x.imag.abs()).amax(-2,keepdim=True)
        exponent=torch.ceil(torch.log2((maximum/lim).clamp_min(2**-100)))
        assert ((exponent>=-128)&(exponent<=127)).all()
        step=torch.exp2(exponent)
        q=torch.complex(torch.round(x.real/step),torch.round(x.imag/step))*step
        parts.append(q.unflatten(-2,g.shape[-3:-1]))
    return sum(parts)


def reported_sinr(g,no,nu=2):
    """CU LMMSE SINR from received effective-channel matrices [B,P,K,R,L]."""
    eye=torch.eye(g.shape[-2],dtype=g.dtype,device=g.device)
    out=[]
    for k in range(g.shape[-3]):
        a=g[...,k,:,:]
        d=a[...,k*nu:(k+1)*nu]
        v=torch.linalg.solve(a@a.mH+no*eye,d)
        z=v.mH@a
        sig=z[...,k*nu:(k+1)*nu].diagonal(dim1=-2,dim2=-1).abs().square()
        den=(z.abs().square().sum(-1)-sig).clamp_min(0)+no*v.abs().square().sum(-2)
        out.append(sig/den.clamp_min(1e-12))
    return torch.stack(out,2).permute(0,2,3,1)


def float_bfp(x,width):
    lim=2**(width-1)-1
    maximum=torch.maximum(x.real.abs(),x.imag.abs()).amax(-2,keepdim=True)
    exponent=torch.ceil(torch.log2((maximum/lim).clamp_min(2**-100)))
    assert ((exponent>=-128)&(exponent<=127)).all()
    step=torch.exp2(exponent)
    return torch.complex(torch.round(x.real/step),torch.round(x.imag/step))*step


def project_ru_ball(w,n):
    parts=w.unflatten(-2,(2,n))
    norm=torch.linalg.vector_norm(parts,dim=(-2,-1),keepdim=True)
    return (parts/torch.maximum(torch.ones_like(norm),norm*math.sqrt(2))).flatten(-3,-2)


def sopip_pipeline(hc,central,n,period,alpha=512.,jcc=5,jap=5,width=9):
    """Independent Algorithms 1/2 implementation, u=1, d=3, perfect local CSI.

    Multi-receive-antenna extension uses H with all K*R rows and L columns
    in the target. Per-RU BALL projection, not forced equality power. period=1
    follows the paper's every-slot pipeline; period=16 is a caching variant.
    Common residual is sent and RU subtracts its cached old H times delivered V
    (the paper's discussion after (9)). Own H/V uploads and V/residual downloads
    use float BFP at the selected width; width=0 is an unlimited unquantized
    reference. This is NOT reproduction of the original network figures.
    """
    slots=len(hc);actual=[];pending={};last=central[0];parsed=None;old_source=0
    encode=(lambda x:x) if width==0 else (lambda x:float_bfp(x,width))
    for t in range(slots):
        if t in pending:
            last,parsed,old_source=pending.pop(t)
        if parsed is None:
            v=last
        else:
            rows=[]
            old=encode(hc[old_source].flatten(-3,-2))
            fresh=hc[t].flatten(-3,-2)
            for ru in range(2):
                a=fresh[...,ru*n:(ru+1)*n]
                prior=last[...,ru*n:(ru+1)*n,:]
                other=parsed-old[...,ru*n:(ru+1)*n]@prior
                for _ in range(jap):
                    prior=prior-a.mH@(a@prior+other)/alpha
                    norm=torch.linalg.vector_norm(prior,dim=(-2,-1),keepdim=True)
                    prior=prior/torch.maximum(torch.ones_like(norm),norm*math.sqrt(2))
                rows.append(prior)
            v=torch.cat(rows,-2)
        actual.append(v)
        # Upload arrives at CU one slot after the actual local H/V snapshot.
        source=t-1
        if source>=0 and source%period==0:
            old=hc[source].flatten(-3,-2)
            uploaded=encode(old)
            uploaded_v=project_ru_ball(torch.cat([encode(actual[source][...,ru*n:(ru+1)*n,:])
                                                for ru in range(2)],-2),n)
            # Same full-power global BD target as the other beamformers.
            ideal_target=ru_power(ORAN_precoder(uploaded.unflatten(-2,(2,4))),n)
            target=uploaded@ideal_target
            updated=uploaded_v
            # Check the chosen fixed inverse step against the evaluated Gram.
            assert float(torch.linalg.matrix_norm(uploaded.mH@uploaded,ord=2).max())<=alpha
            for _ in range(jcc):
                updated=project_ru_ball(updated-uploaded.mH@(uploaded@updated-target)/alpha,n)
            received=project_ru_ball(torch.cat([encode(updated[...,ru*n:(ru+1)*n,:])
                                               for ru in range(2)],-2),n)
            # RU independently caches exactly the own H representation it
            # uploaded, so subtraction uses identical information at both ends.
            residual=encode(uploaded@received-target)
            pending[t+3]=(received,residual,source)
    return actual


def ORAN_precoder(h):
    from oran.precoding import compute_precoder
    return compute_precoder(h,2,'bd')


def continuous_trace(sim, slots, speed, seed):
    """Same fixed UE velocity for all RU links; fixed rays and random phases.

    CDL has fixed angles/delays in a trace, Doppler evolves continuously. Slot
    grid uses OFDM symbol duration including CP, no intra-symbol ICI.
    """
    sionna.phy.config.seed = seed
    for k, row in enumerate(sim.channel.model.links):
        angle = [.3, 1.7][k]
        velocity = torch.tensor([speed*math.cos(angle), speed*math.sin(angle), 0.], device=sim.device)
        for cdl in row:
            cdl._ut_velocity = velocity
    ns = sim.rg.num_ofdm_symbols
    a, tau = sim.channel.model(1, slots*ns, 1/sim.rg.ofdm_symbol_duration)
    # Convert one slot at a time, avoiding the path x frequency x time product.
    hs, csis = [], []
    for t in range(slots):
        h = cir_to_ofdm_channel(sim.channel.frequencies, a[..., t*ns:(t+1)*ns], tau)
        hs.append(h)
        # Boundary observation is available before this slot's W is applied.
        # First-DMRS-symbol CSI would peek into the future within the slot.
        csis.append(sim.channel.csi(ChannelState(h), 0))
    return hs, csis


POLICIES = {
    'stale16': ('stale', 16, 'ordinary'),
    'track16': ('track', 16, 'ordinary'),
    'track16_fastolla': ('track', 16, 'fast'),
    'track16_ageolla': ('track', 16, 'age'),
    'track16_lowpass': ('track', 16, 'lowpass'),
    'track4': ('track', 4, 'ordinary'),
    'pilot_locked16': ('pilot_locked', 16, 'ordinary'),
    'track16_genie': ('track', 16, 'genie'),
    'track16_delta': ('track', 16, 'delta'),
    'track16_g6d1': ('track', 16, 'g6d1'),
    'track16_g9d1': ('track', 16, 'g9d1'),
    'track16_g9d4': ('track', 16, 'g9d4'),
    'track16_g9p1': ('track', 16, 'g9p1'),
    'track16_g6e4d1': ('track', 16, 'g6e4d1'),
    'track16_g9e4d1': ('track', 16, 'g9e4d1'),
    'track16_g9e8d1': ('track', 16, 'g9e8d1'),
    'sopip1': ('sopip1', 16, 'ordinary'),
    'sopip16': ('sopip16', 16, 'ordinary'),
    'sopip1_g9e4d1': ('sopip1', 16, 'g9e4d1'),
}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--drops', type=int, default=8)
    p.add_argument('--seed-start', type=int, default=20263000)
    p.add_argument('--slots', type=int, default=128)
    p.add_argument('--warmup', type=int, default=32)
    p.add_argument('--speeds', type=float, nargs='+', default=[3., 10.])
    p.add_argument('--policies', nargs='+', choices=POLICIES, default=['stale16','track16','track16_ageolla','track4','pilot_locked16','track16_genie'])
    p.add_argument('--snr', type=float, default=20.)
    p.add_argument('--du-period', type=int, default=16)
    p.add_argument('--du-delay', type=int, default=4)
    p.add_argument('--feedback-delay', type=int, default=4)
    p.add_argument('--ack-delay', type=int, default=4)
    p.add_argument('--ridge', type=float, default=1.)
    p.add_argument('--width', type=int, default=9)
    p.add_argument('--local-error',type=float,default=0.,help='nominal variance of independent complex boundary-CSI observation error; not physical SRS')
    p.add_argument('--antennas', type=int, default=4)
    p.add_argument('--num-prb', type=int, default=6)
    p.add_argument('--mode', choices=['numerical', 'crc'], default='numerical')
    p.add_argument('--sopip-alpha',type=float,default=512.)
    p.add_argument('--sopip-jcc',type=int,default=20)
    p.add_argument('--sopip-jap',type=int,default=20)
    p.add_argument('--sopip-width',type=int,choices=[0,6,9],default=9,help='0 means unlimited unquantized SOPIP messages; do not assign finite wire budget')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.warmup >= args.slots or min(args.du_period, args.feedback_delay, args.ack_delay) < 1:
        raise ValueError('Invalid time configuration')
    torch.set_num_threads(2)
    sionna.phy.config.device = args.device
    cfg = ORANConfig(category='B', num_oru=2, num_oru_ant=args.antennas,
                     num_prb=args.num_prb, num_ue_ant=4, num_layers_per_ue=2,
                     uplane=CompressionConfig('bfp', 9), bfw=CompressionConfig('bfp', args.width),
                     device=args.device)
    sim = ORANDownlink(cfg)
    model = AWGNLinkModel(device=args.device)
    reporter = CQIReporter(1, reference_num_re(sim.pdsch_configs[0], cfg.num_prb, 2), model)
    allowed = sim.supported_mcs()
    csirs = CSIRS(cfg, sim.pdsch_configs[0], device=args.device)
    receiver = CSIRSReceiver(csirs, cfg.prg_num_prb, sim.ue.estimator)
    ucodec = IQCodec(cfg.uplane, 1., 12)
    rec = Recorder(args.out)
    (rec.path/'experiment_source.py').write_bytes(Path(__file__).read_bytes())
    (rec.path/'dependency_explore_csirs_precoding.py').write_bytes(Path(__file__).with_name('explore_csirs_precoding.py').read_bytes())
    meta = dict(args=vars(args), config=asdict(cfg), torch=torch.__version__,
                gpu=torch.cuda.get_device_name(), cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
                started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                slot_duration_s=1/cfg.slots_per_second, symbol_duration_s=float(sim.rg.ofdm_symbol_duration),
                assumptions=['ONE full-trace CDL call per independent drop/speed. Fixed angle/delay Doppler trajectory, no spatial consistency.',
                             'Local CSI at slot-boundary symbol 0, exact unless local-error>0; abstract timestamped DU CSI delay, no implemented SRS.',
                             'Optional local-error is independent CN observation error at each boundary; realized NMSE saved separately, no physical SRS or calibration-error model.',
                             'Same physical elapsed slots and noise seeds; pilot slots have zero decoded payload.',
                             'Pilot CQI applied after feedback delay; delayed ACK updates the OLLA group at TRANSMIT time.',
                             'Data W is fixed across each slot and DMRS uses that actual W.',
                             'Both RUs receive each DU version together; no queues/loss/asynchronous arrival.',
                             'BFP applies to DU W only; subsequent local computation is floating point.',
                             'delta is diagnostic fresh GLOBAL effective-SINR side information, never a deployable limited-message algorithm.',
                             'g6/g9 reports are independently quantized local H_l W_l at PRG centers, CU combines delayed messages; g9p1 uses two past reports only.',
                             'numerical mode uses genie SINR and Shannon SE only, no BLER/goodput claim.',
                             'crc mode uses actual CSI-RS, DMRS and LDPC CRC; no HARQ.'])
    (rec.path/'metadata.json').write_text(json.dumps(meta, indent=2))
    (rec.path/'fronthaul_inventory.json').write_text(json.dumps({
        'per_ru_full_grid_wire_rate': asdict(fronthaul_rate(cfg)),
        'note': 'Shared layer U-plane duplicated to both RUs; BFW updates every du-period, uplink CSI and SRS not modeled.'}, indent=2))
    no = 10**(-args.snr/10)
    data_symbols = reference_data_symbols(sim.pdsch_configs[0])
    error_audit=[]
    for drop in range(args.drops):
        seed = args.seed_start+drop
        for speed in args.speeds:
            hs, hc = continuous_trace(sim, args.slots, speed, seed)
            if args.local_error:
                generator=torch.Generator(device=args.device).manual_seed(seed+700000)
                observations=[];signal_energy=error_energy=0.
                for h in hc:
                    error=torch.complex(torch.randn(h.shape,generator=generator,device=args.device),
                                        torch.randn(h.shape,generator=generator,device=args.device))*math.sqrt(args.local_error/2)
                    observations.append(h+error)
                    signal_energy+=float(h.abs().square().sum());error_energy+=float(error.abs().square().sum())
                error_audit.append(dict(drop=drop,speed=speed,realized_nmse=error_energy/signal_energy))
                hc=observations
            # The delayed DU snapshots and W9 are shared by all policies;
            # optional observation error is already included in hc.
            central = {t: quantize(sim.odu.precoder(hc[t]), args.antennas, args.width)
                       for t in range(0, args.slots, args.du_period)}
            weights = {'stale': [], 'track': []}
            versions = []
            for t in range(args.slots):
                ts = max(0, ((t-args.du_delay)//args.du_period)*args.du_period)
                versions.append(ts)
                weights['stale'].append(central[ts])
                weights['track'].append(tracking(hc[ts], hc[t], central[ts], args.antennas, args.ridge))
            needed_kinds={POLICIES[p][0] for p in args.policies}
            for period in [1,16]:
                if f'sopip{period}' in needed_kinds:
                    weights[f'sopip{period}']=sopip_pipeline(hc,central,args.antennas,period,
                        alpha=args.sopip_alpha,jcc=args.sopip_jcc,jap=args.sopip_jap,width=args.sopip_width)
            report_history={}
            for report_kind in needed_kinds:
                if any(POLICIES[p][0]==report_kind and POLICIES[p][2].startswith('g') and POLICIES[p][2]!='genie' for p in args.policies):
                    for width in [6,9]:
                        report_history[(report_kind,width)]=[effective_report(hc[t],weights[report_kind][t],args.antennas,width,no) for t in range(args.slots)]
            for policy in args.policies:
                kind, period, adaptation = POLICIES[policy]
                las = [LinkAdaptation(LinkAdaptationConfig(olla_step_db=2. if adaptation=='fast' else .5),
                        1, cfg.num_ue, int(cfg.num_layers_per_ue*sim.rg.num_data_symbols), reporter, model, allowed)
                       for _ in range(4 if adaptation=='age' else 1)]
                cqi = np.zeros(cfg.num_ue, dtype=int)
                feedback, acks = {}, {}
                measured_t = -1
                last_eff, measured_true = np.zeros(cfg.num_ue), np.zeros(cfg.num_ue)
                locked = weights['track'][0]
                for t in range(args.slots):
                    for when, ack, nack, mcs, group in acks.pop(t, []):
                        las[group].update(ack, nack, mcs)
                    if t in feedback:
                        newc, measured_eff, mt, mt_true = feedback.pop(t)
                        if adaptation=='lowpass' and measured_t >= 0:
                            cqi = np.rint(.5*cqi+.5*newc).astype(int)
                        else:
                            cqi = newc
                        last_eff, measured_t, measured_true = measured_eff, mt, mt_true
                    measurement = t % period == 0
                    if kind=='pilot_locked':
                        if measurement:
                            locked = weights['track'][t]
                        w = locked
                    else:
                        w = weights[kind][t]
                    power=w.unflatten(-2,(2,args.antennas)).abs().square().sum((-2,-1))
                    assert (power <= .5+3e-6).all()
                    if not kind.startswith('sopip'):
                        assert torch.allclose(power,torch.full_like(power,.5),atol=3e-6)
                    one = ChannelState(hs[t])
                    s = lmmse_sinr(one.stacked()[...,data_symbols,:], w, no, sim.odu.prg_of_sc, 2)
                    truec, trueeff = reporter(s[0])
                    se = torch.log2(1+s).mean((-1,-2)).sum(-1)[0].cpu().numpy()
                    if args.mode=='numerical':
                        for k in range(cfg.num_ue):
                            rec.add(dict(drop=drop, seed=seed, speed=speed, slot=t, ue=k, policy=policy,
                                         is_warmup=t<args.warmup, measurement=measurement, du_version=versions[t],
                                         true_cqi=int(truec[k]), true_eff_db=float(trueeff[k]), sum_stream_se=float(se[k])))
                        continue
                    if measurement:
                        sim.set_mcs([14]*cfg.num_ue)
                        sionna.phy.config.seed = seed+100000+t
                        tx = sim.odu(1, hc[t], csi_rs=csirs)
                        grid = apply_precoder(ucodec(tx.x_layers), w, sim.odu.prg_of_sc)
                        sionna.phy.config.seed = seed+200000+t
                        y = sim.channel(grid.unflatten(1,(2,args.antennas)), one, no)
                        measured_s, _ = receiver.sinr(sim.ue.frequency_grid(y))
                        rs_cqi, rs_eff = reporter(measured_s[0])
                        feedback[t+args.feedback_delay] = (rs_cqi, rs_eff, t, trueeff)
                        for k in range(cfg.num_ue):
                            rec.add(dict(drop=drop, seed=seed, speed=speed, slot=t, ue=k, policy=policy,
                                         is_warmup=t<args.warmup, measurement=True, du_version=versions[t],
                                         successful_bits=0, tb_bits=0, crc_ok=None, mcs=None,
                                         cqi=int(rs_cqi[k]), true_cqi=int(truec[k]), true_eff_db=float(trueeff[k]),
                                         pilot_age=0, olla_db=float(las[0].offset_db[k]), sum_stream_se=float(se[k])))
                        continue
                    age = t-measured_t if measured_t >= 0 else -1
                    group = min(3,max(0,(age-args.feedback_delay)//4)) if adaptation=='age' else 0
                    la = las[group]
                    decision_cqi=cqi.copy();report_source=-1
                    if adaptation=='genie':
                        decision_cqi=truec.copy()
                        mcs = la.select_mcs(truec)
                    elif adaptation in ['g6d1','g9d1','g9d4','g9p1','g6e4d1','g9e4d1','g9e8d1']:
                        width=6 if adaptation.startswith('g6') else 9
                        delay=4 if adaptation=='g9d4' else 1
                        source=max(0,t-delay)
                        message_period=8 if 'e8' in adaptation else 4 if 'e4' in adaptation else 1
                        source=(source//message_period)*message_period
                        report_source=source
                        report=report_history[(kind,width)][source]
                        if adaptation=='g9p1' and source>0 and versions[source]==versions[source-1]:
                            report=2*report-report_history[(kind,width)][source-1]
                        report_cqi,_=reporter(reported_sinr(report,no)[0])
                        # Startup CQI0 until first report is causally available.
                        decision_cqi=report_cqi if t>=delay else np.zeros(cfg.num_ue,dtype=int)
                        mcs=la.select_mcs(decision_cqi)
                    elif adaptation=='delta' and measured_t >= 0:
                        # Diagnostic: transport measured CQI to CURRENT actual quality using global truth.
                        corrected = la.sinr_estimate_db(cqi)+trueeff-measured_true
                        thr = la.mcs_threshold_db[la.allowed_mcs]
                        mcs = np.array([la.allowed_mcs[max(0,np.searchsorted(thr,x,side='right')-1)] for x in corrected])
                    else:
                        mcs = la.select_mcs(cqi)
                    sim.set_mcs(mcs.tolist())
                    sionna.phy.config.seed = seed+300000+t
                    tx = sim.odu(1, hc[t])
                    grid = apply_precoder(ucodec(tx.x_layers), w, sim.odu.prg_of_sc)
                    sionna.phy.config.seed = seed+400000+t
                    y = sim.channel(grid.unflatten(1,(2,args.antennas)), one, no)
                    rx = sim.ue(y, no)
                    ack = rx.crc_ok[0].cpu().numpy().astype(int)
                    acks.setdefault(t+args.ack_delay,[]).append((t,ack,1-ack,mcs,group))
                    for k in range(cfg.num_ue):
                        rec.add(dict(drop=drop, seed=seed, speed=speed, slot=t, ue=k, policy=policy,
                                     is_warmup=t<args.warmup, measurement=False, du_version=versions[t],
                                     successful_bits=int(ack[k])*sim.tb_sizes[k], tb_bits=sim.tb_sizes[k], crc_ok=bool(ack[k]),
                                     mcs=int(mcs[k]), cqi=int(cqi[k]), true_cqi=int(truec[k]),
                                     decision_cqi=int(decision_cqi[k]),report_source_slot=report_source,
                                     true_eff_db=float(trueeff[k]), pilot_age=age, olla_db=float(la.offset_db[k]),
                                     sum_stream_se=float(se[k])))
                print(f'drop={drop} speed={speed:g} policy={policy} mode={args.mode} rows={len(rec.rows)}',flush=True)
    (rec.path/'observation_error_audit.json').write_text(json.dumps(error_audit,indent=2))
    rec.finish()


if __name__=='__main__':
    main()
