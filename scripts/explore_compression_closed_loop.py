#!/usr/bin/env python3
"""Static CDL repeated-slot check: does fixed-MCS compression gain survive LA?

Actual CSI-RS or genie effective-channel CQI -> MCS -> DMRS/LDPC CRC -> OLLA.
Same independent drops and air noise across cases. Channels are held fixed
within a trace, not mislabeled as a mobility process. No core modules changed.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import sionna.phy
from oran import ORANConfig, ORANDownlink, CompressionConfig
from oran.bler_model import AWGNLinkModel
from oran.channel import ChannelState
from oran.csi import CQIReporter, lmmse_sinr, reference_num_re, reference_data_symbols
from oran.csi_rs import CSIRS, CSIRSReceiver
from oran.fronthaul import IQCodec
from oran.link_adaptation import LinkAdaptation, LinkAdaptationConfig
from oran.precoding import apply_precoder
from explore_csirs_precoding import Recorder, weight_cases


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    p.add_argument('--drops',type=int,default=16)
    p.add_argument('--seed-start',type=int,default=20262000)
    p.add_argument('--batches',type=int,default=32)
    p.add_argument('--warmup-batches',type=int,default=16)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--measurement-every',type=int,default=2,help='one measurement slot per this many data batches')
    p.add_argument('--snrs',nargs='+',type=float,default=[20])
    p.add_argument('--modes',nargs='+',choices=['rs_olla','rs_no_olla','genie_olla','fixed24'],default=['rs_olla','rs_no_olla','genie_olla'])
    p.add_argument('--measurement-policy',choices=['matched','du9'],default='matched',
                   help='du9: CSI-RS uses DU BFP9 W while data uses the selected reconstructed W')
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    if args.warmup_batches>=args.batches:raise ValueError('Warmup must leave evaluation slots')
    torch.set_num_threads(2)
    sionna.phy.config.device=args.device
    cfg=ORANConfig(category='B',num_prb=12,num_oru_ant=8,mcs_table=1,mcs_index=14,
                   uplane=CompressionConfig('bfp',9),bfw=CompressionConfig('ideal',16),device=args.device)
    rec=Recorder(args.out)
    (rec.path/'experiment_source.py').write_bytes(Path(__file__).read_bytes())
    dependency=Path(__file__).with_name('explore_csirs_precoding.py')
    (rec.path/'dependency_explore_csirs_precoding.py').write_bytes(dependency.read_bytes())
    sim=ORANDownlink(cfg)
    allowed=sim.supported_mcs()
    csirs=CSIRS(cfg,sim.pdsch_configs[0],device=args.device)
    receiver=CSIRSReceiver(csirs,cfg.prg_num_prb,sim.ue.estimator)
    model=AWGNLinkModel(device=args.device)
    reporter=CQIReporter(1,reference_num_re(sim.pdsch_configs[0],cfg.num_prb,2),model)
    num_re_alloc=cfg.num_layers_per_ue*sim.rg.num_data_symbols
    # Resource grid has total layer-independent REs; verify against actual encoder.
    ucodec=IQCodec(CompressionConfig('bfp',9),1.,12)
    meta=dict(args=vars(args),config=asdict(cfg),gpu=torch.cuda.get_device_name(),torch=torch.__version__,
              cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),allowed_mcs=allowed,
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              caveats=['Static independent CDL drops, ideal DU CSI, fixed UE scheduling and layers.',
                       'Actual CSI-RS feedback, DMRS estimates actual compressed W, actual LDPC CRC.',
                       'CQI applied to next data batch; ACK/NACK applied only after preceding data batch.',
                       'Within each batch all transmissions share MCS and pre-batch OLLA state.',
                       'Same air noise and channel per case; bits have same seed but lengths differ across MCS.',
                       'Measurement slots carry punched PDSCH but its CRC not decoded; entire slot excluded from goodput.',
                       'Genie CQI is EESM/AWGN abstraction, not optimal MCS or perfect BLER prediction.',
                       'Known column norm restored to 1/sqrt(4) for all weight cases, no free norm side message.',
                       'matched: pilot/data share W; du9: pilot uses BFP9 W while data uses selected W.',
                       'du9 is a controlled representation mismatch, not a complete RU-tracking or mobility model.',
                       'Fixed pilot sequence repeats, independent AWGN per measurement; no HARQ.'])
    (rec.path/'metadata.json').write_text(json.dumps(meta,indent=2))
    selected={('bfp_norm',3),('phase_channel_norm',3),('bfp_norm',9)}
    for drop in range(args.drops):
        seed=args.seed_start+drop
        sionna.phy.config.seed=seed
        one=sim.channel.sample(1)
        hc=sim.channel.csi(one,sim.csi_symbol)
        w=sim.odu.precoder(hc)
        weights=[(name,width,q) for name,width,q,aligned in weight_cases(w,hc,cfg,[3,9],32) if (name,width) in selected]
        assert len(weights)==3
        measurement_reference=next(q for name,width,q in weights if name=='bfp_norm' and width==9)
        batchstate=ChannelState(one.h_freq.expand(args.batch_size,*one.h_freq.shape[1:]),None)
        batchhc=hc.expand(args.batch_size,*hc.shape[1:])
        for snr in args.snrs:
            no=10**(-snr/10)
            for name,width,q in weights:
                assert torch.allclose(q.abs().square().sum(-2),torch.full_like(q.real.sum(-2),.25),atol=2e-6)
                for mode in args.modes:
                    la=LinkAdaptation(LinkAdaptationConfig(olla=(mode not in ['rs_no_olla','fixed24'])),1,cfg.num_ue,
                                      int(num_re_alloc),reporter,model,allowed)
                    qbatch=q.expand(args.batch_size,*q.shape[1:])
                    genie_s=lmmse_sinr(one.stacked()[...,reference_data_symbols(sim.pdsch_configs[0]),:],q,no,
                                      sim.odu.prg_of_sc,cfg.num_layers_per_ue)
                    genie_cqi,genie_eff=reporter(genie_s[0])
                    for step in range(args.batches):
                        measurement=step%args.measurement_every==0
                        if measurement:
                            # Identical filler MCS, payload seed and AWGN seed for every case.
                            sim.set_mcs([14]*cfg.num_ue)
                            sionna.phy.config.seed=seed+100000+step
                            txm=sim.odu(1,hc,csi_rs=csirs)
                            measured_w=measurement_reference if args.measurement_policy=='du9' else q
                            gridm=apply_precoder(ucodec(txm.x_layers),measured_w,sim.odu.prg_of_sc)
                            sionna.phy.config.seed=seed+200000+step
                            ym=sim.channel(gridm[:,None],one,no)
                            measured_s,_=receiver.sinr(sim.ue.frequency_grid(ym))
                            rs_cqi,rs_eff=reporter(measured_s[0])
                            cqi,eff=(genie_cqi,genie_eff) if mode=='genie_olla' else (rs_cqi,rs_eff)
                        mcs=np.full(cfg.num_ue,24,dtype=int) if mode=='fixed24' else la.select_mcs(cqi)
                        offset_before=la.offset_db.copy()
                        sim.set_mcs(mcs.tolist())
                        sionna.phy.config.seed=seed+300000+step
                        tx=sim.odu(args.batch_size,batchhc)
                        grid=apply_precoder(ucodec(tx.x_layers),qbatch,sim.odu.prg_of_sc)
                        sionna.phy.config.seed=seed+400000+step
                        y=sim.channel(grid[:,None],batchstate,no)
                        rx=sim.ue(y,no)
                        ack=rx.crc_ok.sum(0).cpu().numpy()
                        # MCS/OLLA update happens after all transmissions in batch.
                        la.update(ack,args.batch_size-ack,mcs)
                        for b in range(args.batch_size):
                            for k in range(cfg.num_ue):
                                ok=bool(rx.crc_ok[b,k])
                                rec.add(dict(drop=drop,seed=seed,ue=k,batch=step,slot_in_batch=b,snr_db=snr,
                                             method=name,width=width,mode=mode,is_warmup=step<args.warmup_batches,
                                             measurement_policy=args.measurement_policy,
                                             cqi=int(cqi[k]),genie_cqi=int(genie_cqi[k]),rs_cqi=int(rs_cqi[k]),
                                             effective_sinr_db=float(eff[k]),mcs=int(mcs[k]),
                                             olla_before_db=float(offset_before[k]),olla_after_db=float(la.offset_db[k]),
                                             crc_ok=ok,tb_bits=sim.tb_sizes[k],successful_bits=int(ok)*sim.tb_sizes[k],
                                             measurement_slot_before_batch=measurement))
                    print(f'drop={drop} snr={snr:g} method={name}/{width} mode={mode} rows={len(rec.rows)}',flush=True)
    rec.finish()


if __name__=='__main__':main()
