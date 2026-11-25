"""Independently transmit real PDSCH transport blocks through each selected IQ codec."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parent / "cross_prb_zf"))
from audit_joint import BFP9_BITS, NOISE, POWER_LIMIT, activation, interval, mapping, phase2_selections, read_pair
from check_transport import codec
from validate import channel
from oran.config import CompressionConfig, ORANConfig
from oran.fronthaul import IQCodec
from oran.pdsch import make_pdsch_configs, make_resource_grid

USERS, RUS, PRBS, TONES, SYMBOLS = 4, 16, 4, 12, 14
ALGORITHMS = ("greedy_fixed", "joint_fixed", "greedy_adaptive", "joint_adaptive")
TB_SEED = 20261128
SLOT_S = .0005


def radio_grid(mcs):
    cfg = ORANConfig(num_prb=PRBS, num_ue=USERS, num_ue_ant=1,
                     num_layers_per_ue=1, num_oru=RUS, num_oru_ant=1,
                     dmrs_additional_position=0, mcs_index=mcs, rnti=[101,102,103,104])
    template = make_resource_grid(cfg,make_pdsch_configs(cfg),device="cpu").pilot_pattern
    mask = template.mask[:,0].numpy()
    assert np.array_equal(mask,np.broadcast_to(mask[0],mask.shape))
    pilot_symbols = mask[0].any(1)
    assert pilot_symbols.sum()==1 and np.all(mask[0,pilot_symbols])
    pilots = template.pilots[:,0].numpy().reshape(USERS,-1,PRBS,TONES)
    # The configured CDM groups provide one independent sequence per user's layer in each PRB.
    design = pilots.transpose(2,0,1,3).reshape(PRBS,USERS,-1)
    assert np.all(np.linalg.matrix_rank(design)==USERS)
    return pilot_symbols, pilots, design


def make_precoder(h, masks):
    weights = np.zeros((PRBS,RUS,USERS),complex)
    for b, mask in enumerate(masks):
        on = activation(mask,RUS)
        normal = np.linalg.pinv(h[b][:,on])
        weights[b,on] = normal*np.sqrt(POWER_LIMIT/np.sum(abs(normal)**2,axis=1).max())
        eff = h[b] @ weights[b]
        assert np.max(abs(eff-np.eye(USERS)*eff[0,0])) <= 1e-8*abs(eff[0,0])
    return weights


def apply_iq(h, grid, masks, widths, codecs, compressors, total_cap=None):
    weights = make_precoder(h,masks)
    wave = np.einsum("bau,rutbf->ratbf",weights,grid,optimize=True)
    for b,(width, name) in enumerate(zip(widths,codecs)):
        key=(int(width),str(name))
        if key not in compressors:
            compressors[key] = IQCodec(CompressionConfig(key[1],key[0]),POWER_LIMIT,TONES)
        batch = torch.as_tensor(np.ascontiguousarray(wave[:,:,:,b,:]))
        wave[:,:,:,b,:] = compressors[key](batch).numpy()
    ant_power = (abs(wave)**2).mean(axis=(2,4))
    # The same PRB scalar ensures that the actual compressed waveform, including pilots,
    # respects the per-antenna RF budget for every waveform.
    guard = np.minimum(1.0,POWER_LIMIT/ant_power.max(1))
    if total_cap is not None:
        guard = np.minimum(guard,total_cap/ant_power.sum(1))
    wave *= np.sqrt(guard)[:,None,None,:,None]
    actual = (abs(wave)**2).mean(axis=(2,4))
    assert actual.max() <= POWER_LIMIT + 2e-9
    if total_cap is not None:
        assert np.all(actual.sum(1)<=total_cap+2e-9)
    assert np.all(wave.transpose(0,2,4,3,1)[...,~activation(masks,RUS)]==0)
    return np.einsum("bua,ratbf->rutbf",h,wave,optimize=True),actual.sum(1),float(actual.max())


def equalize(received,pilot_symbols,design):
    observed = received[:,:,pilot_symbols].transpose(0,1,3,2,4).reshape(len(received),USERS,PRBS,-1)
    effective = np.einsum("rubp,bpv->rbuv",observed,np.linalg.pinv(design))
    expected = np.einsum("rbuv,bvp->rubp",effective,design)
    pilot_samples = observed.shape[-1]
    residual = (abs(observed-expected)**2).sum(-1)/(pilot_samples-USERS)
    desired = np.diagonal(effective,axis1=-2,axis2=-1).transpose(0,2,1)
    leakage = (abs(effective*(1-np.eye(USERS)))**2).sum(-1).transpose(0,2,1)
    var = (np.maximum(NOISE,residual) + leakage)/np.maximum(abs(desired)**2,1e-24)
    y = received[:,:,~pilot_symbols]/desired[:,:,None,:,None]
    return y,np.broadcast_to(var[:,:,None,:,None],y.shape)


def encode_pair(d,mcs,tb,encoder,mapper,repeats,device,pilot_symbols,pilot_grid):
    rng = np.random.default_rng([TB_SEED,d,mcs])
    bits = rng.integers(0,2,(repeats,USERS,tb),dtype=np.uint8)
    source = torch.as_tensor(bits,dtype=torch.float32,device=device)
    data = mapper(encoder(source)).cpu().numpy()
    grid = np.empty((repeats,USERS,SYMBOLS,PRBS,TONES),complex)
    grid[:,:,~pilot_symbols] = data.reshape(repeats,USERS,SYMBOLS-1,PRBS,TONES)
    grid[:,:,pilot_symbols] = pilot_grid
    noise = (rng.standard_normal(grid.shape)+1j*rng.standard_normal(grid.shape))*np.sqrt(NOISE/2)
    return bits,grid,noise


def run(pair,k,mcs,repeats,device,limit,match_greedy_power=False):
    indexes = mapping(pair,("drop","k","algorithm"))
    ids = np.unique(pair["drop"][pair["k"]==k])[:limit]
    assert len(ids)>0
    tb,enc,dec,mapper,demap = codec(mcs,device)
    pilot_mask,pilot_grid,design = radio_grid(mcs)
    compressors = {}
    shape = (len(ids),len(ALGORITHMS),repeats,USERS)
    crc, payload = np.empty(shape,bool),np.empty(shape,bool)
    powers = np.empty(shape[:-1]+(PRBS,))
    max_power = 0.
    for d_i,d in enumerate(ids):
        h,_ = channel(int(d),"taps",0.)
        bits,grid,noise = encode_pair(int(d),mcs,tb,enc,mapper,repeats,device,pilot_mask,pilot_grid)
        if d_i==0:
            clean = torch.as_tensor(grid[:,:,~pilot_mask].reshape(repeats,USERS,-1),dtype=torch.complex64,device=device)
            bh,ok = dec(demap(clean,torch.tensor(.001,device=device)))
            assert bool(ok.all()) and np.array_equal(bh.cpu().numpy(),bits)
        cache={}
        for a,alg in enumerate(ALGORITHMS):
            row = indexes[int(d),k,alg]
            masks,widths,names = [pair[f][row] for f in ("mask","width","codec")]
            load = (activation(masks,RUS)*((24*widths.astype(np.int64)+8)*SYMBOLS)[:,None]).sum(0)
            assert np.array_equal(load,pair["link_bits"][row]) and load.max() <= k*BFP9_BITS
            key=(tuple(masks),tuple(widths),tuple(names))
            if key not in cache:
                cap = powers[d_i,0] if match_greedy_power and a>0 else None
                received,power,peak = apply_iq(h,grid,masks,widths,names,compressors,cap)
                y,var = equalize(received+noise,pilot_mask,design)
                y_t = torch.as_tensor(y.reshape(repeats,USERS,-1),dtype=torch.complex64,device=device)
                var_t = torch.as_tensor(var.reshape(repeats,USERS,-1),dtype=torch.float32,device=device)
                bh,ok = dec(demap(y_t,var_t))
                cache[key]=(~ok.cpu().numpy(),(bh.cpu().numpy()!=bits).any(-1),power,peak)
            crc[d_i,a],payload[d_i,a],powers[d_i,a],peak = cache[key]
            max_power=max(max_power,peak)
    lost = crc.mean(axis=(2,3))
    correct = ~(crc | payload)
    goodput = correct.mean(axis=(2,3))*USERS*tb/(SLOT_S*1e6)
    summary = dict(mcs=mcs,tb_bits=tb,layouts=len(ids),per_policy_tbs=int(crc[:,0].size),
                   max_antenna_power_mw=max_power, receiver="LS from four orthogonal NR DMRS ports per PRB",
                   per_prb_total_power_capped_at_greedy_fixed=match_greedy_power,
                   algorithms={a:dict(bler=float(lost[:,i].mean()),goodput_mbps=float(goodput[:,i].mean()),
                                      actual_prb_power=float(powers[:,i].mean()),
                                      undetected_blocks=int(((~crc) & payload)[:,i].sum()))
                               for i,a in enumerate(ALGORITHMS)},
                   paired=dict(joint_vs_greedy_fixed_ci_mbps=interval(goodput[:,1]-goodput[:,0]),
                               joint_vs_greedy_adaptive_ci_mbps=interval(goodput[:,3]-goodput[:,2]),
                               adaptive_vs_fixed_joint_ci_mbps=interval(goodput[:,3]-goodput[:,1]),
                               joint_vs_greedy_fixed_bler_ci=interval(lost[:,1]-lost[:,0]),
                               joint_vs_greedy_adaptive_bler_ci=interval(lost[:,3]-lost[:,2])))
    return summary,dict(crc=crc,payload=payload,power_mw=powers,drop=ids)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--device",default="cuda:1" if torch.cuda.device_count()>1 else
                        ("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--repeats",type=int,default=16)
    parser.add_argument("--limit",type=int)
    parser.add_argument("--mcs",type=int,nargs="+",default=[16,20])
    parser.add_argument("--budget",type=int,nargs="+",default=[2,3])
    parser.add_argument("--match-greedy-power",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    pair = phase2_selections(read_pair(args.input))
    summary,raw = {},{}
    for mcs in args.mcs:
        for k in args.budget:
            label=f"mcs{mcs}_k{k}"
            summary[label],arrays = run(pair,k,mcs,args.repeats,args.device,args.limit,args.match_greedy_power)
            for field,values in arrays.items():
                raw[label+"_"+field]=values
            print(label,summary[label]["algorithms"],flush=True)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"transport.json").write_text(json.dumps(summary,indent=2)+"\n")
    np.savez_compressed(args.out/"transport.npz",**raw)


if __name__=="__main__":
    main()
