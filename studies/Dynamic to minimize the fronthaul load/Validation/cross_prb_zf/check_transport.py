"""Paired Sionna NR PDSCH TB/CRC check using the true effective 4-user channels."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t
from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.nr import TBConfig, TBEncoder, TBDecoder, calculate_tb_size

USERS, PRBS, SC, DATA_SYMS = 4, 4, 12, 13
NOISE = 1e-7  # mW, -70 dBm on the same normalized channel as validate.py
TB_SEED = 20261126


def ci(x):
    x = np.asarray(x)
    mean = float(np.mean(x))
    half = float(t.ppf(.975, len(x)-1)*np.std(x, ddof=1)/np.sqrt(len(x))) if len(x)>1 else 0.
    return [mean,mean-half,mean+half]


def codec(mcs, device):
    cfg = TBConfig(channel_type="PDSCH", mcs_table=1, mcs_index=mcs)
    q, rate = int(cfg.num_bits_per_symbol.item()), float(cfg.target_coderate.item())
    coded = PRBS*SC*DATA_SYMS*q
    tb = int(calculate_tb_size(modulation_order=q, target_coderate=rate,
             num_prbs=PRBS, num_ofdm_symbols=14, num_dmrs_per_prb=12)[0].item())
    enc = TBEncoder(tb,coded,rate,q,num_layers=1,n_rnti=[101,102,103,104],n_id=[1]*USERS,
                    channel_type="PDSCH",device=device)
    dec = TBDecoder(enc,num_bp_iter=12,device=device)
    mapper = Mapper("qam",q,device=device)
    demap = Demapper("maxlog","qam",q,device=device)
    return tb,enc,dec,mapper,demap


@torch.inference_mode()
def simulate(effective, ids, mcs, repeats, chunk, device):
    tb,enc,dec,mapper,demap = codec(mcs,device)
    n = len(ids)
    crc_reject = np.zeros((n,2,repeats,USERS),bool)
    payload_error = np.zeros_like(crc_reject)
    for start in range(0,n,chunk):
        stop = min(start+chunk,n)
        size = stop-start
        # One data/noise stream per drop, shared by both policies and by the CSI-error cases.
        bits_by_drop, noise_by_drop = [], []
        for d in ids[start:stop]:
            rng = np.random.default_rng([TB_SEED,int(d),mcs])
            bits_by_drop.append(rng.integers(0,2,size=(repeats,USERS,tb),dtype=np.uint8))
            shape = (repeats,USERS,DATA_SYMS,PRBS,SC)
            noise_by_drop.append((rng.standard_normal(shape)+1j*rng.standard_normal(shape))*np.sqrt(NOISE/2))
        bits = torch.as_tensor(np.stack(bits_by_drop),dtype=torch.float32,device=device)
        z = torch.as_tensor(np.stack(noise_by_drop),dtype=torch.complex64,device=device)
        source = bits.reshape(size*repeats,USERS,tb)
        x = mapper(enc(source))
        if start==0:
            bh,crc = dec(demap(x[:1],torch.tensor(.001,device=device)))
            assert bool(crc.all()) and bool(torch.all(bh==source[:1])), "clean TB/LLR mapping"
        x = x.reshape(size,repeats,USERS,DATA_SYMS,PRBS,SC)
        e = torch.as_tensor(effective[start:stop],dtype=torch.complex64,device=device)
        y = torch.einsum("dabuk,drktbs->darutbs",e,x)+z[:,None]
        diag = e.diagonal(dim1=-2,dim2=-1).permute(0,1,3,2)
        other = e*(1-torch.eye(USERS,device=device))
        var = (other.abs().square().sum(dim=-1).permute(0,1,3,2)+NOISE)/diag.abs().square()
        equalized = y/diag[:,:,None,:,None,:,None]
        variance = var[:,:,None,:,None,:,None].expand_as(equalized.real)
        flat = (size*2*repeats,USERS,-1)
        decoded, good = dec(demap(equalized.reshape(flat),variance.reshape(flat)))
        good = good.view(size,2,repeats,USERS)
        incorrect = (decoded.view(size,2,repeats,USERS,tb)!=bits[:,None]).any(-1)
        crc_reject[start:stop] = (~good).cpu().numpy()
        payload_error[start:stop] = incorrect.cpu().numpy()
    return tb,crc_reject,payload_error


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--device",default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--repeats",type=int,default=16)
    ap.add_argument("--chunk",type=int,default=1)
    ap.add_argument("--limit",type=int)
    ap.add_argument("--mcs",nargs="+",type=int,default=[20,24])
    ap.add_argument("--cases",nargs="+",default=["taps_0_-70_lowpower","taps_0.01_-70_lowpower","taps_0.1_-70_lowpower"])
    args = ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    metrics, raw = {}, {}
    with np.load(args.input) as data:
        for mcs in args.mcs:
            for label in args.cases:
                ix = np.flatnonzero(data["case"]==label)[:args.limit]
                if not len(ix):
                    raise ValueError(f"No layouts found for {label}")
                ids,eff = data["drop"][ix],data["effective"][ix]
                tb,fail,payload_error = simulate(eff,ids,mcs,args.repeats,args.chunk,args.device)
                same = np.all(eff[:,0]==eff[:,1],axis=(1,2,3))
                assert np.array_equal(fail[same,0],fail[same,1]), "matched waveform control"
                assert np.array_equal(payload_error[same,0],payload_error[same,1]), "matched payload control"
                crc_loss = fail.mean(axis=(2,3))
                undetected = ~fail & payload_error
                delivered = ~(fail | payload_error)
                throughput = delivered.mean(axis=(2,3))*USERS*tb/.0005/1e6
                name = f"{label}_mcs{mcs}"
                row = dict(mcs=mcs,tb_bits=tb,layouts=len(ids),per_policy_tbs=int(fail[:,0].size),
                    repeat_per_ue=args.repeats,bler=crc_loss.mean(0).tolist(),
                    bler_diff_ci=ci(crc_loss[:,1]-crc_loss[:,0]),
                    payload_bler=payload_error.mean(axis=(0,2,3)).tolist(),
                    goodput_mbps=throughput.mean(0).tolist(),goodput_diff_ci_mbps=ci(throughput[:,1]-throughput[:,0]),
                    candidate_only_errors=int(np.sum(~fail[:,0] & fail[:,1])),
                    base_only_errors=int(np.sum(fail[:,0] & ~fail[:,1])),
                    identical_channel_layouts=int(np.sum(same)),undetected_block_errors=int(undetected.sum()),
                    undetected_by_policy=undetected.sum(axis=(0,2,3)).tolist())
                metrics[name] = row
                raw[name] = fail
                raw[name+"_payload_error"] = payload_error
                raw[name+"_ids"] = ids
                print(name, "BLER", row["bler"], "paired diff and CI",row["bler_diff_ci"],flush=True)
    np.savez_compressed(args.out/"crc_pairs.npz",**raw)
    (args.out/"crc.json").write_text(json.dumps(metrics,indent=2)+"\n")


if __name__=="__main__":
    main()
