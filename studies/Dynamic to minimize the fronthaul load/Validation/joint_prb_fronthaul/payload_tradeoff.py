"""Minimize slot IQ payload while meeting the paired fixed-greedy SE benchmark."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from audit_joint import BFP9_BITS, PRBS, interval, mapping, read_pair

BASE_UNIT_BITS = 112
WIDTHS = np.array([5,7,9,5,7,9])
FIXED = 2


def minimize_payload(se, costs, cap, target, incumbent, seconds):
    n, num_ru = costs.shape
    per_candidate = costs.sum(1)
    total = np.tile(per_candidate, PRBS)
    budget = np.tile(costs.T, PRBS)
    assignment = np.repeat(np.eye(PRBS),n,axis=1)
    rows = np.vstack([assignment,budget,se.reshape(1,-1)/target])
    low = np.r_[np.ones(PRBS),np.full(num_ru,-np.inf),1.-1e-9]
    high = np.r_[np.ones(PRBS),np.full(num_ru,cap),np.inf]
    res = milp(total,integrality=np.ones(len(total)), bounds=Bounds(0,1),
               constraints=LinearConstraint(rows,low,high),
               options=dict(time_limit=seconds,mip_rel_gap=1e-6))
    best = np.asarray(incumbent)
    best_cost = float(per_candidate[best].sum())
    if res.x is not None:
        proposed = res.x.reshape(PRBS,n).argmax(1)
        proposed_cost = per_candidate[proposed].sum()
        if (costs[proposed].sum(0).max() <= cap and
                se[np.arange(PRBS),proposed].sum() >= target-2e-7 and proposed_cost <= best_cost):
            best,best_cost = proposed,float(proposed_cost)
    achieved = float(se[np.arange(PRBS),best].sum())
    assert achieved >= target-2e-7 and costs[best].sum(0).max() <= cap
    dual = getattr(res,"mip_dual_bound",None)
    gap = (max(0.,best_cost-dual)/best_cost if dual is not None and np.isfinite(dual) else None)
    return dict(indices=best.tolist(), bits=int(best_cost*BASE_UNIT_BITS),
                mean_prb_se=achieved/PRBS, status=int(res.status), gap=gap)


def simulate(task):
    d,k,se,masks,greedy,seconds = task
    cap = k*BFP9_BITS//BASE_UNIT_BITS
    payload_units = (14*(24*WIDTHS+8))//BASE_UNIT_BITS
    target = float(se[np.arange(PRBS),greedy,FIXED].sum())
    fixed_cost = masks*payload_units[FIXED]
    adaptive_cost = (masks[:,None,:]*payload_units[None,:,None]).reshape(-1,masks.shape[1])
    fixed = minimize_payload(se[:,:,FIXED],fixed_cost,cap,target,greedy,seconds)
    bfp_cost = (masks[:,None,:]*payload_units[None,:3,None]).reshape(-1,masks.shape[1])
    bfp = minimize_payload(se[:,:,:3].reshape(PRBS,-1),bfp_cost,cap,target,
                           np.array(fixed["indices"])*3+FIXED,seconds)
    mask_index, codec_index = np.divmod(bfp["indices"],3)
    adaptive = minimize_payload(se.reshape(PRBS,-1),adaptive_cost,cap,target,
                                mask_index*len(WIDTHS)+codec_index,seconds)
    return dict(drop=d,k=k,target_mean_prb_se=target/PRBS,
                greedy_bits=int(fixed_cost[greedy].sum()*BASE_UNIT_BITS),fixed=fixed,bfp_adaptive=bfp,
                adaptive=adaptive)


def summary(rows):
    sweeps = {}
    for k in (2,3):
        pairs = [r for r in rows if r["k"]==k]
        bits = np.array([[r["greedy_bits"],r["fixed"]["bits"],r["bfp_adaptive"]["bits"],
                          r["adaptive"]["bits"]] for r in pairs])
        sweeps[str(k)] = dict(layouts=len(pairs), mean_bits=bits.mean(0).tolist(),
                             paired_bit_savings_ci=interval(bits[:,1]-bits[:,3]),
                             paired_relative_savings_pct_ci=interval(100*(1-bits[:,3]/bits[:,1])),
                             codec_switch_bit_savings_ci=interval(bits[:,2]-bits[:,3]),
                             certified={a:sum(r[a]["gap"] is not None and r[a]["gap"]<=1e-6 for r in pairs)
                                        for a in ("fixed","bfp_adaptive","adaptive")})
    return dict(metric="Meet the corresponding fixed-greedy mean per-PRB SE, then minimize slot IQ payload",
                net_bit_columns=["greedy_fixed","minimum_fixed","minimum_bfp_adaptive","minimum_adaptive"],
                sweeps=sweeps,rows=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates",type=Path,required=True)
    p.add_argument("--paired",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--workers",type=int,default=6)
    p.add_argument("--time-limit",type=float,default=4.)
    args = p.parse_args()
    pool, pair = read_pair(args.candidates),read_pair(args.paired)
    index = mapping(pair,("drop","k","algorithm","prb"))
    tasks = []
    for d,n in enumerate(pool["num_masks"]):
        for k in (2,3):
            r = [index[d,k,"greedy_fixed",b] for b in range(PRBS)]
            picks = pair["mask_index"][r]
            np.testing.assert_allclose(pool["se"][d,np.arange(PRBS),picks,FIXED],pair["se_per_prb"][r],
                                       rtol=1e-10,atol=1e-10)
            tasks.append((d,k,pool["se"][d,:,:n],pool["masks"][d,:n],picks,args.time_limit))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(simulate,tasks))
    results = summary(rows)
    args.output.write_text(json.dumps(results,indent=2)+"\n")
    for k, stats in results["sweeps"].items():
        print(f"k={k}",stats,flush=True)


if __name__=="__main__":
    main()
