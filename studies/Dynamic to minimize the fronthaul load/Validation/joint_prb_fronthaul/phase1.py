"""Compare serial and joint PRB scheduling at fixed BFP9 fronthaul cost."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import combinations
from math import comb
from pathlib import Path
from time import perf_counter

import numpy as np

from audit_joint import BFP9_BITS, POWER_LIMIT, PRBS, activation, channel, interval, sum_se_from_g
from scheduler import greedy_select, joint_select
import same_total as e8


def short_list(masks, live, gain, q_sum, top):
    """Retain all minimum-rank sets and diverse alternatives at each larger size."""
    num_ru, users = live.shape[1], 4
    sizes = live.sum(1)
    chosen = set(np.flatnonzero(sizes == users))
    for size in range(users + 1, num_ru + 1):
        rows = np.flatnonzero(sizes == size)
        if not len(rows):
            continue
        order = rows[np.argsort(-gain[rows], kind="stable")]
        efficient = rows[np.argsort(q_sum[rows], kind="stable")]
        chosen.update(order[:top])
        chosen.update(efficient[:top])
        for r in range(num_ru):
            for present in (False, True):
                subset = order[live[order,r] == present]
                if len(subset):
                    chosen.add(int(subset[0]))
    return np.array(sorted(chosen),dtype=int)


def candidate_table(h, top):
    users, num_ru = h.shape
    if num_ru == 16:
        masks, live, *_ = e8.candidates(1,users)
        g2, q, _ = e8.score(h,1)
        gain = g2[0]
        valid = np.isfinite(gain)
        masks, live, gain, q = masks[valid],live[valid],gain[valid],q[valid]
        q_sum = q.sum(1)
        chosen = short_list(masks, live, gain, q_sum, top)
        return masks[chosen], live[chosen].astype(np.int16), gain[chosen], q_sum[chosen]
    masks = np.arange(1,1 << num_ru)
    live = activation(masks,num_ru)
    sizes = live.sum(1)
    q = np.zeros((len(masks),num_ru))
    valid = np.zeros(len(masks),bool)
    for size in range(users,num_ru+1):
        rows = np.flatnonzero(sizes == size)
        rs, antennas = np.where(live[rows])
        hs = h[:,antennas.reshape(len(rows),size)].transpose(1,0,2)
        directions = np.linalg.pinv(hs)
        residual = np.max(abs(hs @ directions-np.eye(users)),axis=(1,2))
        valid[rows] = residual < 1e-7
        q[rows[rs],antennas] = np.sum(abs(directions)**2,axis=2).ravel()
    q = q[valid]
    return masks[valid], live[valid].astype(np.int16), POWER_LIMIT/q.max(1), q.sum(1)


@lru_cache(maxsize=2)
def disjoint_pairs(num_ru, users):
    masks = (1 << np.array(list(combinations(range(num_ru),users)))).sum(1)
    a,b = np.where((masks[:,None] & masks[None,:]) == 0)
    return masks[a], masks[b], masks[a] | masks[b]


def tight_select(masks, values, num_ru, users, minimum_masks=None):
    """Exact at cap one, where each RU is used once and every support has size U."""
    start = perf_counter()
    if minimum_masks is None:
        minimum_masks = disjoint_pairs(num_ru,users)
    a,b,unions = minimum_masks
    scores = np.full((PRBS,1<<num_ru),-np.inf)
    for prb,(keys,gain) in enumerate(zip(masks,values)):
        scores[prb,keys] = gain
    pair_scores = [scores[i,a]+scores[i+1,b] for i in (0,2)]
    best = np.full((2,1<<num_ru),-np.inf)
    for i in range(2):
        np.maximum.at(best[i],unions,pair_scores[i])
    all_radios = (1 << num_ru)-1
    joined = best[0] + best[1,all_radios ^ np.arange(1<<num_ru)]
    first = int(joined.argmax())
    assert np.isfinite(joined[first]), "The capacity-one layout is infeasible"
    winners = [np.argmax(np.where(unions == group,pair_scores[i],-np.inf))
               for i,group in enumerate((first,all_radios ^ first))]
    chosen = [a[winners[0]],b[winners[0]],a[winners[1]],b[winners[1]]]
    indices = [int(np.searchsorted(keys,key)) for keys,key in zip(masks,chosen)]
    return dict(indices=indices,value=float(joined[first]),gap=0.,status="exact_partition",
                runtime_s=perf_counter()-start)


def collect(d, case, k, method, outcome, masks, costs, values, q_sum):
    pick = outcome["indices"]
    selected = [int(a[i]) for a,i in zip(masks,pick)]
    used = np.stack([a[i] for a,i in zip(costs,pick)]).sum(0)
    g = np.array([a[i] for a,i in zip(values,pick)])
    power = g * np.array([a[i] for a,i in zip(q_sum,pick)])
    if len(used)<16:
        used = np.pad(used,(0,16-len(used)))
    gap = outcome["gap"]
    return dict(drop=d,case=case,k=k,algorithm=method,mask=selected,g=g,power_mw=power,
                se=sum_se_from_g(g,2 if case.startswith("small") else 4),link_units=used,
                mip_gap=np.nan if gap is None else gap,solver_status=str(outcome["status"]),
                runtime_s=outcome["runtime_s"],candidates=[len(a) for a in masks],
                order=outcome.get("order",[-1]*PRBS))


def simulate(job):
    case,d,top,time_limit = job
    small = case.startswith("small")
    num_ru,users = (8,2) if small else (16,4)
    model = case.rsplit("_",1)[1]
    h = channel(d,model,0.)[0][:,:users,:num_ru]
    tables = [candidate_table(h[0],top)]*PRBS if model == "flat" else [candidate_table(hb,top) for hb in h]
    masks,costs,values,q_sum = map(list,zip(*tables))
    complete_minimum = all(np.sum(c.sum(1)==users) == comb(num_ru,users) for c in costs)

    def can_finish(remaining,next_blocks):
        left = len(next_blocks)
        return (np.minimum(remaining,left).sum(axis=-1) >= users*left)

    tight = tight_select(masks,values,num_ru,users)
    independent = dict(indices=np.array([g.argmax() for g in values]),gap=None,status="independent",runtime_s=0.)
    previous = tight["indices"]
    rows = []
    for k in range(1,PRBS+1):
        budget = np.full(num_ru,k)
        greedy = greedy_select(values,costs,budget,fallback=tight["indices"],
                               can_finish=can_finish if complete_minimum else None)
        g_row = collect(d,case,k,"greedy",greedy,masks,costs,values,q_sum)
        capped = [np.minimum(g, p/q) for g,p,q in zip(values,g_row["power_mw"],q_sum)]
        if k==1:
            joint = tight
            matched = tight_select(masks,capped,num_ru,users)
        elif k==PRBS:
            joint = dict(independent,gap=0.,status="separable")
            matched = dict(joint,indices=[g.argmax() for g in capped])
        else:
            old_value = sum(v[i] for v,i in zip(values,previous))
            fallback = previous if old_value > greedy["value"] else greedy["indices"]
            joint = joint_select(values,costs,budget,fallback=fallback,time_limit=time_limit,mip_rel_gap=1e-6)
            matched = joint_select(capped,costs,budget,fallback=greedy["indices"],time_limit=time_limit,mip_rel_gap=1e-6)
        previous = joint["indices"]
        rows.extend([collect(d,case,k,"independent",independent,masks,costs,values,q_sum),g_row,
                     collect(d,case,k,"joint",joint,masks,costs,values,q_sum),
                     collect(d,case,k,"matched",matched,masks,costs,capped,q_sum)])
    return rows


def summarize(data):
    result = {}
    for case in np.unique(data["case"]):
        result[case] = {}
        for k in np.unique(data["k"]):
            rows = (data["case"]==case)&(data["k"]==k)
            arms = {}
            for name in ("independent","greedy","joint","matched"):
                ix = np.flatnonzero(rows & (data["algorithm"]==name))
                arms[name] = ix[np.argsort(data["drop"][ix])]
            old = data["se"][arms["greedy"]]
            result[case][str(k)] = dict(layouts=len(old),se={a:float(data["se"][ix].mean()) for a,ix in arms.items()},
                joint_gain_ci=interval(data["se"][arms["joint"]]-old),
                matched_gain_ci=interval(data["se"][arms["matched"]]-old),
                candidate_count_range=[int(data["candidates"][rows].min()),int(data["candidates"][rows].max())])
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers",type=int,default=8)
    ap.add_argument("--layouts",type=int,default=96)
    ap.add_argument("--small-layouts",type=int,default=12)
    ap.add_argument("--top",type=int,default=12)
    ap.add_argument("--time-limit",type=float,default=3.)
    ap.add_argument("--output",type=Path,default=Path(__file__).parent/"results/phase1")
    args = ap.parse_args()
    jobs = [(case,d,args.top,args.time_limit)
            for case,n in [("small_taps",args.small_layouts),("small_flat",args.small_layouts),
                           ("main_taps",args.layouts),("main_flat",args.layouts)] for d in range(n)]
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=[]
        for i,records in enumerate(pool.map(simulate,jobs),1):
            rows.extend(records)
            if i in (1,len(jobs)) or i%12==0:
                print(f"{i}/{len(jobs)} layouts elapsed {perf_counter()-start:.1f} s",flush=True)
    data = {key:np.array([r[key] for r in rows]) for key in rows[0]}
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/"paired.npz",**data)
    sources = [Path(__file__),Path(__file__).with_name("scheduler.py")]
    run = dict(arguments=vars(args), seed=20261125, bfp9_bits_per_ru_prb_slot=BFP9_BITS,
               elapsed_s=perf_counter()-start, sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
               sweeps=summarize(data))
    (args.output/"summary.json").write_text(json.dumps(run,indent=2,default=str)+"\n")


if __name__=="__main__":
    main()
