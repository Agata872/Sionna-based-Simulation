"""Independent paired test of cross-PRB user gains on the archived E8 and new channels."""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import combinations
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from scipy.stats import t

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import same_total as e8

ARCHIVE = HERE.parent / "results/same_total_20261124/samples.npz"
OLD_SEED, NEW_SEED = 20261124, 20261125
U, B, M = 4, 4, 16
P = e8.P_MAX_MW
TOL = 1e-9
TAPS_S = np.array([0, .1, .3, .8]) * 1e-6
PDP = np.array([.5, .3, .15, .05])
DF = 360e3
PAIRS = list(combinations(range(B), 2))
MATCHINGS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))


def swap_grid():
    vectors = [np.zeros(U)]
    for step in (1/32, 1/16, 1/8, 1/4):
        for u, v in combinations(range(U), 2):
            change = np.zeros(U)
            change[u], change[v] = step, -step
            vectors.extend([change, -change])
    return np.stack(vectors)


GRID = swap_grid()


@lru_cache(maxsize=1)
def archive():
    with np.load(ARCHIVE) as f:
        return f["val"][:, 0, :, 0, 0, U:], f["key"][:, 0, :, 0, 0, U:]


def channel(d, model, eps):
    seed = OLD_SEED if model == "E8" else NEW_SEED
    iid, beta, _, _ = e8.drop(seed, d, 1, U, B)
    rng = np.random.default_rng([seed, d, 78])

    def cn(shape):
        return (rng.standard_normal(shape) + 1j*rng.standard_normal(shape)) / np.sqrt(2)

    if model in ("E8", "iid"):
        truth = iid
        independent = np.sqrt(beta)[None] * cn(iid.shape)
    else:
        spacing = {"flat": 0, "taps": 1, "gap4": 4, "gap8": 8}[model]
        frequencies = np.arange(B) * DF * spacing
        response = np.exp(-2j*np.pi*frequencies[:, None]*TAPS_S) * np.sqrt(PDP)

        def draw():
            return np.sqrt(beta)[None] * np.einsum("bl,lum->bum", response, cn((len(PDP), U, M)))

        truth, independent = draw(), draw()
    estimate = (1-eps)*truth + np.sqrt(eps*(1-eps))*independent
    return truth, estimate


def mask_weights(h, keys):
    weights = np.zeros((len(keys), M, U), complex)
    for k, key in enumerate(keys):
        antennas = np.flatnonzero((int(key) >> np.arange(M)) & 1)
        weights[k, antennas] = np.linalg.pinv(h[:, antennas])
    return weights


def fresh_pool(h, noise, kappa):
    """Keep the archived-style winner per cardinality and the minimum-power baseline witness."""
    values, q, _ = e8.score(h, 1)
    vals, indices = e8.best_per_m(values, 1, U)
    gains = vals[0, 0, U:]
    keys, _, size, _, _ = e8.candidates(1, U)
    winners = keys[indices[0, 0, U:]]
    required = e8.c2_threshold(gains.max(), noise, kappa)
    min_m = int(np.flatnonzero(gains >= required*(1-1e-11))[0])+U
    ok = np.flatnonzero((size == min_m) & (values[0] >= required*(1-1e-11)))
    lowest_power_key = keys[ok[np.argmin(q[ok].sum(axis=1))]]
    return gains, winners, lowest_power_key


def rates_from_effective(effective, noise):
    desired = abs(np.diagonal(effective, axis1=-2, axis2=-1))**2
    leakage = (abs(effective * (1-np.eye(U)))**2).sum(-1)
    return np.log2(1+desired/(leakage+noise)), desired, leakage


def schedule(h, truth, pools, noise, kappa, strong):
    gains = np.stack([item[0] for item in pools])
    target = kappa*np.log2(1+gains.max(axis=1)/noise)
    required = noise*np.expm1(np.log(2)*target)
    old = (gains >= required[:, None]*(1-1e-11)).argmax(axis=1)
    keys = np.stack([item[1] for item in pools])
    if strong:
        keys = np.column_stack([keys, [item[2] for item in pools]])
        old = np.full(B, keys.shape[1]-1)
    nants = np.array([[int(key).bit_count() for key in row] for row in keys])
    directions = np.stack([mask_weights(h[b], keys[b]) for b in range(B)])
    q = abs(directions)**2
    power_cap = required * q[np.arange(B),old].sum(axis=(1,2))
    base_m = nants[np.arange(B), old]
    best_by_pair = {}
    for i, j in PAIRS:
        swap = min(target[i], target[j])*GRID
        choices, rates, costs = [], [], []
        for b, sign in ((i,1),(j,-1)):
            rate = target[b]+sign*swap
            gain = noise*np.expm1(np.log(2)*rate)
            power = np.einsum("mau,vu->vma", q[b], gain)
            valid = ((power.max(axis=-1) <= P*(1+TOL)) &
                     (power.sum(axis=-1) <= power_cap[b]*(1+TOL)))
            cost = np.where(valid, nants[b], 100)
            choices.append(cost.argmin(1))
            costs.append(cost.min(1))
            rates.append(rate)
        total = costs[0]+costs[1]
        k = int(total.argmin())
        if total[k] < base_m[i]+base_m[j]:
            best_by_pair[i,j] = (total[k], [a[k] for a in choices], [a[k] for a in rates])
        else:
            best_by_pair[i,j] = (base_m[i]+base_m[j], old[[i,j]], np.repeat(target[[i,j],None], U, axis=1))
    pairs = min(MATCHINGS, key=lambda p: sum(best_by_pair[x][0] for x in p))
    new = old.copy()
    r0 = np.repeat(target[:,None], U, axis=1)
    r1 = r0.copy()
    for pair in pairs:
        _, chosen, rates = best_by_pair[pair]
        new[list(pair)] = chosen
        r1[list(pair)] = rates
    planned = np.stack([r0,r1])
    selected = np.stack([old,new])
    chosen_keys = keys[np.arange(B),selected]
    active = nants[np.arange(B),selected]
    gain = noise*np.expm1(np.log(2)*planned)
    w = directions[np.arange(B),selected]*np.sqrt(gain)[:,:,None,:]
    ant_power = (abs(w)**2).sum(axis=-1)
    actual_p = ant_power.sum(axis=-1)
    estimated_e = h@w
    true_e = truth@w
    checked = rates_from_effective(estimated_e, noise)[0]
    err = np.array([np.max(abs(checked-planned)), np.max(abs(checked[0].sum(1)-checked[1].sum(1))),
                    np.max(abs(checked[0].sum(0)-checked[1].sum(0)))])
    assert max(err) < 1e-7 and np.all(active[1].sum()<=active[0].sum())
    assert ant_power.max() <= P*(1+2*TOL) and np.all(actual_p <= power_cap*(1+2*TOL))
    assert np.max(abs(estimated_e*(1-np.eye(U)))) < 1e-7*np.max(abs(estimated_e))
    rates, desired, interference = rates_from_effective(true_e, noise)
    twc = np.log2(1 + desired.sum(axis=1)/(interference.sum(axis=1)+B*noise))
    return dict(key=chosen_keys, count=active, planned=planned, real=rates, twc=twc, effective=true_e,
                power=actual_p, peak=ant_power.max(axis=-1), cap=power_cap, error=err,
                nmse_num=np.sum(abs(h-truth)**2), nmse_den=np.sum(abs(truth)**2))


def worker(job):
    cohort, d = job
    output = []
    states = {"archive": [("E8",0.)],
              "holdout": [("iid",0.),("flat",0.),("taps",0.),("taps",.01),("taps",.1)],
              "spacing": [("gap4",0.),("gap4",.01),("gap8",0.),("gap8",.01)]}[cohort]
    for model, eps in states:
        truth, h = channel(d, model, eps)
        for db in ((-90,-70,-50) if model=="E8" else (-70,)):
            noise = 10.**(db/10)
            if model=="E8":
                values, keys = archive()
                pools = list(zip(values[d], keys[d], [None]*B))
            elif model=="flat":
                pools = [fresh_pool(h[0],noise,.98)]*B
            else:
                pools = [fresh_pool(one,noise,.98) for one in h]
            for strong in ([False,True] if model=="taps" else [model!="E8"]):
                res = schedule(h, truth, pools, noise, .98, strong)
                code = "lowpower" if strong else "winner"
                output.append(dict(case=f"{model}_{eps:g}_{db}_{code}", drop=d, **res))
    return output


def mean_ci(x):
    x = np.asarray(x, float)
    mu = x.mean()
    half = t.ppf(.975,len(x)-1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x)>1 else 0.
    return [float(mu),float(mu-half),float(mu+half)]


def summarize(data):
    result = {}
    for case in np.unique(data["case"]):
        use = data["case"]==case
        c = data["count"][use].mean(axis=-1)
        rates = data["real"][use].mean(axis=2)
        dr = rates[:,1]-rates[:,0]
        twc = data["twc"][use].sum(axis=-1)
        base, new = c.mean(axis=0)
        power = data["power"][use]
        result[case] = dict(drops=int(use.sum()), base=float(base), candidate=float(new),
            delta_ru_ci=mean_ci(c[:,0]-c[:,1]), payload_pct=float(100*(1-new/base)),
            total_7_1_pct_50prb=float(100*.5376*(base-new)/(.5376*base+16*.121)),
            improved_drops=int(np.sum(c[:,1]<c[:,0])), real_sum_base=float(rates[:,0].sum(axis=1).mean()),
            real_sum_delta_ci=mean_ci(dr.sum(1)), per_ue_delta_p05=float(np.quantile(dr,.05)),
            ue_loss_over_5pct=float(np.mean(dr < -.05*rates[:,0]-1e-10)),
            twc_sum_delta_ci=mean_ci(twc[:,1]-twc[:,0]),
            actual_power_base_new_mw=power.mean(axis=(0,2)).tolist(),
            max_total_power_excess_mw=float(np.maximum(0,power[:,1]-data["cap"][use]).max()),
            max_antenna_power_mw=float(data["peak"][use].max()),
            max_rate_check_error=float(data["error"][use].max()),
            measured_nmse=float(data["nmse_num"][use].sum()/data["nmse_den"][use].sum()))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--e8-drops",type=int,default=500)
    ap.add_argument("--holdout-drops",type=int,default=96)
    ap.add_argument("--workers",type=int,default=8)
    ap.add_argument("--spacing-only",action="store_true",
                    help="Use the same four PRBs worth of airtime at spacings of four or eight PRBs.")
    ap.add_argument("--out",type=Path,required=True)
    args = ap.parse_args()
    jobs = [("archive",d) for d in range(args.e8_drops)] + [("holdout",d) for d in range(args.holdout_drops)]
    if args.spacing_only:
        jobs = [("spacing",d) for d in range(args.holdout_drops)]
    args.out.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [row for batch in pool.map(worker,jobs,chunksize=1) for row in batch]
    data = {k:np.stack([r[k] for r in rows]) for k in rows[0]}
    np.savez_compressed(args.out/"paired.npz", **data)
    metrics = summarize(data)
    (args.out/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    for case,s in metrics.items():
        print(case, s["drops"], round(s["base"],3), round(s["candidate"],3), round(s["payload_pct"],2),
              "real sum delta", [round(x,4) for x in s["real_sum_delta_ci"]], flush=True)


if __name__=="__main__":
    main()
