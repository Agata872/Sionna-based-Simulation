"""Independently reconstruct link budgets and RF power, and solve the small ZF cases exactly."""

import argparse
import json
import sys
from itertools import combinations, permutations
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import t

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cross_prb_zf"))
from validate import channel

NOISE = 1e-7
POWER_LIMIT = 10.0
PRBS = 4
RE_PER_SYMBOL = 12
SYMBOLS = 14
BFP9_BITS = SYMBOLS * (2 * RE_PER_SYMBOL * 9 + 8)


def interval(x):
    mean = x.mean()
    half = t.ppf(.975, len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)) if len(x)>1 else 0.0
    return [float(mean), float(mean-half), float(mean+half)]


def activation(masks, num_ru):
    return ((np.asarray(masks, dtype=np.int64)[..., None] >> np.arange(num_ru)) & 1).astype(bool)


def sum_se_from_g(g, users):
    return users * np.log2(1 + np.sum(g, axis=-1)/(PRBS*NOISE))


def direct_zf(h, masks, caps):
    """Use SVD pseudoinverses on the chosen supports, without using candidate code."""
    gains, powers, residuals = [], [], []
    for hb, mask, cap in zip(h, masks, caps):
        on = np.flatnonzero(activation(mask, h.shape[-1]))
        hs = hb[:, on]
        assert len(on) >= h.shape[-2]
        z = np.linalg.pinv(hs)
        q = np.sum(abs(z)**2, axis=1)
        g = min(POWER_LIMIT / np.max(q), cap / q.sum())
        w = np.sqrt(g) * z
        ant_power = np.sum(abs(w)**2, axis=1)
        assert ant_power.max() <= POWER_LIMIT + 1e-10
        gains.append(g)
        powers.append(ant_power.sum())
        residuals.append(np.max(abs(hs @ w - np.sqrt(g)*np.eye(h.shape[-2]))) / np.sqrt(g))
    return np.array(gains), np.array(powers), max(residuals)


def all_small_candidates(h, cap=np.inf):
    """Evaluate every full-rank support of an 8-radio case with independent SVD."""
    users, num_ru = h.shape
    fields = []
    for size in range(users, num_ru + 1):
        on = np.array(list(combinations(range(num_ru), size)), dtype=int)
        hs = h[:, on].transpose(1, 0, 2)
        z = np.linalg.pinv(hs)
        valid = np.max(abs(hs @ z - np.eye(users)), axis=(1,2)) < 1e-7
        q = np.sum(abs(z[valid])**2, axis=2)
        g = np.minimum(POWER_LIMIT / q.max(1), cap / q.sum(1))
        masks = np.sum(1 << on[valid], axis=1)
        fields.append((activation(masks, num_ru).astype(np.int16), g))
    return [np.concatenate([item[i] for item in fields]) for i in (0,1)]


@lru_cache(maxsize=1)
def partition_tables():
    """Lookup tables for dividing 16 radios into four disjoint groups of four."""
    groups = np.array(list(combinations(range(16),4)))
    halves = np.array(list(combinations(range(16),8)))
    choose = np.array(list(combinations(range(8),4)))
    masks = (1 << groups).sum(1)
    half_masks = (1 << halves).sum(1)
    lookup = np.zeros(1 << 16, int)
    lookup[masks] = np.arange(len(groups))
    half_lookup = np.zeros(1 << 16, int)
    half_lookup[half_masks] = np.arange(len(halves))
    partitions = (1 << halves[:,choose]).sum(2)
    split = lookup[partitions], lookup[half_masks[:,None] ^ partitions]
    reverse = half_lookup[65535 ^ half_masks]
    return groups, split, reverse


def exact_tight_main(h, caps):
    """At capacity one, each PRB must use exactly four of the sixteen radios."""
    groups, (left, right), reverse = partition_tables()
    tables = []
    for hb, cap in zip(h, caps):
        hs = hb[:,groups].transpose(1,0,2)
        z = np.linalg.pinv(hs)
        power = np.sum(abs(z)**2,axis=2)
        rank = np.max(abs(hs @ z - np.eye(4)),axis=(1,2)) < 1e-7
        tables.append(np.where(rank, np.minimum(POWER_LIMIT/power.max(1), cap/power.sum(1)), -np.inf))
    first = np.max(tables[0][left] + tables[1][right],axis=1)
    last = np.max(tables[2][left] + tables[3][right],axis=1)
    return float(np.max(first + last[reverse]))


def exact_four_blocks(tables, link_counts):
    """Meet in the middle, encoding radio usage in base 3 for each pair of blocks."""
    num_ru = len(link_counts)
    radix = 3**np.arange(num_ru)

    def combine(a, b):
        usage = (a[0][:, None, :] + b[0][None, :, :]).reshape(-1,num_ru)
        values = (a[1][:,None]+b[1][None,:]).reshape(-1)
        admissible = (usage <= link_counts).all(-1)
        return usage[admissible], values[admissible]

    rhs, rhs_values = combine(*tables[2:])
    best = np.full(3**num_ru, -np.inf)
    np.maximum.at(best, rhs @ radix, rhs_values)
    prefix = best.reshape((3,)*num_ru,order="F")
    for axis in range(num_ru):
        prefix = np.maximum.accumulate(prefix,axis=axis)
    lhs, values = combine(*tables[:2])
    remaining = np.minimum(link_counts-lhs,2)
    return float(np.max(values + prefix.ravel(order="F")[remaining @ radix]))


def check_toy():
    ids = np.array([sum(1 << a for a in s) for s in combinations(range(4),2)])
    cost = activation(ids,4).astype(int)
    g1 = np.array([10,4,9,2,1,2])
    g2 = np.array([2,10,3,9,2,1])
    valid = (cost[:,None,:]+cost[None,:,:] <= 1).all(-1)
    feasible_gain = np.where(valid,g1[:,None]+g2[None,:],-np.inf)
    winner = np.unravel_index(feasible_gain.argmax(),feasible_gain.shape)
    assert tuple(ids[list(winner)])==(9,6)
    assert feasible_gain.max()==18
    assert feasible_gain[g1.argmax(),:].max()==feasible_gain[:,g2.argmax()].max()==11
    return dict(joint=18,greedy_in_both_orders=11)


def serial_opt_of_orders(tables, k, users):
    """Exact per-block choices, with reserve for all future blocks with any U-radio set."""
    num_ru = tables[0][0].shape[1]
    best_value = -np.inf
    best_power_caps = None
    for order in permutations(range(PRBS)):
        balance = np.full(num_ru,k)
        value = 0.
        chosen = [None]*PRBS
        for j,b in enumerate(order):
            use, gain = tables[b]
            remaining = balance-use
            left = PRBS-j-1
            enough = ((remaining>=0).all(1) &
                      (np.minimum(remaining,left).sum(1)>=left*users))
            scored = np.where(enough,gain,-np.inf)
            i = int(scored.argmax())
            assert enough[i]
            balance = remaining[i]
            value += gain[i]
            chosen[b] = int(use[i] @ (1 << np.arange(num_ru)))
        if value > best_value:
            best_value = value
            best_power_caps = chosen
    return best_value,best_power_caps


def tiny_trial(n):
    result={}
    for mode in ("taps","flat"):
        scenarios = []
        for d in range(n):
            h = channel(d,mode,0.)[0][:,:2,:8]
            tables = [all_small_candidates(hb) for hb in h]
            out = []
            for k in range(1,5):
                joined = exact_four_blocks(tables,np.full(8,k))
                greedy,chosen = serial_opt_of_orders(tables,k,2)
                _,cap,resid = direct_zf(h,chosen,np.full(PRBS,np.inf))
                assert resid<1e-8
                matched = exact_four_blocks([all_small_candidates(hb,limit) for hb,limit in zip(h,cap)],np.full(8,k))
                assert greedy <= matched*(1+2e-10) and matched <= joined*(1+2e-10)
                if k==4:
                    reference = sum(values.max() for _,values in tables)
                    assert np.isclose(joined,reference,rtol=1e-12,atol=0)
                out.append(sum_se_from_g(np.array([greedy,joined,matched])[:,None],2))
            scenarios.append(out)
        paired = np.asarray(scenarios)
        result[mode] = {str(k):dict(greedy=float(paired[:,j,0].mean()),
                                   joint=float(paired[:,j,1].mean()),
                                   matched=float(paired[:,j,2].mean()),
                                   delta_ci=interval(paired[:,j,1]-paired[:,j,0]),
                                   matched_delta_ci=interval(paired[:,j,2]-paired[:,j,0]))
                        for j,k in enumerate(range(1,5))}
    return result


def read_pair(path):
    with np.load(path) as data:
        return {k:data[k] for k in data.files}


def mapping(data, fields):
    pairs={(tuple(data[k][i].item() for k in fields)):i for i in range(len(data[fields[0]]))}
    assert len(pairs)==len(data[fields[0]])
    return pairs


def phase2_selections(raw):
    """Group Claude's one-row-per-PRB result into one row per slot and policy."""
    if "prb" not in raw:
        return raw
    index = mapping(raw, ("drop", "k", "algorithm", "prb"))
    groups = sorted({key[:3] for key in index})
    ids = np.array([[index[(*key,b)] for b in range(PRBS)] for key in groups])
    first = ids[:,0]
    links = raw["link_bits"][ids].sum(1)
    used = raw["used_bits"][ids]
    np.testing.assert_array_equal(used, np.broadcast_to(links[:,None,:],used.shape))
    return dict(drop=raw["drop"][first], k=raw["k"][first], algorithm=raw["algorithm"][first],
                mask=raw["mask"][ids] @ (1 << np.arange(16)), width=raw["width"][ids],
                codec=raw["codec"][ids], link_bits=links, power_mw=raw["power_mw"][ids].sum(2),
                antenna_power_mw=raw["power_mw"][ids], se_per_prb=raw["se_per_prb"][ids],
                twc_se=raw["twc_se"][first], desired=raw["signal_mw"][ids],
                distortion=raw["intdist_mw"][ids], mip_gap=raw["mip_gap"][first],
                ecpri_bits=raw["ecpri_bits"][first].sum(1), status=raw["status"][first])


def check_phase2_waveforms(raw):
    """Recompute all selected signals and SE from the saved RF waveform and symbols."""
    for d,h in enumerate(raw["H"]):
        np.testing.assert_array_equal(h,channel(d,"taps",0.)[0])
    h = raw["H"][raw["drop"],raw["prb"]]
    x = raw["x_tx"].astype(complex)
    symbols = raw["qam_idx"][raw["drop"],raw["prb"]]
    levels = raw["qam_levels"]
    s = levels[symbols//8] + 1j*levels[symbols%8]
    y = np.einsum("nua,natf->nutf",h,x,optimize=True)
    normalized = np.mean(abs(s)**2,axis=(2,3))
    gain = np.mean(y*s.conj(),axis=(2,3))/normalized
    signal = abs(gain)**2 * normalized
    residual = np.mean(abs(y-gain[...,None,None]*s)**2,axis=(2,3))
    power = np.mean(abs(x)**2,axis=(2,3))
    se = np.log2(1+signal/(residual+NOISE)).sum(1)
    np.testing.assert_allclose(se,raw["se_per_prb"],rtol=1e-7,atol=2e-7)
    np.testing.assert_allclose(signal,raw["signal_mw"],rtol=3e-6,atol=1e-13)
    np.testing.assert_allclose(residual,raw["intdist_mw"],rtol=3e-5,atol=1e-13)
    np.testing.assert_allclose(power,raw["power_mw"],rtol=1e-7,atol=2e-7)
    assert np.all(x[~raw["mask"]] == 0)
    effect = h @ raw["precoder"]
    scaled = raw["precoder_c"][:,None,None]
    max_zf_error = float(np.max(abs(effect/scaled-np.eye(4))))
    assert max_zf_error < 1e-7
    return dict(prb_waveforms=len(h), max_se_abs_error=float(np.max(abs(se-raw["se_per_prb"]))),
                max_relative_zf_residual=max_zf_error, max_waveform_antenna_power_mw=float(power.max()))


def check_cap_sweep(data, case, indexes, drops):
    report = {}
    for k in (1,2,3,4):
        rows = {name: [indexes[case,d,k,name] for d in drops]
                for name in ("independent", "greedy", "joint", "matched")}
        reference, old, new, matched = (data["se"][i] for i in rows.values())
        assert np.all(new >= old-1e-7) and np.all(matched >= old-1e-7), (case,k)
        assert np.all(reference >= new-1e-7), (case,k)
        if k == PRBS:
            np.testing.assert_allclose(new, reference, atol=1e-8, rtol=1e-9)
            np.testing.assert_allclose(old, reference, atol=1e-8, rtol=1e-9)
        links = data["link_units"].astype(np.int64)
        gain = new - old
        example = int(np.argmax(gain))
        ri, rj = rows["greedy"][example], rows["joint"][example]
        gaps = data["mip_gap"][rows["joint"]]
        matched_gaps = data["mip_gap"][rows["matched"]]
        users = 2 if case.startswith("small") else 4
        upper_se = sum_se_from_g(data["g"][rows["joint"]]*(1+gaps[:,None]),users)
        upper_matched = sum_se_from_g(data["g"][rows["matched"]]*(1+matched_gaps[:,None]),users)
        report[str(k)] = dict(
            layouts=len(drops), independent=float(reference.mean()), greedy=float(old.mean()),
            joint=float(new.mean()), matched=float(matched.mean()), joint_delta_ci=interval(gain),
            matched_delta_ci=interval(matched-old), improved_layouts=int(np.sum(gain>1e-7)),
            mean_joint_gain_ceiling=float((upper_se-old).mean()),
            mean_matched_gain_ceiling=float((upper_matched-old).mean()),
            independent_exceeds_cap=int(np.sum(links[rows["independent"]].max(1)>k)),
            joint_payload_delta_ci_bits=interval((links[rows["joint"]]-links[rows["greedy"]]).sum(1)*BFP9_BITS),
            proven_within_1e_5=int(np.sum(gaps <= 1e-5)),
            matched_proven_within_1e_5=int(np.sum(matched_gaps <= 1e-5)),
            largest_gain=dict(drop=int(drops[example]), difference=float(gain[example]),
                              greedy_mask=data["mask"][ri].tolist(), joint_mask=data["mask"][rj].tolist()))
    return report


def phase1(data):
    indexes = mapping(data, ("case", "drop", "k", "algorithm"))
    max_residual, max_power_excess = 0., 0.
    audited = {}
    small_exact = 0
    tight_exact = 0
    for case in np.unique(data["case"]):
        model = "flat" if "flat" in case else "taps"
        rows = data["case"] == case
        num_ru = data["link_units"].shape[1]
        small = num_ru == 8 or any(word in case for word in ("small", "tiny", "8ru"))
        num_ru, users = (8,2) if small else (16,4)
        rows_for_case = np.flatnonzero(rows)
        seen = np.unique(data["drop"][rows])
        ones, groups = 1 << num_ru, {}
        for d in seen:
            h = channel(int(d), model, 0.)[0][:, :users, :num_ru]
            groups[int(d)] = h
        for i in rows_for_case:
            d, k, method = (data[name][i].item() for name in ("drop", "k", "algorithm"))
            h = groups[d]
            mask = data["mask"][i]
            assert np.all((mask >= 1) & (mask < ones)), (case,d,k,method)
            usage = activation(mask, num_ru).sum(0)
            np.testing.assert_array_equal(usage, data["link_units"][i][:num_ru])
            if method != "independent":
                assert usage.max() <= k, (case,d,k,method,usage)
            cap = (data["power_mw"][indexes[case,d,k,"greedy"]]
                   if method == "matched" else np.full(PRBS, np.inf))
            g, p, residual = direct_zf(h, mask, cap)
            max_residual = max(max_residual, residual)
            max_power_excess = max(max_power_excess, float(np.max(p-cap)))
            np.testing.assert_allclose(g, data["g"][i], rtol=2e-7, atol=2e-15)
            np.testing.assert_allclose(p, data["power_mw"][i], rtol=2e-7, atol=2e-7)
            np.testing.assert_allclose(sum_se_from_g(g, users), data["se"][i], rtol=2e-8, atol=2e-8)
        if small:
            for d,h in groups.items():
                base = [all_small_candidates(hb) for hb in h]
                for k in range(1, PRBS+1):
                    greedy = indexes[case, d, k, "greedy"]
                    local_best, _ = serial_opt_of_orders(base, k, users)
                    np.testing.assert_allclose(data["g"][greedy].sum(), local_best, rtol=2e-7, atol=1e-13)
                    total_cap = data["power_mw"][greedy]
                    matched = [all_small_candidates(hb, cap) for hb,cap in zip(h,total_cap)]
                    for method,tables in (("joint",base), ("matched",matched)):
                        exact = exact_four_blocks(tables, np.full(num_ru,k))
                        predicted = data["g"][indexes[case,d,k,method]].sum()
                        np.testing.assert_allclose(predicted, exact, rtol=2e-6, atol=1e-13)
                        small_exact += 1
        else:
            for d in seen[:12]:
                h = groups[d]
                greedy_row = indexes[case, d, 1, "greedy"]
                for method, cap in (("joint", np.full(PRBS,np.inf)),
                                    ("matched", data["power_mw"][greedy_row])):
                    value = exact_tight_main(h, cap)
                    recorded = data["g"][indexes[case,d,1,method]].sum()
                    np.testing.assert_allclose(recorded, value, rtol=2e-6, atol=1e-13)
                    tight_exact += 1
        joint_indices = np.array([[indexes[case,d,k,"joint"] for k in range(1,5)] for d in seen])
        se = data["se"][joint_indices]
        assert np.all(np.diff(se,axis=1) >= -1e-7), (case, "cap monotonicity")
        audited[case] = check_cap_sweep(data, case, indexes, seen)
    assert max_residual < 1e-7 and max_power_excess < 2e-6
    return dict(rows=len(data["case"]), max_relative_zf_residual=max_residual,
                max_power_excess_mw=max_power_excess, independently_solved_small_rows=small_exact,
                independently_solved_main_tight_rows=tight_exact,
                sweeps=audited)


def phase2(raw):
    data = phase2_selections(raw)
    waveform_check = check_phase2_waveforms(raw) if "x_tx" in raw else None
    policies = ("greedy_fixed", "joint_fixed", "greedy_adaptive", "joint_adaptive")
    indexes = mapping(data, ("drop", "k", "algorithm"))
    masks = data["mask"]
    widths = data["width"].astype(np.int64)
    active = activation(masks,16)
    total = (active * (SYMBOLS * (24 * widths + 8))[...,None]).sum(1)
    np.testing.assert_array_equal(total, data["link_bits"])
    assert np.all(total.max(1) <= data["k"]*BFP9_BITS)
    assert np.all(masks < (1 << 16)) and np.all(active.sum(2) >= 4)
    assert np.isin(widths, [5,7,9]).all()
    names = np.char.lower(data["codec"].astype(str))
    fixed = np.char.endswith(data["algorithm"].astype(str),"_fixed")
    assert np.all(widths[fixed] == 9) and np.all(names[fixed] == "bfp")
    assert np.isin(names, ["bfp", "block_scaling"]).all()
    power = data["power_mw"]
    assert np.all(power <= POWER_LIMIT*active.sum(2) + 1e-6)
    antenna_peak = None
    if "antenna_power_mw" in data:
        ant = data["antenna_power_mw"]
        assert np.all(ant[~active] <= 1e-12) and np.all(ant >= 0)
        np.testing.assert_allclose(ant.sum(2), power, rtol=1e-8, atol=1e-8)
        antenna_peak = float(ant.max())
        assert antenna_peak <= POWER_LIMIT + 2e-6
    per_prb = data["se_per_prb"]
    if per_prb.ndim == 3:
        per_prb = per_prb.sum(2)
    twc = data["twc_se"]
    if twc.ndim == 2:
        twc = twc.sum(1)
    if "desired" in data and "distortion" in data:
        signal, distortion = data["desired"],data["distortion"]
        np.testing.assert_allclose(np.log2(1+signal/(distortion+NOISE)).sum(2), per_prb)
        np.testing.assert_allclose(np.log2(1+signal.sum(1)/(distortion.sum(1)+PRBS*NOISE)).sum(1), twc)
    report = {}
    for k in np.unique(data["k"]):
        drops = np.unique(data["drop"][data["k"]==k])
        rows = np.array([[indexes[d,int(k),label] for label in policies] for d in drops])
        se = per_prb[rows].mean(2)
        twc_se = twc[rows]
        net_bits = total[rows].sum(2)
        assert np.all(se[:,1] >= se[:,0]-1e-7) and np.all(se[:,3] >= se[:,2]-1e-7)
        assert np.all(se[:,3] >= se[:,1]-1e-7)
        comparisons = [(1,0),(3,2),(3,1),(2,0)]
        gaps = data["mip_gap"][rows]
        report[str(k)] = dict(layouts=len(drops), treatment_order=policies,
                              mean_se=se.mean(0).tolist(), mean_twc_se=twc_se.mean(0).tolist(),
                              mean_net_bits=net_bits.mean(0).tolist(),
                              certified_within_1e_5=(gaps <= 1e-5).sum(0).tolist(),
                              widths_used={a:{str(w):int(np.sum(widths[rows[:,i]]==w)) for w in (5,7,9)}
                                           for i,a in enumerate(policies)},
                              differences={f"{policies[i]}_minus_{policies[j]}":dict(
                                  se=interval(se[:,i]-se[:,j]), twc_se=interval(twc_se[:,i]-twc_se[:,j]),
                                  net_bits=interval(net_bits[:,i]-net_bits[:,j])) for i,j in comparisons})
    return dict(rows=len(masks), max_antenna_power_mw=antenna_peak, waveform_check=waveform_check, sweeps=report)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase1",type=Path)
    ap.add_argument("--phase2",type=Path)
    ap.add_argument("--out",type=Path)
    ap.add_argument("--tiny",type=int,default=0,metavar="LAYOUTS")
    args = ap.parse_args()
    audit = dict(toy=check_toy())
    if args.tiny:
        audit["independent_small_grid"]=tiny_trial(args.tiny)
    if args.phase1:
        audit["phase1"] = phase1(read_pair(args.phase1))
    if args.phase2:
        audit["phase2"] = phase2(read_pair(args.phase2))
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2))


if __name__=="__main__":
    main()
