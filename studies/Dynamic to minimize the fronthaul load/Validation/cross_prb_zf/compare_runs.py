"""Compare both Claude implementations against independent outputs and audit every CRC count."""
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

HERE = Path(__file__).resolve().parent
RESULTS = HERE/"results"


def load(path):
    return json.loads(path.read_text())


def compare_claude():
    replay, holdout = [load(HERE/name) for name in ("claude_replay.json","claude_holdout.json")]
    assert replay["meta"]["all_checks_pass"]
    source = {name: HERE/name for name in ("claude_replay.py","claude_holdout.py")}
    source["same_total.py"] = HERE.parent/"same_total.py"
    for name, digest in holdout["meta"]["sources_sha256"].items():
        assert sha256(source[name].read_bytes()).hexdigest() == digest, name
    for name in ("claude_replay","same_total"):
        assert sha256(source[name+".py"].read_bytes()).hexdigest() == replay["meta"]["inputs"][name+"_py_sha256"], name
    original_samples = HERE.parent/"results/same_total_20261124/samples.npz"
    assert sha256(original_samples.read_bytes()).hexdigest() == replay["meta"]["inputs"]["samples_sha256"]
    with np.load(RESULTS/"paired.npz") as f:
        data = {k:f[k] for k in ("case","drop","count","key","power","cap","real","planned")}
    where = {(label,int(d)):i for i,(label,d) in enumerate(zip(data["case"],data["drop"]))}
    assert len(where)==len(data["case"])
    audit = dict(e8_layout_noise=0, holdout_layout_condition=0, e8_count_mismatches=0,
                 holdout_count_mismatches=0, holdout_mask_mismatches=0, max_cap_difference_mw=0.,
                 max_tx_power_difference_mw=0., max_true_se_difference=0., max_predicted_se_difference=0.)

    def update(field, x, y):
        audit[field] = max(audit[field],float(np.max(np.abs(np.asarray(x)-np.asarray(y)))))

    for db,rows in replay["per_drop"].items():
        assert sorted(r["drop"] for r in rows)==list(range(500))
        for r in rows:
            i = where[f"E8_0_{db}_winner",r["drop"]]
            cnt = np.array([r["m_base"],r["m_cand"]])
            audit["e8_layout_noise"] += 1
            audit["e8_count_mismatches"] += int(np.sum(cnt!=data["count"][i]))
            update("max_cap_difference_mw",r["cap_mw"],data["cap"][i])
            if "rates_measured" in r:
                update("max_true_se_difference",r["rates_measured"],data["real"][i,1])
    assert sorted(r["drop"] for r in holdout["per_drop"])==list(range(96))
    for r in holdout["per_drop"]:
        for eps in ("0","0.01"):
            for baseline in ("winner","lowpower"):
                i = where[f"taps_{eps}_-70_{baseline}",r["drop"]]
                choice = r["eps"][eps][baseline]
                audit["holdout_layout_condition"] += 1
                for a,key in enumerate(("base","cand")):
                    out = choice[key]
                    audit["holdout_count_mismatches"] += int(np.sum(out["m"]!=data["count"][i,a]))
                    audit["holdout_mask_mismatches"] += int(np.sum(out["keys"]!=data["key"][i,a]))
                    update("max_true_se_difference",out["se_true"],data["real"][i,a])
                    update("max_predicted_se_difference",out["se_pred"],data["planned"][i,a])
                    update("max_tx_power_difference_mw",out["ptot_mw"],data["power"][i,a])
                update("max_cap_difference_mw",choice["base"]["cap_mw"],data["cap"][i])
    assert (audit["e8_layout_noise"],audit["holdout_layout_condition"])==(1500,384)
    assert all(v==0 for k,v in audit.items() if "mismatch" in k), audit
    assert all(v<1e-8 for k,v in audit.items() if k.startswith("max_")), audit
    return audit


def compare_crc():
    audit = {}

    def paired_interval(values):
        mean = values.mean()
        half = t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))
        return [mean,mean-half,mean+half]

    for folder in (RESULTS, RESULTS/"mcs16",RESULTS/"spacing"):
        with np.load(folder/"crc_pairs.npz") as data:
            for case,row in load(folder/"crc.json").items():
                errors = data[case]
                payload = data[case+"_payload_error"]
                assert errors.shape==(96,2,16,4)
                assert payload.shape==errors.shape and errors.dtype==payload.dtype==np.bool_
                assert row["per_policy_tbs"]==errors[:,0].size
                assert np.array_equal(data[case+"_ids"],np.arange(96))
                bler = errors.mean(axis=(2,3))
                delivered = np.logical_not(errors | payload)
                rates = delivered.mean(axis=(2,3))*4*row["tb_bits"]/.0005/1e6
                undetected = (~errors & payload).sum(axis=(0,2,3))
                assert np.array_equal(undetected,row["undetected_by_policy"])
                assert undetected.sum()==row["undetected_block_errors"]
                assert np.count_nonzero(~errors[:,0] & errors[:,1])==row["candidate_only_errors"]
                assert np.count_nonzero(errors[:,0] & ~errors[:,1])==row["base_only_errors"]
                np.testing.assert_allclose(bler.mean(0),row["bler"],rtol=0,atol=1e-12)
                np.testing.assert_allclose(payload.mean(axis=(0,2,3)),row["payload_bler"],rtol=0,atol=1e-12)
                np.testing.assert_allclose(rates.mean(0),row["goodput_mbps"],rtol=0,atol=1e-10)
                np.testing.assert_allclose(paired_interval(bler[:,1]-bler[:,0]),row["bler_diff_ci"],atol=1e-12)
                np.testing.assert_allclose(paired_interval(rates[:,1]-rates[:,0]),row["goodput_diff_ci_mbps"],atol=1e-10)
                audit[case] = dict(blocks_per_policy=int(errors[:,0].size),crc_undetected_by_policy=undetected.tolist())
    return audit


def frequency_figure():
    cases = [(1, "taps",RESULTS),(4,"gap4",RESULTS/"spacing"),(8,"gap8",RESULTS/"spacing")]
    payload, total, crc16, crc20 = [], [], [], []
    for _,model,folder in cases:
        case = model+"_0.01_-70_lowpower"
        with np.load(folder/"paired.npz") as f:
            counts = f["count"][f["case"]==case].mean(axis=2)
        rng = np.random.default_rng(20261127)
        ds = rng.integers(0,len(counts),size=(10000,len(counts)))
        a,b = counts[ds].mean(1).T
        a0,b0 = counts.mean(0)
        for target,fixed in ((payload,0.),(total,16*.121/.5376)):
            interval = np.percentile(100*(a-b)/(a+fixed),[2.5,97.5])
            target.append([100*(a0-b0)/(a0+fixed),*interval])
        path16 = folder/"mcs16"/"crc.json" if model=="taps" else folder/"crc.json"
        crc16.append(100*np.array(load(path16)[case+"_mcs16"]["bler_diff_ci"]))
        crc20.append(100*np.array(load(folder/"crc.json")[case+"_mcs20"]["bler_diff_ci"]))
    fig,axes = plt.subplots(1,2,figsize=(8.6,3.4),layout="constrained")
    fig.suptitle("Cross-PRB ZF with 1% CSI NMSE (96 paired layouts)",fontsize=11)
    colors = ("#115b84","#c16320")
    for ax,series,names,ylabel in zip(axes,[(payload,total),(crc16,crc20)],
        [("IQ payload","With fixed signalling"),("MCS 16","MCS 20")],
        ["7-1 load reduction (%)","BLER change (percentage points)"]):
        for j,(arr,name) in enumerate(zip(series,names)):
            mean,lo,hi = np.array(arr).T
            pos = np.array([1,4,8])+(j-.5)*.17
            ax.errorbar(pos,mean,yerr=[mean-lo,hi-mean],fmt="o-" if j==0 else "s--",
                        label=name,color=colors[j],capsize=3,lw=1.5,ms=5)
        ax.set_xticks([1,4,8])
        ax.axhline(0,ls=":",color=".45",lw=.8)
        ax.set_xlabel("Spacing between scheduled PRBs (PRBs)")
        ax.set_ylabel(ylabel)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(loc="upper left",frameon=False,fontsize=8)
    fig.savefig(RESULTS/"frequency_spacing.pdf",bbox_inches="tight")
    plt.close(fig)
    return dict(spacing_prbs=[1,4,8], iq_payload_pct=payload,total_7_1_pct=total,
                bler_delta_pp_mcs16=[v.tolist() for v in crc16],bler_delta_pp_mcs20=[v.tolist() for v in crc20])


if __name__=="__main__":
    result=dict(implementation_agreement=compare_claude(),transport_block_checks=compare_crc(),
                estimates_with_95pct_intervals=frequency_figure())
    (RESULTS/"audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result["implementation_agreement"])
