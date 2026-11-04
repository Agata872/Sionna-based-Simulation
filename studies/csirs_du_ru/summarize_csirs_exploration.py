#!/usr/bin/env python3
"""Summarize paired experiments with drop-cluster bootstrap CIs and plots."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def ci_mean(values, rng):
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 2:
        return [float(x.mean())] * 2
    estimates = x[rng.integers(n, size=(3000, n))].mean(1)
    return np.quantile(estimates, [.025, .975]).tolist()


def summarize(directory):
    rows = [json.loads(line) for line in (directory / "rows.jsonl").read_text().splitlines()]
    metadata = json.loads((directory / "metadata.json").read_text())
    mode = metadata["args"]["mode"]
    key = lambda r: (r["seed"], r["drop"], r["ue"], r["snr_db"])
    ideal = {key(r): r for r in rows if r["method"] == "ideal"}
    groups = {}
    for r in rows:
        groups.setdefault((r["snr_db"], r["method"], r["width"]), []).append(r)
    rng = np.random.default_rng(20261006)
    summaries = []
    for (snr, method, width), group in sorted(groups.items()):
        drops = {}
        for r in group:
            drops.setdefault((r["seed"], r["drop"]), []).append(r)
        result = dict(snr_db=snr, method=method, width=width, independent_drops=len(drops), ue_observations=len(group))
        if mode == "numerical":
            se = [sum(r["true_se"] for r in d) for d in drops.values()]
            delta = [sum(r["true_se"] - ideal[key(r)]["true_se"] for r in d) for d in drops.values()]
            prediction = [np.mean([r["prediction_error_db"] for r in d]) for d in drops.values()]
            compression_prediction = [np.mean([r["prediction_error_db"] - ideal[key(r)]["prediction_error_db"] for r in d]) for d in drops.values()]
            ideal_se = np.mean([sum(ideal[key(r)]["true_se"] for r in d) for d in drops.values()])
            result.update(sum_se_mean=float(np.mean(se)), paired_sum_se_delta_mean=float(np.mean(delta)),
                          paired_sum_se_delta_ci95=ci_mean(delta, rng), sum_se_loss_percent=float(-100*np.mean(delta)/ideal_se),
                          prediction_error_db_mean=float(np.mean(prediction)),
                          incremental_prediction_error_db_mean=float(np.mean(compression_prediction)),
                          incremental_prediction_error_db_ci95=ci_mean(compression_prediction, rng),
                          weight_power_mean=float(np.mean([r["weight_power"] for r in group])))
            for field in ("weight_nmse", "pilot_evm_percent"):
                if field in group[0]:
                    result[field + "_mean"] = float(np.mean([r[field] for r in group]))
        else:
            goodput = [sum(r["first_tx_mbps"] for r in d) for d in drops.values()]
            delta = [sum(r["first_tx_mbps"] - ideal[key(r)]["first_tx_mbps"] for r in d) for d in drops.values()]
            ideal_tput = np.mean([sum(ideal[key(r)]["first_tx_mbps"] for r in d) for d in drops.values()])
            bler = [np.mean([1-int(r["crc_ok"]) for r in d]) for d in drops.values()]
            result.update(bler_mean=float(np.mean(bler)), bler_ci95=ci_mean(bler, rng),
                          sum_first_tx_mbps_mean=float(np.mean(goodput)),
                          paired_sum_first_tx_mbps_delta_mean=float(np.mean(delta)),
                          paired_sum_first_tx_mbps_delta_ci95=ci_mean(delta, rng),
                          first_tx_loss_percent=float(-100*np.mean(delta)/max(ideal_tput,1e-12)),
                          total_errors=int(sum(1-int(r["crc_ok"]) for r in group)))
            # Bootstrap is degenerate for zero events; retain a simple per-UE
            # binomial zero-error upper reference, not a cluster-CI replacement.
            if result["total_errors"] == 0:
                result["zero_error_per_ue_binomial_95_upper_reference"] = float(1-.05**(1/len(group)))
        summaries.append(result)
    out = {"mode": mode, "confidence_method": "3000 paired bootstrap draws; all UE observations in each independent CDL drop clustered together", "groups": summaries}
    (directory / "summary.json").write_text(json.dumps(out, indent=2))
    if mode == "crc":
        comparisons = []
        for snr in sorted({r["snr_db"] for r in rows}):
            for width in sorted({r["width"] for r in rows}):
                selected = [r for r in rows if r["method"] == "phase_channel_norm" and r["snr_db"] == snr and r["width"] == width]
                for reference in ("bfp_norm", "phase_nmse_norm"):
                    lookup = {key(r): r for r in rows if r["method"] == reference and r["width"] == width}
                    if not selected or not lookup:
                        continue
                    drops = {}
                    for r in selected:
                        drops.setdefault((r["seed"], r["drop"]), []).append(r["first_tx_mbps"] - lookup[key(r)]["first_tx_mbps"])
                    deltas = [sum(v) for v in drops.values()]
                    comparisons.append(dict(snr_db=snr, width=width, method="phase_channel_norm", reference=reference,
                                            independent_drops=len(deltas), paired_sum_first_tx_mbps_gain=float(np.mean(deltas)),
                                            ci95=ci_mean(deltas, rng)))
        (directory / "objective_comparisons.json").write_text(json.dumps(comparisons, indent=2))
    keys = sorted({k for r in summaries for k in r})
    with (directory / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(summaries)
    plot(directory, mode, summaries)
    return out


def plot(directory, mode, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    snrs = sorted({r["snr_db"] for r in summaries})
    if mode == "numerical":
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
        for snr in snrs:
            rs = sorted((r for r in summaries if r["method"] == "pilot_symbol_bfp" and r["snr_db"] == snr), key=lambda r:r["width"])
            axes[0].plot([r["width"] for r in rs], [r["incremental_prediction_error_db_mean"] for r in rs], "o-", label=f"{snr:g} dB")
            rs = sorted((r for r in summaries if r["method"] == "bfp" and r["snr_db"] == snr), key=lambda r:r["width"])
            axes[1].plot([r["width"] for r in rs], [r["sum_se_loss_percent"] for r in rs], "o-", label=f"{snr:g} dB")
        axes[0].set_ylabel("Incremental measurement bias (dB)")
        axes[0].set_title("Whole pilot-symbol IQ precision")
        axes[1].set_ylabel("Ideal-CSI sum-SE loss (%)")
        axes[1].set_title("Beamforming weight precision")
        name="pilot_vs_weights.png"
    else:
        fig, axes = plt.subplots(1, len(snrs), figsize=(4.1*len(snrs), 4.3), squeeze=False)
        axes=axes[0]
        for ax, snr in zip(axes, snrs):
            normalized = any(r["method"] == "phase_channel_norm" for r in summaries)
            methods = ("bfp_norm", "phase_nmse_norm", "phase_channel_norm") if normalized else ("bfp", "bfp_norm", "phase_nmse", "phase_channel")
            for method in methods:
                rs = sorted((r for r in summaries if r["method"] == method and r["snr_db"] == snr), key=lambda r:r["width"])
                ax.plot([r["width"] for r in rs], [r["bler_mean"] for r in rs], "o-", label=method)
            ref=next(r for r in summaries if r["method"]=="ideal" and r["snr_db"]==snr)
            ax.axhline(ref["bler_mean"], color="k", linestyle=":", label="ideal")
            ax.set_title(f"SNR {snr:g} dB; actual DMRS receiver")
            ax.set_ylabel("First-transmission BLER")
            ax.set_ylim(bottom=0)
        name="crc_weight_baselines.png"
    for ax in axes:
        ax.set_xlabel("BFP mantissa bits")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / name, dpi=160)
    fig.savefig(directory / name.replace(".png", ".pdf"))
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    args=parser.parse_args()
    for directory in args.directories:
        if not (directory/"complete.json").exists():
            raise RuntimeError(f"Run not complete: {directory}")
        summary=summarize(directory)
        print(directory, summary["mode"], len(summary["groups"]), "groups")


if __name__ == "__main__":
    main()
