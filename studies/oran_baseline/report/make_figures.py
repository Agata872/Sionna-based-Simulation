#!/usr/bin/env python3
"""Regenerate the report figures of experiments E1-E3 with English labels.

The original figures in results/ (written by compare_cat_a_b.py, rt_closed_loop.py and rt_compare_compression.py)
carry Chinese labels. This script re-plots them from the stored results.json files, without running any simulation.
The F1 figures are already in English and are copied from the run directory as they are.

Usage (from the repository root, any Python with numpy + matplotlib):
    python studies/oran_baseline/report/make_figures.py
"""

import json
import os
import shutil

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(HERE, "figures")
F1_RUN = os.path.join(RES, "split_fronthaul_load_20261105", "run_115332_drops20_seed2026")

METHOD = {"bfp": "BFP", "bs": "block scaling", "mu": "μ-law"}


def label_of(width):
    if width == "ideal":
        return "ideal FH"
    if width == "none16":
        return "16 bit uncompressed"
    if ":" in width:
        m, w = width.split(":")
        return f"{METHOD[m]} {w} bit"
    return f"BFP {width} bit"


def load(name):
    with open(os.path.join(RES, name, "results.json"), encoding="utf-8") as f:
        return json.load(f)


def required_snr(snr, bler, target=0.1):
    for i in range(1, len(snr)):
        if bler[i - 1] >= target > bler[i]:
            lo, hi = np.log10(max(bler[i - 1], 1e-9)), np.log10(max(bler[i], 1e-9))
            return snr[i - 1] + (np.log10(target) - lo) / (hi - lo) * (snr[i] - snr[i - 1])
    return None


# ---------------------------------------------------------------- E1
def e1(out):
    widths = list(dict.fromkeys(r["width"] for r in out["runs"]))
    colors = {w: plt.cm.tab10(i) for i, w in enumerate(widths)}
    style = {"A": dict(ls="-", marker="o"), "B": dict(ls="--", marker="s")}
    a = out["args"]
    title = (f"MCS table {a['mcs_table']} index {a['mcs']}, {a['num_prb']} PRB, 2 UE × {a['layers_per_ue']} layers, "
             f"8 TX, {a['precoder'].upper()}, CDL-C")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for r in out["runs"]:
        snr = np.array([x["snr_db"] for x in r["results"]])
        y = np.array([np.mean(x["bler"]) for x in r["results"]])
        keep = y > 0
        ax.plot(snr[keep], y[keep], color=colors[r["width"]], **style[r["category"]],
                label=f"Cat {r['category']}, {label_of(r['width'])}")
    ax.set_yscale("log")
    ax.axhline(0.1, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BLER (mean of the two UEs)")
    ax.set_title(title, fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "e1_cat_ab_bler.png"), dpi=150)
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    for r in out["runs"]:
        snr = [x["snr_db"] for x in r["results"]]
        req = required_snr(snr, [float(np.mean(x["bler"])) for x in r["results"]])
        rate = (r["fh_rate_100mhz"]["uplane_total"] + r["fh_rate_100mhz"]["cplane_total"]) / 1e9
        tx_evm = np.mean([x["tx_evm_percent"] for x in r["results"]])
        kw = dict(color=colors[r["width"]], marker=style[r["category"]]["marker"], s=50)
        tag = f"{r['category']}:{r['width']}"
        if req is not None:
            ax1.scatter(rate, req, **kw)
            ax1.annotate(tag, (rate, req), fontsize=7, xytext=(3, 3), textcoords="offset points")
        if tx_evm > 0:
            ax2.scatter(rate, tx_evm, **kw)
            ax2.annotate(tag, (rate, tx_evm), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax1.set_ylabel("SNR required for BLER = 0.1 (dB)")
    ax2.set_ylabel("TX EVM at the O-RU antennas vs ideal FH (%)")
    ax2.set_yscale("log")
    for ax in (ax1, ax2):
        ax.set_xlabel("Fronthaul rate per O-RU at 100 MHz (Gbit/s)")
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.3)
    ax1.set_title("circles: Cat A, squares: Cat B", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "e1_cat_ab_tradeoff.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- E2
def e2(out):
    drops = out["drops"]
    a = out["args"]
    num_ue = a["num_ue"]
    tput = np.array([d["throughput_mbps"][k] for d in drops for k in range(num_ue)], dtype=float)
    pos = np.array([d["positions"][k] for d in drops for k in range(num_ue)])
    cov = out["coverage"]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(np.array(cov["path_gain_db"]), origin="lower", extent=cov["extent"], cmap="Greys_r",
                   vmin=-150, vmax=-60)
    fig.colorbar(im, ax=ax, label="Single-element path gain (dB)", shrink=0.8)
    for p, la in zip(out["config"]["rt"]["oru_positions"], out["config"]["rt"]["oru_look_at"]):
        ax.plot(p[0], p[1], "r^", ms=12)
        d = np.array(la[:2]) - np.array(p[:2])
        d = 30 * d / np.linalg.norm(d)
        ax.annotate("", xy=(p[0] + d[0], p[1] + d[1]), xytext=(p[0], p[1]),
                    arrowprops=dict(arrowstyle="->", color="r"))
    if num_ue > 1:
        for dd in drops:
            q = np.array(dd["positions"])
            ax.plot(q[:, 0], q[:, 1], "-", color="tab:blue", lw=0.5, alpha=0.5)
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=tput, cmap="viridis", s=30, edgecolors="w", linewidths=0.5)
    fig.colorbar(sc, ax=ax, label="UE throughput (Mbit/s)", shrink=0.8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"munich, {num_ue} UE × {a['layers_per_ue']} layers, Cat {a['category']}, "
                 f"U-plane {label_of(a['uplane'])}, MCS table {a['mcs_table']}, {a['num_prb']} PRB", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "e2_rt_mu_map.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- E3
def ue_values(run, key):
    return np.array([v for d in run["drops"] for v in d[key]], dtype=float)


def stats_of(run):
    tput = ue_values(run, "throughput_mbps")
    s = {"mean_tput": tput.mean(), "p5_tput": np.percentile(tput, 5), "bler": ue_values(run, "bler").mean(),
         "olla": ue_values(run, "mean_olla_offset_db").mean(),
         "dcqi": (ue_values(run, "cqi") - ue_values(run, "cqi_genie")).mean()}
    if all("csi_inr_db" in d for d in run["drops"]):
        s["inr"] = 10 * np.log10(np.mean(10 ** (ue_values(run, "csi_inr_db") / 10)))
    return s


def e3(out):
    a = out["args"]
    runs = out["runs"]
    widths = list(dict.fromkeys(r["width"] for r in runs))
    style = {"A": dict(ls="-", marker="o"), "B": dict(ls="--", marker="s")}
    panels = [("mean_tput", "Mean UE throughput (Mbit/s)"), ("p5_tput", "5th-percentile UE throughput (Mbit/s)"),
              ("bler", "First-transmission BLER"), ("olla", "Mean OLLA offset (dB)"),
              ("dcqi", "CQI (CSI-RS) − CQI (genie)"), ("inr", "INR measured on NZP-IMR (dB)")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, (key, ylabel) in zip(axes.flat, panels):
        for cat in ("A", "B"):
            xs, ys = [], []
            for i, w in enumerate(widths):
                run = next((r for r in runs if r["category"] == cat and r["width"] == w), None)
                if run is not None and key in stats_of(run):
                    xs.append(i)
                    ys.append(stats_of(run)[key])
            if xs:
                ax.plot(xs, ys, color="k", label=f"Cat {cat}", **style[cat])
        ax.set_xticks(range(len(widths)))
        ax.set_xticklabels([label_of(w) for w in widths], fontsize=8, rotation=15)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{a['num_ue']} UE × {a['layers_per_ue']} layers, CQI from CSI-RS, MCS table {a['mcs_table']}, "
                 f"{a['num_prb']} PRB, {len(runs[0]['drops'])} drops (solid: Cat A, dashed: Cat B)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "e3_csi_mu_summary.png"), dpi=120)
    plt.close(fig)


def f1():
    """F1 figures are already in English: copy them (constellations as JPEG to keep the repository small)."""
    for name in ("layout", "load_vs_throughput", "bler_dcqi", "load_scaling"):
        shutil.copy(os.path.join(F1_RUN, f"fig_{name}.png"), os.path.join(OUT, f"f1_{name}.png"))
    src = os.path.join(F1_RUN, "fig_constellations.png")
    dst = os.path.join(OUT, "f1_constellations.jpg")
    try:
        from PIL import Image
        Image.open(src).convert("RGB").save(dst, quality=80)
    except ImportError:
        os.system(f'convert "{src}" -sampling-factor 4:2:0 -quality 80 "{dst}"')


def main():
    os.makedirs(OUT, exist_ok=True)
    e1(load("cat_ab"))
    e2(load("rt_mu"))
    e3(load("csi_mu"))
    f1()
    print("Figures written to", OUT)


if __name__ == "__main__":
    main()
