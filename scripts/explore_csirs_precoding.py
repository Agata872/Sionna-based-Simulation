#!/usr/bin/env python3
"""Paired stage-0/1 O-RAN experiments; existing oran/ modules remain intact.

Numerical mode isolates whole CSI-RS-symbol IQ quantization and weight
quantization. CRC mode validates data-weight effects with the actual DMRS
receiver. Channel realizations, payload bits and noise are paired across cases.
Run via source env.sh and select one physical GPU with CUDA_VISIBLE_DEVICES.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import sionna.phy
from oran import ORANConfig, CompressionConfig, ORANDownlink
from oran.csi import lmmse_sinr
from oran.csi_rs import CSIRS, CSIRSReceiver
from oran.fronthaul import IQCodec, fronthaul_rate
from oran.precoding import apply_precoder


IDEAL = CompressionConfig("ideal", 16)


def args_of():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["numerical", "crc"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--seed-start", type=int, default=101)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-prb", type=int, default=12)
    p.add_argument("--snrs", type=float, nargs="+", default=[10, 20, 30])
    p.add_argument("--widths", type=int, nargs="+", default=[3, 4, 5, 6, 7, 9, 12, 16])
    p.add_argument("--phase-candidates", type=int, default=32)
    p.add_argument("--mcs", type=int, default=14)
    p.add_argument("--num-ue", type=int, default=2)
    p.add_argument("--antennas", type=int, default=8)
    p.add_argument("--azimuth", type=float, default=120)
    p.add_argument("--cdl", default="C")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def metadata(args, cfg):
    return dict(args=vars(args), config=asdict(cfg), started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                python=platform.python_version(), torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
                git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                caveats=["Static independent CDL drops, ideal transmitter CSI, fixed MCS; no aging or SRS estimation.",
                         "Analytic Ethernet cost; no packet queue/deadline model.",
                         "Pilot IQ precision changes entire pilot-bearing OFDM symbol, not arbitrary REs.",
                         "Phase search is a globally informed diagnostic baseline, not a novelty claim.",
                         "First-transmission goodput only; no HARQ or pilot-slot amortization in CRC mode."])


class Recorder:
    def __init__(self, directory):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=True)
        if (self.path / "rows.jsonl").exists():
            raise FileExistsError(f"Use a fresh output directory: {self.path}")
        self.f = (self.path / "rows.jsonl").open("w")
        self.rows = []

    def add(self, row):
        self.rows.append(row)
        self.f.write(json.dumps(row, allow_nan=False) + "\n")
        self.f.flush()

    def finish(self):
        self.f.close()
        keys = sorted({k for r in self.rows for k in r})
        with (self.path / "rows.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(self.rows)
        (self.path / "complete.json").write_text(json.dumps({"rows": len(self.rows), "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))


def codec(cfg, width):
    return IQCodec(CompressionConfig("bfp", width), 1 / (cfg.num_tx_ant * cfg.num_layers), cfg.num_oru_ant)


def quantize_weight(w, cfg, width):
    return codec(cfg, width)(w.transpose(-1, -2)).transpose(-1, -2)


def phase_quantize(w, h_csi, cfg, width, nphase, objective):
    """Independent column phase search, transmitting chosen reconstructed W directly.

    A phase per PRG/column is harmless for ideal precoding; all reference and
    data signals use the chosen quantized W. No phase side message is required.
    NMSE selection aligns phases only when computing its diagnostic objective.
    The other selector minimizes channel-weighted reconstruction distortion.
    Both use the same BFP bit budget; the second needs global CSI at the DU.
    """
    real = torch.arange(nphase, device=w.device) * (2 * math.pi / nphase)
    phases = torch.polar(torch.ones_like(real), real).to(w.dtype)
    wc = w.unsqueeze(0) * phases[:, None, None, None, None]
    q = quantize_weight(wc, cfg, width)
    aligned_err = q * phases.conj()[:, None, None, None, None] - w.unsqueeze(0)
    if objective == "nmse":
        costs = aligned_err.abs().square().sum(-2)
    else:
        # All measured UE channels; scale-independent global channel metric.
        transformed = torch.einsum("bpkrn,tbpnl->tbpkrl", h_csi, aligned_err)
        costs = transformed.abs().square().sum((-3, -2))
    index = costs.argmin(0)  # [B, PRG, L]
    gather_index = index.unsqueeze(0).unsqueeze(-2).expand(1, *q.shape[1:])
    chosen = q.gather(0, gather_index).squeeze(0)
    theta = phases[index].unsqueeze(-2)
    return chosen, chosen * theta.conj()


def weight_cases(w, h_csi, cfg, widths, nphase):
    yield "ideal", 16, w, w
    for width in widths:
        q = quantize_weight(w, cfg, width)
        yield "bfp", width, q, q
        if width <= 9:
            target_norm = torch.linalg.vector_norm(w, dim=-2, keepdim=True)
            normed = q * target_norm / torch.linalg.vector_norm(q, dim=-2, keepdim=True).clamp_min(1e-12)
            yield "bfp_norm", width, normed, normed
        if width <= 7:
            for objective in ("nmse", "channel"):
                q, aligned = phase_quantize(w, h_csi, cfg, width, nphase, objective)
                yield "phase_" + objective, width, q, aligned
                scale = torch.linalg.vector_norm(w, dim=-2, keepdim=True) / torch.linalg.vector_norm(q, dim=-2, keepdim=True).clamp_min(1e-12)
                yield "phase_" + objective + "_norm", width, q * scale, aligned * scale


def mean_sinr_db(s):
    # Geometric mean per-layer SINR in dB, not an EESM or BLER prediction.
    return 10 * torch.log10(s.clamp_min(1e-12)).mean(tuple(range(2, s.ndim)))


def se_of(s):
    return torch.log2(1 + s).sum(2).mean(tuple(range(2, s.ndim - 1)))


def cost_inventory(args, cfg, csirs, out):
    """Exact logical-block counts and baseline analytic wire cost.

    CSI-RS-specific sparse/structural options here are payload lower references,
    not implemented standards-compatible packet formats. Baseline includes
    full-grid U-plane and SE11, with weights shared by pilot/data every slot.
    """
    rows = []
    for antennas in (8, 32, 64):
        for prg in (2, 4, 8, 0):
            for period in (1, 8, 20):
                c = ORANConfig(category="B", num_prb=51, num_oru_ant=antennas, num_ue=args.num_ue, prg_size=prg,
                               uplane=CompressionConfig("bfp", 9), bfw=CompressionConfig("bfp", 9))
                rate = fronthaul_rate(c)
                wbits = c.num_prg * c.num_layers * (8 + 2 * antennas * 9)
                uall = c.num_layers * c.num_prb * 14 * (8 + 24 * 9)
                pilot_symbol = c.num_layers * c.num_prb * (8 + 24 * 9)
                active_pilot = int((csirs._layers != 0).sum().item() / args.num_prb * c.num_prb)
                rows.append(dict(antennas=antennas, prg_size=prg, period_slots=period,
                                 baseline_u_payload_gbps=rate.uplane_payload / 1e9,
                                 baseline_u_wire_gbps=rate.uplane_total / 1e9,
                                 shared_weights_payload_gbps=wbits * c.slots_per_second / 1e9,
                                 baseline_c_wire_gbps=rate.cplane_total / 1e9,
                                 pilot_symbol_share_u_payload=pilot_symbol / (period * uall),
                                 pilot_qpsk_sample_bits_per_measurement=2 * active_pilot,
                                 note="No additional W_RS counted: current W reused. QPSK figure excludes signaling/CDM reuse."))
    (out / "cost_inventory.json").write_text(json.dumps(rows, indent=2))


@torch.no_grad()
def numerical(args, cfg, sim, recorder):
    csirs = CSIRS(cfg, sim.pdsch_configs[0], device=args.device)
    receiver = CSIRSReceiver(csirs, cfg.prg_num_prb, sim.ue.estimator)
    cost_inventory(args, cfg, csirs, recorder.path)
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        sionna.phy.config.seed = seed
        state = sim.channel.sample(args.batch_size)
        hc = sim.channel.csi(state, sim.csi_symbol)
        tx = sim.odu(args.batch_size, hc, csi_rs=csirs)
        ideal_grid = apply_precoder(tx.x_layers, tx.w, sim.odu.prg_of_sc)
        assert torch.equal(sim.orus[0].frequency_domain(tx.x_layers, tx.w), ideal_grid)
        cases = [("ideal", 16, ideal_grid, tx.w, 0.0)]
        for width in args.widths:
            c = IQCodec(CompressionConfig("bfp", width), 1, 12)
            xp = tx.x_layers.clone()
            xp[:, :, csirs.symbol:csirs.symbol+1] = c(xp[:, :, csirs.symbol:csirs.symbol+1])
            pilot_positions = csirs._layers != 0
            ref = tx.x_layers[:, :, csirs.symbol, :][:, pilot_positions]
            rec = xp[:, :, csirs.symbol, :][:, pilot_positions]
            evm = ((rec - ref).abs().square().sum() / ref.abs().square().sum()).sqrt().item() * 100
            cases.append(("pilot_symbol_bfp", width, apply_precoder(xp, tx.w, sim.odu.prg_of_sc), tx.w, evm))
        for method, width, q, aligned in weight_cases(tx.w, hc, cfg, args.widths, args.phase_candidates):
            if method == "ideal":
                continue
            nmse = ((aligned - tx.w).abs().square().sum((-1, -2)) / tx.w.abs().square().sum((-1, -2))).mean(1)
            cases.append((method, width, apply_precoder(tx.x_layers, q, sim.odu.prg_of_sc), q, nmse))
        for snr in args.snrs:
            no = 10 ** (-snr / 10)
            ref_s = lmmse_sinr(state.stacked(), tx.w, no, sim.odu.prg_of_sc, cfg.num_layers_per_ue)
            ref_se, ref_db = se_of(ref_s), mean_sinr_db(ref_s)
            for method, width, grid, w, err in cases:
                sionna.phy.config.seed = seed + 100000 + int(snr * 10)
                y = sim.channel(grid[:, None], state, no)
                measured_s, _ = receiver.sinr(y)
                true_s = lmmse_sinr(state.stacked(), w, no, sim.odu.prg_of_sc, cfg.num_layers_per_ue)
                true_se, true_db = se_of(true_s), mean_sinr_db(true_s)
                measured_db = mean_sinr_db(measured_s)
                power = w.abs().square().sum((-1, -2)).mean(1)
                for b in range(args.batch_size):
                    for k in range(cfg.num_ue):
                        row = dict(seed=seed, drop=b, ue=k, snr_db=snr, method=method, width=width,
                                   true_se=float(true_se[b, k]), ideal_se=float(ref_se[b, k]),
                                   true_sinr_db=float(true_db[b, k]), ideal_sinr_db=float(ref_db[b, k]),
                                   measured_sinr_db=float(measured_db[b, k]),
                                   prediction_error_db=float(measured_db[b, k] - true_db[b, k]),
                                   weight_power=float(power[b]))
                        if method == "pilot_symbol_bfp":
                            row["pilot_evm_percent"] = err
                        elif method != "ideal":
                            row["weight_nmse"] = float(err[b])
                        recorder.add(row)
            print(f"numerical seed={seed} snr={snr:g} rows={len(recorder.rows)}", flush=True)


@torch.no_grad()
def crc(args, cfg, sim, recorder):
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        sionna.phy.config.seed = seed
        state = sim.channel.sample(args.batch_size)
        hc = sim.channel.csi(state, sim.csi_symbol)
        tx = sim.odu(args.batch_size, hc)
        cases = list(weight_cases(tx.w, hc, cfg, args.widths, args.phase_candidates))
        for snr in args.snrs:
            no = 10 ** (-snr / 10)
            for method, width, w, aligned in cases:
                grid = apply_precoder(tx.x_layers, w, sim.odu.prg_of_sc)
                sionna.phy.config.seed = seed + 100000 + int(snr * 10)
                y = sim.channel(grid[:, None], state, no)
                rx = sim.ue(y, no)  # DMRS estimates the actual transmitted W.
                for b in range(args.batch_size):
                    for k in range(cfg.num_ue):
                        good = bool(rx.crc_ok[b, k])
                        recorder.add(dict(seed=seed, drop=b, ue=k, snr_db=snr, method=method, width=width,
                                          mcs=args.mcs, crc_ok=good, tb_bits=sim.tb_sizes[k],
                                          first_tx_mbps=int(good) * sim.tb_sizes[k] * cfg.slots_per_second / 1e6,
                                          weight_power=float(w[b].abs().square().sum((-1, -2)).mean()),
                                          rx_evm_percent=float(100 * (((rx.x_hat[b, k] - tx.x_data[b, k]).abs().square().sum() /
                                                                       tx.x_data[b, k].abs().square().sum()).sqrt()))))
            print(f"crc seed={seed} snr={snr:g} rows={len(recorder.rows)}", flush=True)


def main():
    args = args_of()
    torch.set_num_threads(2)
    sionna.phy.config.device = args.device
    cfg = ORANConfig(category="B", num_prb=args.num_prb, num_ue=args.num_ue, num_oru_ant=args.antennas, mcs_index=args.mcs,
                     channel_model="cdl", cdl_model=args.cdl, ue_azimuth_range=args.azimuth,
                     uplane=IDEAL, bfw=IDEAL, device=args.device)
    recorder = Recorder(args.out)
    (recorder.path / "metadata.json").write_text(json.dumps(metadata(args, cfg), indent=2))
    started = time.time()
    sim = ORANDownlink(cfg)
    (numerical if args.mode == "numerical" else crc)(args, cfg, sim, recorder)
    recorder.finish()
    print(f"complete rows={len(recorder.rows)} wall_s={time.time()-started:.1f}", flush=True)


if __name__ == "__main__":
    main()
