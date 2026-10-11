#!/usr/bin/env bash
# Paper-1 experiment queue (results/paper1_20261008). Runs the named jobs one after another on one GPU.
# Usage: bash scripts/paper1_jobs.sh <gpu> <job> [<job> ...]
# A job whose run directory already has complete.json is skipped; an incomplete one is moved aside
# (<name>.partial-<utc>) and rerun. Every run keeps metadata.json, the source snapshot and a log.
cd "$(dirname "$0")/.."
source env.sh
set -u
R=results/paper1_20261008
mkdir -p "$R"
GPU=$1; shift

# Final method and fair accounting for every scheme: noise-matched BFP9, combinatorial index sets.
FIN="--bfp 9 --bfp-kappa 2 --vq-scale measured --vq-cap --idx-coding comb --min-gain-db -115 \
--region-samples 64 --region-shape gauss --region-floor 1 --budgets 4 8 16 32 64 128"
ALL="ad_dft twin_res urts_code urts_vq grid_ecb grid_vq site_klt"

run() {
    local name=$1; shift
    if [ -f "$R/$name/complete.json" ]; then echo "skip $name (complete)"; return; fi
    if [ -d "$R/$name" ]; then mv "$R/$name" "$R/$name.partial-$(date -u +%Y%m%dT%H%M%SZ)"; fi
    echo "start $name gpu=$GPU $(date -u +%FT%TZ)"
    OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=$GPU python scripts/explore_twin_cellfree.py --out "$R/$name" "$@" \
        > "$R/$name.log" 2>&1
    echo "end $name exit=$? $(date -u +%FT%TZ)"
}

for job in "$@"; do
    case $job in
        # E1 main comparison, all baselines, paired with the stage-5..7 drops (same seeds)
        k8_nominal)   run main_k8_nominal_16  --reality nominal $FIN --drops 16 --pos-errors 0 2 5 --schemes $ALL --seed 20269300 ;;
        k8_diffuse)   run main_k8_diffuse_16  --reality diffuse $FIN --drops 16 --pos-errors 0 2 5 --schemes $ALL --seed 20269300 ;;
        k16_nominal)  run main_k16_nominal_8  --reality nominal $FIN --num-ue 16 --drops 8 --pos-errors 0 2 --schemes $ALL --seed 20269616 ;;
        k16_diffuse)  run main_k16_diffuse_8  --reality diffuse $FIN --num-ue 16 --drops 8 --pos-errors 0 2 --schemes $ALL --seed 20269616 ;;
        k32_nominal)  run main_k32_nominal_6  --reality nominal $FIN --num-ue 32 --drops 6 --pos-errors 0 2 --schemes $ALL --seed 20269632 ;;
        k32_diffuse)  run main_k32_diffuse_6  --reality diffuse $FIN --num-ue 32 --drops 6 --pos-errors 0 2 --schemes $ALL --seed 20269632 ;;
        # E2 Cai et al. as published: 4 grid vertices only
        cai4_nominal) run cai4_k8_nominal_16  --reality nominal $FIN --grid-samples 4 --drops 16 --pos-errors 0 2 5 --schemes grid_ecb grid_vq --seed 20269300 ;;
        cai4_diffuse) run cai4_k8_diffuse_16  --reality diffuse $FIN --grid-samples 4 --drops 16 --pos-errors 0 2 5 --schemes grid_ecb grid_vq --seed 20269300 ;;
        # E3 per-O-RU power constraint at the O-DU precoder
        perru_nominal) run perru_k8_nominal_16 --reality nominal $FIN --precoder rzf_perru --drops 16 --pos-errors 0 2 5 --schemes ad_dft urts_code urts_vq grid_vq --seed 20269300 ;;
        perru_diffuse) run perru_k8_diffuse_16 --reality diffuse $FIN --precoder rzf_perru --drops 16 --pos-errors 0 2 5 --schemes ad_dft urts_code urts_vq grid_vq --seed 20269300 ;;
        *) echo "unknown job $job" ;;
    esac
done
