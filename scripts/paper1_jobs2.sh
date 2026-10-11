#!/usr/bin/env bash
# Paper-1 experiment queue, second wave (results/paper1_20261008); same conventions as paper1_jobs.sh.
# Usage: bash scripts/paper1_jobs2.sh <gpu> [--after <pid>] <job> [<job> ...]
#   --after <pid>: wait until that process (a first-wave queue on the same GPU) has exited.
cd "$(dirname "$0")/.."
source env.sh
set -u
R=results/paper1_20261008
mkdir -p "$R"
GPU=$1; shift
if [ "${1:-}" = "--after" ]; then
    while kill -0 "$2" 2>/dev/null; do sleep 60; done
    shift 2
fi

FIN="--bfp 9 --bfp-kappa 2 --vq-scale measured --vq-cap --idx-coding comb --min-gain-db -115 \
--region-samples 64 --region-shape gauss --region-floor 1 --budgets 4 8 16 32 64 128"
TRK="--codec urts_vq --bfp 9 --bfp-kappa 2 --vq-scale measured --vq-cap --idx-coding comb --drops 8 --seed 20269900 \
--sigma0 5 10 --occasions 3 --budgets 4 8 16 32 64 --budget-loc 32 --loc-mode posterior --loc-temp 0.001 --acq-samples 192 --region-samples 64"
MM="ad_dft twin_res urts_code urts_vq grid_vq"
ALL="ad_dft twin_res urts_code urts_vq grid_ecb grid_vq site_klt"

go() {   # go <script> <name> <args...>
    local script=$1 name=$2; shift 2
    if [ -f "$R/$name/complete.json" ]; then echo "skip $name (complete)"; return; fi
    if [ -d "$R/$name" ]; then mv "$R/$name" "$R/$name.partial-$(date -u +%Y%m%dT%H%M%SZ)"; fi
    echo "start $name gpu=$GPU $(date -u +%FT%TZ)"
    OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=$GPU python "scripts/$script" --out "$R/$name" "$@" > "$R/$name.log" 2>&1
    echo "end $name exit=$? $(date -u +%FT%TZ)"
}
CF=explore_twin_cellfree.py

for job in "$@"; do
    case $job in
        # E1/E2 reruns after the GPU out-of-memory failures (same arguments as paper1_jobs.sh)
        k8_nominal)   go $CF main_k8_nominal_16 --reality nominal $FIN --drops 16 --pos-errors 0 2 5 --schemes $ALL --seed 20269300 ;;
        k16_nominal)  go $CF main_k16_nominal_8 --reality nominal $FIN --num-ue 16 --drops 8 --pos-errors 0 2 --schemes $ALL --seed 20269616 ;;
        cai4_nominal) go $CF cai4_k8_nominal_16 --reality nominal $FIN --grid-samples 4 --drops 16 --pos-errors 0 2 5 --schemes grid_ecb grid_vq --seed 20269300 ;;
        # E4 closed-loop O-DU localization with the final method (stage-6 configuration otherwise)
        track_nominal) go explore_twin_tracking.py track_final_nominal_8 $TRK --reality nominal ;;
        track_diffuse) go explore_twin_tracking.py track_final_diffuse_8 $TRK --reality diffuse ;;
        # E5 data-driven baselines: twin-trained autoencoders without / with the estimated position (paired drops)
        ae_nominal) go $CF ae_k8_nominal_16 --reality nominal $FIN --drops 16 --pos-errors 0 2 5 --schemes site_klt twin_ae twin_aep --site-samples 16384 --seed 20269300 ;;
        ae_diffuse) go $CF ae_k8_diffuse_16 --reality diffuse $FIN --drops 16 --pos-errors 0 2 5 --schemes site_klt twin_ae twin_aep --site-samples 16384 --seed 20269300 ;;
        # E6 effective dimension of the region subspace (theory T2)
        dim) go paper1_region_dimension.py dim_region_16 --centres 16 --region-samples 256 --stds 0.25 0.5 1 2 3 5 8 --arrays 4x4 8x8 --prbs 52 104 ;;
        # E7 twin mismatch with the unified accounting and the grid ablation (6 drops; the first 3 pair with mmfinal_*)
        mm_nominal)    go $CF mm_nominal_6    --reality nominal $FIN --drops 6 --pos-errors 0 2 --schemes $MM --seed 20269700 ;;
        mm_geom05)     go $CF mm_geom05_6     --reality nominal $FIN --real-geom 0.5 --drops 6 --pos-errors 0 2 --schemes $MM --seed 20269700 ;;
        mm_mat03)      go $CF mm_mat03_6      --reality nominal $FIN --real-material 0.3 --drops 6 --pos-errors 0 2 --schemes $MM --seed 20269700 ;;
        mm_cars40)     go $CF mm_cars40_6     --reality nominal $FIN --real-cars 40 --drops 6 --pos-errors 0 2 --schemes $MM --seed 20269700 ;;
        mm_diffusetw0) go $CF mm_diffusetw0_6 --reality diffuse $FIN --drops 6 --pos-errors 0 2 --schemes $MM --seed 20269700 ;;
        mm_all)        go $CF mm_all_6        --reality diffuse $FIN --real-geom 0.5 --real-material 0.3 --real-cars 40 --drops 6 --pos-errors 0 2 --schemes $MM --seed 20269700 ;;
        *) echo "unknown job $job" ;;
    esac
done
