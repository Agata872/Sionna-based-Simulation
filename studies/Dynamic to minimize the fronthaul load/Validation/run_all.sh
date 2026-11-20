#!/usr/bin/env bash
# 一键运行：第一轮补跑 M = 8，以及第二轮 (a) 最强 UE 映射、(b)+(c) ZF/RZF 与 CSI 误差。
# 在 herdr 面板中运行，SSH 断开后任务继续。预计约 25 分钟（48 进程），主要耗时在 precoder 的 M = 8。
# 进度：results/run_all_20261119.log；状态：results/run_all_20261119.status（running / done / failed: <步骤>）。
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
PY="/home/user/Sionna-based Simulation/.venv/bin/python3"
WORKERS="${WORKERS:-48}"
LOG=results/run_all_20261119.log
STATUS=results/run_all_20261119.status
mkdir -p results
exec > >(tee -a "$LOG") 2>&1

STEP=start
trap 'echo "failed: $STEP" > "$STATUS"; echo "== FAILED at $STEP $(date "+%F %T")"' ERR
echo running > "$STATUS"
echo "== start $(date '+%F %T'), workers = $WORKERS"

STEP=tests;         "$PY" test_antsel.py && "$PY" test_round2.py
STEP=round1_M8;     "$PY" run_round1.py --M 8 --samples 1000 --workers "$WORKERS" --out results/round1_mrt_M8_20261119
STEP=round2a;       "$PY" run_round2.py --exp mapping  --M 1 2 4 6 8 --samples 1000 --workers "$WORKERS"
STEP=round2bc;      "$PY" run_round2.py --exp precoder --M 1 2 4 6 8 --samples 1000 --workers "$WORKERS"

echo done > "$STATUS"
echo "== ALL DONE $(date '+%F %T')"
