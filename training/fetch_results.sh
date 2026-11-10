#!/usr/bin/env bash
# Main PC (Git Bash): copy summary, metrics, train logs and figures of one snapshot from the sub-PC into
# experiments/2026-10-29_dsnet_finetune/results/<snap>/  (weights and predicted masks stay on the sub-PC).
#   usage: bash fetch_results.sh snap_YYYYMMDD_HHMM
set -euo pipefail
SNAP=$1
EXP="${EXP:-$(cd "$(dirname "$0")/.." && pwd)}"
TRAIN_HOST=${TRAIN_HOST:-subpc}   # 학습용 PC ssh 이름
DST="$EXP/results/$SNAP"
R=mentorpi/finetune/runs/$SNAP
mkdir -p "$DST/runs" "$DST/figures"
scp -q "$TRAIN_HOST:$R/summary.md" "$TRAIN_HOST:$R/summary.json" "$TRAIN_HOST:$R/cv_plan.json" "$TRAIN_HOST:$R/baselines/baselines.json" "$DST/"
scp -q "$TRAIN_HOST:$R/figures/*.png" "$DST/figures/"
for d in $(ssh $TRAIN_HOST "cd ~/$R && ls -d cv_hold*/ main_train*/ 2>/dev/null | tr -d /" < /dev/null); do
  mkdir -p "$DST/runs/$d"
  scp -q "$TRAIN_HOST:$R/$d/metrics.json" "$TRAIN_HOST:$R/$d/train_log.csv" "$DST/runs/$d/"
done
echo "copied to $DST"
