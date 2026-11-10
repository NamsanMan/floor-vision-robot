#!/usr/bin/env bash
# Sub-PC one-command re-run for one label snapshot copied from the main PC.
#   usage: bash ~/mentorpi/code/finetune/run_cv.sh ~/mentorpi/data/ft150_user/<snap> [seeds, default 0,1,2]
#   <snap>/labels/{train,test}/*.json = labelme JSONs copied from the main PC (a read-only snapshot).
# Steps: masks of reviewed frames -> EoMT-L pre-label masks (once) -> baselines -> leave-one-position-out CV
#        (every group with >= 10 reviewed frames; 4 folds once spot4 is labeled) + main model (A+B+C -> spot4)
#        -> summary.md/json + figures. Runs are sequential on the GPU. Re-running the same snapshot skips runs
#        whose metrics.json exists, so always use a NEW snapshot directory after the labels change.
# Outputs: <snap>/masks/{train,test}, ~/mentorpi/finetune/runs/<snap>/{baselines,cv_*,main_*,summary.*,figures}
# Optional env (for quick pipeline tests only): RUNS_ROOT=<dir> instead of ~/mentorpi/finetune/runs,
#   FT_EXTRA="--epochs 1" extra ft_train.py arguments (changes the recipe - never for real results).
set -euo pipefail
SNAP=$(realpath "$1")
SEEDS=${2:-0,1,2}
NAME=$(basename "$SNAP")
PY=$HOME/mentorpi/venv-teacher/bin/python
FT=$HOME/mentorpi/code/finetune
PRE=$HOME/mentorpi/data/ft150_user/prelabel_masks
OUT=${RUNS_ROOT:-$HOME/mentorpi/finetune/runs}/$NAME
EXTRA=()
[ -n "${FT_EXTRA:-}" ] && EXTRA=(-- $FT_EXTRA)
mkdir -p "$OUT"
cd "$FT"
echo "[1/5] labels -> masks (reviewed frames only)  $(date)"
for sp in train test; do "$PY" labels_to_masks.py "$SNAP/labels/$sp" "$SNAP/masks/$sp"; done
echo "[2/5] EoMT-L pre-label masks from the original JSONs (read-only; made once)"
for sp in train test; do
  [ -f "$PRE/$sp/objects.json" ] || "$PY" labels_to_masks.py "$HOME/mentorpi/data/ft150/labels/$sp" "$PRE/$sp" --all
done
echo "[3/5] baselines (zero-shot DS-Net, EoMT-L pre-label)  $(date)"
"$PY" baselines.py --masks "$SNAP/masks" --prelabel "$PRE" --out "$OUT/baselines"
echo "[4/5] CV + main model  $(date)"
"$PY" cv.py --masks "$SNAP/masks" --out-root "$OUT" --seeds "$SEEDS" "${EXTRA[@]}"
echo "[5/5] summary + figures  $(date)"
"$PY" summarize.py --root "$OUT" > /dev/null
"$PY" make_figure.py --root "$OUT" --masks "$SNAP/masks" --prelabel "$PRE"
echo "done $(date): $OUT/summary.md"
