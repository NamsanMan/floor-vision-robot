#!/usr/bin/env bash
# Main PC (Git Bash): take a read-only snapshot of the labelme JSONs, copy it to the sub-PC, start run_cv.sh there.
#   usage: bash sync_labels_and_run.sh          (snapshot name = snap_YYYYMMDD_HHMM)
# The labeling folder is only read (cp). If labelme was saving a file at that moment the JSON check fails:
# just run the script again.
set -euo pipefail
LAB="${LAB:-$(cd "$(dirname "$0")/labeling" && pwd)}"
TRAIN_HOST=${TRAIN_HOST:-subpc}   # 학습용 PC ssh 이름
SNAP=snap_$(date +%Y%m%d_%H%M)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/labels/train" "$TMP/labels/test"
cp -p "$LAB"/ft150/labels/train/*.json "$TMP/labels/train/"
cp -p "$LAB"/ft150/labels/test/*.json "$TMP/labels/test/"
python - "$TMP" <<'EOF'
import glob, json, sys
for sp in ("train", "test"):
    fs = sorted(glob.glob(f"{sys.argv[1]}/labels/{sp}/*.json"))
    rev = sum(bool((json.load(open(f, encoding="utf-8")).get("flags") or {}).get("reviewed")) for f in fs)
    print(f"{sp}: {rev}/{len(fs)} reviewed")
EOF
ssh $TRAIN_HOST "mkdir -p ~/mentorpi/data/ft150_user/$SNAP/labels ~/mentorpi/finetune/runs" < /dev/null
scp -q -r "$TMP/labels/train" "$TMP/labels/test" "$TRAIN_HOST:mentorpi/data/ft150_user/$SNAP/labels/"
scp -q "$LAB/labels_to_masks.py" "$TRAIN_HOST:mentorpi/code/finetune/labels_to_masks.py"   # keep the converter in sync
ssh $TRAIN_HOST "nohup bash ~/mentorpi/code/finetune/run_cv.sh ~/mentorpi/data/ft150_user/$SNAP > ~/mentorpi/finetune/runs/run_$SNAP.log 2>&1 < /dev/null &" < /dev/null
echo "started $SNAP on the sub-PC. progress: ssh $TRAIN_HOST 'tail -n 30 ~/mentorpi/finetune/runs/run_$SNAP.log'"
echo "when it says done: bash fetch_results.sh $SNAP"
