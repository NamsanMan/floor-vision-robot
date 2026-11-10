#!/bin/bash
# VLA 실험 "말로 가리킨 물체 앞으로 가기" 한 번 실행 (메인컴 Git Bash, 2026-10-31)
#   bash vla_run.sh "주황 상자" [--go]
# 로봇: 카메라 한 장 + 지도 위치(snap.py) → 메인컴: DS-Net으로 물체 후보(vla_proposals.py, env mot) → CLIP으로 고르기·목표 계산
# (vla_ground.py, 기본 env) → 목표를 로봇에 보냄 → --go면 시험 도구로 카메라 조건 목표 주행(기록 남음).
# 전제: Nav2·인식 노드가 켜져 있고, 출발 위치를 스캔 맞춤으로 넣어 AMCL이 맞는 상태.
set -e
Q="$1"; GO="$2"
H=${ROBOT_SSH:-pi@mentorpi.local}
V="$(cd "$(dirname "$0")" && pwd)"
E="$(cd "$V/../.." && pwd)"
W="$E/results/vla_$(date +%Y%m%d)/$(date +%H%M%S)"; mkdir -p "$W"
PY_MOT=${PY_MOT:-python}     # onnxruntime 1.23 환경
PY_BASE=${PY_BASE:-python}    # torch + transformers 환경
export PYTHONIOENCODING=utf-8
echo "== 1. 로봇에서 사진 + 위치"
ssh $H "docker exec -u ubuntu MentorPi bash -c 'source /opt/ros/humble/setup.bash; python3 /home/ubuntu/shared/vla/snap.py'" | tail -1
scp -q $H:/home/pi/docker/tmp/vla/snap.jpg $H:/home/pi/docker/tmp/vla/snap.json "$W/"
echo "== 2. 바닥 분할로 물체 후보"
"$PY_MOT" "$V/vla_proposals.py" "$W/snap.jpg" --model "$E/robot/offload/models/ft_240x320_int8_gelu.onnx" --out "$W/proposals.json" | tail -1
echo "== 3. CLIP으로 '$Q' 찾기"
"$PY_BASE" "$V/vla_ground.py" "$Q" --snap "$W/snap.jpg" --props "$W/proposals.json" --pose "$W/snap.json" --out "$W/goal_point.json" 2>&1 | grep -E "후보 [0-9]|→"
[ -f "$W/goal_point.json" ] || { echo "목표를 못 만들어 멈춤 (그림: $W/goal_point_vis.jpg)"; exit 1; }
scp -q "$W/goal_point.json" $H:/home/pi/docker/tmp/drive_tests/goal_point.json
echo "목표 보냄 · 기록 폴더 $W"
if [ "$GO" = "--go" ]; then
  echo "== 4. 목표 주행 (카메라 조건)"
  ssh $H "bash /home/pi/docker/tmp/drive_test/dt.sh goal --cond cam --obj multi --to-click --no-frames --timeout 90 --note 'VLA: $Q'" | tail -6
fi
