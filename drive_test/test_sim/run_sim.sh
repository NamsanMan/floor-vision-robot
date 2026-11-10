#!/bin/bash
# 주행 시험 도구 모의 점검 (WSL Ubuntu-22.04 + ROS 2 Humble, 2026-10-30). 로봇에 올리지 않음.
#   wsl -d Ubuntu-22.04 bash <이 파일 경로> <robot 폴더 경로>
# 모의 로봇(mock_robot.py) + 진짜 인식 노드(가짜 onnxruntime) + 진짜 감시 노드 + drive_test.py로 시험 순서를 한 번 돌린다.
# 로봇·다른 PC와 섞이지 않게 ROS_DOMAIN_ID=77, 이 컴퓨터 안에서만 통신. 모의 로봇 조작은 simctl.py(ros2 CLI가 WSL에서 멈춤).
SRC="$1"
S=/tmp/dt_sim
pkill -f mock_robot.py; pkill -f floor_seg_node.py; pkill -f floor_watchdog.py
rm -rf $S && mkdir -p $S/navlogs $S/fs && cp -r "$SRC" $S/code
C=$S/code
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1 PYTHONIOENCODING=utf-8
export DT_ROOT=$S/drive_tests DT_NAVLOG=$S/navlogs
export PYTHONPATH=$C/drive_test/test_sim/fake_ort:$C:$PYTHONPATH
python3 -u $C/drive_test/test_sim/mock_robot.py > $S/navlogs/nav_sim.log 2>&1 &
python3 -u $C/floor_seg_node.py --ros-args -p out_dir:=$S/fs > $S/node.log 2>&1 &
NODE=$!
python3 -u $C/floor_watchdog.py > $S/watchdog.log 2>&1 &
sleep 6
DT() { timeout 90 python3 -u $C/drive_test/drive_test.py "$@"; }
SC() { timeout 20 python3 -u $C/drive_test/test_sim/simctl.py "$@"; }
echo "== 1. 조건 바꾸기"; DT cond cam; DT cond lidar; DT cond cam_nomem
echo "== 2. 직진 정지: 카메라+기억, 책 1.5 m (정지 예상)"; SC obs 1.61 1; DT straight --cond cam --obj book --at 1.5; SC still
SC carry
echo "== 3. 들어서 옮긴 직후 같은 시험 (옛 기억을 안 지우면 출발하자마자 섬)"; DT straight --cond cam --obj book --at 1.5
SC carry
echo "== 4. 라이다만 (충돌 예상)"; DT straight --cond lidar --obj book --at 1.5; SC still
SC carry
echo "== 5. 카메라·기억 끔 (감속하지만 충돌 예상)"; DT straight --cond cam_nomem --obj book --at 1.5
SC carry
echo "== 6. 장애물 없음 (끝까지 감 예상)"; SC obs 1.61 0; DT straight --cond cam --obj none --at 1.5
SC carry
echo "== 7. 점검 모드"; DT straight --cond cam --obj none --dry
echo "== 8. 출발 위치 저장"; SC init; sleep 1; DT pose save
echo "== 9. 목표 주행: 카메라, 책 1.2 m (모의 Nav2는 회피를 못 해 Collision Monitor가 세움 → 25초 내내 서 있어야 함 = 기억 만료 수정 확인)"
SC obs 1.31 1
DT goal --cond cam --obj book --at 1.2 --timeout 25; SC still
SC carry
echo "== 10. 목표 주행: 라이다만 (도착, 겹침 예상)"; DT goal --cond lidar --obj book --at 1.2
SC carry
echo "== 11. 제자리 기록"; DT pose reset; DT idle --cond cam --secs 4 --tag sim
echo "== 12. 비상 정지 (주행 5초 뒤 stop)"; SC obs 1.61 0
DT straight --cond lidar --obj none --at 3.0 &
J=$!; sleep 7; DT stop; wait $J; SC still
SC carry
echo "== 13. 감시 노드: 주행 중 인식 노드를 끔 (감시 정지 → 헛정지 예상)"
DT straight --cond cam --obj none --at 1.5 &
J=$!; sleep 7; kill -INT $NODE; wait $J; SC still
echo "== 14. 메모·표"; DT note 2 "A4로 잰 거리 4.5cm"; DT table
echo "== 기록 파일"; ls $S/drive_tests/*/ | head -30; ls $S/drive_tests/*/02_*/frames | head -3; head -c 400 $S/drive_tests/*/02_*/frames/frames.csv; echo
echo "== 인식 노드 로그 끝"; tail -n 6 $S/node.log
echo "== 감시 노드 로그"; cat $S/watchdog.log
pkill -f mock_robot.py; pkill -f floor_watchdog.py
echo "== 끝"
