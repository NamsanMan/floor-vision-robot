#!/bin/bash
# 바닥 분할 인식 노드 + 감시 노드 시작/정지 (로봇 컨테이너 ubuntu 사용자). 호스트에서:
#   docker exec -u ubuntu MentorPi bash /home/ubuntu/shared/floor_seg/run_floor_seg.sh start [ros-args...] | stop | status
#   (ros-args는 인식 노드에만, 예: -p max_rate:=0.0 -p threads:=2). 감시 노드만 끄기/켜기: stop_node / start_node (감시 시험용)
# onnxruntime은 ~/floorseg_pkgs(bench_venv에서 onnxruntime 패키지만 복사)로 넣는다. numpy는 ~/.local의 1.24.3을 그대로 씀.
D=/home/ubuntu/shared/floor_seg
env_setup() {
  source /opt/ros/humble/setup.bash
  [ -f ~/ros2_ws/install/setup.bash ] && source ~/ros2_ws/install/setup.bash
  export PYTHONPATH=$HOME/floorseg_pkgs:$PYTHONPATH
  mkdir -p $D/logs $D/debug
}
alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
start_one() {   # $1 = 이름(node|watchdog), $2 = 스크립트, 나머지 = ros-args
  local name=$1 script=$2; shift 2
  if alive $D/$name.pid; then echo "$name 이미 실행 중 pid $(cat $D/$name.pid)"; return; fi
  local log=$D/logs/${name}_$(date +%Y%m%d_%H%M%S).log
  nohup python3 $D/$script ${1:+--ros-args} "$@" > $log 2>&1 < /dev/null &
  echo $! > $D/$name.pid
  echo "$name 시작 pid $(cat $D/$name.pid) 로그 $log"
}
stop_one() {
  if alive $D/$1.pid; then kill -INT "$(cat $D/$1.pid)"; sleep 2; kill "$(cat $D/$1.pid)" 2>/dev/null; echo "$1 정지"; else echo "$1 실행 중 아님"; fi
  rm -f $D/$1.pid
}
case "$1" in
  start) env_setup; start_one node floor_seg_node.py "${@:2}"; start_one watchdog floor_watchdog.py ;;
  stop) stop_one node; stop_one watchdog ;;
  start_node) env_setup; start_one node floor_seg_node.py "${@:2}" ;;
  stop_node) stop_one node ;;
  status)
    for n in node watchdog; do
      if alive $D/$n.pid; then echo "$n 실행 중 pid $(cat $D/$n.pid)"; tail -n 2 "$(ls -t $D/logs/${n}_*.log 2>/dev/null | head -1)"; else echo "$n 실행 중 아님"; fi
    done ;;
  *) echo "사용법: $0 start [ros-args] | stop | start_node | stop_node | status" ;;
esac
