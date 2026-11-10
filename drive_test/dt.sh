#!/bin/bash
# 주행 시험 명령 (로봇 호스트 pi에서 실행, 2026-10-30): bash /home/pi/docker/tmp/drive_test/dt.sh <명령> [옵션]
#   예: dt.sh cond cam | dt.sh pose save | dt.sh straight --cond cam --obj book | dt.sh goal --cond lidar --obj book | dt.sh stop
#   명령·옵션 설명은 drive_test.py 첫머리. 컨테이너 안에서 ROS 환경을 잡고 실행한다.
#   인자는 "$@"로 그대로 넘김(변수에 명령을 담으면 호스트 zsh가 단어를 안 쪼갬, 2026-10-29).
exec docker exec -u ubuntu -e PYTHONIOENCODING=utf-8 MentorPi bash -c \
  'source /opt/ros/humble/setup.bash; exec python3 -u /home/ubuntu/shared/drive_test/drive_test.py "$@"' drive_test "$@"
