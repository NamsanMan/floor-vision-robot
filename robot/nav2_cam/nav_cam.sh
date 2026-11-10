#!/bin/bash
# 카메라 연결 Nav2 시작/정지 (로봇 호스트 pi에서 실행). 2026-10-29
#   bash /home/pi/docker/tmp/nav2_cam/nav_cam.sh start [지도이름=room_v2] | stop | status
# start: 기본 구동(start_node.service) 정지 → 컨테이너 ROS 정리 → nav_cam.launch.py (카메라·LiDAR·모터 + Nav2 + Collision Monitor)
# stop : Nav2 정리 → 기본 구동 다시 시작. 재시작 때 컨트롤러 보드 초기화 부저가 한 번 울린다.
# 바닥 분할 노드(floor_seg)는 명령줄에 'ros'가 없어 .stop_ros.sh에 안 죽고, 카메라가 다시 뜨면 이어서 받는다.
H=/home/pi/docker/tmp/nav2_cam
case "$1" in
  start)
    MAP=${2:-room_v2}
    sudo systemctl stop start_node.service
    docker exec -u ubuntu -w /home/ubuntu MentorPi /bin/zsh -c "~/.stop_ros.sh" > /dev/null 2>&1
    sleep 2
    LOG=/home/ubuntu/shared/nav2_cam/logs/nav_$(date +%Y%m%d_%H%M%S).log
    docker exec -u ubuntu -d -w /home/ubuntu MentorPi /bin/zsh -ic "ros2 launch /home/ubuntu/shared/nav2_cam/nav_cam.launch.py map:=$MAP > $LOG 2>&1"
    echo "nav_cam 시작 (지도 $MAP), 로그 $LOG" ;;
  stop)
    docker exec -u ubuntu -w /home/ubuntu MentorPi /bin/zsh -c "~/.stop_ros.sh" > /dev/null 2>&1
    sleep 2
    sudo systemctl start start_node.service
    echo "nav_cam 정지, 기본 구동 다시 시작" ;;
  status)
    docker exec -u ubuntu MentorPi /bin/zsh -ic "ros2 lifecycle get /collision_monitor; ros2 lifecycle get /local_costmap/local_costmap" 2>/dev/null | grep -v -E "^\s*$|環境|environment|---|LIDAR|CAMERA|MACHINE|HOST|MASTER|ROS_DOMAIN|ASR|VERSION" ;;
  *) echo "사용법: $0 start [지도] | stop | status" ;;
esac
