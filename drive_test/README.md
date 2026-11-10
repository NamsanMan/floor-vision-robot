# drive_test

주행 test를 매번 같은 절차로 돌리고 기록하려고 만든 도구다. 로봇 컨테이너 안에서 돌고, `dt.sh`가 호스트에서 docker exec로 띄운다.

| 명령 | |
|---|---|
| `cond cam / lidar / cam_nomem` | 인식 노드 조건 바꾸기. LiDAR only는 출력만 비우고, blind-spot memory를 끄면 지금 보이는 point만 쓴다 |
| `pose save / reset / click` | 출발 위치 저장, AMCL 재설정(들어갔는지 확인하고 안 됐으면 다시 보냄), RViz에서 찍은 point를 목적지로 |
| `straight --cond cam --obj book` | Collision Monitor만 켜고 초속 10 cm로 직진. 정지·충돌을 판정 |
| `goal --cond cam [--to-click]` | Nav2 목표 주행. 도착·포기·시간 초과 |
| `idle --secs 10 --tag light_off` | 제자리 기록 (조명 test, PC 추론 지연 측정) |
| `table` | 회차 결과표 |

회차마다 `log.csv`(0.05초 간격 위치·명령·정면 거리)와 `events.json`(Collision Monitor·Nav2 로그 요약, 감시 신호, 지연, CPU)을 남긴다. `--no-frames`를 빼면 처리한 프레임도 JPEG로 저장한다.

바퀴 odom이 이동 거리를 실제보다 18% 길게 재서, 도구 안에서는 이동 거리에 0.85를 곱해 쓴다(첫 주행 날 실측한 값).

`test_sim/`은 로봇 없이 WSL에서 도구를 먼저 돌려 보려고 만든 모의 로봇(mock)이다. 여기서 로봇이 서 있는 동안에도 blind-spot memory가 만료돼 정지가 풀리는 버그를 찾았고, 그래서 유지 시간을 움직인 시간으로만 세게 바꿨다.
