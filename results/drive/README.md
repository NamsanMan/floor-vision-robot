# 주행 test (2026-10-31)

장애물을 학습 때와 다른 물건과 배치로 놓고 test를 진행하였다. 장애물은 흰 책(바닥과 대비가 낮다), 종이컵, 운동화, 검은 상자이고, 목적지에는 주황 상자를 놓았다. 검은 상자와 주황 상자 말고는 모두 LiDAR 스캔 높이(16 cm)보다 낮다.

멈춘 거리는 측정하지 않고 섰는지 부딪혔는지로 판정했다. 나머지 숫자는 test 도구(`drive_test/`)가 남긴 기록이다.

## 직진 후 정지 test

Collision Monitor만 켜고 초속 10 cm로 1.49 m 앞의 흰 책을 향해 직진했다.

| 회차 | 조건 | 결과 |
|---|---|---|
| 05 | LiDAR only | 책을 못 보고 충돌 |
| 04 | Camera with blind-spot memory | 34 cm 앞부터 감속, 책 앞에서 정지 |
| 06 | Camera without blind-spot memory | 감속은 했지만 책이 사각지대로 들어가자 point가 사라져 충돌 |

회차 01은 장애물 없이 달리다가 바닥에 남겨 둔 A4 종이에 섰다. Floor segmentation은 물체의 높이를 모르니 납작한 물체에도 선다. 종이를 치운 02는 끝까지 헛정지가 없었다. 03은 무효다. 바퀴 odom이 실제 이동 거리보다 18% 길게 재는 바람에, test 도구가 책에 닿기 전에 충돌로 판정하고 끝냈다(실제로는 20 cm 남아 있었다). 그 뒤로는 odom 거리에 0.85를 곱해 판정했다.

## 목표 주행 test

Nav2(TEB), 최고 초속 25 cm.

| 회차 | 조건 | 결과 |
|---|---|---|
| 07 | LiDAR only, 2.5 m 앞 | 14초 만에 도착. 컵과 책을 밀고 일직선으로 감 |
| 08 | Camera with blind-spot memory | 장애물 앞에서 서고 못 돌아감, 60초 시간 초과 |
| 09 | Camera with blind-spot memory | 40초 만에 도착, 옆으로 59 cm 돌아감 |
| 10 | LiDAR only, 주황 상자 앞 | 23초 만에 도착, 컵과 책을 밀고 감 |
| 11 | Camera with blind-spot memory | 장애물 앞에서 서고 못 돌아감, 90초 시간 초과 |
| 12 | Camera with blind-spot memory | 41초 만에 도착, 66 cm 돌아감 |
| 14 | Camera with blind-spot memory | 1.07 m 돌아갔지만 목적지 앞 구석에서 Nav2가 포기 |
| 15 | Camera with blind-spot memory, PC 추론 | 장애물을 돌아갔지만 목적지 앞 구석에서 Nav2가 포기 |

카메라 조건은 한 번도 장애물과 부딪히지 않았다. 08과 11이 못 돌아간 이유는 나중에 찾았다. Blind-spot memory point cloud(odom 좌표)의 관측 소스에 `sensor_frame`이 없어서, Nav2가 거리 제한(2 m)을 로봇이 아니라 odom 원점에서 쟀다. 로봇을 손으로 출발선에 되돌려도 odom은 줄지 않으니, 회차가 쌓일수록 장애물이 범위 밖으로 밀려나 코스트맵에서 빠졌다. 출발할 때 odom 원점과의 거리가 09는 0 m였고 08은 2.71 m, 11은 3.52 m였다. 경로 제어기는 계속 앞으로 가려 하고 Collision Monitor는 계속 막는 deadlock이었다.

`sensor_frame: ascamera_color_0`을 넣고 11과 같은 조건(원점에서 3.36 m)으로 다시 달린 14에서는 deadlock 없이 장애물을 돌아갔다. 다만 주황 상자 앞이 빽빽해서 마지막 구간에서 TEB와 global planner가 길을 못 찾았다. 13은 무효다(목표 주행 명령 안의 AMCL 재설정이 반영되지 않아 바로 도착으로 끝남).

그림: `paths_goal_2p5m.png`(07과 09), `paths_orange_box.png`(10과 11).

15·16회차는 PC 추론 주행과 연결 끊김 test라 자세한 내용은 [offload](../offload)에, 17·18회차는 [vla](../vla)에 정리했다.

## 다가가는 동안 판정의 불안정성 (라벨 없이)

직진 03·04·05에서 저장한 프레임마다 "지금까지 간 거리 + 카메라가 잰 정면 거리"로 가만히 있는 책의 위치를 추정하고, 그 값이 얼마나 흔들리는지 봤다(`approach_stability.png`, `approach_stability.json`, `analysis/approach_stability.py`).

| 로봇 중심~책 | 1.30~1.51 m | 1.00~1.30 m | 0.60~1.00 m | 0.32~0.60 m |
|---|---|---|---|---|
| 놓친 프레임 | 70% | 5% | 0% | 0% |

- 처음 확실히 잡히는 거리(reliable detection range)는 약 1.3 m다. Blind-spot memory는 사각지대(로봇 중심에서 0.29 m)에 들어가기 전에 한 번만 보면 되니 여유가 1 m쯤 있다.
- 1 m 넘게 떨어져 있을 때는 추정이 0~28 cm 멀게 흔들리다가, 0.95 m 안쪽부터 ±5 cm 안으로 모인다. 먼 거리의 바닥 투영 해상도와 흰 책 접지선의 낮은 대비 때문이다.
- 0.8 m 안에서의 흔들림(표준편차)은 1.7~2.3 cm였고, 빈 바닥 74프레임에서 헛검출은 없었다.

## 그 밖에

- 주행 중에 프레임을 JPEG로 저장하면 인식 노드의 메인 스레드가 98%까지 차서, 목표 주행에서는 저장을 껐다.
- 명령을 띄울 때마다 DDS discovery로 순간 부하가 생겨 출발 전에 3초 기다리게 했다.
- 로봇을 들어 옮기면 AMCL 위치가 지도 밖으로 밀려나서, 스캔을 지도에 맞춰 위치를 다시 잡았다(`amcl_lost_scan_match.png`, `robot/tools/scan_match_init.py`).
- RViz를 붙일 때마다 Nav2가 과부하로 무너져서 주행 중에는 RViz를 쓰지 않았다.

## 파일

`runs/`: 회차별 `log.csv`, `events.json`과 전체 `results.csv`. `map/`: test한 방 지도. 프레임 JPEG는 넣지 않았다.
