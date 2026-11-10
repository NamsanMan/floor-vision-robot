# YOLO11n을 Pi 5에서 돌려 보기

강의 실습(YOLO11 + NCNN)을 하면서 속도와 정확도를 직접 측정하였다. 로봇 카메라 시점의 낮은 물체를 COCO detection model이 거의 못 잡는 걸 보고 floor segmentation으로 방향을 잡게 되었다.

## 속도 (모델만, 480×640 입력, 중앙값)

| PyTorch | ONNX Runtime | NCNN |
|---|---|---|
| 348 ms | 134 ms | 64 ms |

전처리와 후처리까지 포함한 전체 시간은 PyTorch에서 428 ms, NCNN에서 70 ms로 6.1배 차이다. NCNN이 빠른 이유는 대부분 FP16이었다. Pi 5(Cortex-A76)에서는 ncnn 기본 옵션이 FP16 저장과 연산을 켜기 때문에, FP32로 내보낸 모델도 실제로는 FP16으로 돈다. FP16을 끄면 141~169 ms로 ONNX Runtime FP32와 비슷해진다.

## FP16이 정확도에 미치는 영향 (COCO val2017 5,000장, 640×640)

NCNN FP32 mAP 39.12, FP16 39.08로 0.04점 차이였고, 지연은 238 ms에서 118 ms로 줄었다. 성능 저하가 거의 없이 속도를 두 배로 높일 수 있다.

## ROS로 돌렸을 때

강의 설정(영상 구독의 QoS를 best effort로)에서는 detection이 초당 4.0번에 그쳤다. 카메라는 초당 12.8장을 내는데, 0.9 MB짜리 원본 영상이 best effort subscriber에게 가는 도중 많이 버려졌다. reliable로 바꾸자 8.3번이 됐다(PyTorch는 2.2번). 그래서 이 로봇에서는 원본 영상을 reliable로 구독하고, 최신 1장만 받게 했다(depth 1).

## 낮은 물체 감지에서 발생한 문제

바닥에서 10 cm 높이 카메라로 찍은 종이컵 더미는 confidence를 0.01까지 내려도 cup으로 잡히지 않았다(`results/live_ncnn_conf005.jpg`: 낮은 점수로 의자 같은 엉뚱한 것만 잡힌다). 일주일 뒤 같은 시점에서 다시 봤을 때도 책은 0.20(기본 임계 0.25 아래)이었고, 운동화는 COCO에 클래스가 없었다. LiDAR도 못 보고 detection model도 못 잡는 물체라서, 물체를 알아보는 대신 '바닥이 아닌 곳'을 보기로 했다.

## 파일

`scripts/`: 속도 측정, COCO 평가, NCNN FP16/FP32 비교, ROS 실행 측정, 그림. 설치 스크립트와 강의 launch를 고치는 스크립트는 강의 자료와 얽혀 있어서 넣지 않았다. `figures/`, `results/`(측정 JSON, 로봇 카메라 프레임).
