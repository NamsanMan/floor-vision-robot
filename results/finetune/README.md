# Fine-tuning 결과

평가는 학습에 안 쓴 자리에서 했다. 촬영 자리 네 곳(A\~D)을 나눠 한 곳을 빼고 학습하는 4-fold cross-validation을 시드 3개로 돌렸다. 이와 별도로 A\~C로만 학습한 모델을 D(spot4)에서 채점한 것이 test set 결과다. 평가 시의 해상도는 라벨과 같은 480×640이다.
CamVid로 pretrain한 내 모델의 weight에서 시작해 fine-tuning했다.

## 바닥 IoU

| | 360×480 입력 | 240×320 입력 (로봇 설정) |
|---|---|---|
| Fine-tuning 전 (CamVid 차도+인도를 바닥으로 봄) | 0.703 | 0.834 |
| Fine-tuning, 자리별 4-fold cross-validation 평균 | **0.962** ± 0.029 | **0.953** ± 0.030 |
| Fine-tuning, D(spot4) test set | 0.982 | 0.974 |

Fine-tuning 전 모델도 바닥은 대체로 찾지만 물체의 밑부분을 바닥으로 칠하는 일이 많았다. Fine-tuning 뒤에도 가장 어려운 자리는 A였다(0.919). 장애물 쪽으로 직진하며 흰 책에 바짝 다가간 장면이 많은 곳인데, IoU가 가장 낮은 프레임(0.33)도 흰 책이 화면 아래를 채운 장면이다.

## 접지선 평가

바닥 IoU가 높다고 해도 물체와 바닥이 만나게 되는 접지선(ground contact line)까지 인식이 정확하다는 보장이 없다. 주행에서 중요한 것은 "바닥과 물체가 만나는 정확한 지점"이다. 이는 현재 화면의 column 방향으로 물체의 거리를 재는 방식을 이용했기 때문이다. 따라서 해당 부분을 따로 측정했다(spot4, 물체 51개).

| | 접지선 recall | 거리 오차 중앙값 | p90 | unsafe column |
|---|---|---|---|---|
| Fine-tuning 360×480 (시드 3개) | 0.90~0.96 | 1.2 cm | 13 cm | 4.3~5.3% |
| Fine-tuning 240×320 (시드 3개) | 0.94~0.96 | 1.9~2.5 cm | 15 cm | 2.8~4.5% |
| Fine-tuning 전 360×480 | 0.65 | 10.1 cm | 131 cm | 7.9% |
| Fine-tuning 전 240×320 | 0.71 | 7.2 cm | 95 cm | 9.3% |
| EoMT-L을 이용한 auto-label (참고) | 0.73 | | | 6.2% |

- 접지선 recall: 물체마다 column별로 맨 아래 8픽셀을 보고, 그 절반 이상을 '바닥 아님'으로 맞히면 잡은 것으로 셌다.
- 거리 오차: column마다 바닥이 끊기는 첫 줄을 카메라 보정으로 앞 거리로 바꿔 정답과 비교했다(정답이 2.5 m 안쪽인 column).
- unsafe column: 실제보다 10 cm 넘게 멀리까지 바닥이라고 본 column.

EoMT-L은 foundation model(DINOv2) 기반의 segmentation 모델이다. 이를 이용한 auto-label은 케이블 12개 중 5개, 신발 5개 중 1개만 잡았다. auto-labeling 후 직접 손으로 고친 라벨로 학습한 3.7M 모델은 케이블 12/12, 신발 4~5/5를 잡았다.

## Pi 5에 올리기 (240×320, ONNX Runtime 1.23.2 CPU, 2스레드)

| | 지연 평균 / p95 | 바닥 IoU (spot4) | 크기 |
|---|---|---|---|
| FP32 | 141.5 / 161 ms | 0.975 | 14.3 MB |
| INT8 | 103.4 / 118 ms | | |
| INT8 + GELU fusion | **83.8 / 94 ms** | 0.977 | 4.2 MB |

INT8은 정적 양자화(QDQ, per-channel weight quantization, percentile calibration on 48 images)다. ONNX Runtime이 양자화된 그래프에서는 GELU를 자동으로 묶지 못해서 직접 묶었다. FP32와 픽셀 단위로 99.75% 같다. 카메라의 fps는 약 12.5이고, 모델은 한 장 처리에 84 ms(초당 약 12장)가 걸려 둘이 속도는 비슷했다. 로봇에서는 Nav2와 CPU를 나눠 쓰려고 처리를 초당 5장으로 제한했다.

## 파일

- `summary.json`, `baselines.json`, `contact_metrics.json`, `cv_plan.json`: 위 표의 원자료
- `runs/`: 학습 회차마다 `metrics.json`, `train_log.csv`
- `ft_240x320_deploy.json`: 배포 측정
- `figures/`: cross-validation 예측, spot4 예측
- `robot_scene/`: 로봇에 올린 뒤의 화면(경계 point)과 RViz에서 본 코스트맵
