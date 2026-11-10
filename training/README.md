# training

내가 연구에서 제안한 경량 segmentation 모델 DS-Net(SegFormer MiT-B0 구조, 약 3.7M 파라미터, CamVid 학습)을 바닥에서 10 cm 높이 카메라 시점에 맞게 2클래스(바닥 / 바닥 아님)로 fine-tuning했다. 학습은 RTX 3060이 있는 PC에서 했다.

## 데이터

연구실 통로 네 곳에 장애물(책, 검은 상자, 운동화, 컵, 주황 상자 등)을 놓고 로봇 카메라(640×480)로 찍었다. 자리마다 제자리에서 한 바퀴 돌고, 장애물 쪽으로 몇 번 직진했다.

150장을 라벨링했다. 학습 120장(spot1~3 회전 92장, 직진 28장), test 30장(spot4, 학습에 안 쓴 자리)이다. EoMT-L(DINOv2 기반 segmentation 모델)이 칠한 바닥 auto-label을 labelme에서 손으로 고치는 식으로 했고, 규칙은 이렇게 정했다.

- 라벨이 겹치는 경우에는 물체 > ignore > 바닥 순으로 우선순위를 매겼다. 칠하지 않은 곳은 바닥 아님이다.
- LiDAR가 놓칠 만한 바닥 위 낮은 물건(책, 신발, 케이블, 컵, 낮은 상자, 의자·책상 다리)을 물체로 그렸다.
- 물체가 바닥에 닿는 아래쪽 선은 특히 정확히 그렸다.

auto-label이 자주 틀린 곳은 바닥 색과 비슷한 흰 책, 바닥에 누운 검은 케이블, 가까운 흰 신발 밑창이었다.

데이터셋과 weight는 저장소에 없다. 연구실 내부 사진이고, weight는 투고 중인 논문의 모델을 이용했기 때문이다.

## 학습 설정

- 초기 weight: 논문 DS-Net CamVid 체크포인트 전체. Classifier layer만 2클래스 1×1 conv로 새로 만든다.
- 입력 360×480.
- Data augmentation: 좌우 반전, 밝기·대비·채도 ±0.3, 크기 0.75~1.25배.
- cross-entropy(ignore 255), AdamW(인코더 6e-5, 디코더 6e-4, weight decay 0.01), poly, batch 8, 60 epoch, AMP.
- 마지막 epoch weight를 그대로 평가한다. 한 번 학습(60 epoch)에 60~77초.

## 파일

| 파일 | |
|---|---|
| `dsnet_model.py` | HuggingFace SegFormer(mit-b0 설정)로 DS-Net 구조를 만들고 논문 체크포인트를 불러온다. transformers 4.x/5.x 모듈 이름 차이를 변환한다 |
| `ft_train.py`, `cv.py` | 학습, 촬영 자리별 cross-validation |
| `baselines.py` | Fine-tuning 전 모델(CamVid 차도·인도를 바닥으로), auto-label과 비교 |
| `contact_metrics.py` | 접지선 recall, column별 주행 가능 거리 오차 |
| `deploy_int8.py`, `run_ft_bench.sh` | ONNX 변환, INT8 양자화와 GELU fusion, Pi 5 속도 측정 |
| `summarize.py`, `make_figure.py` | 표와 그림 |
| `run_cv.sh`, `sync_labels_and_run.sh`, `fetch_results.sh` | 라벨을 학습 PC로 보내 cross-validation을 한 번에 돌리고 결과를 가져오기 |
| `labeling/` | labelme JSON → 마스크(0 바닥 아님, 1 바닥, 255 ignore), 진행 확인 |

결과는 [results/finetune](../results/finetune)에 있다.
