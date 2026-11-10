"""가짜 onnxruntime — 모의 시험(WSL) 전용, 로봇에 올리지 말 것 (2026-10-30).
입력 영상에서 빨간 부분 = 바닥 아님, 나머지 = 바닥. 출력 모양은 진짜 모델과 같은 (1, 2, H/4, W/4) 로짓."""
import time

import numpy as np

__version__ = "fake-for-sim"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


class SessionOptions:
    intra_op_num_threads = 1
    inter_op_num_threads = 1


class _Input:
    name = "input"


class InferenceSession:
    def __init__(self, path, sess_options=None, providers=None):
        self.path = path

    def get_inputs(self):
        return [_Input()]

    def run(self, output_names, feed):
        x = next(iter(feed.values()))[0] * STD + MEAN              # (3, H, W) RGB 0~1
        red = (x[0] > 0.6) & (x[1] < 0.35) & (x[2] < 0.35)
        h, w = red.shape
        red4 = red.reshape(h // 4, 4, w // 4, 4).mean((1, 3)) > 0.5
        lg1 = np.where(red4, -3.0, 3.0).astype(np.float32)
        time.sleep(0.06)                                           # Pi 5 추론 시간 흉내
        return [np.stack([-lg1, lg1])[None]]
