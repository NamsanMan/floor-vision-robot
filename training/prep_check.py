"""전처리 차이 확인 (서브컴): 학습·평가는 PIL bilinear 축소 + torch 업샘플, 로봇 ROS 노드는 OpenCV를 쓴다.
OpenCV 축소(AREA·LINEAR) + OpenCV 업샘플로 바꿨을 때 spot4 test 지표가 달라지는지 INT8+GELU ONNX로 비교.
usage: ~/mentorpi/venv/bin/python prep_check.py   (~/mentorpi/code/finetune 에서)"""
import glob
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG_ROOT, eval_input, load_rgb  # noqa: E402
from contact_metrics import evaluate, load_gt  # noqa: E402

M = os.path.expanduser("~/mentorpi/finetune/deploy/ft_240x320_int8_gelu.onnx")
sess = ort.InferenceSession(M, providers=["CPUExecutionProvider"])
names = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(IMG_ROOT, "test", "*.jpg")))
gt, obj, objn = load_gt(os.path.expanduser("~/mentorpi/data/ft150_user/snap_20261029_1419/masks"), names)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def cv_prep(path, interp):
    rgb = cv2.cvtColor(cv2.resize(cv2.imread(path), (320, 240), interpolation=interp), cv2.COLOR_BGR2RGB)
    return ((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]


def floor_of(x, up):
    lg = sess.run(None, {"input": x})[0]
    if up == "torch":
        return F.interpolate(torch.from_numpy(lg), size=(480, 640), mode="bilinear", align_corners=False)[0].argmax(0).numpy() == 1
    return cv2.resize(lg[0, 1] - lg[0, 0], (640, 480), interpolation=cv2.INTER_LINEAR) > 0


path = lambda n: os.path.join(IMG_ROOT, "test", n + ".jpg")  # noqa: E731
ref = {n: floor_of(eval_input(load_rgb(path(n)), (240, 320)).unsqueeze(0).numpy(), "torch") for n in names}
for tag, interp in (("PIL bilinear + torch up", None), ("cv2 AREA + cv2 up", cv2.INTER_AREA), ("cv2 LINEAR + cv2 up", cv2.INTER_LINEAR)):
    pr = ref if interp is None else {n: floor_of(cv_prep(path(n), interp), "cv") for n in names}
    r = evaluate(pr, gt, obj, objn, names)
    agree = float(np.mean([(pr[n] == ref[n]).mean() for n in names]))
    print(f"{tag:<26} agree_vs_PIL {agree:.4f}  floorIoU {r['floor_iou']}  contact {r['contact_recall@50']}  "
          f"err_med {r['freespace_abs_err_cm_median']}cm  p90 {r['freespace_abs_err_cm_p90']}cm  danger {r['danger_cols_over10cm']}")
