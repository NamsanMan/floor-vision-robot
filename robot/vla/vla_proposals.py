#!/usr/bin/env python3
"""VLA 실험 2단계 (메인컴, conda env mot = 로봇과 같은 ONNX Runtime, 2026-10-31):
우리 바닥 분할(DS-Net INT8)로 '바닥에 닿은 물체 덩어리'를 찾아 후보로 낸다. 열마다 바닥이 끊기는 첫 점(인식 노드와 같은 규칙)을
바닥 투영(IPM)으로 로봇 기준 좌표로 바꾸고, 붙어 있는 열끼리 묶어 덩어리마다 접지점과 자를 영역(물체 높이 가정)을 계산.
usage: python vla_proposals.py snap.jpg [--model ../offload/models/ft_240x320_int8_gelu.onnx] [--out proposals.json]"""
import argparse
import json
import math
import os

import cv2
import numpy as np
import onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def imread_u(path):
    """Windows OpenCV는 한글 경로를 못 열어서 바이트로 읽어 디코드(2026-10-31)"""
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(path, img):
    cv2.imencode(os.path.splitext(path)[1], img)[1].tofile(path)


class FloorGeometry:
    """floor_seg_node.py의 같은 이름 클래스와 같은 수식(ROS 없이 쓰려고 복사)"""

    def __init__(self, fx=590.77, fy=590.51, cx=333.39, cy=232.73, h=0.106, pitch_deg=2.45, roll_deg=0.0, cam_x=0.061):
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        self.Kinv = np.linalg.inv(self.K)
        p, r = math.radians(pitch_deg), math.radians(roll_deg)
        n = np.array([math.sin(r) * math.cos(p), math.cos(r) * math.cos(p), math.sin(p)])
        fwd = np.array([0, 0, 1.0]) - n[2] * n
        fwd /= np.linalg.norm(fwd)
        left = -np.cross(n, fwd)
        left = left if left[0] < 0 else -left
        self.n, self.fwd, self.left, self.h, self.cam_x = n, fwd, left, h, cam_x
        self.foot = h * n

    def pixel_to_base(self, u, v):
        r = self.Kinv @ np.stack([u, v, np.ones_like(u)]).astype(np.float64)
        den = self.n @ r
        ok = den > 1e-6
        X = (self.h / np.where(ok, den, 1.0)) * r
        d = X - self.foot[:, None]
        return self.cam_x + self.fwd @ d, self.left @ d, ok


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model", default=os.path.join(here, "..", "offload", "models", "ft_240x320_int8_gelu.onnx"))
    ap.add_argument("--col-step", type=int, default=2)
    ap.add_argument("--max-range", type=float, default=4.0)
    ap.add_argument("--obj-height", type=float, default=0.22, help="자를 영역 높이로 가정할 물체 높이(m). 낮은 장애물 기준")
    ap.add_argument("--out", default="proposals.json")
    a = ap.parse_args()
    bgr = imread_u(a.image)
    H, W = bgr.shape[:2]
    sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    x = ((cv2.cvtColor(cv2.resize(bgr, (320, 240), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
          - MEAN) / STD).transpose(2, 0, 1)[None]
    lg = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    floor = cv2.resize(lg[1] - lg[0], (W, H), interpolation=cv2.INTER_LINEAR) > 0
    cols = np.arange(0, W, a.col_step)
    nf = ~floor[:, cols]
    run = nf.copy()
    for k in range(1, 3):                                      # 3픽셀 연속 비바닥(인식 노드와 같음)
        run[k:] &= nf[:-k]
    run[:2] = False
    idx = run[::-1].argmax(0)
    found = run[::-1][idx, np.arange(len(cols))]
    rows = np.where(found, H - 1 - idx, -1)
    geo = FloorGeometry()
    bx, by, ok = geo.pixel_to_base(cols.astype(float), np.maximum(rows, 0).astype(float))
    valid = found & ok & (bx - geo.cam_x < a.max_range) & (rows > 0)
    # 붙어 있는 열끼리 묶기: 다음 열의 접지점이 가까우면 같은 덩어리
    segs, cur = [], []
    for i in range(len(cols)):
        if not valid[i]:
            if cur:
                segs.append(cur)
            cur = []
            continue
        if cur:
            j = cur[-1]
            gap = math.hypot(bx[i] - bx[j], by[i] - by[j])
            if gap > 0.10 + 0.05 * bx[i]:
                segs.append(cur)
                cur = []
        cur.append(i)
    if cur:
        segs.append(cur)
    props = []
    for sg in segs:
        if len(sg) < 3:
            continue
        u0, u1 = int(cols[sg[0]]), int(cols[sg[-1]])
        vb = int(np.max(rows[sg]))
        x_med, y_med = float(np.median(bx[sg])), float(np.median(by[sg]))
        zc = max(0.15, x_med - geo.cam_x)
        hp = int(geo.K[1, 1] * a.obj_height / zc)
        pad = int(0.15 * (u1 - u0)) + 6
        box = [max(0, u0 - pad), max(0, vb - hp), min(W - 1, u1 + pad), min(H - 1, vb + 8)]
        if box[2] - box[0] < 16 or box[3] - box[1] < 16:
            continue
        props.append({"cols": [u0, u1], "contact_row": vb, "base_xy": [round(x_med, 3), round(y_med, 3)],
                      "nearest_x": round(float(np.min(bx[sg])), 3), "box": box})
    json.dump({"image": os.path.abspath(a.image), "proposals": props}, open(a.out, "w"), indent=1)
    vis = bgr.copy()
    vis[floor] = (0.6 * vis[floor] + 0.4 * np.array([60, 200, 60])).astype(np.uint8)
    for k, p in enumerate(props):
        b = p["box"]
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 200, 255), 2)
        cv2.putText(vis, f"{k}: {p['base_xy'][0]:.2f}m", (b[0], max(12, b[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    imwrite_u(os.path.splitext(a.out)[0] + "_vis.jpg", vis)
    print(f"후보 {len(props)}개: " + ", ".join(f"{k}({p['base_xy'][0]:.2f}m, {p['base_xy'][1]:+.2f})" for k, p in enumerate(props)))


if __name__ == "__main__":
    main()
