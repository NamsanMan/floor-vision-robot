#!/usr/bin/env python3
"""주행 중 바닥 분할 판정의 프레임 간 흔들림 — 라벨 없이 (메인컴, 2026-10-31).
가만히 있는 장애물(출발선에서 앞쪽 끝까지 at m)을 향해 직진한 회차들의 frames.csv(프레임마다 촬영 시각 odom 위치 + 정면 거리)로,
매 프레임 "로봇이 간 거리 + 카메라가 잰 거리" = 장애물 위치 추정을 만들고 그것이 얼마나 흔들리는지, 보여야 할 때 몇 번 놓쳤는지 센다.
odom은 2026-10-31 실측으로 18% 많이 세므로 × odom_scale로 보정. 장애물 없는 회차는 헛검출 비율만 센다.
usage: python approach_stability.py <회차 상위 폴더> --obstacle 03 04 05 --empty 02 [--at 1.49] [--out approach_stability.png]"""
import argparse
import csv
import glob
import json
import math
import os

import cv2
import numpy as np

FRONT_X, NEAR_VISIBLE, MAX_SEEN = 0.11, 0.30, 1.56    # 로봇 앞면, 화면 맨 아래 줄 바닥(base x), 인식 범위 끝(base x)


def load(d):
    rows = list(csv.DictReader(open(os.path.join(d, "frames", "frames.csv"), encoding="utf-8")))
    ev = json.load(open(os.path.join(d, "events.json"), encoding="utf-8"))
    return rows, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("--obstacle", nargs="+", default=[])
    ap.add_argument("--empty", nargs="+", default=[])
    ap.add_argument("--at", type=float, default=1.49)
    ap.add_argument("--odom-scale", type=float, default=0.85)
    ap.add_argument("--out", default="approach_stability.png")
    a = ap.parse_args()
    true_x = FRONT_X + a.at                                 # 출발 때 로봇 중심에서 장애물 앞 끝까지(m)
    curves, summary = [], []
    for num in a.obstacle:
        d = glob.glob(os.path.join(a.runs_dir, f"{num}_*"))[0]
        rows, ev = load(d)
        x0, y0, yaw0 = (float(rows[0][k]) for k in ("odom_x", "odom_y", "odom_yaw"))
        c, s = math.cos(yaw0), math.sin(yaw0)
        est, gap, miss, seen_should, jit, prev = [], [], 0, 0, [], None
        for r in rows:
            travel = a.odom_scale * (c * (float(r["odom_x"]) - x0) + s * (float(r["odom_y"]) - y0))
            dist_true = true_x - travel                     # 지금 로봇 중심에서 장애물 앞 끝까지(실제, 근사)
            f = float(r["front_m"])
            visible = NEAR_VISIBLE + 0.02 < dist_true < MAX_SEEN - 0.05
            if visible:
                seen_should += 1
                if f >= 9.0:
                    miss += 1
            if f < 9.0 and visible and f > NEAR_VISIBLE:
                e = travel + f                               # 장애물 앞 끝 위치 추정(출발 로봇 중심 기준)
                est.append(e)
                gap.append(dist_true)
                if prev is not None:
                    jit.append(abs((f - prev[1]) + (travel - prev[0])))   # 정지 물체면 '거리 변화 + 이동'이 0이어야 함
                prev = (travel, f)
        est, gap = np.array(est), np.array(gap)
        near = gap < 0.8
        summary.append({"run": num, "cond": ev["args"]["cond"], "frames_should_see": seen_should, "missed": miss,
                        "est_median_m": round(float(np.median(est)), 3), "bias_cm": round(float(np.median(est) - true_x) * 100, 1),
                        "std_cm": round(float(est.std()) * 100, 1), "std_near_cm": round(float(est[near].std()) * 100, 1) if near.sum() > 2 else None,
                        "jitter_p50_cm": round(float(np.median(jit)) * 100, 1) if jit else None,
                        "jitter_p95_cm": round(float(np.percentile(jit, 95)) * 100, 1) if jit else None})
        curves.append((num, gap, est))
    for num in a.empty:
        d = glob.glob(os.path.join(a.runs_dir, f"{num}_*"))[0]
        rows, ev = load(d)
        fp = sum(1 for r in rows if float(r["front_m"]) < 9.0)
        summary.append({"run": num, "cond": ev["args"]["cond"], "empty_frames": len(rows), "false_detections": fp})
    for sm in summary:
        print(sm)
    json.dump({"true_x_m": true_x, "odom_scale": a.odom_scale, "runs": summary}, open(os.path.splitext(a.out)[0] + ".json", "w"), ensure_ascii=False, indent=1)

    # 그림: 가로 = 로봇~장애물 실제 거리, 세로 = 장애물 위치 추정 오차(cm)
    W, H, L, B = 760, 420, 70, 50
    img = np.full((H, W, 3), 255, np.uint8)
    def to_px(g, e_cm):
        return int(L + (1.6 - g) / 1.3 * (W - L - 20)), int(H - B - (e_cm + 30) / 60 * (H - B - 30))
    for e_cm in (-30, -15, 0, 15, 30):
        y = to_px(1.6, e_cm)[1]
        cv2.line(img, (L, y), (W - 20, y), (225, 225, 225) if e_cm else (150, 150, 150), 1)
        cv2.putText(img, f"{e_cm:+d}", (10, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
    for g in (1.5, 1.2, 0.9, 0.6, 0.3):
        x = to_px(g, 0)[0]
        cv2.line(img, (x, 30), (x, H - B), (235, 235, 235), 1)
        cv2.putText(img, f"{g:.1f}", (x - 12, H - B + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
    cols = [(220, 120, 30), (40, 160, 40), (40, 40, 220)]
    for i, (num, gap, est) in enumerate(curves):
        for g, e in zip(gap, est):
            cv2.circle(img, to_px(g, (e - true_x) * 100), 3, cols[i % 3], -1)
        cv2.putText(img, f"run {num}", (W - 110, 50 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cols[i % 3], 2)
    cv2.putText(img, "distance robot center -> obstacle (m)  ->  approaching", (L, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
    cv2.putText(img, "obstacle position estimate error (cm), white book, frame by frame", (L, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
    cv2.imwrite(a.out, img)
    print("그림 저장:", a.out)


if __name__ == "__main__":
    main()
