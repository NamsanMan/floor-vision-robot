#!/usr/bin/env python3
"""LiDAR 스캔 한 장을 지도에 맞춰 로봇의 실제 위치를 찾는다 (메인컴, 2026-10-31).
배경: 직진 시험 때마다 로봇을 들어 출발선으로 되돌렸더니, AMCL은 바퀴 기록만 따라가 지도 밖(x 4.6 m)까지 밀려나 있었다.
방법: 지도의 장애물 칸에서 거리장(likelihood field, σ 8cm)을 만들고, 빈칸 위 모든 위치 × 2도 간격 방향을 전부 시험해
      스캔 점이 장애물 근처에 가장 많이 떨어지는 자세를 고른 뒤 주변을 1cm·0.5도로 다시 찾는다(상관 스캔 매칭).
usage: python scan_match_init.py <지도.pgm> <스캔.json(scan_grab.py)> [--origin -2.62 -1.42] [--res 0.05]
       [--wrong x y yaw도] [--out 그림.png]  → 찾은 위치를 출력하고 best_pose.json, 그림 저장"""
import argparse
import json
import math
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgm")
    ap.add_argument("scan")
    ap.add_argument("--origin", type=float, nargs=2, default=[-2.62, -1.42])
    ap.add_argument("--res", type=float, default=0.05)
    ap.add_argument("--sigma", type=float, default=0.08)
    ap.add_argument("--wrong", type=float, nargs=3, metavar=("X", "Y", "YAW_DEG"), help="비교로 그릴 틀린 위치(예: 그때 AMCL 추정)")
    ap.add_argument("--out", default="scan_match.png")
    a = ap.parse_args()
    im = cv2.imread(a.pgm, cv2.IMREAD_UNCHANGED)
    h, w = im.shape
    res, (ox, oy) = a.res, a.origin
    occ = im == 0
    d = cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 5) * res
    like = np.exp(-(d ** 2) / (2 * a.sigma ** 2))
    s = json.load(open(a.scan))
    r = np.array(s["ranges"])
    ang = s["angle_min"] + np.arange(len(r)) * s["angle_increment"]
    ok = np.isfinite(r) & (r > max(0.12, s["range_min"])) & (r < min(6.0, s["range_max"]))
    P = np.stack([s["lidar_x"] + r[ok] * np.cos(ang[ok] + s["lidar_yaw"]), s["lidar_y"] + r[ok] * np.sin(ang[ok] + s["lidar_yaw"])], 1)

    def score(C, yaw):
        c, si = math.cos(yaw), math.sin(yaw)
        W = C[:, None, :] + (P @ np.array([[c, si], [-si, c]]))[None]
        ix = ((W[..., 0] - ox) / res).astype(int)
        iy = (h - 1 - ((W[..., 1] - oy) / res)).astype(int)
        inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        v = np.zeros(ix.shape)
        v[inside] = like[iy[inside], ix[inside]]
        return v.mean(1)

    fy, fx = np.nonzero(im == 254)
    C = np.stack([ox + (fx + 0.5) * res, oy + (h - fy - 0.5) * res], 1)
    coarse = []
    for yd in np.arange(-180, 180, 2.0):
        sc = score(C, math.radians(yd))
        k = sc.argmax()
        coarse.append((sc[k], C[k, 0], C[k, 1], yd))
    coarse.sort(reverse=True)
    _, x0, y0, yd0 = coarse[0]
    G = np.stack(np.meshgrid(np.arange(x0 - 0.1, x0 + 0.101, 0.01), np.arange(y0 - 0.1, y0 + 0.101, 0.01)), -1).reshape(-1, 2)
    fine = []
    for yd in np.arange(yd0 - 3, yd0 + 3.01, 0.5):
        sc = score(G, math.radians(yd))
        k = sc.argmax()
        fine.append((sc[k], G[k, 0], G[k, 1], yd))
    sb, xb, yb, ydb = max(fine)
    far = next((c for c in coarse if math.hypot(c[1] - xb, c[2] - yb) > 0.5), None)
    print(f"찾은 위치: x={xb:.3f} y={yb:.3f} 방향 {ydb:.1f}도, 점수 {sb:.3f}"
          + (f" (0.5m 넘게 떨어진 다른 후보 최고 {far[0]:.3f})" if far else ""))
    json.dump({"x": xb, "y": yb, "yaw_deg": ydb, "score": sb}, open(os.path.join(os.path.dirname(os.path.abspath(a.out)), "best_pose.json"), "w"))

    # 그림: 지도 + 틀린 위치(빨강)와 찾은 위치(초록)에서 본 스캔. 틀린 위치가 지도 밖이면 캔버스를 넓힘
    S, pad = 8, 0
    if a.wrong:
        pad = max(0, int(((a.wrong[0] + 1.5 - ox) / res - w)))
    base = cv2.copyMakeBorder(im, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=205)
    big = cv2.cvtColor(cv2.resize(base, ((w + pad) * S, h * S), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)

    def to_px(x, y):
        return int((x - ox) / res * S), int((h - (y - oy) / res) * S)

    def draw(x, y, yd, col):
        c, si = math.cos(math.radians(yd)), math.sin(math.radians(yd))
        for wx, wy in P @ np.array([[c, si], [-si, c]]) + [x, y]:
            cv2.circle(big, to_px(wx, wy), 2, col, -1)
        cx, cy = to_px(x, y)
        cv2.circle(big, (cx, cy), 8, col, 2)
        cv2.line(big, (cx, cy), (int(cx + 30 * c), int(cy - 30 * si)), col, 3)

    if a.wrong:
        draw(*a.wrong, (0, 0, 230))
    draw(xb, yb, ydb, (0, 160, 0))
    cv2.putText(big, "green: scan matched to map (true pose)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 0), 2)
    if a.wrong:
        cv2.putText(big, "red: AMCL estimate after carrying the robot back (lost)", (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 210), 2)
    cv2.imwrite(a.out, big)
    print(f"그림 저장: {a.out}")


if __name__ == "__main__":
    main()
