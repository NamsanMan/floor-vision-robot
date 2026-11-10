#!/usr/bin/env python3
"""주행 시험 회차들의 경로(AMCL map 좌표)를 지도 위에 겹쳐 그린다 (메인컴, 2026-10-31).
usage: python plot_drive_paths.py <지도.pgm> <회차 폴더 상위> --runs 07:red:라벨 09:green:라벨 ... [--out 그림.png] [--title 글]
       색은 red/green/blue/orange/purple. 각 회차 폴더의 log.csv(map_x, map_y)와 events.json(목표)을 읽는다."""
import argparse
import csv
import glob
import json
import os

import cv2
import numpy as np

COL = {"red": (40, 40, 220), "green": (40, 160, 40), "blue": (220, 120, 30), "orange": (0, 140, 255), "purple": (160, 60, 160)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgm")
    ap.add_argument("runs_dir")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--origin", type=float, nargs=2, default=[-2.62, -1.42])
    ap.add_argument("--res", type=float, default=0.05)
    ap.add_argument("--out", default="paths.png")
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    im = cv2.imread(a.pgm, cv2.IMREAD_UNCHANGED)
    h, w = im.shape
    S = 8
    big = cv2.cvtColor(cv2.resize(im, (w * S, h * S), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    ox, oy = a.origin

    def px(x, y):
        return int((x - ox) / a.res * S), int((h - (y - oy) / a.res) * S)

    y_text = 22
    if a.title:
        cv2.putText(big, a.title, (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
        y_text += 26
    for spec in a.runs:
        num, color, label = (spec.split(":", 2) + [""])[:3]
        d = glob.glob(os.path.join(a.runs_dir, f"{num}_*"))[0]
        rows = [r for r in csv.DictReader(open(os.path.join(d, "log.csv"), encoding="utf-8")) if r.get("map_x")]
        P = np.array([px(float(r["map_x"]), float(r["map_y"])) for r in rows], np.int32)
        c = COL[color]
        if len(P) > 1:
            cv2.polylines(big, [P], False, c, 3)
            cv2.circle(big, tuple(P[0]), 7, c, 2)
            cv2.drawMarker(big, tuple(P[-1]), c, cv2.MARKER_TILTED_CROSS, 16, 3)
        ev = json.load(open(os.path.join(d, "events.json"), encoding="utf-8"))
        g = ev.get("extra", {}).get("goal_map")
        if g:
            cv2.drawMarker(big, px(*g), (0, 0, 0), cv2.MARKER_STAR, 22, 2)
        res = {"도착": "arrived", "시간 초과": "timeout (stuck)", "Nav2가 포기(aborted)": "aborted", "취소됨": "canceled"}.get(
            ev.get("result", ""), "")                                   # OpenCV 글꼴은 한글을 못 그림
        cv2.putText(big, f"{label} - {res}" if label else res, (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
        y_text += 24
    cv2.putText(big, "o start   x end   * goal", (10, h * S - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
    cv2.imwrite(a.out, big)
    print(f"그림 저장: {a.out}")


if __name__ == "__main__":
    main()
