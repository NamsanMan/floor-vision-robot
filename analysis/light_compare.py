#!/usr/bin/env python3
"""실제 조명 열화 시험: 불 켬 vs 불 끔 (메인컴, conda env mot, 2026-10-31).
로봇을 장애물(책·컵·신발) 앞에 세워 두고 같은 자리에서 불 끔 → 불 켬 순서로 10초씩 기록한 두 회차(idle)를 라벨 없이 비교한다.
  (1) 영상 밝기  (2) 정면 거리·잡힌 비율(노드 출력)  (3) 열마다 경계 → 바닥 투영 거리(노드가 frames.csv에 남긴 경계 행)
  (4) 저장 프레임을 같은 INT8 모델로 다시 돌린 바닥 판정: 인식 범위(1.5 m) 안 화면 영역의 바닥 비율, 불 켬 결과 대비 IoU
      (+ 참고: 불 끔 프레임을 소프트웨어로 밝혀서 넣으면 되살아나는지)
usage: python light_compare.py <세션 폴더> --on 02 --off 01 [--out light_compare.png]"""
import argparse
import csv
import glob
import json
import math
import os

import cv2
import matplotlib
import numpy as np
import onnxruntime as ort

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
FX, FY, CX, CY = 590.77, 590.51, 333.39, 232.73        # 노드와 같은 카메라 보정·설치값
CAM_H, PITCH, CAM_X = 0.106, 2.45, 0.061
MAX_RANGE, FRONT_HW, COL_STEP, RUN_PX = 1.5, 0.12, 4, 3
H, W = 480, 640


class Geo:
    """floor_seg_node.py FloorGeometry와 같은 바닥 투영(roll 0)"""

    def __init__(self):
        self.K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]])
        self.Kinv = np.linalg.inv(self.K)
        p = math.radians(PITCH)
        n = np.array([0.0, math.cos(p), math.sin(p)])
        fwd = np.array([0, 0, 1.0]) - n[2] * n
        fwd /= np.linalg.norm(fwd)
        left = -np.cross(n, fwd)
        left = left if left[0] < 0 else -left
        self.n, self.fwd, self.left, self.foot = n, fwd, left, CAM_H * n

    def pixel_to_base(self, u, v):
        r = self.Kinv @ np.stack([u, v, np.ones_like(u)]).astype(np.float64)
        den = self.n @ r
        ok = den > 1e-6
        X = (CAM_H / np.where(ok, den, 1.0)) * r
        d = X - self.foot[:, None]
        return CAM_X + self.fwd @ d, self.left @ d, ok

    def row_of_distance(self, x):
        X = self.foot + self.fwd * (x - CAM_X)
        return FY * X[1] / X[2] + CY


def imread_u(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def load(root, num):
    d = glob.glob(os.path.join(root, f"{num}_*"))[0]
    rows = list(csv.DictReader(open(os.path.join(d, "frames", "frames.csv"), encoding="utf-8")))
    for r in rows:
        r["jpg"] = os.path.join(d, "frames", r["stamp"] + ".jpg")
    rows = [r for r in rows if os.path.exists(r["jpg"])]
    return d, rows


def boundary(floor):
    """floor_seg_node.py boundary()와 같은 규칙: 열마다 아래에서 위로 RUN_PX 연속 '바닥 아님'이 처음 나오는 행, 없으면 -1"""
    cols = np.arange(0, W, COL_STEP)
    nf = ~floor[:, cols]
    run = nf.copy()
    for k in range(1, RUN_PX):
        run[k:] &= nf[:-k]
    run[:RUN_PX - 1] = False
    flip = run[::-1]
    idx = flip.argmax(0)
    return np.where(flip[idx, np.arange(len(cols))], H - 1 - idx, -1)


def col_dist(geo, v_far, rows):
    """노드와 같은 규칙: 경계가 max_range 안이면 그 바닥 점의 앞 거리(base x), 없거나 멀면 max_range 끝. 정면 폭 안 최소 = 정면 거리"""
    rows = np.array([int(v) for v in rows.split()] if isinstance(rows, str) else rows, float)
    cols = np.arange(0, W, COL_STEP, dtype=float)
    bx, by, ok = geo.pixel_to_base(cols, np.where(rows >= 0, rows, H - 1))
    hit = (rows >= max(v_far, 0)) & ok & (bx - CAM_X <= MAX_RANGE + 1e-6)
    fr = bx[hit & (np.abs(by) <= FRONT_HW)]
    return np.where(hit, bx, CAM_X + MAX_RANGE), hit, (float(fr.min()) if len(fr) else 9.9)


def blur_person(img):
    """불 켬 영상 오른쪽 위 책상에 앉은 사람 — 블로그·git용 그림에서만 흐림(원본 프레임은 그대로)"""
    out = img.copy()
    out[0:215, 415:575] = cv2.GaussianBlur(out[0:215, 415:575], (61, 61), 0)
    return out


def infer(sess, inp, bgr):
    rgb = cv2.cvtColor(cv2.resize(bgr, (320, 240), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
    lg = sess.run(None, {inp: x})[0][0]
    return cv2.resize(lg[1] - lg[0], (W, H), interpolation=cv2.INTER_LINEAR) > 0


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--on", required=True)
    ap.add_argument("--off", required=True)
    ap.add_argument("--model", default=os.path.join(here, "offload", "models", "ft_240x320_int8_gelu.onnx"))
    ap.add_argument("--gain", type=float, default=15.0, help="불 끔 프레임을 소프트웨어로 밝히는 배율(그림·참고 시험)")
    ap.add_argument("--out", default="light_compare.png")
    a = ap.parse_args()
    geo = Geo()
    v_far = geo.row_of_distance(CAM_X + MAX_RANGE)
    r0 = int(math.ceil(v_far))                                    # 이 행부터 아래 = 장애물이 없다면 1.5 m 안 바닥이 보일 영역
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(a.model, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    cols = np.arange(0, W, COL_STEP)
    bearing = np.degrees(np.arctan((CX - cols) / FX))             # 왼쪽 +
    res, data = {}, {}
    for key, num in (("on", a.on), ("off", a.off)):
        d, rows = load(a.session, num)
        imgs = [imread_u(r["jpg"]) for r in rows]
        gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in imgs]
        masks = np.stack([infer(sess, inp, im) for im in imgs])
        dist = np.stack([col_dist(geo, v_far, r["rows_every_4px"])[0] for r in rows])
        off_rows = np.stack([boundary(m) for m in masks])           # 저장 JPEG로 다시 낸 경계(로봇 원본 프레임 결과와 맞는지 점검)
        node_rows = np.stack([np.array([int(v) for v in r["rows_every_4px"].split()]) for r in rows])
        same = (off_rows >= 0) & (node_rows >= 0)
        front = np.array([float(r["front_m"]) for r in rows])
        det = front[front < 9.0]
        data[key] = {"imgs": imgs, "masks": masks, "dist": dist, "rows": rows}
        res[key] = {"run": os.path.basename(d), "frames": len(rows),
                    "brightness_mean": round(float(np.median([g.mean() for g in gray])), 1),
                    "brightness_floor_region": round(float(np.median([g[r0:].mean() for g in gray])), 1),
                    "front_det_rate": round(len(det) / max(1, len(front)), 3),
                    "front_med_m": round(float(np.median(det)), 3) if len(det) else None,
                    "front_std_cm": round(float(det.std()) * 100, 1) if len(det) else None,
                    "floor_ratio_in_range": round(float(masks[:, r0:].mean()), 3),
                    "col_dist_med_m": round(float(np.median(np.median(dist, 0))), 3),
                    "cols_blocked_at_blind_edge": round(float((np.median(dist, 0) < 0.32).mean()), 3),
                    "offline_vs_robot_row_diff_med_px": round(float(np.median(np.abs(off_rows - node_rows)[same])), 1) if same.any() else None}
    ref = np.median(data["on"]["masks"].astype(np.uint8), 0) > 0.5          # 불 켬 프레임 다수결 = 같은 장면의 기준 판정
    def iou(m):
        a_, b_ = m[r0:], ref[r0:]
        return float((a_ & b_).sum() / max(1, (a_ | b_).sum()))
    res["on"]["floor_iou_vs_on"] = round(float(np.median([iou(m) for m in data["on"]["masks"]])), 3)
    res["off"]["floor_iou_vs_on"] = round(float(np.median([iou(m) for m in data["off"]["masks"]])), 3)
    bright = [np.clip(im.astype(np.float32) * a.gain, 0, 255).astype(np.uint8) for im in data["off"]["imgs"]]
    bm = np.stack([infer(sess, inp, im) for im in bright])
    bres = [col_dist(geo, v_far, boundary(m)) for m in bm]
    bdist = np.stack([b_[0] for b_ in bres])
    bfront = np.array([b_[2] for b_ in bres])
    bdet = bfront[bfront < 9.0]
    data["bright"] = {"dist": bdist}
    res["off_brightened"] = {"gain": a.gain, "floor_ratio_in_range": round(float(bm[:, r0:].mean()), 3),
                             "floor_iou_vs_on": round(float(np.median([iou(m) for m in bm])), 3),
                             "front_det_rate": round(len(bdet) / max(1, len(bfront)), 3),
                             "front_med_m": round(float(np.median(bdet)), 3) if len(bdet) else None,
                             "front_p10_p90_m": [round(float(np.percentile(bdet, q)), 3) for q in (10, 90)] if len(bdet) else None,
                             "cols_blocked_at_blind_edge": round(float((np.median(bdist, 0) < 0.32).mean()), 3),
                             "note": "참고용: 저장된 JPEG(밝기 2/255 근처)를 소프트웨어로 밝혀 같은 모델에 넣은 결과. 로봇에서 돈 값 아님"}
    res["geometry"] = {"max_range_row": round(v_far, 1), "nearest_visible_floor_x": round(float(geo.pixel_to_base(
        np.array([CX]), np.array([H - 1.0]))[0][0]), 3), "max_range_x": CAM_X + MAX_RANGE}

    # ---------- 그림
    k_on, k_off = len(data["on"]["rows"]) // 2, len(data["off"]["rows"]) // 2
    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.9], hspace=0.16, wspace=0.04)

    def panel(ax, img, mask, rows_str, title):
        vis = blur_person(img)
        if mask is not None:
            vis[mask] = (0.45 * vis[mask] + 0.55 * np.array([60, 200, 60])).astype(np.uint8)
        if rows_str is not None:
            _, hit, _ = col_dist(geo, v_far, rows_str)
            rr = np.array([int(v) for v in rows_str.split()])
            for c, r_, h_ in zip(cols, rr, hit):
                if h_:
                    cv2.circle(vis, (int(c), int(r_)), 3, (0, 0, 255), -1)
        cv2.line(vis, (0, r0), (W - 1, r0), (0, 200, 255), 1)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    on, off, br = res["on"], res["off"], res["off_brightened"]
    panel(fig.add_subplot(gs[0, 0]), data["on"]["imgs"][k_on], data["on"]["masks"][k_on], data["on"]["rows"][k_on]["rows_every_4px"],
          f"(a) Lights ON - brightness {on['brightness_mean']:.0f}/255\ncup at {on['front_med_m']:.2f} m: every frame / white book: read as floor")
    panel(fig.add_subplot(gs[0, 1]), data["off"]["imgs"][k_off], None, None,
          f"(b) Lights OFF - as captured\nbrightness {off['brightness_mean']:.1f}/255")
    panel(fig.add_subplot(gs[0, 2]), bright[k_off], data["off"]["masks"][k_off], data["off"]["rows"][k_off]["rows_every_4px"],
          f"(c) Lights OFF - displayed x{a.gain:.0f}, model output of (b)\nno floor found -> 'blocked' at {off['front_med_m']:.2f} m")
    ax = fig.add_subplot(gs[1, :])
    for key, col, ls, lab in (("on", "tab:green", "-", "lights ON (robot)"), ("off", "black", "-", "lights OFF (robot)"),
                              ("bright", "tab:purple", "--", f"lights OFF frames x{a.gain:.0f} in software (offline, reference)")):
        dist = data[key]["dist"]
        ax.fill_between(bearing, np.percentile(dist, 10, 0), np.percentile(dist, 90, 0), color=col, alpha=0.12, lw=0)
        ax.plot(bearing, np.median(dist, 0), color=col, ls=ls, lw=2, label=lab)
    nx = res["geometry"]["nearest_visible_floor_x"]
    ax.axhline(nx, color="tab:red", ls=":", lw=1)
    ax.text(bearing[0] - 0.5, nx - 0.07, f"nearest visible floor {nx:.2f} m (closer = blind zone)", color="tab:red", fontsize=9)
    ax.axhline(CAM_X + MAX_RANGE, color="gray", ls=":", lw=1)
    ax.text(bearing[0] - 0.5, CAM_X + MAX_RANGE + 0.03, f"max range {CAM_X + MAX_RANGE:.2f} m (= no obstacle in this column)",
            color="gray", fontsize=9)
    ax.annotate("cup", (-6, 1.12), (-6, 0.85), ha="center", fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("white book\n(missed)", (8, 1.56), (8, 1.22), ha="center", fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlim(bearing[0], bearing[-1])
    ax.set_ylim(0, 1.75)
    ax.set_xlabel("bearing from camera axis (deg, left +)")
    ax.set_ylabel("free floor per column (m, robot center)")
    ax.legend(loc="center right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.3)
    ax.set_title(f"(d) Per-column free-space distance (median, band p10-p90)   |   floor IoU vs lights-on: "
                 f"OFF {off['floor_iou_vs_on']:.2f}, OFF x{a.gain:.0f} {br['floor_iou_vs_on']:.2f}", fontsize=10)
    fig.suptitle("Real low-light test - robot stationary, same spot, only the room lights change", fontsize=13)
    fig.savefig(a.out, dpi=110, bbox_inches="tight")
    js = os.path.splitext(a.out)[0] + ".json"
    json.dump(res, open(js, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for key in ("on", "off"):
        r = res[key]
        print(f"[{'불 켬' if key == 'on' else '불 끔'}] {r['run']} {r['frames']}장 · 밝기 {r['brightness_mean']}/255(바닥 영역 {r['brightness_floor_region']})"
              f" · 정면 {r['front_med_m']} m(표준편차 {r['front_std_cm']} cm, 잡힘 {r['front_det_rate'] * 100:.0f}%)"
              f" · 1.5 m 영역 바닥 비율 {r['floor_ratio_in_range'] * 100:.0f}% · 불 켬 대비 IoU {r['floor_iou_vs_on']:.2f}"
              f" · 열별 거리 중앙값 {r['col_dist_med_m']} m · 사각지대 끝에서 막힌 열 {r['cols_blocked_at_blind_edge'] * 100:.0f}%")
    b = res["off_brightened"]
    print(f"[참고] 불 끔 ×{b['gain']:.0f} 밝힘 → 바닥 비율 {b['floor_ratio_in_range'] * 100:.0f}% · 불 켬 대비 IoU {b['floor_iou_vs_on']:.2f}"
          f" · 정면 {b['front_med_m']} m(p10~p90 {b['front_p10_p90_m']}, 잡힘 {b['front_det_rate'] * 100:.0f}%)"
          f" · 사각지대 끝에서 막힌 열 {b['cols_blocked_at_blind_edge'] * 100:.0f}%")
    print(f"[점검] 저장 JPEG로 다시 낸 경계 vs 로봇 경계, 행 차이 중앙값: 불 켬 {res['on']['offline_vs_robot_row_diff_med_px']} px"
          f" · 불 끔 {res['off']['offline_vs_robot_row_diff_med_px']} px")
    print(f"저장: {a.out}, {js}")


if __name__ == "__main__":
    main()
