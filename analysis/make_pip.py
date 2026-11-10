#!/usr/bin/env python3
"""로봇 카메라 PiP 클립 만들기 (메인컴, conda env mot, 2026-10-31)

직진 정지 시험(2026-10-31) 회차마다 인식 노드가 저장한 카메라 프레임(frames/<stamp>.jpg)에 바닥 분할 결과를 겹쳐
포트폴리오 데모 영상에 넣을 작은 PiP(화면 속 화면) 클립과, 폰 영상과 시각을 맞출 JSON을 만든다.
  초록 반투명  저장 JPEG를 같은 INT8 모델(ft_240x320_int8_gelu.onnx)로 다시 돌린 바닥 판정
  빨간 점      노드가 frames.csv에 남긴 경계 행(rows_every_4px) 중 인식 범위(카메라에서 1.5 m) 안 점
               = 로봇이 실제로 쓴 장애물 점(노드 디버그 그림과 같은 규칙)
  주황 선      1.5 m 인식 범위 행(이 선보다 위 = 1.5 m보다 먼 바닥이라 무시)
  위쪽 머리띠  회차·조건(영문), 출발 후 경과 시간 t = 촬영 시각 - t0_epoch, 정면 장애물 거리(frames.csv front_m, 9 이상 = 없음),
               장애물 출력 끔(frames.csv mute=1, 라이다만 회차)이면 그 표시
영상  640x540 (머리띠 60 + 카메라 640x480), 10 fps 고정, libx264 CRF 18, yuv420p (BT.709로 변환·표기)
      저장 프레임 하나를 다음 프레임 촬영 시각까지 반복해서 실제 시간을 유지한다:
      영상 시각 τ(초)의 화면 = 촬영 시각 ≤ first_frame_epoch + τ 인 가장 최근 저장 프레임  →  epoch = first_frame_epoch + τ
      (마지막 저장 프레임은 출력 한 장 = 0.1초만 보임. 출력 프레임 수 = ceil((마지막 - 처음) × 10) + 1)
JSON  run, t0_epoch, first_frame_epoch, last_frame_epoch, fps_out, duration_s, n_frames
      + 참고: first_frame_t_s(영상 0초 = 회차 시각 몇 초), 출력 프레임 수, 결과·메모(results.csv), Collision Monitor 사건, 점검 값
      시각은 모두 로봇(라즈베리파이) 시계 epoch 초. 촬영 시각 = 카메라 header.stamp, t0_epoch = drive_test.py time.time() (같은 시계)
미리보기  _preview.jpg = 회차마다 3장(처음 잡음/중간/마지막). 인코딩된 MP4에서 다시 꺼낸 프레임이라 최종 결과 그대로다

usage (Git Bash):
  python make_pip.py              # 01~06 (프레임 없는 회차는 건너뜀)
  python make_pip.py --runs 04 05  # 일부 회차만(미리보기도 그 회차만)
  옵션: --session <회차 폴더들이 든 폴더>  --out <출력 폴더>  --model <onnx>  --ffmpeg <ffmpeg.exe>  --no-preview
  기본 경로는 이 파일 위치(videos/tools) 기준
    세션 = experiments/2026-10-29_dsnet_finetune/results/drive_20261031/20261031
    모델 = experiments/2026-10-29_dsnet_finetune/robot/offload/models/ft_240x320_int8_gelu.onnx
    출력 = videos/work/pip/run0X_robot_view.mp4 + .json, _preview.jpg
주의: OpenCV는 윈도우에서 한글 경로를 못 열어서 np.fromfile + cv2.imdecode로 읽는다. ffmpeg는 CPU(libx264)만 쓴다(NVENC 안 됨).
      인코딩은 한 번에 하나씩(회차 순서대로) 돈다. 기하·모델·경계 규칙은 robot/light_compare.py(= floor_seg_node.py)와 같다.
"""
import argparse
import csv
import glob
import json
import math
import os
import subprocess

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))                       # .../Mentorpi와 ROS2
EXP = os.path.join(ROOT, "experiments", "2026-10-29_dsnet_finetune")
DEF_SESSION = os.path.join(EXP, "results", "drive_20261031", "20261031")
DEF_MODEL = os.path.join(EXP, "robot", "offload", "models", "ft_240x320_int8_gelu.onnx")
DEF_OUT = os.path.join(ROOT, "videos", "work", "pip")
DEF_FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FONT = "C:/Windows/Fonts/malgunbd.ttf"

# ---------- light_compare.py / floor_seg_node.py와 같은 카메라 보정·설치값과 규칙
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
FX, FY, CX, CY = 590.77, 590.51, 333.39, 232.73
CAM_H, PITCH, CAM_X = 0.106, 2.45, 0.061
MAX_RANGE, FRONT_HW, COL_STEP, RUN_PX = 1.5, 0.12, 4, 3
H, W = 480, 640
COLS = np.arange(0, W, COL_STEP)

# ---------- 영상
FPS = 10
HDR_H = 60                                    # 머리띠 높이 → 전체 640x540
OUT_W, OUT_H = W, H + HDR_H
GREEN = np.array([60, 200, 60], np.float32)   # BGR, 바닥 겹침 색
ALPHA = 0.40                                  # 바닥 겹침 불투명도(노드 디버그 그림 save_debug와 같은 0.4)
RED, ORANGE = (0, 0, 255), (0, 140, 255)      # BGR
COND_EN = {"cam": "camera + memory", "lidar": "LiDAR only", "cam_nomem": "camera, no memory"}
X264 = ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-threads", "6"]


class Geo:
    """floor_seg_node.py FloorGeometry와 같은 바닥 투영(roll 0) — light_compare.py Geo 그대로"""

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


def boundary(floor):
    """floor_seg_node.py boundary()와 같은 규칙: 열마다 아래에서 위로 RUN_PX 연속 '바닥 아님'이 처음 나오는 행, 없으면 -1"""
    nf = ~floor[:, COLS]
    run = nf.copy()
    for k in range(1, RUN_PX):
        run[k:] &= nf[:-k]
    run[:RUN_PX - 1] = False
    flip = run[::-1]
    idx = flip.argmax(0)
    return np.where(flip[idx, np.arange(len(COLS))], H - 1 - idx, -1)


def col_hits(geo, v_far, rows):
    """노드와 같은 규칙: 경계가 1.5 m 안인 열(= 장애물 점) 표시와 정면 폭(±0.12 m) 안 최소 앞 거리(없으면 9.9)"""
    rows = np.asarray(rows, float)
    bx, by, ok = geo.pixel_to_base(COLS.astype(float), np.where(rows >= 0, rows, H - 1))
    hit = (rows >= max(v_far, 0)) & ok & (bx - CAM_X <= MAX_RANGE + 1e-6)
    fr = bx[hit & (np.abs(by) <= FRONT_HW)]
    return hit, (float(fr.min()) if len(fr) else 9.9)


def infer(sess, inp, bgr):
    """노드 infer_floor()와 같은 전처리: 320x240 INTER_AREA → RGB 정규화 → 로짓 차를 640x480으로 키워 > 0 = 바닥"""
    rgb = cv2.cvtColor(cv2.resize(bgr, (320, 240), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
    lg = sess.run(None, {inp: x})[0][0]
    return cv2.resize(lg[1] - lg[0], (W, H), interpolation=cv2.INTER_LINEAR) > 0


def imread_u(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def fmt_t(t):
    return f"{0.0 if abs(t) < 0.05 else t:.1f}"


# ---------- 회차 읽기
def load_run(session, num):
    """회차 폴더·events.json·frames.csv(JPEG 있는 줄만)·results.csv 한 줄. 프레임이 없으면 rows = []와 이유"""
    ds = sorted(glob.glob(os.path.join(session, f"{num}_*")))
    if not ds:
        return None, "회차 폴더 없음"
    d = ds[0]
    ev = json.load(open(os.path.join(d, "events.json"), encoding="utf-8"))
    res = {}
    rp = os.path.join(session, "results.csv")
    if os.path.exists(rp):
        res = next((r for r in csv.DictReader(open(rp, encoding="utf-8")) if r["run"].strip() == str(int(num))), {})
    run = {"num": num, "dir": d, "name": os.path.basename(d), "ev": ev, "res": res, "rows": []}
    fcsv = os.path.join(d, "frames", "frames.csv")
    if not os.path.exists(fcsv):
        return run, "프레임 없음(--no-frames로 돈 회차)" if ev["args"].get("no_frames") else "frames.csv 없음"
    rows = list(csv.DictReader(open(fcsv, encoding="utf-8")))
    run["rows"] = [r for r in rows if os.path.exists(os.path.join(d, "frames", r["stamp"] + ".jpg"))]
    run["n_missing_jpg"] = len(rows) - len(run["rows"])
    if not run["rows"]:
        return run, "JPEG 없음"
    run["rows"].sort(key=lambda r: int(r["stamp"]))
    return run, ""


# ---------- 그리기
class Painter:
    def __init__(self, r_line):
        self.r_line = r_line
        self.f_main = ImageFont.truetype(FONT, 19)
        self.f_sub = ImageFont.truetype(FONT, 17)
        self.f_note = ImageFont.truetype(FONT, 15)
        self.f_tag = ImageFont.truetype(FONT, 13)
        self.x_t = OUT_W - 10 - int(math.ceil(self.f_main.getlength("t = 00.0 s")))   # 시간 글자 시작 고정(숫자가 바뀌어도 안 흔들림)

    def label_font(self, label):
        """회차 이름이 시간 글자와 겹치지 않을 크기"""
        for size in range(19, 12, -1):
            f = ImageFont.truetype(FONT, size)
            if 10 + f.getlength(label) <= self.x_t - 14:
                return f
        return f

    def overlay(self, bgr, floor, hit, rows):
        vis = bgr.copy()
        vis[floor] = (vis[floor].astype(np.float32) * (1 - ALPHA) + GREEN * ALPHA).astype(np.uint8)
        vis[self.r_line:self.r_line + 2] = ORANGE                              # 1.5 m 행부터 2px(1px은 yuv420p 색 솎음에 묻힘)
        for c, r in zip(COLS[hit], rows[hit]):
            cv2.circle(vis, (int(c), int(r)), 3, RED, -1, cv2.LINE_AA)
        return vis

    def compose(self, vis, label_font, label, t, front, muted):
        rgb = np.empty((OUT_H, OUT_W, 3), np.uint8)
        rgb[:HDR_H] = (20, 20, 20)
        rgb[HDR_H:] = vis[:, :, ::-1]
        im = Image.fromarray(rgb)
        d = ImageDraw.Draw(im)
        d.line([(0, HDR_H - 1), (OUT_W - 1, HDR_H - 1)], fill=(70, 70, 70))
        d.text((10, 26), label, font=label_font, fill=(245, 245, 245), anchor="ls")
        d.text((self.x_t, 26), f"t = {fmt_t(t)} s", font=self.f_main, fill=(255, 255, 255), anchor="ls")
        if front < 9.0:
            d.text((10, 51), f"front obstacle {front:.2f} m", font=self.f_sub, fill=(255, 200, 60), anchor="ls")
        else:
            d.text((10, 51), "no obstacle ahead", font=self.f_sub, fill=(150, 210, 150), anchor="ls")
        if muted:
            d.text((OUT_W - 10, 51), "camera output muted (LiDAR-only run)", font=self.f_note, fill=(255, 120, 100), anchor="rs")
        d.text((OUT_W - 6, HDR_H + self.r_line - 3), "1.5 m", font=self.f_tag, fill=(255, 150, 40), anchor="rs",
               stroke_width=1, stroke_fill=(0, 0, 0))
        return np.ascontiguousarray(np.asarray(im)[:, :, ::-1])


# ---------- ffmpeg
def ffprobe_of(ffmpeg):
    return os.path.join(os.path.dirname(ffmpeg), "ffprobe.exe" if ffmpeg.lower().endswith(".exe") else "ffprobe")


def open_encoder(ffmpeg, out):
    """BGR 원시 프레임을 stdin으로 받아 10 fps 고정 MP4로. RGB→YUV는 BT.709로 변환하고 같은 값으로 표기
    (accurate_rnd 없으면 swscale이 버림 반올림이라 Y·Cb·Cr이 약 -0.5씩 치우침 → R·B가 1.5단계 어두워짐, 2026-10-31 실측)"""
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{OUT_H}", "-framerate", str(FPS), "-i", "-",
           "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=bicubic+accurate_rnd+full_chroma_int", *X264,
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
           "-movflags", "+faststart", out]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def probe(ffmpeg, mp4):
    cmd = [ffprobe_of(ffmpeg), "-v", "error", "-select_streams", "v:0", "-count_frames",
           "-show_entries", "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames,color_space:format=duration",
           "-of", "json", mp4]
    j = json.loads(subprocess.run(cmd, capture_output=True, check=True).stdout.decode("utf-8"))
    s = j["streams"][0]
    return {"duration_s": round(float(j["format"]["duration"]), 3), "frames": int(s["nb_read_frames"]), "codec": s["codec_name"],
            "size": f"{s['width']}x{s['height']}", "pix_fmt": s["pix_fmt"], "fps": s["r_frame_rate"], "color_space": s.get("color_space")}


def grab(ffmpeg, mp4, k):
    """인코딩된 MP4에서 k번째 출력 프레임을 BGR로 꺼냄(미리보기·화질 점검용)"""
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", mp4,
           "-vf", f"select=eq(n\\,{k}),scale=flags=bicubic+accurate_rnd+full_chroma_int,format=bgr24", "-frames:v", "1",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    buf = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(buf, np.uint8)[:OUT_H * OUT_W * 3].reshape(OUT_H, OUT_W, 3)


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse == 0 else round(10 * math.log10(255.0 ** 2 / mse), 1)


# ---------- 회차 하나
def make_clip(run, sess, inp, geo, v_far, painter, a):
    rows, ev, num = run["rows"], run["ev"], run["num"]
    t0 = float(ev["t0_epoch"])
    cond = ev["args"].get("cond", "")
    label = f"Robot camera · run {num} · {COND_EN.get(cond, cond)}"
    lfont = painter.label_font(label)
    stamps_ns = np.array([int(r["stamp"]) for r in rows], np.int64)
    first, last = int(rows[0]["stamp"]) / 10 ** 9, int(rows[-1]["stamp"]) / 10 ** 9     # 정수 나눗셈 = 가장 가까운 float
    rel = (stamps_ns - stamps_ns[0]) * 1e-9                                  # 영상 시각(초) = 촬영 시각 - 첫 프레임
    t_run = stamps_ns * 1e-9 - t0                                            # 회차 시각(초) = 촬영 시각 - t0_epoch
    fronts = np.array([float(r["front_m"]) for r in rows])
    n_out = int(math.ceil(rel[-1] * FPS - 1e-9)) + 1
    idx = np.searchsorted(rel, np.arange(n_out) / FPS + 1e-6, side="right") - 1   # 출력 프레임 k → 그때까지 가장 최근 저장 프레임
    counts = np.bincount(idx, minlength=len(rows))
    k_first = np.searchsorted(idx, np.arange(len(rows)))                     # 저장 프레임 i가 처음 나오는 출력 프레임

    # 미리보기 3장: 처음 잡음(없으면 출발) / 그와 마지막 사이 가운데 / 마지막
    det = np.where((fronts < 9.0) & (t_run >= 0))[0]
    i_a = int(det[0]) if len(det) else int(np.argmax(t_run >= 0)) if (t_run >= 0).any() else 0
    i_c = len(rows) - 1
    i_b = int(np.argmin(np.abs(t_run - (t_run[i_a] + t_run[i_c]) / 2)))
    picks = [(i_a, "처음 잡음" if len(det) else "출발"), (i_b, "중간"), (i_c, "마지막")]

    out = os.path.join(a.out, f"run{num}_robot_view.mp4")
    enc = open_encoder(a.ffmpeg, out)
    rendered, d_rows, near, front_err = {}, [], [], []
    try:
        for i, r in enumerate(rows):
            bgr = imread_u(os.path.join(run["dir"], "frames", r["stamp"] + ".jpg"))
            if bgr is None or bgr.shape[:2] != (H, W):
                raise RuntimeError(f"프레임을 못 읽음/크기 다름: {r['stamp']}.jpg")
            floor = infer(sess, inp, bgr)
            node_rows = np.array([int(v) for v in r["rows_every_4px"].split()])
            hit, front_re = col_hits(geo, v_far, node_rows)
            front_err.append(abs(min(front_re, 9.9) - min(float(r["front_m"]), 9.9)))
            off_rows = boundary(floor)                                       # 다시 돌린 판정의 경계 ↔ 노드 경계(초록과 점이 맞는지 점검)
            same = (off_rows >= 0) & (node_rows >= 0)
            if same.any():
                diff = np.abs(off_rows - node_rows)[same]
                d_rows.append(float(np.median(diff)))
                near.append(float((diff <= 2).mean()))
            vis = painter.overlay(bgr, floor, hit, node_rows)
            frame = painter.compose(vis, lfont, label, t_run[i], float(r["front_m"]), r.get("mute", "0").strip() == "1")
            if any(i == p[0] for p in picks):
                rendered[i] = frame
            buf = frame.tobytes()
            for _ in range(int(counts[i])):
                enc.stdin.write(buf)
    finally:
        enc.stdin.close()
        rc = enc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg 인코딩 실패(종료 코드 {rc}): {out}")

    pr = probe(a.ffmpeg, out)
    thumbs, ps = [], []
    for i, cap in picks:
        g = grab(a.ffmpeg, out, int(k_first[i]))
        ps.append(psnr(g, rendered[i]))
        thumbs.append((g, f"{cap} · t = {fmt_t(t_run[i])} s"))

    res = run["res"]
    cm = [{"t_s": c[0], "kind": c[1]} for c in ev.get("cm_log", [])]
    meta = {
        "run": run["name"],
        "t0_epoch": t0,
        "first_frame_epoch": float(first),
        "last_frame_epoch": float(last),
        "fps_out": FPS,
        "duration_s": round(n_out / FPS, 3),
        "n_frames": len(rows),
        # ---- 이하 참고
        "video": os.path.basename(out),
        "label": label,
        "size": [OUT_W, OUT_H],
        "camera_box_xywh": [0, HDR_H, W, H],
        "n_out_frames": n_out,
        "first_frame_stamp_ns": int(rows[0]["stamp"]),
        "last_frame_stamp_ns": int(rows[-1]["stamp"]),
        "first_frame_t_s": round(float(t_run[0]), 3),
        "last_frame_t_s": round(float(t_run[-1]), 3),
        "time_mapping": "epoch = first_frame_epoch + 영상 시각 τ, 회차 시각 t = epoch - t0_epoch (로봇 시계). "
                        "출력 프레임 k(τ = k/10)는 촬영 시각 ≤ first_frame_epoch + τ 인 가장 최근 저장 프레임",
        "cond": cond,
        "obj": ev["args"].get("obj"),
        "result": res.get("result") or ev.get("result"),
        "note": res.get("note", ""),
        "stop_t_s": ev.get("extra", {}).get("stop_t"),
        "cm_events": cm,
        "first_obstacle_t_s": round(float(t_run[det[0]]), 3) if len(det) else None,
        "first_obstacle_front_m": round(float(fronts[det[0]]), 3) if len(det) else None,
        "check": {
            "every_frame_shown": bool((counts >= 1).all()),
            "hold_out_frames_min_max": [int(counts.min()), int(counts.max())],
            "src_interval_ms_median_max": [round(float(np.median(np.diff(rel))) * 1e3), round(float(np.diff(rel).max()) * 1e3)]
            if len(rows) > 1 else None,
            "front_from_node_rows_max_err_m": round(float(max(front_err)), 4),
            "offline_vs_node_boundary_med_px": round(float(np.median(d_rows)), 1) if d_rows else None,
            "offline_vs_node_boundary_within_2px": round(float(np.mean(near)), 3) if near else None,
            "ffprobe": pr,
            "preview_psnr_db": ps,
        },
    }
    js = os.path.splitext(out)[0] + ".json"
    json.dump(meta, open(js, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    ok_dur = abs(pr["duration_s"] - n_out / FPS) < 0.051 and pr["frames"] == n_out
    print(f"  저장 {os.path.basename(out)} + .json · 저장 프레임 {len(rows)}장 → 출력 {n_out}장(10 fps) · 길이 {n_out / FPS:.1f} s"
          f" (촬영 처음~마지막 {rel[-1]:.2f} s, 회차 시각 {t_run[0]:+.2f} ~ {t_run[-1]:+.2f} s)")
    print(f"  ffprobe {pr['duration_s']:.3f} s · {pr['frames']}장 · {pr['size']} {pr['codec']} {pr['pix_fmt']} {pr['fps']} fps"
          f" {pr['color_space']} → {'맞음' if ok_dur else '어긋남!'} · 모든 저장 프레임 표시 {'예' if meta['check']['every_frame_shown'] else '아니오!'}")
    print(f"  점검: 노드 경계로 다시 낸 정면 거리 최대 오차 {meta['check']['front_from_node_rows_max_err_m']} m"
          f" · 다시 돌린 경계 vs 노드 경계 차 중앙 {meta['check']['offline_vs_node_boundary_med_px']} px"
          f"(2px 안 {meta['check']['offline_vs_node_boundary_within_2px']}) · MP4 미리보기 PSNR {ps} dB"
          + (f" · 처음 잡음 t = {meta['first_obstacle_t_s']} s ({meta['first_obstacle_front_m']} m)" if len(det) else " · 정면 장애물 없음"))
    return meta, thumbs


def preview_sheet(items, path):
    """회차마다 한 줄, 3장(0.6배)씩 + 아래 설명"""
    s = 0.6
    tw, th, cap, gap, top = int(OUT_W * s), int(OUT_H * s), 24, 8, 44
    ncol = 3
    sheet = Image.new("RGB", (gap + ncol * (tw + gap), top + len(items) * (th + cap + gap)), (245, 245, 245))
    d = ImageDraw.Draw(sheet)
    f_t, f_c = ImageFont.truetype(FONT, 20), ImageFont.truetype(FONT, 15)
    d.text((gap, 30), "PiP 미리보기 · 2026-10-31 직진 정지 시험 · 인코딩된 MP4에서 꺼낸 프레임(회차마다 3장)", font=f_t,
           fill=(30, 30, 30), anchor="ls")
    for r, (num, thumbs) in enumerate(items):
        y = top + r * (th + cap + gap)
        for c, (img, text) in enumerate(thumbs[:ncol]):
            x = gap + c * (tw + gap)
            sm = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
            sheet.paste(Image.fromarray(np.ascontiguousarray(sm[:, :, ::-1])), (x, y))
            d.text((x + 2, y + th + 17), f"run {num} · {text}", font=f_c, fill=(40, 40, 40), anchor="ls")
    sheet.save(path, quality=90)


def main():
    ap = argparse.ArgumentParser(description="로봇 카메라 PiP 클립(바닥 분할 겹침) 만들기")
    ap.add_argument("--session", default=DEF_SESSION, help="회차 폴더(01_straight_... 등)가 든 폴더")
    ap.add_argument("--runs", nargs="+", default=["01", "02", "03", "04", "05", "06"])
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--model", default=DEF_MODEL)
    ap.add_argument("--ffmpeg", default=DEF_FFMPEG)
    ap.add_argument("--no-preview", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    geo = Geo()
    v_far = geo.row_of_distance(CAM_X + MAX_RANGE)
    r_line = int(math.ceil(v_far))                                           # 이 행부터 아래 = 1.5 m 안 바닥
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(a.model, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    painter = Painter(r_line)
    print(f"모델 {os.path.basename(a.model)} · 1.5 m 행 {v_far:.1f} → 주황 선 {r_line}~{r_line + 1}행 · 출력 {a.out}")
    items, skipped = [], []
    for num in a.runs:
        num = f"{int(num):02d}"
        run, why = load_run(a.session, num)
        if why:
            print(f"[{num}] 건너뜀: {why}" + (f" ({run['name']})" if run else ""))
            skipped.append(num)
            continue
        print(f"[{num}] {run['name']} · 프레임 {len(run['rows'])}장" + (f" (JPEG 없는 줄 {run['n_missing_jpg']}개 뺌)" if run["n_missing_jpg"] else ""))
        _, thumbs = make_clip(run, sess, inp, geo, v_far, painter, a)
        items.append((num, thumbs))
    if items and not a.no_preview:
        path = os.path.join(a.out, "_preview.jpg")
        preview_sheet(items, path)
        print(f"미리보기 저장: {path}")
    print(f"끝: 만든 회차 {[n for n, _ in items]}" + (f" · 건너뛴 회차 {skipped}" if skipped else ""))


if __name__ == "__main__":
    main()
