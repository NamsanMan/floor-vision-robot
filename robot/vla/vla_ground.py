#!/usr/bin/env python3
"""VLA 실험 3단계 (메인컴, 기본 conda 환경: torch + transformers, 캐시된 CLIP ViT-B/16, 2026-10-31):
말로 한 물체(예: "주황 상자")를 2단계 후보 중에서 CLIP으로 고르고, 그 접지점을 지도 좌표로 바꿔 '물체 앞'을 Nav2 목표로 만든다.
끝에서 끝까지 행동을 내는 VLA 모델이 아니라 '시각-언어 그라운딩 + 바닥 분할 + Nav2'를 잇는 모듈식 구성.
usage: python vla_ground.py "주황 상자" --snap snap.jpg --props proposals.json --pose snap.json [--standoff 0.35] [--out goal_point.json]"""
import argparse
import json
import math
import os

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel, CLIPProcessor, CLIPTokenizerFast

KO = {"주황 상자": "orange box", "주황색 상자": "orange box", "주황색 박스": "orange box", "주황 박스": "orange box",
      "검은 상자": "black box", "검은색 상자": "black box", "검은색 박스": "black box", "검은 박스": "black box",
      "신발": "sneaker", "운동화": "sneaker", "종이컵": "paper cup", "컵": "paper cup", "책": "white book", "흰 책": "white book",
      "의자": "chair", "책상": "desk", "컴퓨터": "computer case", "PC": "computer case", "본체": "computer case"}


def imread_u(path):
    """Windows OpenCV는 한글 경로를 못 열어서 바이트로 읽어 디코드(2026-10-31)"""
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(path, img):
    cv2.imencode(os.path.splitext(path)[1], img)[1].tofile(path)


def square_crop(bgr, box):
    """CLIP은 짧은 변을 224로 맞춘 뒤 가운데 정사각형만 보므로, 넓거나 긴 영역은 물체가 잘린다 → 정사각형으로 패딩해 전체를 넣음"""
    x0, y0, x1, y1 = box
    crop = bgr[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    s = max(h, w)
    sq = np.full((s, s, 3), crop.reshape(-1, 3).mean(0), np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop
    return Image.fromarray(cv2.cvtColor(sq, cv2.COLOR_BGR2RGB))
LABELS = ["orange box", "black box", "sneaker", "paper cup", "white book", "desk", "computer case", "chair", "wall", "floor", "cables"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--snap", default="snap.jpg")
    ap.add_argument("--props", default="proposals.json")
    ap.add_argument("--pose", default="snap.json")
    ap.add_argument("--standoff", type=float, default=0.35, help="물체 접지점에서 로봇 중심까지 남길 거리(m)")
    ap.add_argument("--min-prob", type=float, default=0.30)
    ap.add_argument("--out", default="goal_point.json")
    a = ap.parse_args()
    target = KO.get(a.query.strip(), a.query.strip())
    labels = LABELS if target in LABELS else LABELS + [target]
    bgr = imread_u(a.snap)
    props = json.load(open(a.props, encoding="utf-8"))["proposals"]
    pose = json.load(open(a.pose, encoding="utf-8")).get("map_pose")
    if not props:
        raise SystemExit("후보가 없습니다")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", local_files_only=True).to(dev).eval()
    # 캐시에 preprocessor_config.json이 없어(가중치·토크나이저만 있음) CLIP 표준 전처리를 직접 지정 — 새로 내려받지 않음
    ip = CLIPImageProcessor(size={"shortest_edge": 224}, crop_size={"height": 224, "width": 224}, resample=3,
                            image_mean=[0.48145466, 0.4578275, 0.40821073], image_std=[0.26862954, 0.26130258, 0.27577711])
    proc = CLIPProcessor(image_processor=ip,
                         tokenizer=CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch16", local_files_only=True))
    crops = [square_crop(bgr, p["box"]) for p in props]
    with torch.no_grad():
        inp = proc(text=[f"a photo of a {t}" for t in labels], images=crops, return_tensors="pt", padding=True).to(dev)
        prob = model(**inp).logits_per_image.softmax(-1).cpu().numpy()   # (후보, 문장)
    ti = labels.index(target)
    k = int(prob[:, ti].argmax())
    best = float(prob[k, ti])
    for i, p in enumerate(props):
        top = labels[int(prob[i].argmax())]
        print(f"  후보 {i}: 앞 {p['base_xy'][0]:.2f} m, 옆 {p['base_xy'][1]:+.2f} m → '{target}' {prob[i, ti]:.2f} (가장 닮은 것: {top} {prob[i].max():.2f})")
    ok = best >= a.min_prob
    ox, oy = props[k]["base_xy"]
    res = {"query": a.query, "target": target, "proposal": k, "prob": round(best, 3), "object_base": [ox, oy]}
    if ok and pose:
        c, s = math.cos(pose[2]), math.sin(pose[2])
        mx, my = pose[0] + c * ox - s * oy, pose[1] + s * ox + c * oy           # 물체 접지점(지도)
        dx, dy = mx - pose[0], my - pose[1]
        d = math.hypot(dx, dy)
        gx, gy = mx - a.standoff * dx / d, my - a.standoff * dy / d             # 로봇 쪽으로 standoff만큼 앞
        res.update({"xy": [round(gx, 3), round(gy, 3)], "frame": "map", "object_map": [round(mx, 3), round(my, 3)],
                    "robot_map": [round(v, 3) for v in pose]})
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)   # 로봇이 UTF-8로 읽음
        print(f"→ '{a.query}'({target}) = 후보 {k}, 확률 {best:.2f} · 물체 지도 ({mx:.2f}, {my:.2f}) · 목표 ({gx:.2f}, {gy:.2f}) 저장: {a.out}")
    else:
        print(f"→ 목표 안 만듦: " + (f"가장 높은 확률 {best:.2f} < {a.min_prob}" if not ok else "지도 위치 없음(snap.json)"))
    vis = bgr.copy()
    for i, p in enumerate(props):
        b = p["box"]
        col = (0, 0, 255) if (i == k and ok) else (180, 180, 180)
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), col, 3 if i == k else 1)
        cv2.putText(vis, f"{prob[i, ti]:.2f}", (b[0] + 2, b[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    cv2.putText(vis, f'"{target}"  ->  ' + (f"p={best:.2f}, {ox:.2f} m ahead" if ok else "not found"), (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    imwrite_u(os.path.splitext(a.out)[0] + "_vis.jpg", vis)


if __name__ == "__main__":
    main()
