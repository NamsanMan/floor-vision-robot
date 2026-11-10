"""labelme JSON -> 학습·채점 마스크. 겹침 우선순위: 물체 > ignore > floor, 아무것도 없으면 '바닥 아님'.
출력(사진마다): <name>_mask.png (0 바닥 아님, 1 바닥, 255 ignore), <name>_obj.png (0 없음, 1..N 물체 번호), objects.json(번호 -> 라벨, reviewed 여부)
- 초벌(EoMT-L)만 있고 사람이 'reviewed'를 체크하지 않은 사진은 기본으로 건너뜀(--all이면 포함). 학습·채점 코드는 objects.json의 키만 쓸 것.
- 모양을 칠하는 규칙은 labelme(utils.shape_to_mask·shapes_to_label, PIL)과 똑같다: polygon·rectangle·oriented_rectangle·circle·mask.
  선·점처럼 영역이 아닌 모양이 있으면 아무것도 쓰지 않고 멈춘다(먼저 전부 검사한 뒤, 출력 폴더의 옛 마스크를 지우고 쓴다).
- numpy·Pillow만 쓴다 → labelme conda env에서도, 서브컴 venv에서도 돈다.
usage: python labels_to_masks.py <labels_dir> <out_dir> [--all]"""
import argparse
import base64
import glob
import io
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("labels_dir")
ap.add_argument("out")
ap.add_argument("--all", action="store_true", help="reviewed 체크 안 된 초벌도 포함")
a = ap.parse_args()


def shape_mask(s, H, W, where):
    t = s.get("shape_type") or "polygon"
    xy = [tuple(map(float, p)) for p in s["points"]]
    if t == "mask":                                                     # labelme shapes_to_label과 같음
        (x1, y1), (x2, y2) = np.asarray(s["points"]).astype(int)
        crop = np.array(Image.open(io.BytesIO(base64.b64decode(s["mask"])))).astype(bool)
        if crop.ndim == 3:
            crop = crop[..., 0]
        m = np.zeros((H, W), bool)
        m[y1:y2 + 1, x1:x2 + 1] = crop[:max(0, min(y2 + 1, H) - y1), :max(0, min(x2 + 1, W) - x1)]
        return m
    img = Image.fromarray(np.zeros((H, W), np.uint8))
    d = ImageDraw.Draw(img)
    if t == "circle":
        (cx, cy), (px, py) = xy[:2]
        r = math.hypot(cx - px, cy - py)
        d.ellipse(((cx - r, cy - r), (cx + r, cy + r)), outline=1, fill=1)
    elif t == "rectangle":
        (x0, y0), (x1, y1) = xy[:2]
        d.rectangle(((min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1))), outline=1, fill=1)
    elif t in ("polygon", "oriented_rectangle"):
        if len(xy) < 3:
            raise ValueError(f"{where}: {t} has only {len(xy)} points")
        d.polygon(xy=xy, outline=1, fill=1)
    else:
        raise ValueError(f"{where}: unsupported shape_type {t!r} (선·점은 영역이 아님 - 다각형으로 다시 그릴 것)")
    return np.array(img, bool)


# 1) 전부 읽고 칠해 보기 (하나라도 틀리면 아무것도 안 씀)
frames, skipped, warns = [], [], []
try:
    for jf in sorted(glob.glob(os.path.join(a.labels_dir, "*.json"))):
        d = json.load(open(jf, encoding="utf-8"))
        name = os.path.splitext(os.path.basename(jf))[0]
        reviewed = bool((d.get("flags") or {}).get("reviewed", False))
        if not reviewed and not a.all:
            skipped.append(name)
            continue
        H, W = d["imageHeight"], d["imageWidth"]
        shapes = d.get("shapes", [])
        ms = [shape_mask(s, H, W, f"{name} shape {k} ({s['label']})") for k, s in enumerate(shapes)]
        mask = np.zeros((H, W), np.uint8)
        obj = np.zeros((H, W), np.uint8)
        for s, m in zip(shapes, ms):                   # 바닥
            if s["label"] == "floor":
                mask[m] = 1
        for s, m in zip(shapes, ms):                   # ignore
            if s["label"] == "ignore":
                mask[m] = 255
        objs, om = [], []
        for s, m in zip(shapes, ms):                   # 물체 (가장 우선)
            if s["label"].startswith("obstacle"):
                objs.append(s["label"])
                om.append(m)
                obj[m] = len(objs)
                mask[m] = 0
        for i in range(len(om)):                       # 같은 물체를 두 번 그린 흔적
            for j in range(i + 1, len(om)):
                inter = int((om[i] & om[j]).sum())
                if inter > 0.5 * min(om[i].sum(), om[j].sum()):
                    warns.append(f"{name}: obstacle {i + 1} ({objs[i]}) and {j + 1} ({objs[j]}) overlap {inter}px")
        frames.append((name, mask, obj, {"reviewed": reviewed, "objects": objs, "object_px": [int(m.sum()) for m in om],
                                         "floor_frac": round(float((mask == 1).mean()), 3),
                                         "ignore_frac": round(float((mask == 255).mean()), 3)}))
except (ValueError, KeyError) as e:
    sys.exit(f"ERROR (nothing written): {e}")

# 2) 옛 출력 지우고 쓰기
os.makedirs(a.out, exist_ok=True)
for old in glob.glob(os.path.join(a.out, "*_mask.png")) + glob.glob(os.path.join(a.out, "*_obj.png")) + glob.glob(os.path.join(a.out, "objects.json")):
    os.remove(old)
for name, mask, obj, info in frames:
    Image.fromarray(mask).save(os.path.join(a.out, name + "_mask.png"))
    Image.fromarray(obj).save(os.path.join(a.out, name + "_obj.png"))
json.dump({n: info for n, _, _, info in frames}, open(os.path.join(a.out, "objects.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
for w in warns:
    print("WARN", w)
print(f"{len(frames)} label files -> {a.out}  (skipped {len(skipped)} not reviewed)")
