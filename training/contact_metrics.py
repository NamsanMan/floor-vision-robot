"""엄격한 지표 (서브컴, spot4 test 30장): 물체 '윗부분은 원래 바닥 아님'이라 후해지는 물체별 recall을 대신한다.
1) 접지 띠 recall: 물체마다 열별 가장 아래 8px(바닥에 닿는 띠)만 보고, 그중 '바닥 아님'으로 맞힌 비율이 50% 이상이면 잡음
2) 주행 가능 거리 오차: 화면 열마다 아래에서 위로 올라가며 바닥이 끊기는 첫 줄(3px 연속 '바닥 아님')을 찾고,
   카메라 보정(높이 0.106m, 2.45도 아래, fy·cy)으로 앞 거리(m)로 바꿔 정답과 비교. 정답 거리 2.5m 안쪽 열만,
   정답 경계가 ignore면 그 열은 뺌. '위험' = 모델이 실제보다 10cm 넘게 더 멀리까지 바닥이라고 한 열(장애물을 늦게 봄)
비교: 미세조정 DS-Net(메인 모델 시드 0~2), 미세조정 전 DS-Net(Road+Sidewalk), EoMT-L 초벌(참고: 라벨이 이걸 고친 것)
다른 스크립트(deploy_int8.py)는 load_gt()·evaluate()를 가져다 쓴다.
usage: ~/mentorpi/venv-teacher/bin/python contact_metrics.py <snap_dir_of_runs> <masks_root> <prelabel_masks_root> <out.json>"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image

LABEL_HW = (480, 640)
H_CAM, PITCH, FY, CY = 0.106, math.radians(2.45), 590.51, 232.73
BAND, RUN, ZMAX, DANGER = 8, 3, 2.5, 0.10


def z_of_row(v):
    a = PITCH + math.atan((v - CY) / FY)
    return H_CAM / math.tan(a) if a > 1e-3 else float("inf")


ZROW = np.array([z_of_row(v) for v in range(LABEL_HW[0])])


def load_gt(masks_root, names, split="test"):
    gt = {n: np.array(Image.open(os.path.join(masks_root, split, n + "_mask.png"))) for n in names}
    obj = {n: np.array(Image.open(os.path.join(masks_root, split, n + "_obj.png"))) for n in names}
    objn = json.load(open(os.path.join(masks_root, split, "objects.json"), encoding="utf-8"))
    return gt, obj, objn


def boundary_rows(floor):
    """열마다 아래에서 위로: 바닥이 끊기는 첫 줄(RUN px 연속 비바닥)"""
    Hh, W = floor.shape
    rows = np.full(W, -1)
    for x in range(W):
        col = floor[:, x]
        run = 0
        for y in range(Hh - 1, -1, -1):
            if not col[y]:
                run += 1
                if run == RUN:
                    rows[x] = y + RUN - 1
                    break
            else:
                run = 0
    return rows


def evaluate(pred_floor_by_name, gt_all, obj_all, objn, names):
    caught, fracs, per_type = [], [], {}
    errs, danger = [], 0
    inter = union = 0
    for n in names:
        pf = pred_floor_by_name[n].astype(bool)
        gt, obj = gt_all[n], obj_all[n]
        valid = gt != 255
        inter += int((pf & (gt == 1) & valid).sum()); union += int(((pf | (gt == 1)) & valid).sum())
        for k, lab in enumerate(objn[n]["objects"], start=1):     # 1) 접지 띠
            m = obj == k
            if not m.any():
                continue
            ys, xs = np.nonzero(m)
            bottom = np.full(LABEL_HW[1], -1)
            np.maximum.at(bottom, xs, ys)
            band = m & (np.arange(LABEL_HW[0])[:, None] >= (bottom[None, :] - BAND + 1)) & (bottom[None, :] >= 0)
            f = float((~pf[band]).mean())
            fracs.append(f); caught.append(f >= 0.5)
            per_type.setdefault(lab.replace("obstacle_", ""), []).append(f >= 0.5)
        rg = boundary_rows(gt == 1)                               # 2) 주행 가능 거리
        rp = boundary_rows(pf)
        for x in range(LABEL_HW[1]):
            if rg[x] < 0 or rp[x] < 0:
                continue
            if (gt[max(rg[x] - RUN + 1, 0):rg[x] + 1, x] == 255).any():   # 정답 경계가 ignore면 뺌
                continue
            zg, zp = ZROW[rg[x]], ZROW[rp[x]]
            if not (zg <= ZMAX):
                continue
            zp = min(zp, 5.0)
            errs.append(zp - zg)
            danger += (zp - zg) > DANGER
    errs = np.array(errs)
    return {"floor_iou": round(inter / max(union, 1), 4),
            "contact_recall@50": round(float(np.mean(caught)), 4), "contact_notfloor_frac_mean": round(float(np.mean(fracs)), 4),
            "n_objects": len(caught), "contact_recall_by_type": {t: f"{sum(v)}/{len(v)}" for t, v in sorted(per_type.items())},
            "freespace_abs_err_cm_median": round(float(np.median(np.abs(errs))) * 100, 1),
            "freespace_abs_err_cm_p90": round(float(np.percentile(np.abs(errs), 90)) * 100, 1),
            "danger_cols_over10cm": round(float(danger / len(errs)), 4), "n_cols": int(len(errs))}


def main():
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import image_frames, predict_logits
    from dsnet_model import CAMVID_ROAD, CAMVID_SIDEWALK, FT_CLASSES, load_camvid_dsnet, load_finetuned

    runs, masks_root, pre_root, out_p = sys.argv[1:5]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    floor_idx = FT_CLASSES.index("floor")
    frames = image_frames("test")
    names = [f["name"] for f in frames]
    gt, obj, objn = load_gt(masks_root, names)
    results = {}
    for hw in ((360, 480), (240, 320)):
        res, seeds = {}, []
        for s in range(3):
            net = load_finetuned(os.path.join(runs, f"main_trainABC_s{s}", "model_final.pth"))[0].to(dev)
            preds = {f["name"]: (lg.argmax(0).numpy() == floor_idx) for f, lg in predict_logits(net, frames, hw, dev)}
            seeds.append(evaluate(preds, gt, obj, objn, names))
        res["fine-tuned (s0)"] = seeds[0]
        res["fine-tuned seeds"] = {k: [r[k] for r in seeds] for k in ("contact_recall@50", "freespace_abs_err_cm_median", "danger_cols_over10cm")}
        zs, _ = load_camvid_dsnet()
        zs = zs.to(dev)
        preds = {f["name"]: np.isin(lg.argmax(0).numpy(), (CAMVID_ROAD, CAMVID_SIDEWALK)) for f, lg in predict_logits(zs, frames, hw, dev)}
        res["zero-shot road+sidewalk"] = evaluate(preds, gt, obj, objn, names)
        results[f"{hw[0]}x{hw[1]}"] = res
    pre = {n: np.array(Image.open(os.path.join(pre_root, "test", n + "_mask.png"))) == 1 for n in names}
    results["EoMT-L draft (reference)"] = evaluate(pre, gt, obj, objn, names)
    json.dump(results, open(out_p, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(results, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
