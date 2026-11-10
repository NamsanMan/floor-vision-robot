"""Baselines on every labeled position group, scored against the user's labels exactly like the fine-tuned model:
  (a) DS-Net zero-shot = the CamVid checkpoint as is, floor = CamVid Road (3), or Road + Sidewalk (3, 4)
      (the earlier zero-shot experiments of this project counted Road + Sidewalk as drivable);
  (b) EoMT-L pre-label masks (the automatic first draft the user corrected) vs the user's labels.
Also saves the zero-shot 12-class argmax maps (labeled frames + every test image) for figures.

usage: python baselines.py --masks <snap>/masks --prelabel <prelabel_masks_root> --out <runs>/<snap>/baselines
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (Evaluator, dump_json, image_frames, labeled_frames, load_mask, parse_hw,  # noqa: E402
                    predict_logits, save_mask_png)
from dsnet_model import CAMVID_CLASSES, CAMVID_ROAD, CAMVID_SIDEWALK, load_camvid_dsnet  # noqa: E402

MAPPINGS = {"road": (CAMVID_ROAD,), "road+sidewalk": (CAMVID_ROAD, CAMVID_SIDEWALK)}


def results(evs):
    out = {g: e.result() for g, e in evs.items() if g != "all"}
    out["all"] = evs["all"].result()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True)
    ap.add_argument("--prelabel", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-hw", default="360x480,240x320")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "zs_class"), exist_ok=True)
    dev = torch.device("cuda")
    frames = labeled_frames(a.masks)
    groups = sorted({f["group"] for f in frames})
    print("labeled frames per group:", {g: sum(f["group"] == g for f in frames) for g in groups}, flush=True)
    res = {"groups": {g: [f["name"] for f in frames if f["group"] == g] for g in groups},
           "zero_shot": {}, "eomt_prelabel": {}, "zero_shot_class_hist": {}, "user_correction": {}}

    net, info = load_camvid_dsnet()
    net.to(dev)
    res["zero_shot_init"] = info
    hws = [parse_hw(s) for s in a.eval_hw.split(",") if s]
    for k, hw in enumerate(hws):
        key = f"{hw[0]}x{hw[1]}"
        evs = {m: {g: Evaluator() for g in groups + ["all"]} for m in MAPPINGS}
        hist = {g: np.zeros((2, len(CAMVID_CLASSES)), np.int64) for g in groups + ["all"]}
        for f, lg in predict_logits(net, frames, hw, dev):
            cls = lg.argmax(0).numpy()
            gt, obj = load_mask(f["mask"]), load_mask(f["obj"])
            objs = (f["info"] or {}).get("objects")
            for m, ids in MAPPINGS.items():
                pred = np.isin(cls, ids)
                for g in (f["group"], "all"):
                    evs[m][g].add(f["name"], pred, gt, obj, objs)
            for g in (f["group"], "all"):
                for t in (0, 1):
                    hist[g][t] += np.bincount(cls[gt == t], minlength=len(CAMVID_CLASSES))
            if k == 0:
                save_mask_png(os.path.join(a.out, "zs_class", f["name"] + ".png"), cls)
        res["zero_shot"][key] = {m: results(evs[m]) for m in MAPPINGS}
        res["zero_shot_class_hist"][key] = {
            g: {t_name: {c: round(float(h[t][i] / max(1, h[t].sum())), 4) for i, c in enumerate(CAMVID_CLASSES)}
                for t, t_name in ((1, "gt_floor_px"), (0, "gt_notfloor_px"))} for g, h in hist.items()}
    # zero-shot class maps for every test image (spot4), labeled or not, for figures
    done = {f["name"] for f in frames}
    todo = [f for f in image_frames("test") if f["name"] not in done]
    for f, lg in predict_logits(net, todo, hws[0], dev):
        save_mask_png(os.path.join(a.out, "zs_class", f["name"] + ".png"), lg.argmax(0).numpy())

    evs = {g: Evaluator() for g in groups + ["all"]}
    for f in frames:
        pre = load_mask(os.path.join(a.prelabel, f["split"], f["name"] + "_mask.png"))
        gt, obj = load_mask(f["mask"]), load_mask(f["obj"])
        objs = (f["info"] or {}).get("objects")
        for g in (f["group"], "all"):
            evs[g].add(f["name"], pre == 1, gt, obj, objs)
        changed = int(((pre == 1) != (gt == 1)).sum())
        res["user_correction"][f["name"]] = {"changed_px": changed, "changed_frac": round(changed / gt.size, 5)}
    res["eomt_prelabel"]["480x640"] = results(evs)
    for g in groups + ["all"]:
        names = [f["name"] for f in frames if g == "all" or f["group"] == g]
        ch = [res["user_correction"][n]["changed_px"] for n in names]
        res["user_correction"][f"_group_{g}"] = {"frames": len(names), "frames_changed": int(sum(c > 0 for c in ch)),
                                                 "changed_px_frac": round(float(np.sum(ch)) / (len(names) * 480 * 640), 5)}
    dump_json(res, os.path.join(a.out, "baselines.json"))
    for key, r in res["zero_shot"].items():
        for m, rr in r.items():
            print(f"zero-shot {m:14s} [{key}] " + "  ".join(
                f"{g}: fIoU {rr[g]['summary']['floor_iou']:.4f} mIoU {rr[g]['summary']['miou']:.4f}" for g in groups + ["all"]))
    rr = res["eomt_prelabel"]["480x640"]
    print("EoMT-L pre-label       " + "  ".join(
        f"{g}: fIoU {rr[g]['summary']['floor_iou']:.4f} mIoU {rr[g]['summary']['miou']:.4f}" for g in groups + ["all"]))


if __name__ == "__main__":
    main()
