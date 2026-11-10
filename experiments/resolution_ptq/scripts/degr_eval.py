"""FP32 vs INT8 (PTQ, calibrated on training images) on the repo's unseen-degradation CamVid sets.

Reuses the ONNX files written by ptq_eval.py and the same protocol (360x480 label, antialiased
downsizing to the model input, evaluate_all). Images of a condition are decoded once and every
model is evaluated on the same tensors.

Caller sets before import: SWEEP_DATASET=camvid, SWEEP_FOLD=<fold>, SWEEP_RESULT_BASE_DIR=<scratch>,
HF_HUB_OFFLINE=1.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = os.environ.get("DSNET_REPO", "dsnet_src")   # 논문 학습 코드(비공개) 위치
sys.path.insert(0, REPO)

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from data_loader import CamVidDataset, SegmentationTransform
from evaluate import evaluate_all

ROOTS = [Path(os.environ.get("CAMVID_DEGR", "data/CamVid_unseen_degradation")),
         Path(os.environ.get("CAMVID_DEGR", "data/CamVid_unseen_degradation") + "_edge")]
LABEL_FALLBACK = ROOTS[0]


def down(x, size):
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)


class OrtModel(nn.Module):
    def __init__(self, path, size):
        super().__init__()
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.name = self.sess.get_inputs()[0].name
        self.size = size

    def forward(self, x):
        x = down(x, self.size).cpu().numpy().astype(np.float32)
        outs = [self.sess.run(None, {self.name: x[i:i + 1]})[0] for i in range(x.shape[0])]
        return torch.from_numpy(np.concatenate(outs, 0))


def conditions(fold):
    jobs = []
    for root in ROOTS:
        parent = root / fold / "test" / "images"
        labels = root / fold / "test" / "labels"
        if not labels.is_dir():
            labels = LABEL_FALLBACK / fold / "test" / "labels"
        for d in sorted(p for p in parent.iterdir() if p.is_dir()):
            if any(d.glob("*.png")):
                jobs.append((root.name, d.name, d, labels))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ptq-dir", required=True)
    ap.add_argument("--sizes", default="360x480,240x320")
    ap.add_argument("--variants", default="fp32,int8_minmax,int8_percentile")
    args = ap.parse_args()
    ptq = Path(args.ptq_dir)
    torch.set_grad_enabled(False)

    models = {}
    for s in args.sizes.split(","):
        h, w = map(int, s.split("x"))
        for v in args.variants.split(","):
            models[(s, v)] = OrtModel(ptq / f"{args.tag}_{s}_{v}.onnx", (h, w))

    out = ptq / f"{args.tag}_degr.json"
    res = {"tag": args.tag, "fold": config.FOLD, "rows": []}
    tf = SegmentationTransform(config.DATA.INPUT_RESOLUTION)
    jobs = conditions(config.FOLD)
    print(f"{args.tag}: fold={config.FOLD} conditions={len(jobs)} models={len(models)}", flush=True)
    for root, cond, img_dir, lbl_dir in jobs:
        t0 = time.time()
        ds = CamVidDataset(str(img_dir), str(lbl_dir), transform=tf)
        data = [(img, mask.to(torch.uint8)) for img, mask in (ds[i] for i in range(len(ds)))]
        loader = DataLoader(data, batch_size=8, shuffle=False, num_workers=0)
        row = {"root": root, "condition": cond, "n": len(ds)}
        for (s, v), model in models.items():
            row[f"{s}/{v}"] = round(float(evaluate_all(model, loader, torch.device("cpu"))["mIoU"]), 4)
        res["rows"].append(row)
        out.write_text(json.dumps(res, indent=1), encoding="utf-8")
        parts = []
        for s in args.sizes.split(","):
            f32 = row[f"{s}/fp32"]
            parts.append(f"{s} fp32 {f32 * 100:5.2f} " + " ".join(
                f"{v.replace('int8_', '')} {(row[f'{s}/{v}'] - f32) * 100:+5.2f}"
                for v in args.variants.split(",") if v != "fp32"))
        print(f"  {cond:<30} " + " | ".join(parts) + f"  ({time.time() - t0:.0f}s)", flush=True)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
