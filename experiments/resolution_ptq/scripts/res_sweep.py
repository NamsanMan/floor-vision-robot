"""DS-Net (FP32) mIoU vs input resolution, reusing the repo's own loader, model and metric.

The loader keeps producing 360x480 images and labels (paper protocol). A wrapper downsizes
the image to the target size (antialiased, like cv2.INTER_AREA on the robot) before the
model; evaluate_all then upsamples logits back to the 360x480 label, as in the paper.

Caller sets before import: SWEEP_DATASET=camvid, SWEEP_FOLD=A_set|B_set,
SWEEP_RESULT_BASE_DIR=<scratch> (config mkdirs a log dir there), HF_HUB_OFFLINE=1.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = os.environ.get("DSNET_REPO", "dsnet_src")   # 논문 학습 코드(비공개) 위치
sys.path.insert(0, REPO)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from data_loader import CamVidDataset, SegmentationTransform
from eval_kd_checkpoint import load_model_checkpoint
from evaluate import evaluate_all
from models import create_model

SIZES = [(360, 480), (300, 400), (270, 360), (240, 320), (180, 240), (120, 160)]
PI_MS = {(360, 480): 346.4, (240, 320): 152.6, (180, 240): 79.1}  # Pi 5, ORT FP32, 2 threads, ROS running


class Resize(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.size = None

    def forward(self, x):
        if self.size is not None and tuple(x.shape[-2:]) != self.size:
            x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False, antialias=True)
        return self.model(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", default="lr,hr")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = create_model(config.KD.STUDENT_NAME).to(dev)
    info = load_model_checkpoint(net, Path(args.ckpt), dev)
    net.eval()
    model = Resize(net).eval()

    dirs = {"lr": config.DATA.TEST_IMG_DIR, "hr": config.DATA.TEST_HR_DIR / "images"}
    names = config.DATA.CLASS_NAMES
    res = {"fold": config.FOLD, "ckpt": args.ckpt, "epoch": info.get("epoch"),
           "student": config.KD.STUDENT_NAME, "device": str(dev), "runs": []}
    print(f"fold={config.FOLD} student={config.KD.STUDENT_NAME} epoch={info.get('epoch')} device={dev}")

    for view in args.views.split(","):
        ds = CamVidDataset(images_dir=str(dirs[view]), masks_dir=str(config.DATA.TEST_LABEL_DIR),
                           transform=SegmentationTransform(config.DATA.INPUT_RESOLUTION))
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        for size in SIZES:
            model.size = size
            t0 = time.time()
            m = evaluate_all(model, loader, dev)
            pc = {names[c]: (None if v != v else round(float(v), 4))
                  for c, v in enumerate(m["per_class_iou"]) if c != config.DATA.IGNORE_INDEX}
            run = {"view": view, "size": list(size), "n": len(ds), "mIoU": round(float(m["mIoU"]), 4),
                   "PixelAcc": round(float(m["PixelAcc"]), 4), "per_class": pc,
                   "pi_ms": PI_MS.get(size)}
            res["runs"].append(run)
            print(f"{view} {size[0]:>3}x{size[1]:<3} mIoU {run['mIoU']:.4f}  PA {run['PixelAcc']:.4f}  "
                  f"({time.time() - t0:.0f}s, n={len(ds)})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved", args.out)


if __name__ == "__main__":
    main()
