"""YOLO11n-seg: PyTorch vs NCNN(480x640) 마스크 비교 (예시 사진, 같은 레터박스)."""
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import ultralytics
from ultralytics import YOLO

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("/home/ubuntu/shared/yolo_setup/results")
A = Path(ultralytics.__file__).parent / "assets"
models = {"PyTorch .pt": YOLO("/home/ubuntu/yolo11n-seg.pt"),
          "NCNN 480x640 (fp16)": YOLO("/home/ubuntu/yolo11n-seg_ncnn_model", task="segment")}
res = {}
fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
for r, name in enumerate(["bus", "zidane"]):
    img = cv2.imread(str(A / f"{name}.jpg"))
    outs = {k: m.predict(img, imgsz=(480, 640), conf=0.25, rect=False, retina_masks=True, verbose=False)[0]
            for k, m in models.items()}
    (kp, rp), (kn, rn) = outs.items()
    ious = []
    if rp.masks is not None and rn.masks is not None:
        mp, mn = rp.masks.data.numpy() > 0.5, rn.masks.data.numpy() > 0.5
        for i, c in enumerate(rp.boxes.cls.numpy()):
            cand = [j for j, c2 in enumerate(rn.boxes.cls.numpy()) if c2 == c]
            if cand:
                ious.append(max((mp[i] & mn[j]).sum() / max(1, (mp[i] | mn[j]).sum()) for j in cand))
    res[name] = {"n_pytorch": len(rp.boxes), "n_ncnn": len(rn.boxes),
                 "mask_iou_mean": round(float(np.mean(ious)), 4) if ious else None,
                 "mask_iou_min": round(float(np.min(ious)), 4) if ious else None}
    for c, (k, rr) in enumerate(outs.items()):
        axes[r][c].imshow(rr.plot()[:, :, ::-1])
        axes[r][c].set_axis_off()
        t = f"{name} | {k}: {len(rr.boxes)} instances"
        if c == 1 and ious:
            t += f"\nmask IoU vs PyTorch: mean {res[name]['mask_iou_mean']:.3f}, min {res[name]['mask_iou_min']:.3f}"
        axes[r][c].set_title(t, fontsize=9.5)
fig.suptitle("YOLO11n-seg (conf >= 0.25): PyTorch vs NCNN", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig_seg.png", dpi=105)
print(json.dumps(res))
json.dump(res, open(OUT / "seg_check.json", "w"), indent=2)
