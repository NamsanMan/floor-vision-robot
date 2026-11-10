"""COCO val2017 mAP on Pi 5: NCNN 기본(FP16 연산) vs NCNN FP32(fp16 끔) vs PyTorch.
usage: val_coco.py <ncnn_fp16|ncnn_fp32|pt> <full|500>
같은 640x640 레터박스(rect=False), batch 1, conf 0.001 / iou 0.7 (ultralytics 기본 val 설정)."""
import json
import sys
import time
from pathlib import Path

import ncnn
from ultralytics import YOLO

variant, subset = sys.argv[1], sys.argv[2]
S = Path("/home/ubuntu/shared/yolo_setup")
COCO = Path("/home/ubuntu/shared/datasets/coco")

if variant == "ncnn_fp32":
    # ultralytics는 ncnn.Net()을 그대로 만들고 옵션을 안 건드린다 -> 로드 전에 fp16을 끄도록 클래스를 바꿔치기
    class Net32(ncnn.Net):
        def __init__(self):
            super().__init__()
            for k in ("use_fp16_packed", "use_fp16_storage", "use_fp16_arithmetic", "use_bf16_storage"):
                if hasattr(self.opt, k):
                    setattr(self.opt, k, False)
    ncnn.Net = Net32

# 데이터 yaml (이름은 모델에서, 라벨 인덱스는 ultralytics coco2017labels와 동일한 80클래스 순서)
names = YOLO("/home/ubuntu/yolo11n.pt").names
images = sorted((COCO / "images/val2017").glob("*.jpg"))
lst = COCO / ("val2017_all.txt" if subset == "full" else "val2017_first500.txt")
lst.write_text("\n".join(str(p) for p in (images if subset == "full" else images[:500])) + "\n")
data = S / f"coco_val_{subset}.yaml"
data.write_text(f"path: {COCO}\ntrain: {lst}\nval: {lst}\nnames:\n" +
                "".join(f"  {k}: {v}\n" for k, v in names.items()))

weights = str(S / "bench640/yolo11n_ncnn_model") if variant.startswith("ncnn") else "/home/ubuntu/yolo11n.pt"
model = YOLO(weights, task="detect")
t0 = time.time()
m = model.val(data=str(data), imgsz=640, batch=1, rect=False, device="cpu", plots=False, verbose=False,
              project=str(S / "val_runs"), name=f"{variant}_{subset}", exist_ok=True)
res = {"variant": variant, "subset": subset, "images": len(images) if subset == "full" else 500,
       "mAP50-95": round(float(m.box.map), 5), "mAP50": round(float(m.box.map50), 5),
       "precision": round(float(m.box.mp), 5), "recall": round(float(m.box.mr), 5),
       "speed_ms": {k: round(v, 2) for k, v in m.speed.items()}, "wall_min": round((time.time() - t0) / 60, 1)}
if variant == "ncnn_fp32":
    res["fp16_off_check"] = {k: getattr(ncnn.Net().opt, k) for k in ("use_fp16_storage", "use_fp16_arithmetic")}
print("RESULT " + json.dumps(res))
json.dump(res, open(S / f"results/val_{variant}_{subset}.json", "w"), indent=2)
