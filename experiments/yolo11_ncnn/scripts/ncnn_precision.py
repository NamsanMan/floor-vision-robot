"""NCNN 기본 옵션이 Pi 5(Cortex-A76, ARMv8.2 FP16)에서 FP16 연산을 켜는지 확인하고,
FP16을 끈 순수 FP32 NCNN과 속도·수치를 비교한다. 입력: bus.jpg를 ultralytics와 같은 480x640 레터박스로."""
import json
import subprocess
import time
from pathlib import Path

import cv2
import ncnn
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.nn.modules import C2f, Detect

S = Path("/home/ubuntu/shared/yolo_setup")
MODEL = "/home/ubuntu/yolo11n_ncnn_model"
FP16_KEYS = ("use_fp16_packed", "use_fp16_storage", "use_fp16_arithmetic", "use_bf16_storage")
WARMUP, ITERS = 20, 50

img = LetterBox((480, 640), auto=False)(image=cv2.imread(str(Path(ultralytics.__file__).parent / "assets/bus.jpg")))
x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0

net_pt = YOLO("/home/ubuntu/yolo11n.pt").model.fuse(verbose=False).eval().float()
for m in net_pt.modules():
    if isinstance(m, Detect):
        m.dynamic, m.export, m.format = False, True, "onnx"
    elif isinstance(m, C2f) and hasattr(m, "forward_split"):
        m.forward = m.forward_split
with torch.inference_mode():
    y = net_pt(torch.from_numpy(x)[None])
y = (y[0] if isinstance(y, (list, tuple)) else y).numpy()[0]  # (84, 6300)
np.save(S / "x_bus_480x640.npy", x[None])
np.save(S / "y_pt_bus_480x640.npy", y[None])
relevant = y[4:].max(0) > 0.25  # 실제 검출 후보가 되는 앵커만


def make(threads, fp32):
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = threads
    if fp32:
        for k in FP16_KEYS:
            if hasattr(net.opt, k):
                setattr(net.opt, k, False)
    net.load_param(f"{MODEL}/model.ncnn.param")
    net.load_model(f"{MODEL}/model.ncnn.bin")
    return net


def run(net):
    with net.create_extractor() as ex:
        ex.input("in0", ncnn.Mat(x))
        _, out = ex.extract("out0")
    return np.array(out).reshape(y.shape)


def diffs(o):
    return {"score_max_abs": round(float(np.abs(o[4:] - y[4:]).max()), 5),
            "box_max_abs_px_all_anchors": round(float(np.abs(o[:4] - y[:4]).max()), 3),
            "box_max_abs_px_conf025_anchors": round(float(np.abs(o[:4, relevant] - y[:4, relevant]).max()), 3),
            "n_conf025_anchors": int(relevant.sum())}


def bench(net):
    for _ in range(WARMUP):
        run(net)
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        run(net)
        ts.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(ts)
    return {"mean_ms": round(float(a.mean()), 2), "median_ms": round(float(np.median(a)), 2),
            "p95_ms": round(float(np.percentile(a, 95)), 2), "fps_mean": round(1000 / float(a.mean()), 2)}


R = {"ncnn": ncnn.__version__,
     "default_opts": {k: getattr(ncnn.Net().opt, k) for k in dir(ncnn.Net().opt) if k.startswith("use_")},
     "numeric_bus": {}, "latency_480x640": []}
for fp32 in (False, True):
    label = "NCNN FP32 (fp16 off)" if fp32 else "NCNN default"
    R["numeric_bus"][label] = diffs(run(make(4, fp32)))
    for threads in (4, 2):
        R["latency_480x640"].append({"variant": label, "threads": threads, **bench(make(threads, fp32))})
        print(R["latency_480x640"][-1], flush=True)
chk = subprocess.run(["/home/ubuntu/bench_venv/bin/python", str(S / "ort_check.py"), str(S / "onnx/yolo11n_480x640.onnx"),
                      str(S / "x_bus_480x640.npy"), str(S / "y_pt_bus_480x640.npy")], capture_output=True, text=True)
R["numeric_bus"]["ONNX Runtime (FP32)"] = json.loads(chk.stdout.strip().splitlines()[-1]) if chk.returncode == 0 else chk.stderr[-300:]
print(json.dumps({k: v for k, v in R.items() if k != "latency_480x640"}, indent=1))
json.dump(R, open(S / "results/ncnn_precision.json", "w"), indent=2)
