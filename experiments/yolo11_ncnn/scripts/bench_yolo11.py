"""YOLO11n 최적화 검증 (Pi 5 CPU, bringup 가동 중, 로봇 정지 상태).
1) PyTorch vs NCNN 검출 결과 일치  2) 모델만 지연: PyTorch / NCNN / ONNX Runtime  3) predict 전체 지연
측정 규약은 기존 벤치마크와 동일: 20 warm-up + 50 timed, batch 1."""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import ncnn
import numpy as np
import torch
import torch.onnx._internal.onnx_proto_utils as _opu
import ultralytics
from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Detect

S = Path("/home/ubuntu/shared/yolo_setup")
OUT = S / "results"
OUT.mkdir(parents=True, exist_ok=True)
PT = "/home/ubuntu/yolo11n.pt"
NCNN = {(480, 640): "/home/ubuntu/yolo11n_ncnn_model", (640, 640): str(S / "bench640/yolo11n_ncnn_model")}
VENV_PY = "/home/ubuntu/bench_venv/bin/python"
ORT_BENCH = "/home/ubuntu/shared/bench/bench_ort.py"
WARMUP, ITERS = 20, 50
ASSETS = Path(ultralytics.__file__).parent / "assets"
IMAGES = {
    "robot_corridor": "/home/ubuntu/shared/capture/20261007_163444/rgb_0.png",
    "robot_box": "/home/ubuntu/shared/capture/20261007_164937/rgb_0.png",
    "bus": str(ASSETS / "bus.jpg"),
    "zidane": str(ASSETS / "zidane.jpg"),
}
R = {"env": {}, "consistency": {}, "numeric": {}, "model_only": [], "e2e": {}}


def temp_c():
    try:
        return int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000
    except OSError:
        return None


def stats(ts):
    a = np.asarray(ts)
    return {"mean_ms": round(float(a.mean()), 2), "median_ms": round(float(np.median(a)), 2),
            "p95_ms": round(float(np.percentile(a, 95)), 2), "min_ms": round(float(a.min()), 2),
            "max_ms": round(float(a.max()), 2), "std_ms": round(float(a.std()), 2),
            "fps_mean": round(1000.0 / float(a.mean()), 2)}


def timeit(fn):
    for _ in range(WARMUP):
        fn()
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return stats(ts)


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def compare(rp, rn):
    P = list(zip(rp.boxes.xyxy.numpy(), rp.boxes.cls.numpy().astype(int), rp.boxes.conf.numpy()))
    N = list(zip(rn.boxes.xyxy.numpy(), rn.boxes.cls.numpy().astype(int), rn.boxes.conf.numpy()))
    used, ious, dconf = set(), [], []
    for b, c, s in sorted(P, key=lambda t: -t[2]):
        best, bj = 0.0, None
        for j, (b2, c2, _) in enumerate(N):
            if j not in used and c2 == c and iou(b, b2) > best:
                best, bj = iou(b, b2), j
        if bj is not None and best >= 0.5:
            used.add(bj)
            ious.append(best)
            dconf.append(abs(float(s) - float(N[bj][2])))
    names = rp.names
    return {"n_pytorch": len(P), "n_ncnn": len(N), "matched": len(ious),
            "mean_iou": round(float(np.mean(ious)), 4) if ious else None,
            "min_iou": round(float(np.min(ious)), 4) if ious else None,
            "max_conf_diff": round(float(np.max(dconf)), 4) if dconf else None,
            "pytorch": sorted(f"{names[c]} {s:.2f}" for _, c, s in P),
            "ncnn": sorted(f"{names[c]} {s:.2f}" for _, c, s in N)}


def export_graph(net):
    """ultralytics Exporter와 같은 그래프(Detect export 모드, C2f forward_split)."""
    for m in net.modules():
        if isinstance(m, Detect):
            m.dynamic, m.export, m.format = False, True, "onnx"
        elif isinstance(m, C2f) and hasattr(m, "forward_split"):
            m.forward = m.forward_split
    return net


def ncnn_runner(model_dir, threads):
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = threads
    net.load_param(f"{model_dir}/model.ncnn.param")
    net.load_model(f"{model_dir}/model.ncnn.bin")

    def run(x_chw):
        mat = ncnn.Mat(x_chw)
        with net.create_extractor() as ex:
            if hasattr(ex, "set_num_threads"):
                ex.set_num_threads(threads)
            ex.input("in0", mat)
            _, out = ex.extract("out0")
        return np.array(out)
    return run


def first(y):
    return (y[0] if isinstance(y, (list, tuple)) else y).numpy()


R["env"] = {"ultralytics": ultralytics.__version__, "torch": torch.__version__, "ncnn": ncnn.__version__,
            "numpy": np.__version__, "load_avg_start": os.getloadavg(), "temp_start_c": temp_c(),
            "protocol": f"{WARMUP} warm-up + {ITERS} timed, batch 1, ROS bringup running, robot idle"}

# 0) 640x640 NCNN (Ultralytics 공식 벤치마크와 같은 입력, 비교용) - 480x640 모델은 덮어쓰지 않도록 별도 폴더
d640 = S / "bench640"
if not (d640 / "yolo11n_ncnn_model" / "model.ncnn.bin").exists():
    d640.mkdir(exist_ok=True)
    shutil.copy(PT, d640 / "yolo11n.pt")
    cwd = os.getcwd()
    os.chdir(d640)
    YOLO(str(d640 / "yolo11n.pt")).export(format="ncnn", imgsz=640, device="cpu")
    os.chdir(cwd)

# 1) 검출 결과 일치 (같은 480x640 레터박스로 맞춤: rect=False)
pt_model, nc_model = YOLO(PT), YOLO(NCNN[(480, 640)], task="detect")
for name, path in IMAGES.items():
    if not os.path.exists(path):
        print("skip (missing):", path)
        continue
    img = cv2.imread(path)
    kw = dict(imgsz=(480, 640), conf=0.25, rect=False, verbose=False)
    rp, rn = pt_model.predict(img, **kw)[0], nc_model.predict(img, **kw)[0]
    R["consistency"][name] = compare(rp, rn)
    cv2.imwrite(str(OUT / f"det_{name}_pytorch.jpg"), rp.plot())
    cv2.imwrite(str(OUT / f"det_{name}_ncnn.jpg"), rn.plot())
    print(name, {k: v for k, v in R["consistency"][name].items() if k not in ("pytorch", "ncnn")}, flush=True)

# 2) 모델만 지연 + 수치 일치 (같은 입력 텐서)
_opu._add_onnxscript_fn = lambda model_bytes, custom_opsets: model_bytes  # onnx 패키지 없이 export (export_det.py와 동일)
net = export_graph(YOLO(PT).model.fuse(verbose=False).eval().float())
frame = cv2.imread(IMAGES["robot_box"])
(S / "onnx").mkdir(exist_ok=True)
for (h, w) in [(480, 640), (640, 640)]:
    img = frame if (h, w) == frame.shape[:2] else cv2.resize(frame, (w, h))
    x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0
    xt = torch.from_numpy(x)[None]
    with torch.inference_mode():
        y_pt = first(net(xt))[0]
    onnx_f = S / "onnx" / f"yolo11n_{h}x{w}.onnx"
    if not onnx_f.exists():
        torch.onnx.export(net, torch.zeros(1, 3, h, w), str(onnx_f), opset_version=17,
                          input_names=["images"], output_names=["output0"], do_constant_folding=True)
    np.save(S / f"x_{h}x{w}.npy", x[None])
    np.save(S / f"y_pt_{h}x{w}.npy", y_pt[None])
    y_nc = ncnn_runner(NCNN[(h, w)], 4)(x)
    chk = subprocess.run([VENV_PY, str(S / "ort_check.py"), str(onnx_f), str(S / f"x_{h}x{w}.npy"),
                          str(S / f"y_pt_{h}x{w}.npy")], capture_output=True, text=True)
    R["numeric"][f"{h}x{w}"] = {
        "output_shape": list(y_pt.shape),
        "ncnn_vs_pytorch_max_abs_diff": float(np.abs(y_nc.reshape(y_pt.shape) - y_pt).max()),
        "ort": json.loads(chk.stdout.strip().splitlines()[-1]) if chk.returncode == 0 else chk.stderr[-500:]}
    print("numeric", h, w, R["numeric"][f"{h}x{w}"], flush=True)
    for threads in (4, 2):
        torch.set_num_threads(threads)
        with torch.inference_mode():
            st = timeit(lambda: net(xt))
        R["model_only"].append({"backend": "PyTorch", "size": f"{h}x{w}", "threads": threads, **st, "temp_c": temp_c()})
        print(R["model_only"][-1], flush=True)
        runner = ncnn_runner(NCNN[(h, w)], threads)
        st = timeit(lambda: runner(x))
        R["model_only"].append({"backend": "NCNN", "size": f"{h}x{w}", "threads": threads, **st, "temp_c": temp_c()})
        print(R["model_only"][-1], flush=True)
        o = subprocess.run([VENV_PY, ORT_BENCH, str(onnx_f), str(threads)], capture_output=True, text=True)
        st = json.loads(o.stdout.strip().splitlines()[-1])
        R["model_only"].append({"backend": "ONNX Runtime", "size": f"{h}x{w}", "threads": threads,
                                **{k: v for k, v in st.items() if k not in ("threads", "input")}, "temp_c": temp_c()})
        print(R["model_only"][-1], flush=True)
torch.set_num_threads(4)

# 3) predict 전체 (전처리 + 추론 + NMS), yolo_ros 기본 threshold 0.5, 로봇 카메라 프레임 640x480
for label, model in [("PyTorch", pt_model), ("NCNN", nc_model)]:
    speeds = []

    def run():
        r = model.predict(frame, imgsz=(480, 640), conf=0.5, verbose=False)[0]
        speeds.append(r.speed)
    st = timeit(run)
    stages = {k: round(float(np.mean([s[k] for s in speeds[WARMUP:]])), 2) for k in speeds[0]}
    R["e2e"][label] = {**st, "stages_ms": stages, "temp_c": temp_c()}
    print("e2e", label, R["e2e"][label], flush=True)

R["env"]["load_avg_end"] = os.getloadavg()
R["env"]["temp_end_c"] = temp_c()
json.dump(R, open(OUT / "bench_yolo11.json", "w"), indent=2, ensure_ascii=False)
print("saved", OUT / "bench_yolo11.json")
