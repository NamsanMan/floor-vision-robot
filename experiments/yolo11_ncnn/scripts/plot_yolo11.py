"""bench_yolo11.json + ncnn_precision.json -> fig_latency.png, fig_detections.png"""
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("/home/ubuntu/shared/yolo_setup/results")
R = json.load(open(OUT / "bench_yolo11.json"))
P = json.load(open(OUT / "ncnn_precision.json"))
CAMERA_MS = 1000 / 8.88  # ascamera RGB 실측 8.88 Hz
OFFICIAL_640 = {"PyTorch": 388, "ONNX Runtime": 191, "NCNN (default: fp16)": 94}  # Ultralytics docs, Pi 5, YOLO11n 640
COLORS = {"PyTorch": "#9e9e9e", "ONNX Runtime": "#5b8def", "NCNN FP32 (fp16 off)": "#f3c08f",
          "NCNN (default: fp16)": "#e8833a"}


def rows_for(size):
    rows = []
    for b, key in [("PyTorch", "PyTorch"), ("ONNX Runtime", "ONNX Runtime"), ("NCNN (default: fp16)", "NCNN")]:
        for t in (2, 4):
            r = next((r for r in R["model_only"] if r["size"] == size and r["backend"] == key and r["threads"] == t), None)
            if r:
                rows.append((b, t, r["mean_ms"], r["median_ms"], r["p95_ms"]))
        if size == "480x640" and b == "ONNX Runtime":
            for t in (2, 4):
                r = next(r for r in P["latency_480x640"] if r["variant"] == "NCNN FP32 (fp16 off)" and r["threads"] == t)
                rows.append(("NCNN FP32 (fp16 off)", t, r["mean_ms"], r["median_ms"], r["p95_ms"]))
    return rows


fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), gridspec_kw={"width_ratios": [1.1, 1, 1]})
for ax, size in zip(axes[:2], ["480x640", "640x640"]):
    rows = rows_for(size)
    xmax = max(r[4] for r in rows) * 1.45
    for i, (b, t, mean, med, p95) in enumerate(rows):
        ax.barh(i, mean, color=COLORS[b], alpha=0.6 if t == 2 else 1.0, edgecolor="black" if t == 4 else "none", lw=0.6)
        ax.plot([mean, p95], [i, i], color="black", lw=1)
        ax.plot([p95], [i], marker="|", color="black", ms=8)
        ax.text(p95 + xmax * 0.015, i, f"mean {mean:.0f} / med {med:.0f} / p95 {p95:.0f}", va="center", fontsize=8)
        if size == "640x640" and t == 4 and b in OFFICIAL_640:
            ax.plot([OFFICIAL_640[b]], [i], marker="D", color="crimson", ms=6, zorder=5)
    if size == "640x640":
        ax.plot([], [], marker="D", color="crimson", ls="", label="Ultralytics docs, Pi 5 (course table)")
        ax.legend(loc="lower right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{b}\n{t} threads" for b, t, *_ in rows], fontsize=8.3)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.set_xlabel("ms per frame (bar = mean, whisker = p95)")
    ax.set_title(f"Model only - YOLO11n, input {size}", fontsize=10.5)
    ax.grid(axis="x", alpha=0.3)

ax = axes[2]
e2e = R["e2e"]
names = ["PyTorch", "NCNN"]
for i, n in enumerate(names):
    left = 0.0
    for k, c in [("preprocess", "#bdbdbd"), ("inference", COLORS["PyTorch"] if n == "PyTorch" else COLORS["NCNN (default: fp16)"]),
                 ("postprocess", "#7e57c2")]:
        v = e2e[n]["stages_ms"][k]
        ax.barh(i, v, left=left, color=c, edgecolor="white")
        left += v
    ax.plot([e2e[n]["mean_ms"], e2e[n]["p95_ms"]], [i, i], color="black", lw=1)
    ax.plot([e2e[n]["p95_ms"]], [i], marker="|", color="black", ms=8)
    ax.text(e2e[n]["p95_ms"] + 8, i, f"mean {e2e[n]['mean_ms']:.0f} ms = {e2e[n]['fps_mean']:.1f} FPS\np95 {e2e[n]['p95_ms']:.0f} ms",
            va="center", fontsize=8.5)
ax.axvline(CAMERA_MS, color="crimson", ls="--", lw=1.2)
ax.text(CAMERA_MS + 4, 0.5, f"camera frame interval\n{CAMERA_MS:.0f} ms (8.9 Hz)", color="crimson", fontsize=8.5, va="center")
ax.set_yticks(range(len(names)))
ax.set_yticklabels(["PyTorch .pt", "NCNN 480x640\n(launch default now)"], fontsize=8.5)
ax.invert_yaxis()
ax.set_xlim(0, 1150)
ax.set_xlabel("ms per frame: preprocess | inference | postprocess (NMS)")
ax.set_title("Full ultralytics predict, robot frame 640x480, conf 0.5", fontsize=10.5)
ax.grid(axis="x", alpha=0.3)
env = R["env"]
fig.suptitle(f"Raspberry Pi 5 CPU, ROS bringup running, robot idle | ultralytics {env['ultralytics']}, torch {env['torch']}, "
             f"ncnn {env['ncnn']} | 20 warm-up + 50 timed | SoC {env['temp_start_c']:.0f}-{env['temp_end_c']:.0f} C, no throttling",
             fontsize=9.5)
fig.tight_layout()
fig.savefig(OUT / "fig_latency.png", dpi=130)

imgs = [n for n in ["robot_box", "bus", "zidane"] if n in R["consistency"]]
fig, axes = plt.subplots(len(imgs), 2, figsize=(11, 3.9 * len(imgs)))
for r, name in enumerate(imgs):
    c = R["consistency"][name]
    for col, (be, key) in enumerate([("PyTorch .pt", "pytorch"), ("NCNN 480x640 (fp16)", "ncnn")]):
        im = cv2.imread(str(OUT / f"det_{name}_{key}.jpg"))[:, :, ::-1]
        a = axes[r][col]
        a.imshow(im)
        a.set_axis_off()
        n = c["n_pytorch"] if key == "pytorch" else c["n_ncnn"]
        title = f"{name} | {be}: {n} boxes"
        if key == "ncnn" and c["matched"]:
            title += f"\nmatched {c['matched']}/{c['n_pytorch']}, mean IoU {c['mean_iou']:.3f}, max |d conf| {c['max_conf_diff']:.3f}"
        a.set_title(title, fontsize=9)
fig.suptitle("YOLO11n detections, conf >= 0.25, same 480x640 letterbox: PyTorch vs NCNN", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig_detections.png", dpi=110)
print("saved figures")
