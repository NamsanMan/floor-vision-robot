"""yolo_ros 실제 실행 결과(live_*.json) + 스냅샷 -> fig_live.png"""
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("/home/ubuntu/shared/yolo_setup/results")
CAMERA_HZ = 12.8  # cam_rate.py: reliable 구독으로 YOLO 실행 중 측정 (YOLO 없이 12.4 Hz, 간격 81 ms)
runs = [("NCNN\nbest effort (course)", "live_ncnn", "#f3c08f"),
        ("NCNN\nreliable", "live_ncnn_rel", "#e8833a"),
        ("PyTorch\nreliable", "live_pt_rel", "#9e9e9e")]
R = {k: json.load(open(OUT / f"{k}.json")) for _, k, _ in runs}

fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
ax = fig.add_subplot(gs[0, 0])
for i, (lab, k, c) in enumerate(runs):
    ax.bar(i, R[k]["hz"], color=c)
    ax.text(i, R[k]["hz"] + 0.25, f"{R[k]['hz']:.1f} Hz", ha="center", fontsize=10)
ax.axhline(CAMERA_HZ, color="crimson", ls="--", lw=1.2)
ax.text(2.45, CAMERA_HZ + 0.25, f"camera {CAMERA_HZ} Hz", color="crimson", ha="right", fontsize=9)
ax.set_xticks(range(len(runs)))
ax.set_xticklabels([r[0] for r in runs], fontsize=9.5)
ax.set_ylim(0, CAMERA_HZ * 1.2)
ax.set_ylabel("/yolo/detections rate (Hz)")
ax.set_title("Detection rate in the running ROS pipeline (30 s each)", fontsize=11)
ax.grid(axis="y", alpha=0.3)

ax = fig.add_subplot(gs[0, 1])
for i, (lab, k, c) in enumerate(runs):
    m, p = R[k]["capture_to_detection_ms_mean"], R[k]["capture_to_detection_ms_p95"]
    ax.bar(i, m, color=c)
    ax.plot([i, i], [m, p], color="black", lw=1)
    ax.plot([i], [p], marker="_", color="black", ms=14)
    ax.text(i + 0.08, p + 12, f"mean {m:.0f} / p95 {p:.0f} ms", ha="center", fontsize=9)
ax.set_xticks(range(len(runs)))
ax.set_xticklabels([r[0] for r in runs], fontsize=9.5)
ax.set_ylabel("camera stamp -> detection published (ms)")
ax.set_title("System latency (bar = mean, whisker = p95)", fontsize=11)
ax.set_ylim(0, 760)
ax.grid(axis="y", alpha=0.3)

for col, (f, title) in enumerate([
        ("live_ncnn.jpg", "Run 1 (darker, mean brightness 77): chair score 0.27 < threshold 0.5 -> no box"),
        ("live_ncnn_rel.jpg", "Later (brighter, 101): chair 0.61-0.76 -> detected, track ID 1")]):
    a = fig.add_subplot(gs[1, col])
    a.imshow(cv2.imread(str(OUT / f))[:, :, ::-1])
    a.set_axis_off()
    a.set_title(title, fontsize=10)
    a.annotate("stacked paper cups: no 'cup' even at conf 0.05", xy=(352, 180), xytext=(150, 400), fontsize=9.5,
               color="white", arrowprops=dict(arrowstyle="->", color="white"),
               bbox=dict(boxstyle="round", fc="black", alpha=0.6))
fig.suptitle("yolo_ros on Raspberry Pi 5 (YOLO11n, 480x640, tracking + debug nodes on, camera 640x480 bgr8 ~0.9 MB/frame)",
             fontsize=11.5)
fig.tight_layout()
fig.savefig(OUT / "fig_live.png", dpi=110)
print("saved", OUT / "fig_live.png")
