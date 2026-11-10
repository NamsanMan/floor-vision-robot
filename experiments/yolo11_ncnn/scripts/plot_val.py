"""val_*.json -> 표 출력 + fig_map.png (FP16이 정확도를 얼마나 깎는가)"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path("/home/ubuntu/shared/yolo_setup/results")
rows = []
for variant, label in [("pt", "PyTorch FP32"), ("ncnn_fp32", "NCNN FP32 (fp16 off)"), ("ncnn_fp16", "NCNN default (fp16)")]:
    for subset in ("500", "full"):
        f = R / f"val_{variant}_{subset}.json"
        if f.exists():
            d = json.load(open(f))
            rows.append((label, subset, d))
            print(f"{label:22s} {subset:>4s}  mAP50-95 {100 * d['mAP50-95']:.2f}  mAP50 {100 * d['mAP50']:.2f}  "
                  f"P {100 * d['precision']:.1f}  R {100 * d['recall']:.1f}  inference {d['speed_ms']['inference']:.0f} ms  "
                  f"wall {d['wall_min']} min")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
colors = {"PyTorch FP32": "#9e9e9e", "NCNN FP32 (fp16 off)": "#f3c08f", "NCNN default (fp16)": "#e8833a"}
for ax, subset, title in [(axes[0], "500", "first 500 images (all three)"), (axes[1], "full", "all 5,000 images")]:
    sel = [(lab, d) for lab, s, d in rows if s == subset]
    for i, (lab, d) in enumerate(sel):
        v = 100 * d["mAP50-95"]
        ax.bar(i, v, color=colors[lab])
        ax.text(i, v + 0.3, f"{v:.2f}\n({d['speed_ms']['inference']:.0f} ms)", ha="center", fontsize=9.5)
    ax.set_xticks(range(len(sel)))
    ax.set_xticklabels([lab for lab, _ in sel], fontsize=9)
    lo = min(100 * d["mAP50-95"] for _, d in sel) if sel else 0
    ax.set_ylim(max(0, lo - 6), lo + 4)
    ax.set_ylabel("COCO val2017 mAP50-95 (%)")
    ax.set_title(f"YOLO11n 640x640 on Pi 5 - {title}", fontsize=10.5)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Does NCNN's default FP16 cost accuracy? (label: mAP, inference ms per image)", fontsize=11)
fig.tight_layout()
fig.savefig(R / "fig_map.png", dpi=120)
print("saved fig_map.png")
