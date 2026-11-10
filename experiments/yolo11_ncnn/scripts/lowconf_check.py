"""실행 중 캡처한 장면에서 임계값을 낮추면 무엇이 몇 점으로 나오는지 (yolo_ros 기본 threshold 0.5)."""
import sys

import cv2
from ultralytics import YOLO

img = cv2.imread(sys.argv[1])
for name, path in [("NCNN", "/home/ubuntu/yolo11n_ncnn_model"), ("PyTorch", "/home/ubuntu/yolo11n.pt")]:
    r = YOLO(path, task="detect").predict(img, imgsz=(480, 640), conf=0.05, verbose=False)[0]
    found = sorted(((float(s), r.names[int(c)], [int(v) for v in b]) for s, c, b in
                    zip(r.boxes.conf, r.boxes.cls, r.boxes.xyxy)), reverse=True)
    print(f"{name}: " + (", ".join(f"{n} {s:.2f} @x{b[0]}-{b[2]},y{b[1]}-{b[3]}" for s, n, b in found[:8]) or "nothing >= 0.05"))
    if name == "NCNN":
        cv2.imwrite(sys.argv[2], r.plot(conf=True, line_width=1))
