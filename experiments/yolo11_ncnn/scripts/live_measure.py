"""/yolo/detections를 N초 구독해 실제 파이프라인의 주기, 카메라 촬영 시각 기준 지연, 검출 클래스를 잰다."""
import collections
import json
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from yolo_msgs.msg import DetectionArray

label, duration = sys.argv[1], float(sys.argv[2])
rclpy.init()
node = Node("yolo_live_probe")
arrivals, latency_ms, n_dets = [], [], []
counts, best = collections.Counter(), {}


def cb(msg):
    arrivals.append(time.monotonic())
    latency_ms.append((node.get_clock().now() - Time.from_msg(msg.header.stamp)).nanoseconds / 1e6)
    n_dets.append(len(msg.detections))
    for d in msg.detections:
        counts[d.class_name] += 1
        best[d.class_name] = round(max(best.get(d.class_name, 0.0), d.score), 3)


node.create_subscription(DetectionArray, "/yolo/detections", cb, 10)
t_end = time.monotonic() + duration
while time.monotonic() < t_end:
    rclpy.spin_once(node, timeout_sec=0.1)

period = np.diff(arrivals) * 1000.0 if len(arrivals) > 1 else np.array([0.0])
lat = np.asarray(latency_ms) if latency_ms else np.array([0.0])
res = {"label": label, "duration_s": duration, "messages": len(arrivals),
       "hz": round((len(arrivals) - 1) / (arrivals[-1] - arrivals[0]), 2) if len(arrivals) > 1 else 0.0,
       "period_ms_mean": round(float(period.mean()), 1), "period_ms_p95": round(float(np.percentile(period, 95)), 1),
       "capture_to_detection_ms_mean": round(float(lat.mean()), 1),
       "capture_to_detection_ms_p95": round(float(np.percentile(lat, 95)), 1),
       "frames_with_detection_pct": round(100.0 * sum(1 for n in n_dets if n) / max(1, len(n_dets)), 1),
       "detections_by_class": dict(counts.most_common()), "best_score": best}
print(json.dumps(res, ensure_ascii=False))
json.dump(res, open(f"/home/ubuntu/shared/yolo_setup/results/live_{label}.json", "w"), indent=2, ensure_ascii=False)
node.destroy_node()
rclpy.shutdown()
