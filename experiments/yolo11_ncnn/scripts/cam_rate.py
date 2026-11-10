"""카메라 원본 토픽의 발행 주기: reliable 구독(손실 없이 전부 받음) + 직렬화 바이트에서 header.stamp만 직접 읽음."""
import struct
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

duration = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
rclpy.init()
node = Node("cam_rate_probe")
arrival, stamp = [], []


def cb(raw):
    arrival.append(time.monotonic())
    sec, nsec = struct.unpack_from("<iI", raw, 4)  # CDR 헤더 4바이트 뒤가 header.stamp
    stamp.append(sec + nsec * 1e-9)


node.create_subscription(Image, "/ascamera/camera_publisher/rgb0/image", cb,
                         QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE), raw=True)
det = []
try:
    from yolo_msgs.msg import DetectionArray
    node.create_subscription(DetectionArray, "/yolo/detections", lambda m: det.append(time.monotonic()), 10)
except ImportError:
    pass
t_end = time.monotonic() + duration
while time.monotonic() < t_end:
    rclpy.spin_once(node, timeout_sec=0.05)
d = np.diff(stamp) * 1000
det_hz = (len(det) - 1) / (det[-1] - det[0]) if len(det) > 1 else 0.0
print(f"frames {len(stamp)} in {duration:.0f}s | arrival {(len(arrival) - 1) / (arrival[-1] - arrival[0]):.2f} Hz | "
      f"stamp interval mean {d.mean():.0f} / median {np.median(d):.0f} / p95 {np.percentile(d, 95):.0f} / max {d.max():.0f} ms"
      f" | detections {det_hz:.2f} Hz")
node.destroy_node()
rclpy.shutdown()
