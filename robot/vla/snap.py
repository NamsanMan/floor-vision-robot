#!/usr/bin/env python3
"""VLA 실험 1단계 (로봇 컨테이너, 2026-10-31): 카메라 한 장 + 그 순간 로봇의 지도 위치를 저장.
usage: python3 snap.py [출력 폴더=/home/ubuntu/shared/vla]  → snap.jpg, snap.json(촬영 시각, map 위치)"""
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener

out = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/shared/vla"
os.makedirs(out, exist_ok=True)
rclpy.init()
n = rclpy.create_node("vla_snap")
buf = Buffer()
TransformListener(buf, n)
imgs = []
n.create_subscription(Image, "/ascamera/camera_publisher/rgb0/image", lambda m: imgs.append(m),
                      QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST))
end = time.time() + 1.5                                       # TF 버퍼 채우기
while time.time() < end or not imgs:
    rclpy.spin_once(n, timeout_sec=0.05)
m = imgs[-1]
end = time.time() + 1.0                                       # 영상 시각 뒤 TF까지 받기
while time.time() < end:
    rclpy.spin_once(n, timeout_sec=0.05)
pose = None
for when in (Time.from_msg(m.header.stamp), Time()):
    try:
        t = buf.lookup_transform("map", "base_footprint", when)
        q = t.transform.rotation
        pose = [t.transform.translation.x, t.transform.translation.y,
                math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))]
        break
    except Exception:
        continue
img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step // 3, 3)[:, :m.width]
bgr = img if m.encoding == "bgr8" else img[:, :, ::-1]
cv2.imwrite(os.path.join(out, "snap.jpg"), np.ascontiguousarray(bgr), [cv2.IMWRITE_JPEG_QUALITY, 95])
json.dump({"stamp_ns": Time.from_msg(m.header.stamp).nanoseconds, "map_pose": pose}, open(os.path.join(out, "snap.json"), "w"))
print(f"저장: {out}/snap.jpg · map 위치 " + (f"({pose[0]:.3f}, {pose[1]:.3f}, {math.degrees(pose[2]):.1f}도)" if pose else "없음(AMCL 확인)"))
n.destroy_node()
rclpy.try_shutdown()
