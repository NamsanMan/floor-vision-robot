#!/usr/bin/env python3
"""/tf에 실제로 오가는 (부모 → 자식) 좌표계 쌍과 빈도 (로봇 컨테이너, 2026-10-31). map→odom이 안 보일 때 확인용.
usage: python3 tf_frames_check.py [초=4]"""
import collections
import sys
import time

import rclpy
from tf2_msgs.msg import TFMessage

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
rclpy.init()
n = rclpy.create_node("tf_frames_check")
cnt = collections.Counter()
n.create_subscription(TFMessage, "/tf", lambda m: cnt.update((t.header.frame_id, t.child_frame_id) for t in m.transforms), 100)
end = time.time() + secs
while time.time() < end:
    rclpy.spin_once(n, timeout_sec=0.05)
for (p, c), k in sorted(cnt.items()):
    print(f"{p} → {c}: {k / secs:.1f}/초")
if not any(p == "map" for p, _ in cnt):
    print("map → (무엇)도 없음")
n.destroy_node()
rclpy.try_shutdown()
