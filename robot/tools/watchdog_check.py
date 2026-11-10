#!/usr/bin/env python3
"""감시 노드 출력 확인 (로봇 컨테이너): /floor_obstacles/watchdog 점 개수를 0.1초 간격으로 dur초 동안 기록해,
정지 신호(점 9개)가 언제 켜지고 꺼졌는지 출력한다. usage: python3 watchdog_check.py <초>"""
import sys
import time

import rclpy
from sensor_msgs.msg import PointCloud2

dur = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
rclpy.init()
n = rclpy.create_node("watchdog_check")
st = {"n": None, "t": None}
n.create_subscription(PointCloud2, "/floor_obstacles/watchdog", lambda m: st.update(n=m.width * m.height, t=time.time()), 5)
t0, prev, changes, samples = time.time(), None, [], 0
while time.time() - t0 < dur:
    rclpy.spin_once(n, timeout_sec=0.05)
    if st["n"] is not None and st["n"] != prev:
        changes.append((time.time() - t0, st["n"]))
        prev = st["n"]
    samples += 1
print("감시 신호 변화 (시작 후 초, 점 개수):", [(round(a, 1), b) for a, b in changes] or "메시지 없음")
n.destroy_node()
rclpy.try_shutdown()
