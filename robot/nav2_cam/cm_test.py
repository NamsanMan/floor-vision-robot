#!/usr/bin/env python3
"""Collision Monitor 반응 시험 (로봇 컨테이너, 2026-10-29). 로봇을 움직이지 않도록 아주 작은 속도(기본 0.001 m/s = 1 mm/s,
모터가 돌지 않는 크기)를 cmd_vel_smoothed(Collision Monitor 입력)에 dur초 동안 10Hz로 넣고, Collision Monitor 출력(/cmd_vel)을 기록한다.
  출력이 0이면 = 정지, 입력의 0.3배면 = 감속(PolygonSlow), 입력 그대로면 = 통과. 끝나면 0을 몇 번 보내 마무리.
usage: python3 cm_test.py <태그> [초=2] [속도=0.001]"""
import collections
import sys
import time

import rclpy
from geometry_msgs.msg import Twist

tag = sys.argv[1] if len(sys.argv) > 1 else "test"
dur = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
v = float(sys.argv[3]) if len(sys.argv) > 3 else 0.001
assert abs(v) <= 0.005, "안전: 5 mm/s 넘는 속도는 쓰지 않음"
rclpy.init()
n = rclpy.create_node("cm_test")
pub = n.create_publisher(Twist, "/cmd_vel_smoothed", 10)
got = []
n.create_subscription(Twist, "/cmd_vel", lambda m: got.append(round(m.linear.x, 6)), 10)
t0 = time.time()
while time.time() - t0 < 0.5:                      # 연결 잡기
    rclpy.spin_once(n, timeout_sec=0.05)
got.clear()
msg = Twist()
msg.linear.x = v
t0, last = time.time(), 0.0
while time.time() - t0 < dur:
    if time.time() - last >= 0.1:
        pub.publish(msg)
        last = time.time()
    rclpy.spin_once(n, timeout_sec=0.02)
stop = Twist()
for _ in range(5):
    pub.publish(stop)
    rclpy.spin_once(n, timeout_sec=0.05)
c = collections.Counter(got[:-5] if len(got) > 5 else got)
kind = "출력 없음" if not c else ("정지(0)" if set(c) == {0.0} else ("통과" if max(c) >= v * 0.99 else "감속"))
print(f"[{tag}] 입력 {v} m/s × {dur:.0f}초 → /cmd_vel 출력 {dict(c)} → {kind}", flush=True)
n.destroy_node()
rclpy.try_shutdown()
