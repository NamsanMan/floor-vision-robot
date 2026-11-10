#!/usr/bin/env python3
"""LiDAR 스캔 한 장 + base_footprint 기준 LiDAR 장착 위치를 JSON으로 저장 (로봇 컨테이너, 2026-10-31).
지도에 스캔을 맞춰 초기 위치를 계산하는 데 쓴다(AMCL이 엉뚱한 곳에서 시작했을 때).
usage: python3 scan_grab.py [출력=/home/ubuntu/shared/floor_seg/scan_now.json]"""
import json
import math
import sys
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener

out = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/shared/floor_seg/scan_now.json"
rclpy.init()
n = rclpy.create_node("scan_grab")
buf = Buffer()
TransformListener(buf, n)
scans = []
n.create_subscription(LaserScan, "/scan_raw", lambda m: scans.append(m), qos_profile_sensor_data)
while len(scans) < 5:
    rclpy.spin_once(n, timeout_sec=0.1)
s = scans[-1]
t, end = None, time.time() + 10.0                             # 정적 TF(base→lidar)가 늦게 올 수 있어 기다림(2026-10-31: 부팅 직후 실패)
while t is None:
    try:
        t = buf.lookup_transform("base_footprint", s.header.frame_id, Time())
    except Exception:
        if time.time() > end:
            raise
        rclpy.spin_once(n, timeout_sec=0.1)
q = t.transform.rotation
yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
json.dump({"ranges": [float(r) for r in s.ranges], "angle_min": s.angle_min, "angle_increment": s.angle_increment,
           "range_min": s.range_min, "range_max": s.range_max,
           "lidar_x": t.transform.translation.x, "lidar_y": t.transform.translation.y, "lidar_yaw": yaw}, open(out, "w"))
print(f"저장: {out} · 점 {len(s.ranges)}개 · LiDAR 장착 ({t.transform.translation.x:.3f}, {t.transform.translation.y:.3f}), {math.degrees(yaw):.1f}도")
n.destroy_node()
rclpy.try_shutdown()
