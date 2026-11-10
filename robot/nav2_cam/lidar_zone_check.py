#!/usr/bin/env python3
"""Collision Monitor 정지·감속 구역 안의 LiDAR 점 확인 (로봇 컨테이너, 2026-10-29).
/scan_raw를 TF로 base_footprint 좌표에 옮겨, PolygonStop(x 0~0.16, |y|≤0.10)·PolygonSlow(x 0~0.40, |y|≤0.14) 안에 든 점을 센다.
로봇 자기 몸(카메라 받침 등)이 LiDAR에 잡히는지, 아니면 실제로 가까운 물체인지 각도·거리로 본다.
usage: python3 lidar_zone_check.py [스캔 수=10]"""
import math
import sys

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
rclpy.init()
n = rclpy.create_node("lidar_zone_check")
buf = Buffer()
TransformListener(buf, n)
scans = []
n.create_subscription(LaserScan, "/scan_raw", lambda m: scans.append(m), qos_profile_sensor_data)
while len(scans) < N + 3:
    rclpy.spin_once(n, timeout_sec=0.1)
t = buf.lookup_transform("base_footprint", scans[-1].header.frame_id, Time())
tx, ty = t.transform.translation.x, t.transform.translation.y
q = t.transform.rotation
yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
print(f"LiDAR 프레임 {scans[-1].header.frame_id}: base 기준 위치 ({tx:.3f}, {ty:.3f}), 회전 {math.degrees(yaw):.1f}도, "
      f"range_min {scans[-1].range_min:.3f} m")
hits = {"PolygonStop": [], "PolygonSlow": []}
for s in scans[3:]:
    r = np.array(s.ranges, float)
    a = s.angle_min + np.arange(len(r)) * s.angle_increment
    ok = np.isfinite(r) & (r >= s.range_min) & (r <= s.range_max)
    x = tx + np.cos(a + yaw) * r
    y = ty + np.sin(a + yaw) * r
    for name, (xm, ym) in (("PolygonStop", (0.16, 0.10)), ("PolygonSlow", (0.40, 0.14))):
        m = ok & (x >= 0) & (x <= xm) & (np.abs(y) <= ym)
        hits[name] += [(round(math.degrees(ai), 1), round(ri, 3), round(xi, 3), round(yi, 3)) for ai, ri, xi, yi in zip(a[m], r[m], x[m], y[m])]
for name, h in hits.items():
    print(f"{name}: 스캔 {N}개 동안 점 {len(h)}개 (스캔당 {len(h) / N:.1f})")
    if h:
        arr = np.array(h)
        print(f"   거리 {arr[:, 1].min():.3f}~{arr[:, 1].max():.3f} m, LiDAR 각도 {arr[:, 0].min():.1f}~{arr[:, 0].max():.1f}도, "
              f"base x {arr[:, 2].min():.3f}~{arr[:, 2].max():.3f}, y {arr[:, 3].min():.3f}~{arr[:, 3].max():.3f}")
        print("   예:", h[:6])
n.destroy_node()
rclpy.try_shutdown()
