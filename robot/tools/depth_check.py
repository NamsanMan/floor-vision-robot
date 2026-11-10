#!/usr/bin/env python3
"""줄자 대신 깊이 카메라로 인식 노드 거리 확인 (로봇 컨테이너, 2026-10-29).
/floor_obstacles/points(base_footprint 바닥 점)를 같은 카메라 보정으로 화면에 되돌려 놓고, 그 점 바로 위(물체 앞면, 3px 위)의
깊이(mm, RGB에 정렬됨)와 '보정으로 계산한 카메라 앞 거리'를 비교한다. 깊이는 화면 아래쪽(가까운 바닥)을 못 재므로 0.45~1.6m만.
usage: python3 depth_check.py [초=6]"""
import sys
import time

import numpy as np
import rclpy
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2

sys.path.insert(0, "/home/ubuntu/shared/floor_seg")
from floor_seg_node import FloorGeometry  # noqa: E402

geo = FloorGeometry(590.77, 590.51, 333.39, 232.73, 0.106, 2.45, 0.0, 0.061)
dur = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
rclpy.init()
node = rclpy.create_node("depth_check")
state = {"depth": None}
rows = []


def on_depth(m):
    if m.encoding in ("16UC1", "mono16"):
        state["depth"] = np.frombuffer(m.data, np.uint16).reshape(m.height, m.width)


def on_pts(m):
    d = state["depth"]
    if d is None:
        return
    p = np.array([(x, y) for x, y, _ in point_cloud2.read_points(m, field_names=("x", "y", "z"), skip_nans=True)])
    if not len(p):
        return
    u, v, ok = geo.base_to_pixel(p[:, 0], p[:, 1])
    X = geo.foot[:, None] + geo.fwd[:, None] * (p[:, 0] - geo.cam_x) + geo.left[:, None] * p[:, 1]   # 카메라 좌표 바닥 점
    for ui, vi, zc in zip(u.astype(int), v.astype(int) - 3, X[2]):
        if 0 <= ui < d.shape[1] and 0 <= vi < d.shape[0] and 0.45 <= zc <= 1.6:
            patch = d[max(vi - 1, 0):vi + 2, max(ui - 1, 0):ui + 2].astype(float)
            patch = patch[patch > 0]
            if len(patch):
                rows.append((zc, np.median(patch) / 1000.0, ui))


qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
node.create_subscription(Image, "/ascamera/camera_publisher/depth0/image_raw", on_depth, qos)
node.create_subscription(PointCloud2, "/floor_obstacles/points", on_pts, 5)
t0 = time.time()
while time.time() - t0 < dur:
    rclpy.spin_once(node, timeout_sec=0.1)
r = np.array(rows)
if len(r):
    e = r[:, 1] - r[:, 0]
    print(f"비교한 점 {len(r)}개 (보정으로 계산한 카메라 앞 거리 0.45~1.6 m)")
    print(f"깊이 - 보정 계산: 중앙값 {np.median(e) * 100:+.1f} cm, |오차| 중앙값 {np.median(np.abs(e)) * 100:.1f} cm, 상위10% {np.percentile(np.abs(e), 90) * 100:.1f} cm")
    for lo, hi in ((0.45, 0.8), (0.8, 1.2), (1.2, 1.6)):
        m = (r[:, 0] >= lo) & (r[:, 0] < hi)
        if m.any():
            print(f"  거리 {lo:.2f}-{hi:.2f} m: {m.sum():4d}개, 오차 중앙값 {np.median(e[m]) * 100:+.1f} cm")
    for lo, hi, nm in ((0, 270, "화면 왼쪽 u<270"), (270, 340, "화면 가운데"), (340, 640, "화면 오른쪽 u>=340")):
        m = (r[:, 2] >= lo) & (r[:, 2] < hi)
        if m.any():
            print(f"  {nm:<16} {m.sum():4d}개, 보정 거리 중앙값 {np.median(r[m, 0]):.2f} m, 오차 중앙값 {np.median(e[m]) * 100:+.1f} cm")
else:
    print("비교할 점 없음 (깊이 영상이 없거나 장애물이 0.45~1.6 m 밖)")
node.destroy_node()
rclpy.try_shutdown()
