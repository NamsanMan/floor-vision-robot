#!/usr/bin/env python3
"""odom TF와 카메라 영상의 시각 비교 (로봇 컨테이너, 2026-10-30). 인식 노드가 '영상 촬영 시각의 odom 위치'를 TF에서 못 찾아
최근 위치로 대체하는 비율이 높아서, 두 시각이 지금 시각보다 얼마나 늦는지 잰다.
usage: python3 tf_stamp_check.py [초=5]"""
import sys
import time

import numpy as np
import rclpy
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
rclpy.init()
n = rclpy.create_node("tf_stamp_check")
tf_age, img_age, tf_gap, last = [], [], [], [None]


def on_tf(m):
    now = n.get_clock().now().nanoseconds
    for t in m.transforms:
        if t.header.frame_id == "odom" and t.child_frame_id == "base_footprint":
            s = Time.from_msg(t.header.stamp).nanoseconds
            tf_age.append((now - s) / 1e6)
            if last[0] is not None:
                tf_gap.append((s - last[0]) / 1e6)
            last[0] = s


def on_img(m):
    img_age.append((n.get_clock().now().nanoseconds - Time.from_msg(m.header.stamp).nanoseconds) / 1e6)


n.create_subscription(TFMessage, "/tf", on_tf, 100)
n.create_subscription(Image, "/ascamera/camera_publisher/rgb0/image", on_img,
                      QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST))
end = time.time() + secs
while time.time() < end:
    rclpy.spin_once(n, timeout_sec=0.05)
q = lambda a: f"중앙값 {np.median(a):.0f} ms (최소 {np.min(a):.0f}, 최대 {np.max(a):.0f}, {len(a)}개)" if a else "없음"  # noqa: E731
print(f"odom→base_footprint TF가 지금보다 늦은 정도: {q(tf_age)}")
print(f"  TF 간격: {q(tf_gap)}")
print(f"카메라 영상이 지금보다 늦은 정도: {q(img_age)}")
if tf_age and img_age:
    print(f"→ 영상 촬영 시각의 TF가 버퍼에 있으려면 TF 지연 < 영상 지연이어야 함: TF {np.median(tf_age):.0f} ms vs 영상 {np.median(img_age):.0f} ms")
n.destroy_node()
rclpy.try_shutdown()
