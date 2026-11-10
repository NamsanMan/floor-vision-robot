#!/usr/bin/env python3
"""RGB 영상 한 장 저장 (로봇 컨테이너, 2026-10-31) — 지금 카메라가 어디를 보는지 확인용.
usage: python3 grab_frame.py [--out /home/ubuntu/shared/floor_seg/now.jpg]"""
import argparse
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/ascamera/camera_publisher/rgb0/image")
    ap.add_argument("--out", default="/home/ubuntu/shared/floor_seg/now.jpg")
    a = ap.parse_args()
    rclpy.init()
    n = Node("grab_frame")
    got = []

    def cb(msg):
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step // 3, 3)[:, :msg.width]
        got.append(img if msg.encoding == "bgr8" else img[:, :, ::-1])

    n.create_subscription(Image, a.topic, cb, qos_profile_sensor_data)
    t0 = time.time()
    while len(got) < 5 and time.time() - t0 < 10:            # 몇 장 받고 마지막 것
        rclpy.spin_once(n, timeout_sec=0.1)
    if got:
        cv2.imwrite(a.out, got[-1])
        print(f"저장: {a.out}, 밝기 {got[-1].mean():.1f}")
    else:
        print("영상 못 받음")
    n.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
