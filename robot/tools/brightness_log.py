#!/usr/bin/env python3
"""카메라 자동 노출 확인 (로봇 컨테이너, 2026-10-31): RGB 영상의 프레임별 밝기를 정해진 시간 동안 CSV로 남긴다.
카메라 드라이버(ascamera)는 ROS에 노출·감도 설정이 없고 제조사 설정 파일은 암호화돼 있어서, 동작으로 확인한다.
  고정 노출: 불을 끄면 밝기가 계단처럼 떨어진 채 그대로, 켜면 원래 값으로 바로 돌아옴
  자동 노출: 끈 직후 떨어졌다가 몇 초에 걸쳐 다시 올라감, 켠 직후 과노출(하얗게)됐다가 내려옴. 노출이 길어지면 초당 장수가 줄기도 함
usage: python3 brightness_log.py [--secs 240] [--out /home/ubuntu/shared/floor_seg/ae_test.csv]
       일찍 끝내기: 호스트에서 touch /home/pi/docker/tmp/floor_seg/ae_stop (CSV는 그때까지 받은 것으로 저장)"""
import argparse
import csv
import os
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=240.0)
    ap.add_argument("--topic", default="/ascamera/camera_publisher/rgb0/image")
    ap.add_argument("--out", default="/home/ubuntu/shared/floor_seg/ae_test.csv")
    ap.add_argument("--stop-file", default="/home/ubuntu/shared/floor_seg/ae_stop")
    a = ap.parse_args()
    rclpy.init()
    n = Node("brightness_log")
    rows = []

    def cb(msg):
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step // 3, 3)[::2, :msg.width:2]   # 2칸마다(가볍게)
        g = img.mean(axis=2)
        h = g.shape[0]
        rows.append((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, time.time(),
                     round(float(g.mean()), 2), round(float(g[h // 2:].mean()), 2), round(float(g[:h // 3].mean()), 2),
                     round(float(np.percentile(g, 99)), 1), round(float((g >= 250).mean()), 4)))

    n.create_subscription(Image, a.topic, cb, qos_profile_sensor_data)
    if os.path.exists(a.stop_file):
        os.remove(a.stop_file)
    t0 = last = time.time()
    print(f"기록 시작: {a.secs:.0f}초 동안 {a.topic} 프레임 밝기 (일찍 끝내려면 {a.stop_file} 만들기)", flush=True)
    try:
        while time.time() - t0 < a.secs and not os.path.exists(a.stop_file):
            rclpy.spin_once(n, timeout_sec=0.1)
            if time.time() - last >= 10 and rows:
                last = time.time()
                print(f"  {last - t0:4.0f}초 · 프레임 {len(rows)} · 지금 밝기 {rows[-1][2]:.1f}", flush=True)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stamp", "recv", "mean", "bottom_half", "top_third", "p99", "sat_frac"])
        w.writerows(rows)
    m = np.array([r[2] for r in rows]) if rows else np.zeros(1)
    st = np.array([r[0] for r in rows]) if rows else np.zeros(1)
    fps = (len(st) - 1) / max(1e-6, st[-1] - st[0]) if len(st) > 1 else 0.0
    print(f"끝: 프레임 {len(rows)}, 평균 {fps:.1f} fps, 밝기 최소 {m.min():.1f} / 최대 {m.max():.1f} → {a.out}", flush=True)
    n.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
