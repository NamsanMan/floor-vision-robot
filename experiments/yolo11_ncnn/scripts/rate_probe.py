"""카메라 원본 토픽(직렬화된 채로 받아 부담 최소)과 /yolo/detections의 실제 주기를 동시에 잰다."""
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from yolo_msgs.msg import DetectionArray

duration = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
rclpy.init()
node = Node("rate_probe")
cam_arrival, cam_stamp, det_arrival = [], [], []


def cam_cb(raw):
    cam_arrival.append(time.monotonic())
    if len(cam_stamp) < 400:
        m = deserialize_message(raw, Image)
        cam_stamp.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)


node.create_subscription(Image, "/ascamera/camera_publisher/rgb0/image", cam_cb, qos_profile_sensor_data, raw=True)
node.create_subscription(DetectionArray, "/yolo/detections", lambda m: det_arrival.append(time.monotonic()), 10)
t_end = time.monotonic() + duration
while time.monotonic() < t_end:
    rclpy.spin_once(node, timeout_sec=0.05)


def hz(ts):
    return round((len(ts) - 1) / (ts[-1] - ts[0]), 2) if len(ts) > 1 else 0.0


print(f"camera arrivals {hz(cam_arrival)} Hz | camera header-stamp rate {hz(cam_stamp)} Hz "
      f"(stamp interval median {np.median(np.diff(cam_stamp)) * 1000:.0f} ms) | detections {hz(det_arrival)} Hz")
node.destroy_node()
rclpy.shutdown()
