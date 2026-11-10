"""지금 카메라 원본 프레임 1장 vs 앞서 저장한 JPEG 스냅샷: 밝기와 의자 점수 비교 (같은 장면인데 검출이 달랐던 이유 확인)."""
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from ultralytics import YOLO

S = "/home/ubuntu/shared/yolo_setup/results"
rclpy.init()
node = Node("frame_grab")
got = []
node.create_subscription(Image, "/ascamera/camera_publisher/rgb0/image", lambda m: got.append(m),
                         QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
t_end = time.monotonic() + 10
while not got and time.monotonic() < t_end:
    rclpy.spin_once(node, timeout_sec=0.1)
msg = got[-1]
raw = CvBridge().imgmsg_to_cv2(msg, desired_encoding="bgr8")
cv2.imwrite(f"{S}/raw_now.png", raw)
_, enc = cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, 95])
images = {"raw frame now (encoding %s)" % msg.encoding: raw,
          "same frame as JPEG q95": cv2.imdecode(enc, cv2.IMREAD_COLOR),
          "snapshot run1 (best effort)": cv2.imread(f"{S}/live_ncnn.jpg")}
model = YOLO("/home/ubuntu/yolo11n_ncnn_model", task="detect")
for name, im in images.items():
    r = model.predict(im, imgsz=(480, 640), conf=0.05, verbose=False)[0]
    chairs = sorted((float(s) for s, c in zip(r.boxes.conf, r.boxes.cls) if r.names[int(c)] == "chair"), reverse=True)
    print(f"{name:32s} mean brightness {im.mean():6.1f} | right-chair region {im[0:224, 400:600].mean():6.1f} | "
          f"chair scores {[round(s, 2) for s in chairs[:3]]}")
node.destroy_node()
rclpy.shutdown()
