#!/usr/bin/env python3
"""모의 로봇 — WSL Humble에서 drive_test.py·floor_seg_node.py·floor_watchdog.py를 점검하는 용도. 로봇에 올리지 않음 (2026-10-30).
  · 진짜 위치(world)와 odom 위치를 따로 적분. /mock/carry_back = 사람이 로봇을 들어 출발선(world 원점)에 놓음 — odom은 모른다
  · TF odom→base_footprint 50Hz, map→odom (AMCL 흉내: /initialpose를 받으면 map 위치가 그 값이 되게)
  · 가짜 카메라 /ascamera/camera_publisher/rgb0/image (bgr8 640×480, 15Hz): 회색 바닥 + 빨간 장애물 바닥면(카메라 보정식으로 투영,
    카메라 바로 앞 사각지대는 화면 밖으로 나감)
  · 가짜 Collision Monitor: /cmd_vel_smoothed → /cmd_vel. 기억 점(odom)·감시 점(base)이 정지 구역(x 0~0.16, |y|≤0.10)에 2개 이상 = 0,
    감속 구역(x 0~0.40, |y|≤0.14) = 0.3배. source_timeout 1초(넘으면 그 소스 무시), stop_pub_timeout 2초.
    'Robot to ...' 로그는 표준 출력 → 가짜 Nav2 로그 파일
  · 가짜 NavigateToPose(직선으로 목표를 향함, 회피 없음 → /cmd_vel_smoothed), 코스트맵 지우기 서비스, /collision_monitor/get_state
  · 모터 드라이버 흉내: 마지막 /cmd_vel을 계속 유지(최악 가정) → 시험 도구가 끝에 정지 명령을 꼭 보내는지 확인할 수 있다
파라미터: obs_x(장애물 앞쪽 끝 world x), obs_l, obs_w, obs_on"""
import math
import time

import cv2
import numpy as np
import rclpy
from floor_seg_node import FloorGeometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, Twist
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ClearEntireCostmap
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster


def compose(a, b):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2])


def inv(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-(c * a[0] + s * a[1]), -(-s * a[0] + c * a[1]), -a[2])


def cloud(m):
    if m.width * m.height == 0:
        return np.zeros((0, 2))
    return np.frombuffer(bytes(m.data), np.float32).reshape(-1, m.point_step // 4)[:, :2].astype(float)


def clip_x(poly, x0):
    out = []
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]
        if p[0] >= x0:
            out.append(p)
        if (p[0] >= x0) != (q[0] >= x0):
            t = (x0 - p[0]) / (q[0] - p[0])
            out.append((x0, p[1] + t * (q[1] - p[1])))
    return out


class Mock(Node):
    def __init__(self):
        super().__init__("mock_robot")
        self.cbg = ReentrantCallbackGroup()
        for k, v in (("obs_x", 1.61), ("obs_l", 0.26), ("obs_w", 0.19), ("obs_on", True)):
            self.declare_parameter(k, v)
        self.geo = FloorGeometry(590.77, 590.51, 333.39, 232.73, 0.106, 2.45, 0.0, 0.061)
        self.true, self.odom, self.map_T_odom = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], (0.0, 0.0, 0.0)
        self.cmd, self.t_last = (0.0, 0.0, 0.0), time.time()
        self.mem, self.mem_t, self.wd, self.wd_t = np.zeros((0, 2)), 0.0, np.zeros((0, 2)), 0.0
        self.cm_state, self.stop_since = "continue", None
        self.cm_log = rclpy.logging.get_logger("collision_monitor")
        self.tfb = TransformBroadcaster(self)
        self.pub_img = self.create_publisher(Image, "/ascamera/camera_publisher/rgb0/image", 2)
        self.pub_out = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_in = self.create_publisher(Twist, "/cmd_vel_smoothed", 10)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 10, callback_group=self.cbg)
        self.create_subscription(Twist, "/cmd_vel_smoothed", self.on_cm_in, 10, callback_group=self.cbg)
        self.create_subscription(PointCloud2, "/floor_obstacles/memory", self.on_mem, 10, callback_group=self.cbg)
        self.create_subscription(PointCloud2, "/floor_obstacles/watchdog", self.on_wd, 10, callback_group=self.cbg)
        self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.on_init, 10, callback_group=self.cbg)
        self.create_service(GetState, "/collision_monitor/get_state", self.on_state, callback_group=self.cbg)
        self.create_service(ClearEntireCostmap, "/local_costmap/clear_entirely_local_costmap", self.on_clear, callback_group=self.cbg)
        self.create_service(ClearEntireCostmap, "/global_costmap/clear_entirely_global_costmap", self.on_clear, callback_group=self.cbg)
        self.create_service(Trigger, "/mock/carry_back", self.on_carry, callback_group=self.cbg)
        ActionServer(self, NavigateToPose, "navigate_to_pose", execute_callback=self.exec_nav, callback_group=self.cbg,
                     goal_callback=lambda g: GoalResponse.ACCEPT, cancel_callback=lambda g: CancelResponse.ACCEPT)
        self.create_timer(0.02, self.step, callback_group=self.cbg)
        self.create_timer(1 / 15, self.camera, callback_group=self.cbg)
        self.get_logger().info("모의 로봇 시작")

    # ----- 움직임·TF
    def on_cmd(self, m):
        self.cmd = (m.linear.x, m.linear.y, m.angular.z)

    def step(self):
        now = time.time()
        dt, self.t_last = now - self.t_last, now
        vx, vy, wz = self.cmd
        for p in (self.true, self.odom):
            c, s = math.cos(p[2]), math.sin(p[2])
            p[0] += (c * vx - s * vy) * dt
            p[1] += (s * vx + c * vy) * dt
            p[2] += wz * dt
        st = self.get_clock().now().to_msg()
        out = []
        for parent, child, p in (("odom", "base_footprint", self.odom), ("map", "odom", self.map_T_odom)):
            t = TransformStamped()
            t.header.stamp, t.header.frame_id, t.child_frame_id = st, parent, child
            t.transform.translation.x, t.transform.translation.y = float(p[0]), float(p[1])
            t.transform.rotation.z, t.transform.rotation.w = math.sin(p[2] / 2), math.cos(p[2] / 2)
            out.append(t)
        self.tfb.sendTransform(out)

    def on_init(self, m):
        q = m.pose.pose.orientation
        p = (m.pose.pose.position.x, m.pose.pose.position.y, math.atan2(2 * q.w * q.z, 1 - 2 * q.z * q.z))
        self.map_T_odom = compose(p, inv(tuple(self.odom)))
        self.get_logger().info(f"initialpose → map ({p[0]:.2f}, {p[1]:.2f})")

    def on_carry(self, req, res):
        self.true[:] = [0.0, 0.0, 0.0]
        res.success, res.message = True, "로봇을 들어 출발선에 놓음(odom은 그대로)"
        return res

    # ----- 가짜 카메라
    def camera(self):
        img = np.full((480, 640, 3), 128, np.uint8)
        if self.get_parameter("obs_on").value:
            x0, L, W = (self.get_parameter(k).value for k in ("obs_x", "obs_l", "obs_w"))
            tx, ty, th = self.true
            c, s = math.cos(th), math.sin(th)
            poly = []
            for X, Y in ((x0, -W / 2), (x0 + L, -W / 2), (x0 + L, W / 2), (x0, W / 2)):
                poly.append((c * (X - tx) + s * (Y - ty), -s * (X - tx) + c * (Y - ty)))
            poly = clip_x(poly, self.geo.cam_x + 0.02)
            if len(poly) >= 3:
                u, v, ok = self.geo.base_to_pixel(np.array([p[0] for p in poly]), np.array([p[1] for p in poly]))
                pts = np.stack([np.clip(u, -5000, 5000), np.clip(v, -5000, 5000)], 1).astype(np.int32)
                cv2.fillPoly(img, [pts], (0, 0, 255))
        m = Image()
        m.header.stamp, m.header.frame_id = self.get_clock().now().to_msg(), "ascamera_color_0"
        m.height, m.width, m.encoding, m.step = 480, 640, "bgr8", 640 * 3
        m.data = img.tobytes()
        self.pub_img.publish(m)

    # ----- 가짜 Collision Monitor
    def on_mem(self, m):
        self.mem, self.mem_t = cloud(m), time.time()

    def on_wd(self, m):
        self.wd, self.wd_t = cloud(m), time.time()

    def on_cm_in(self, m):
        now = time.time()
        pts = []
        if now - self.mem_t < 1.0 and len(self.mem):
            ox, oy, th = self.odom
            c, s = math.cos(th), math.sin(th)
            dx, dy = self.mem[:, 0] - ox, self.mem[:, 1] - oy
            pts.append(np.stack([c * dx + s * dy, -s * dx + c * dy], 1))
        if now - self.wd_t < 1.0 and len(self.wd):
            pts.append(self.wd)
        P = np.vstack(pts) if pts else np.zeros((0, 2))
        n_stop = int(((P[:, 0] >= 0) & (P[:, 0] <= 0.16) & (np.abs(P[:, 1]) <= 0.10)).sum())
        n_slow = int(((P[:, 0] >= 0) & (P[:, 0] <= 0.40) & (np.abs(P[:, 1]) <= 0.14)).sum())
        state = "stop" if n_stop >= 2 else "slowdown" if n_slow >= 2 else "continue"
        if state != self.cm_state:
            self.cm_log.info({"stop": "Robot to stop due to PolygonStop polygon",
                              "slowdown": "Robot to slowdown for 30.000000 percents due to PolygonSlow polygon",
                              "continue": "Robot to continue normal operation"}[state])
            self.cm_state = state
            self.stop_since = now if state == "stop" else None
        out = Twist()
        if state == "stop":
            if now - self.stop_since > 2.0:                           # stop_pub_timeout: 2초 뒤엔 0도 안 보냄
                return
        elif state == "slowdown":
            out.linear.x, out.linear.y, out.angular.z = m.linear.x * 0.3, m.linear.y * 0.3, m.angular.z * 0.3
        else:
            out = m
        self.pub_out.publish(out)

    def on_state(self, req, res):
        res.current_state.id, res.current_state.label = 3, "active"
        return res

    def on_clear(self, req, res):
        self.get_logger().info("코스트맵 지움")
        return res

    # ----- 가짜 Nav2 목표 (직선)
    def exec_nav(self, gh):
        gx, gy = gh.request.pose.pose.position.x, gh.request.pose.pose.position.y
        fb = NavigateToPose.Feedback()
        while rclpy.ok():
            if gh.is_cancel_requested:
                self.pub_in.publish(Twist())
                gh.canceled()
                return NavigateToPose.Result()
            mx, my, mth = compose(self.map_T_odom, tuple(self.odom))
            dx, dy = gx - mx, gy - my
            d = math.hypot(dx, dy)
            if d < 0.2:
                self.pub_in.publish(Twist())
                gh.succeed()
                return NavigateToPose.Result()
            c, s = math.cos(mth), math.sin(mth)
            k = min(0.25, d) / d
            t = Twist()
            t.linear.x, t.linear.y = (c * dx + s * dy) * k, (-s * dx + c * dy) * k
            self.pub_in.publish(t)
            fb.distance_remaining = float(d)
            gh.publish_feedback(fb)
            time.sleep(0.05)
        return NavigateToPose.Result()


def main():
    rclpy.init()
    node = Mock()
    ex = MultiThreadedExecutor(num_threads=6)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
