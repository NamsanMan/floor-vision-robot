#!/usr/bin/env python3
"""모의 로봇 조작 도우미 (WSL 모의 시험 전용, 2026-10-30). ros2 CLI(param set)가 WSL에서 멈춰서 rclpy로 직접 부른다.
usage: simctl.py obs <x> <on: 1|0> | carry | init | still"""
import math
import sys
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rcl_interfaces.srv import SetParameters
from rclpy.parameter import Parameter
from rclpy.time import Time
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener


def call(n, cli, req, timeout=5.0):
    if not cli.wait_for_service(timeout_sec=timeout):
        return None
    fut = cli.call_async(req)
    end = time.time() + timeout
    while not fut.done() and time.time() < end:
        rclpy.spin_once(n, timeout_sec=0.05)
    return fut.result()


rclpy.init()
n = rclpy.create_node("simctl")
cmd = sys.argv[1]
if cmd == "obs":
    req = SetParameters.Request()
    req.parameters = [Parameter("obs_x", value=float(sys.argv[2])).to_parameter_msg(),
                      Parameter("obs_on", value=sys.argv[3] == "1").to_parameter_msg()]
    r = call(n, n.create_client(SetParameters, "/mock_robot/set_parameters"), req)
    print(f"   [모의] 장애물 앞쪽 끝 x={sys.argv[2]} {'있음' if sys.argv[3] == '1' else '없음'}" + ("" if r else " — 실패"))
elif cmd == "carry":
    r = call(n, n.create_client(Trigger, "/mock/carry_back"), Trigger.Request())
    print("   [모의] " + (r.message if r else "carry_back 실패"))
elif cmd == "init":
    pub = n.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
    end = time.time() + 3
    while pub.get_subscription_count() == 0 and time.time() < end:
        rclpy.spin_once(n, timeout_sec=0.1)
    m = PoseWithCovarianceStamped()
    m.header.frame_id = "map"
    m.pose.pose.orientation.w = 1.0
    pub.publish(m)
    rclpy.spin_once(n, timeout_sec=0.3)
    print("   [모의] RViz 2D Pose Estimate 흉내: map 원점")
elif cmd == "still":
    b = Buffer()
    TransformListener(b, n)

    def pos():
        end = time.time() + 1.0
        while time.time() < end:
            rclpy.spin_once(n, timeout_sec=0.05)
        t = b.lookup_transform("odom", "base_footprint", Time()).transform.translation
        return t.x, t.y

    p0 = pos()
    p1 = pos()
    print(f"   [확인] 1초 동안 움직인 거리 {math.hypot(p1[0] - p0[0], p1[1] - p0[1]) * 100:.1f} cm")
n.destroy_node()
rclpy.try_shutdown()
