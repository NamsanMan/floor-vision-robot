#!/usr/bin/env python3
"""바닥 분할 감시 노드 (로봇 컨테이너, 2026-10-29) — '조용한 실패'를 '멈추는 실패'로 바꾸는 장치.
2026-10-29 실험에서 CPU 과부하 때 카메라 장애물 점이 늦게 도착해 Nav2가 7분간 버렸는데도 Nav2는 정상으로 보였다.
그래서 카메라 데이터가 끊기거나 늦어지면 Collision Monitor의 정지 구역 안에 가짜 장애물 점을 넣어 로봇을 세운다.
  감시: /floor_obstacles/memory  (받은 지 stale_rx_s 넘게 지나면 = 끊김, 메시지 시각이 stale_age_s 넘게 오래되면 = 늦음)
  발행: /floor_obstacles/watchdog PointCloud2 (base_footprint, 10Hz) — 정상 = 빈 점구름, 이상 = 정지 구역 안 점 9개
        Collision Monitor 설정에 pointcloud 소스로 넣는다. base_footprint 좌표라 TF가 밀려도 변환이 필요 없다.
의존: 시스템 python3 + rclpy만"""
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


class FloorWatchdog(Node):
    def __init__(self):
        super().__init__("floor_watchdog")
        P = lambda name, val: self.declare_parameter(name, val).value  # noqa: E731
        self.stale_rx = P("stale_rx_s", 0.6)                # 5Hz(0.2초 간격) 기준 3번 연속 못 받으면 → 바로 정지
        self.stale_age = P("stale_age_s", 0.8)              # 메시지 시각이 0.8초 넘게 오래됨 (정상 지연 중앙값 0.12, p95 0.16초)
        self.persist = P("persist_s", 0.3)                  # 지연은 0.3초 넘게 이어질 때만 정지. 2026-10-29: 0.5초 문턱·즉시 반응이면
                                                            # 시험 스크립트를 띄운 순간의 한 프레임 지연(0.50초)에도 헛정지했음
        self.grace = P("startup_grace_s", 10.0)             # 시작 직후 인식 노드가 뜰 때까지는 봐줌
        x0, x1 = P("stop_x_min", 0.12), P("stop_x_max", 0.15)   # Collision Monitor PolygonStop(앞면 0.11 + 5cm) 안
        self.stop_pts = [(x0 + (x1 - x0) * i / 2, -0.04 + 0.04 * j, 0.05) for i in range(3) for j in range(3)]
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(PointCloud2, "/floor_obstacles/memory", self.on_mem, qos)
        self.pub = self.create_publisher(PointCloud2, "/floor_obstacles/watchdog", 5)
        self.t_start = time.monotonic()
        self.last_rx = None
        self.last_age = 0.0
        self.late_since = None
        self.stopping = False
        self.n_trips = 0
        self.create_timer(0.1, self.tick)
        self.get_logger().info(f"감시 시작: 끊김 {self.stale_rx}s면 바로, 지연 {self.stale_age}s가 {self.persist}s 이어지면 "
                               f"정지 구역에 점 {len(self.stop_pts)}개")

    def on_mem(self, msg):
        self.last_rx = time.monotonic()
        self.last_age = (self.get_clock().now().nanoseconds - Time.from_msg(msg.header.stamp).nanoseconds) * 1e-9

    def tick(self):
        now = time.monotonic()
        if self.last_rx is None:
            bad = now - self.t_start > self.grace
            why = "인식 노드 데이터가 한 번도 안 옴"
        elif now - self.last_rx > self.stale_rx:
            bad, why = True, f"데이터 끊김 {now - self.last_rx:.2f}s"
        elif self.last_age > self.stale_age:
            self.late_since = self.late_since or now
            bad = now - self.late_since >= self.persist      # 잠깐 튄 지연은 넘기고, 이어지는 지연만 정지
            why = f"데이터 지연 {self.last_age:.2f}s가 {now - self.late_since:.1f}s 이어짐"
        else:
            bad, why = False, ""
        if not (self.last_rx is not None and self.last_age > self.stale_age):
            self.late_since = None
        if bad != self.stopping:
            self.stopping = bad
            if bad:
                self.n_trips += 1
                self.get_logger().warn(f"정지 신호 켬 ({why}) — 누적 {self.n_trips}회")
            else:
                self.get_logger().info("카메라 데이터 회복 — 정지 신호 끔")
        hdr = Header(stamp=self.get_clock().now().to_msg(), frame_id="base_footprint")
        self.pub.publish(point_cloud2.create_cloud_xyz32(hdr, self.stop_pts if bad else []))


def main():
    rclpy.init()
    node = FloorWatchdog()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):         # 정지 신호(SIGINT)로 끌 때 나오는 예외는 정상
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
