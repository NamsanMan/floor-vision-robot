#!/usr/bin/env python3
"""바닥 분할 인식 노드 (로봇 컨테이너, 2026-10-29)
카메라 → 미세조정 DS-Net INT8(ONNX Runtime) → 화면 열마다 '바닥이 끊기는 첫 점' → 카메라 보정(높이 0.106m, 2.45도 아래)으로
base_footprint 바닥 좌표(m) → 장애물 점.
  /floor_obstacles/points          PointCloud2 (base_footprint) 지금 프레임의 장애물 점 (열 col_step칸마다, max_range 안)
  /floor_obstacles/memory          PointCloud2 (odom) 장애물 점 기억: 카메라에 지금 보이는 곳은 새 관측으로 바꾸고,
                                   사각지대(가장 가까운 보이는 바닥보다 가까운 곳)·화면 밖·max_range 밖은 expire초 동안 유지 (ID 없음).
                                   expire는 로봇이 움직인 시간만 센다(2026-10-30: 서 있는 동안에도 줄어들면, Collision Monitor가
                                   장애물 앞에 세운 지 10초 뒤 기억이 사라져 정지가 풀리고 밀고 들어감 — 모의 시험에서 발견)
  /floor_obstacles/clearing        PointCloud2 (base_footprint) 코스트맵 '지우기' 끝점: 열마다 장애물 점, 없으면 max_range 바닥 점.
                                   코스트맵 장애물 층은 광선이 지나간 칸만 비우므로, 장애물이 없는 열에도 끝점이 있어야 옛 표시가 지워진다
  /floor_obstacles/front_distance  Float32 로봇 폭(±front_half_width) 안에서 가장 가까운 장애물까지 앞 거리(m, base 원점 기준; 없으면 9.9)
디버그: debug_every 프레임마다 겹친 그림 JPEG(링 버퍼), 프레임마다 지연 CSV(전처리/추론/후처리 ms, 카메라 시각과의 차)
주행 시험용(2026-10-30, 실행 중 ros2 param set으로 바꿈):
  mute             True = 장애물 점·기억을 빈 점구름으로 발행('라이다만' 조건). 추론·지우기 끝점·정면 거리는 그대로라 CPU 부하가 같고,
                   카메라 층·감시 노드도 데이터가 계속 와서 정상으로 본다(노드를 끄면 카메라 층이 '갱신 안 됨' → Nav2 제어기가 멈춤)
  memory_expire_s  0 = 기억 끔(지금 보이는 점만)
  record_dir       경로 = 처리한 프레임 JPEG + 프레임별 경계 행·촬영 시각 로봇 위치 CSV 저장(판정 흔들림 분석용), 빈 문자열 = 끔
  서비스 ~/clear_memory (std_srvs/Trigger): 기억 비우기 — 로봇을 들어 옮기면 odom은 그대로라 옛 기억이 로봇 바로 앞에 남는다
온보드 vs 오프로딩(2026-10-30):
  backend          local = 로봇에서 추론 / remote = 메인컴에서 추론(offload/offload_pc.py). remote면 240x320으로 줄인 JPEG를 보내고
                   바닥-비바닥 로짓 차(float16, 60x80)를 받는다. 나머지(경계·바닥 좌표·기억·발행)는 같은 코드.
                   메인컴 방화벽을 안 건드리려고 노드가 remote_port에서 기다리고 메인컴이 접속해 온다.
                   응답이 remote_timeout_s 안에 안 오면 그 프레임은 발행하지 않음 → 이어지면 감시 노드가 로봇을 세운다(로컬로 몰래 대체하지 않음)
의존: 시스템 python3 + rclpy, onnxruntime은 ~/floorseg_pkgs(bench_venv에서 onnxruntime만 복사)로 PYTHONPATH에 넣음 — 시스템 패키지는 안 건드림
"""
import csv
import math
import os
import socket
import struct
import threading
import time

import cv2
import numpy as np
import onnxruntime as ort
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
REQ = struct.Struct("<4sII")              # 오프로딩 요청: 'FSEG', 번호, JPEG 길이
REP = struct.Struct("<4sIfHHI")           # 응답: 'FSEG', 번호, 메인컴 처리 ms, 높이, 너비, 데이터 길이(float16 로짓 차)


def recv_exact(c, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = c.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("연결 끊김")
        buf += chunk
    return bytes(buf)


class FloorGeometry:
    """카메라 보정 → 화면 점 (u,v) ↔ base_footprint 바닥 (x,y). 수식은 stage1_analyze.py의 ipm()·to_cam()과 같다."""

    def __init__(self, fx, fy, cx, cy, h, pitch_deg, roll_deg, cam_x):
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        self.Kinv = np.linalg.inv(self.K)
        p, r = math.radians(pitch_deg), math.radians(roll_deg)
        n = np.array([math.sin(r) * math.cos(p), math.cos(r) * math.cos(p), math.sin(p)])   # 카메라 좌표(y 아래, z 앞)의 바닥 법선
        fwd = np.array([0, 0, 1.0]) - n[2] * n
        fwd /= np.linalg.norm(fwd)
        left = -np.cross(n, fwd)
        left = left if left[0] < 0 else -left
        self.n, self.fwd, self.left, self.h, self.cam_x = n, fwd, left, h, cam_x
        self.foot = h * n

    def pixel_to_base(self, u, v):
        r = self.Kinv @ np.stack([u, v, np.ones_like(u)]).astype(np.float64)        # 3xN 광선
        den = self.n @ r
        ok = den > 1e-6                                                            # 지평선 아래만
        X = (self.h / np.where(ok, den, 1.0)) * r
        d = X - self.foot[:, None]
        return self.cam_x + self.fwd @ d, self.left @ d, ok

    def base_to_pixel(self, x, y):
        X = self.foot[:, None] + self.fwd[:, None] * (x - self.cam_x) + self.left[:, None] * y
        z = np.maximum(X[2], 1e-6)
        return self.K[0, 0] * X[0] / z + self.K[0, 2], self.K[1, 1] * X[1] / z + self.K[1, 2], X[2] > 1e-3

    def row_of_distance(self, x):
        """정면(u=cx)에서 base 앞 거리 x(m)에 해당하는 화면 행"""
        return float(self.base_to_pixel(np.array([x]), np.array([0.0]))[1][0])


class FloorSegNode(Node):
    def __init__(self):
        super().__init__("floor_seg")
        P = lambda name, val: self.declare_parameter(name, val).value  # noqa: E731
        self.model_path = P("model", "/home/ubuntu/shared/bench/ft/ft_240x320_int8_gelu.onnx")
        threads = P("threads", 2)
        self.in_hw = (P("in_h", 240), P("in_w", 320))
        self.col_step = P("col_step", 4)
        self.run_px = P("run_px", 3)
        self.max_range = P("max_range", 1.5)
        self.z_mark = P("z_mark", 0.05)
        self.expire = P("memory_expire_s", 10.0)                         # 로봇이 움직인 시간 기준(서 있는 동안은 안 줄어듦). 0 = 기억 끔
        self.front_hw = P("front_half_width", 0.12)
        self.debug_every = P("debug_every", 15)
        self.max_rate = P("max_rate", 5.0)                               # 처리 상한(Hz). 0 = 들어오는 대로. 기본 5Hz: Nav2와 같이 돌 때 CPU 여유(2026-10-29 조용한 실패 대책)
        self.mute = P("mute", False)                                     # 주행 시험 '라이다만' 조건 (위 설명)
        self.backend = P("backend", "local")                             # 온보드(local) vs 오프로딩(remote) (위 설명)
        self.remote_timeout = P("remote_timeout_s", 0.5)
        self.jpeg_q = P("remote_jpeg_quality", 90)
        self.conn, self.conn_lock, self.req_id, self.n_fail = None, threading.Lock(), 0, 0
        self.srv = None
        try:
            self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.srv.bind(("0.0.0.0", P("remote_port", 5601)))
            self.srv.listen(1)
            threading.Thread(target=self.accept_loop, daemon=True).start()
        except OSError as e:
            self.get_logger().warn(f"오프로딩 포트를 못 열었음({e}) — backend=remote는 쓸 수 없음")
        P("record_dir", "")
        self.record_dir, self.rec_f, self.rec = "", None, None
        self.t_proc = 0.0
        self.out_dir = P("out_dir", "/home/ubuntu/shared/floor_seg")
        self.geo = FloorGeometry(P("fx", 590.77), P("fy", 590.51), P("cx", 333.39), P("cy", 232.73),
                                 P("cam_height", 0.106), P("pitch_deg", 2.45), P("roll_deg", 0.0), P("cam_x", 0.061))
        self.v_far = self.geo.row_of_distance(self.geo.cam_x + self.max_range)   # 카메라에서 max_range보다 먼 바닥이 시작되는 행

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(self.model_path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Image, "/ascamera/camera_publisher/rgb0/image", self.on_image, qos)
        self.pub_pts = self.create_publisher(PointCloud2, "/floor_obstacles/points", 5)
        self.pub_clr = self.create_publisher(PointCloud2, "/floor_obstacles/clearing", 5)
        self.pub_mem = self.create_publisher(PointCloud2, "/floor_obstacles/memory", 5)
        self.pub_front = self.create_publisher(Float32, "/floor_obstacles/front_distance", 5)
        self.tfb = Buffer()
        # TF는 전용 노드·스레드에서 받음(2026-10-30): 이 노드는 실행 흐름이 하나라 추론하는 약 90ms 동안 TF를 못 받아,
        # 영상 촬영 시각보다 나중 TF가 버퍼에 없어서 촬영 시각 위치를 거의 못 찾았음(로봇 실측 3/43).
        # 로봇의 tf2_ros는 node=None을 지원하지 않는 옛 Humble 빌드라 전용 노드·스레드를 직접 만든다(spin_thread=True는 종료 때
        # 컨텍스트가 먼저 닫히면 그 스레드에서 RCLError를 찍음)
        self.tf_node = rclpy.create_node("floor_seg_tf")
        self.tfl = TransformListener(self.tfb, self.tf_node)
        self.tf_exec = SingleThreadedExecutor()
        self.tf_exec.add_node(self.tf_node)
        threading.Thread(target=self.spin_tf, daemon=True).start()
        self.mem = np.zeros((0, 3))                                      # x_odom, y_odom, 본 순간의 mclock
        self.mclock, self.last_pose = 0.0, None                          # 기억 만료용 '로봇이 움직인 시간' 시계(초)
        self.n_pose = {"stamp": 0, "latest": 0, "none": 0}
        self.add_on_set_parameters_callback(self.on_params)
        self.create_service(Trigger, "~/clear_memory", self.on_clear_memory)

        os.makedirs(os.path.join(self.out_dir, "debug"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "logs"), exist_ok=True)
        self.csv_f = open(os.path.join(self.out_dir, "logs", time.strftime("timing_%Y%m%d_%H%M%S.csv")), "w", newline="")
        self.csv = csv.writer(self.csv_f)
        self.csv.writerow(["stamp", "lag_ms", "pre_ms", "inf_ms", "post_ms", "total_ms", "n_pts", "n_mem", "front_m", "odom_ok",
                           "backend", "server_ms", "up_bytes", "remote_fail"])     # remote면 inf_ms = 왕복(전송 + 메인컴 처리)
        self.n, self.t_last_print, self.win = 0, time.time(), []
        self.get_logger().info(f"모델 {os.path.basename(self.model_path)} 입력 {self.in_hw} 스레드 {threads}, "
                               f"최대 거리 {self.max_range} m = 화면 {self.v_far:.1f}행 아래, 가장 가까운 보이는 바닥 x = "
                               f"{float(self.geo.pixel_to_base(np.array([self.geo.K[0, 2]]), np.array([479.0]))[0][0]):.3f} m")

    # ---------- 영상 → 바닥 → 경계 점
    def infer_floor(self, bgr):
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(cv2.resize(bgr, (self.in_hw[1], self.in_hw[0]), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
        t1 = time.perf_counter()
        lg = self.sess.run(None, {self.inp: x})[0][0]                    # (2, h/4, w/4), 0 = 바닥 아님, 1 = 바닥
        t2 = time.perf_counter()
        floor = cv2.resize(lg[1] - lg[0], (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR) > 0
        return floor, (t1 - t0) * 1e3, (t2 - t1) * 1e3

    def spin_tf(self):
        try:
            self.tf_exec.spin()
        except Exception:                                                # 종료 때 컨텍스트가 먼저 닫히며 나는 예외는 정상
            pass

    # ---------- 오프로딩 (메인컴에서 추론)
    def accept_loop(self):
        while True:
            try:
                c, addr = self.srv.accept()
            except OSError:
                return                                                   # 노드 종료로 소켓이 닫힘
            c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            c.settimeout(self.remote_timeout)
            with self.conn_lock:
                old, self.conn = self.conn, c
            if old:
                old.close()
            self.get_logger().info(f"오프로딩 서버(메인컴 {addr[0]}) 연결됨")

    def drop_conn(self, why):
        with self.conn_lock:
            c, self.conn = self.conn, None
        if c:
            try:
                c.close()
            except OSError:
                pass
        self.get_logger().warn(why, throttle_duration_sec=2)

    def infer_remote(self, bgr):
        """240x320으로 줄여 JPEG로 보내고 바닥-비바닥 로짓 차를 받음. 실패하면 None(이 프레임은 발행 안 함)"""
        t0 = time.perf_counter()
        small = cv2.resize(bgr, (self.in_hw[1], self.in_hw[0]), interpolation=cv2.INTER_AREA)
        jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_q])[1].tobytes()
        t1 = time.perf_counter()
        with self.conn_lock:
            c = self.conn
        if c is None:
            self.n_fail += 1
            self.get_logger().warn("오프로딩 서버(메인컴) 연결 없음 — 프레임 발행 안 함", throttle_duration_sec=2)
            return None
        self.req_id = (self.req_id + 1) % 2 ** 32
        try:
            c.sendall(REQ.pack(b"FSEG", self.req_id, len(jpg)) + jpg)
            magic, rid, server_ms, h, w, n = REP.unpack(recv_exact(c, REP.size))
            if magic != b"FSEG" or rid != self.req_id:
                raise ConnectionError("응답 번호 어긋남")
            diff = np.frombuffer(recv_exact(c, n), np.float16).reshape(h, w).astype(np.float32)
        except (OSError, ConnectionError, struct.error, ValueError) as e:
            self.n_fail += 1
            self.drop_conn(f"오프로딩 응답 실패({type(e).__name__}: {e}) — 연결을 끊고 메인컴이 다시 접속하길 기다림")
            return None
        t2 = time.perf_counter()
        floor = cv2.resize(diff, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR) > 0
        return floor, (t1 - t0) * 1e3, (t2 - t1) * 1e3, server_ms, len(jpg)

    def boundary(self, floor):
        """열마다 아래에서 위로: run_px 연속 '바닥 아님'이 처음 나오는 행(그 묶음의 가장 아래 행). 없으면 -1"""
        H = floor.shape[0]
        cols = np.arange(0, floor.shape[1], self.col_step)
        nf = ~floor[:, cols]
        run = nf.copy()
        for k in range(1, self.run_px):                                  # 행 y, y-1, ..., y-run+1 이 모두 비바닥
            run[k:] &= nf[:-k]
        run[:self.run_px - 1] = False
        flip = run[::-1]
        idx = flip.argmax(0)
        found = flip[idx, np.arange(len(cols))]
        rows = np.where(found, H - 1 - idx, -1)
        return cols, rows

    # ---------- 주행 시험용 스위치
    def on_params(self, params):
        for p in params:
            if p.name == "mute":
                self.mute = bool(p.value)
                self.get_logger().info("장애물 출력 " + ("끔 (라이다만 조건, 추론은 계속)" if self.mute else "켬"))
            elif p.name == "memory_expire_s":
                self.expire = float(p.value)
                self.get_logger().info(f"기억 유지 {self.expire:.1f}초" + (" = 기억 끔" if self.expire <= 0 else ""))
            elif p.name == "record_dir":
                self.set_record(str(p.value))
            elif p.name == "backend":
                if p.value not in ("local", "remote"):
                    return SetParametersResult(successful=False, reason="backend는 local 또는 remote")
                self.backend = p.value
                self.get_logger().info("추론 위치: " + ("메인컴(오프로딩)" if self.backend == "remote" else "로봇(온보드)"))
        return SetParametersResult(successful=True)

    def set_record(self, d):
        if self.rec_f:
            self.rec_f.close()
            self.rec_f = self.rec = None
            self.get_logger().info(f"프레임 기록 끝: {self.record_dir}")
        self.record_dir = d
        if d:
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, "frames.csv")
            new = not os.path.exists(path)
            self.rec_f = open(path, "a", newline="")
            self.rec = csv.writer(self.rec_f)
            if new:
                self.rec.writerow(["stamp", "pose_src", "odom_x", "odom_y", "odom_yaw", "lag_ms", "front_m", "n_pts", "n_mem",
                                   "mute", "expire_s", "backend", f"rows_every_{self.col_step}px"])
            self.get_logger().info(f"프레임 기록 시작: {d}")

    def on_clear_memory(self, req, res):
        res.message = f"기억 점 {len(self.mem)}개 지움"
        self.mem = np.zeros((0, 3))
        res.success = True
        self.get_logger().info(res.message)
        return res

    # ---------- 기억
    def odom_pose(self, stamp):
        """영상 촬영 시각의 odom 위치. 그 시각 TF가 아직 없으면 가장 최근 위치.
        2026-10-30: 전에는 늘 최근 위치를 써서, 처리 지연(약 0.12초) 동안 움직인 만큼 기억 점이 앞으로 밀렸음(0.25 m/s면 약 3cm)"""
        for when, src in ((Time.from_msg(stamp), "stamp"), (Time(), "latest")):
            try:
                t = self.tfb.lookup_transform("odom", "base_footprint", when)
            except Exception:
                continue
            q = t.transform.rotation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            return (t.transform.translation.x, t.transform.translation.y, yaw), src
        return None, "none"

    def update_memory(self, pts_base, pose, now):
        if pose is None:
            return
        ox, oy, yaw = pose
        if self.last_pose is not None:                                   # 지난 프레임보다 0.5cm 또는 0.6도 넘게 움직였을 때만 시계가 감
            lx, ly, lyaw, lt = self.last_pose
            if math.hypot(ox - lx, oy - ly) > 0.005 or abs(math.remainder(yaw - lyaw, 2 * math.pi)) > 0.01:
                self.mclock += now - lt
        self.last_pose = (ox, oy, yaw, now)
        c, s = math.cos(yaw), math.sin(yaw)
        if len(self.mem):
            dx, dy = self.mem[:, 0] - ox, self.mem[:, 1] - oy
            bx, by = c * dx + s * dy, -s * dx + c * dy                    # odom → 지금 base
            u, v, front = self.geo.base_to_pixel(bx, by)
            # 지금 카메라가 보는 바닥 → 새 관측으로 교체. 맨 아래 줄(479)에서 잡힌 점이 반올림으로 479.x가 되어 사각지대로
            # 오인되면 10초 동안 겹겹이 쌓였음(2026-10-29) → 아래쪽에 1.5px 여유
            visible = front & (u >= -1) & (u < 641) & (v >= self.v_far - 1) & (v <= 480.5)
            keep = (~visible) & (self.mclock - self.mem[:, 2] < self.expire) & (np.hypot(bx, by) < 4.0)
            self.mem = self.mem[keep]
        if len(pts_base):
            ex, ey = ox + c * pts_base[:, 0] - s * pts_base[:, 1], oy + s * pts_base[:, 0] + c * pts_base[:, 1]
            self.mem = np.vstack([self.mem, np.stack([ex, ey, np.full(len(ex), self.mclock)], 1)])

    # ---------- 콜백
    def on_image(self, msg):
        t_start = time.perf_counter()
        if self.max_rate > 0 and t_start - self.t_proc < 1.0 / self.max_rate:
            return                                                       # 처리 상한: 이번 프레임은 건너뜀
        self.t_proc = t_start
        if msg.encoding not in ("bgr8", "rgb8"):
            self.get_logger().warn(f"지원하지 않는 영상 형식 {msg.encoding}", throttle_duration_sec=5)
            return
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step // 3, 3)[:, :msg.width]
        bgr = img if msg.encoding == "bgr8" else img[:, :, ::-1]
        remote = self.backend == "remote"
        if remote:
            r = self.infer_remote(bgr)
            if r is None:
                return                                                   # 발행 안 함 → 0.6초 넘게 이어지면 감시 노드가 로봇을 세움
            floor, pre_ms, inf_ms, server_ms, up = r
        else:
            floor, pre_ms, inf_ms = self.infer_floor(bgr)
            server_ms, up = 0.0, 0
        cols, rows = self.boundary(floor)
        sel = rows >= max(self.v_far, 0)
        bx, by, ok = self.geo.pixel_to_base(cols[sel].astype(float), rows[sel].astype(float))
        ok &= (bx - self.geo.cam_x) <= self.max_range + 1e-6
        pts = np.stack([bx[ok], by[ok]], 1) if ok.any() else np.zeros((0, 2))
        front = pts[np.abs(pts[:, 1]) <= self.front_hw, 0] if len(pts) else np.zeros(0)
        front_m = float(front.min()) if len(front) else 9.9

        now = self.get_clock().now()
        pose, pose_src = self.odom_pose(msg.header.stamp)
        self.n_pose[pose_src] += 1
        self.update_memory(pts, pose, now.nanoseconds * 1e-9)
        hdr = msg.header
        hdr.frame_id = "base_footprint"
        self.pub_pts.publish(point_cloud2.create_cloud_xyz32(hdr, [] if self.mute else [(float(x), float(y), self.z_mark) for x, y in pts]))
        r_end = np.where(sel, rows, int(math.ceil(self.v_far))).astype(float)   # 장애물이 없거나 max_range 밖인 열 = max_range 바닥 점
        cx_, cy_, cok = self.geo.pixel_to_base(cols.astype(float), r_end)
        self.pub_clr.publish(point_cloud2.create_cloud_xyz32(hdr, [(float(x), float(y), self.z_mark) for x, y in zip(cx_[cok], cy_[cok])]))
        if pose is not None:
            hdr_m = msg.header.__class__(stamp=msg.header.stamp, frame_id="odom")
            self.pub_mem.publish(point_cloud2.create_cloud_xyz32(
                hdr_m, [] if self.mute else [(float(x), float(y), self.z_mark) for x, y, _ in self.mem]))
        self.pub_front.publish(Float32(data=front_m))
        post_ms = (time.perf_counter() - t_start) * 1e3 - pre_ms - inf_ms
        total_ms = (time.perf_counter() - t_start) * 1e3
        stamp_ns = Time.from_msg(msg.header.stamp).nanoseconds
        lag_ms = (now.nanoseconds - stamp_ns) / 1e6
        self.csv.writerow([f"{stamp_ns}", f"{lag_ms:.1f}", f"{pre_ms:.1f}", f"{inf_ms:.1f}",
                           f"{post_ms:.1f}", f"{total_ms:.1f}", len(pts), len(self.mem), f"{front_m:.3f}", int(pose is not None),
                           self.backend, f"{server_ms:.1f}", up, self.n_fail])
        if self.rec:                                                     # 주행 시험 회차 기록 (프레임 간 판정 흔들림 분석용)
            cv2.imwrite(os.path.join(self.record_dir, f"{stamp_ns}.jpg"), np.ascontiguousarray(bgr), [cv2.IMWRITE_JPEG_QUALITY, 92])
            px, py, pyaw = pose if pose is not None else (float("nan"),) * 3
            self.rec.writerow([stamp_ns, pose_src, f"{px:.4f}", f"{py:.4f}", f"{pyaw:.4f}", f"{lag_ms:.1f}", f"{front_m:.3f}", len(pts),
                               len(self.mem), int(self.mute), self.expire, self.backend, " ".join(map(str, rows.tolist()))])
        self.n += 1
        self.win.append(total_ms)
        if self.debug_every and self.n % self.debug_every == 0:
            self.save_debug(bgr, floor, cols[sel][ok], rows[sel][ok], front_m)
        if time.time() - self.t_last_print > 5:
            dt = time.time() - self.t_last_print
            how = f"메인컴 왕복 {inf_ms:.0f} ms (메인컴 처리 {server_ms:.0f}, 실패 누적 {self.n_fail})" if remote else f"추론 {inf_ms:.0f}"
            self.get_logger().info(f"{len(self.win) / dt:.1f} fps, 노드 {np.mean(self.win):.0f} ms ({how}), "
                                   f"장애물 점 {len(pts)}, 기억 {len(self.mem)}, 정면 {front_m:.2f} m, odom {'정상' if pose else '없음'}"
                                   f" (촬영 시각 위치 {self.n_pose['stamp']}/최근 위치 {self.n_pose['latest']}/없음 {self.n_pose['none']})"
                                   + (", 장애물 출력 끔" if self.mute else "") + (", 기억 끔" if self.expire <= 0 else "")
                                   + (", 프레임 기록 중" if self.rec else ""))
            self.csv_f.flush()
            if self.rec_f:
                self.rec_f.flush()
            self.t_last_print, self.win = time.time(), []

    def save_debug(self, bgr, floor, cols, rows, front_m):
        vis = bgr.copy()
        vis[floor] = (0.6 * vis[floor] + 0.4 * np.array([60, 200, 60])).astype(np.uint8)
        for u, v in zip(cols, rows):
            cv2.circle(vis, (int(u), int(v)), 3, (0, 0, 255), -1)
        cv2.line(vis, (0, int(self.v_far)), (639, int(self.v_far)), (255, 200, 0), 1)
        cv2.putText(vis, f"front {front_m:.2f} m  pts {len(cols)}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        k = (self.n // self.debug_every) % 200
        cv2.imwrite(os.path.join(self.out_dir, "debug", f"dbg_{k:03d}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])

    def destroy_node(self):
        self.csv_f.close()
        if self.rec_f:
            self.rec_f.close()
        for s in (self.conn, self.srv):
            if s:
                try:
                    s.close()
                except OSError:
                    pass
        self.tf_exec.shutdown(timeout_sec=1.0)
        self.tf_node.destroy_node()
        super().destroy_node()


def main():
    rclpy.init()
    node = FloorSegNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):    # SIGINT로 끌 때 콜백 도중 발행이 실패하는 것은 정상
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
