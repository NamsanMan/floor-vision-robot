#!/usr/bin/env python3
"""주행 시험 도구 (로봇 컨테이너, 2026-10-30) — 낮은 장애물 정지·회피 시험을 한 줄 명령으로 돌리고 회차마다 기록한다.
호스트(pi)에서: bash /home/pi/docker/tmp/drive_test/dt.sh <명령> [옵션]

조건: Nav2는 한 번만 켜 두고 인식 노드 파라미터만 바꾼다. 노드를 끄면 카메라 층이 '갱신 안 됨'(expected_update_rate 1초)이 되어
      Nav2 제어기가 코스트맵을 기다리며 멈추므로 끄지 않는다.
  lidar      장애물 출력 끔(mute). 추론은 그대로라 CPU 부하가 같다 → 코스트맵·Collision Monitor엔 LiDAR만
  cam        카메라 + 장애물 점 기억(10초)
  cam_nomem  카메라, 기억 끔(지금 보이는 점만) → 사각지대로 들어간 장애물을 Collision Monitor가 놓치는지
명령:
  cond <조건>                  조건만 바꾸기
  pose save | reset | click    RViz 2D Pose Estimate로 출발 위치를 맞춘 뒤 save / 로봇을 들어 출발선에 다시 놓은 뒤 reset /
                               RViz Publish Point로 찍은 목적지 저장(click) → goal --to-click
  straight --cond C --obj O [--at 1.5] [--speed 0.10]
      시험 ①: Collision Monitor 입력(cmd_vel_smoothed)에 직진 명령, 경로 계획 없음.
      끝 = Collision Monitor가 1초 넘게 세움(정지) / 장애물 앞쪽 끝 + 3cm까지 감(충돌) / 시간 초과
  goal --cond C --obj O [--at 1.2] [--dist 2.5]
      시험 ②: 저장한 출발 위치에서 앞으로 dist m 목표를 Nav2에 보냄. 보내기 전 AMCL 위치를 출발 위치로 다시 맞추고
      코스트맵·기억을 비움(로봇을 들어 옮기면 odom은 그대로라 옛 장애물이 로봇 바로 앞에 남는다)
  idle --cond C [--secs 10] [--tag 이름]   제자리 기록(점검·조명 비교·온보드 vs 오프로딩 지연·CPU)
  --backend local|remote       추론 위치(기본 local=로봇). remote = 메인컴(offload/offload_pc.py가 접속해 있어야 함)
  note <회차> <메모...>   결과에 메모 추가(A4로 잰 거리 등)  /  table   결과 표
  stop                     비상: 진행 중인 시험 중단 + Nav2 목표 전부 취소 + 정지 명령
  공통: --note 메모, --no-frames(프레임 저장 끔), --session 이름(기본 오늘 날짜), --dry(straight: 명령을 빈 토픽으로, 점검용)
--at = 출발선(로봇 앞면)에서 장애물 앞쪽 끝까지(m). 장애물 크기(진행 방향 길이·폭)는 OBJ 근사값, --size L W로 바꿈.
기록: /home/ubuntu/shared/drive_tests/<날짜>/results.csv(회차당 한 줄), <회차>_<시험>_<조건>_<물체>/log.csv(20Hz 상태)·
      events.json(Collision Monitor 로그·감시 신호·카메라 거리)·frames/(인식 노드가 저장한 프레임, 판정 흔들림 분석용)
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import signal
import subprocess
import sys
import time

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped, Twist
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ClearEntireCostmap
from rcl_interfaces.srv import SetParameters
from rclpy.action import ActionClient
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

ROOT = os.environ.get("DT_ROOT", "/home/ubuntu/shared/drive_tests")
NAV_LOGS = os.environ.get("DT_NAVLOG", "/home/ubuntu/shared/nav2_cam/logs")
FS_DIR = os.environ.get("DT_FSDIR", "/home/ubuntu/shared/floor_seg")      # 인식 노드 기록(timing_*.csv)·node.pid
NODE = "/floor_seg"
FRONT_X, HALF_L, HALF_W = 0.11, 0.11, 0.09        # 로봇 외곽(base_footprint 기준, nav2 footprint x ±0.11, y ±0.09). 앞면 = x 0.11
NEAR_VISIBLE = 0.30                                # 화면 맨 아래 줄 바닥 = base 앞 0.287 m. 이보다 가까운 정면 거리는 장애물 앞 끝이 아니라 화면 끝
MAX_STRAIGHT_SPEED = 0.15
ODOM_SCALE = 0.85            # 실제 이동 / odom 이동. 2026-10-31 실측: odom 1.52 m인데 실제 1.29 m(A4) — 메카넘 바퀴가 미끄러져 많이 셈.
                             # 시험 도구의 이동·남은 거리·충돌 판정에만 쓴다(Collision Monitor·기억은 odom 그대로 → 일찍 서는 쪽이라 안전)
OBJ = {"none": (0.0, 0.0), "book": (0.26, 0.19), "shoe": (0.27, 0.10), "cable": (0.25, 0.25),   # 진행 방향 길이·폭(m) 근사
       "cup": (0.08, 0.08), "multi": (0.0, 0.0)}                    # multi = 여러 개 흩어 놓은 종합 주행(여유 계산 안 함, 영상으로 판단)
OBJ_KO = {"none": "장애물 없음", "book": "책", "shoe": "운동화", "cable": "케이블", "cup": "종이컵", "multi": "여러 개(종합)"}
CONDS = {"lidar": (True, 10.0), "cam": (False, 10.0), "cam_nomem": (False, 0.0)}                # (mute, memory_expire_s)
COND_KO = {"lidar": "라이다만", "cam": "카메라+기억", "cam_nomem": "카메라·기억 끔"}
TEST_KO = {"straight": "시험① 직진 정지", "goal": "시험② 목표 회피", "idle": "제자리 기록"}
LOGRE = re.compile(r"\[(INFO|WARN|ERROR)\] \[(\d+\.\d+)\] \[([^\]]+)\]: (.*)")
PROBLEMS = {"LiDAR 드롭": r"dropping message: frame .lidar_frame", "카메라 점 드롭": r"dropping message: frame .(base_footprint|odom)",
            "카메라 층 갱신 안 됨": r"floor_obstacles/(memory|clearing) observation buffer has not been updated",
            "주기 못 맞춤": r"missed its desired rate|Failed to meet update rate", "TF 오류": r"Extrapolation Error|extrapolation into the",
            "CM 소스 무시": r"Ignoring the source", "진행 실패": r"Failed to make progress"}
LOG_COLS = ["t", "along", "lateral", "dyaw", "odom_x", "odom_y", "odom_yaw", "map_x", "map_y", "map_yaw", "cmd_in_vx",
            "cmd_out_vx", "cmd_out_vy", "cmd_out_wz", "cmd_out_age", "front_m", "front_age", "mem_front_m", "mem_n", "wd_n", "nav_dist"]
RESULT_COLS = ["run", "time", "test", "cond", "backend", "obj", "at_m", "result", "travel_m", "gap_odom_cm", "gap_cam_cm", "clear_cm",
               "first_det_m", "slow_gap_cm", "duration_s", "cm_stop", "cm_slow", "cm_approach", "wd_trips", "recoveries",
               "fps", "lag_p50_ms", "lag_p95_ms", "lag_p99_ms", "lag_max_ms", "inf_p50_ms", "server_p50_ms", "remote_fail",
               "cpu_pct", "node_cores", "problems", "frames", "note"]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def cloud_xy(m):
    if m.width * m.height == 0:
        return np.zeros((0, 2))
    a = np.frombuffer(bytes(m.data), np.float32).reshape(-1, m.point_step // 4)    # 우리 노드의 xyz32 점구름 (x, y = 앞 두 칸)
    return a[:, :2].astype(float)


class Tester:
    def __init__(self):
        n = self.n = rclpy.create_node("drive_test")
        self.tfb = Buffer()
        self.tfl = TransformListener(self.tfb, n)
        self.pub_cmd = n.create_publisher(Twist, "/cmd_vel_smoothed", 10)
        self.pub_init = n.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
        self.front, self.out, self.cin = (9.9, 0.0), None, None
        self.n_out, self.wd, self.mem = 0, (0, 0.0), np.zeros((0, 2))
        self.rows, self.fronts, self.wd_trips, self.t0 = [], [], [], time.time()
        self.scale = ODOM_SCALE
        n.create_subscription(Float32, "/floor_obstacles/front_distance", self.on_front, 10)
        n.create_subscription(Twist, "/cmd_vel", self.on_out, 10)
        n.create_subscription(Twist, "/cmd_vel_smoothed", self.on_in, 10)
        n.create_subscription(PointCloud2, "/floor_obstacles/watchdog", self.on_wd, 10)
        n.create_subscription(PointCloud2, "/floor_obstacles/memory", self.on_mem, 10)
        self.cli_param = n.create_client(SetParameters, NODE + "/set_parameters")
        self.cli_clear = n.create_client(Trigger, NODE + "/clear_memory")
        self.cli_cm = n.create_client(GetState, "/collision_monitor/get_state")
        self.cli_lc = n.create_client(ClearEntireCostmap, "/local_costmap/clear_entirely_local_costmap")
        self.cli_gc = n.create_client(ClearEntireCostmap, "/global_costmap/clear_entirely_global_costmap")
        self.cli_cancel = n.create_client(CancelGoal, "/navigate_to_pose/_action/cancel_goal")
        self.nav = ActionClient(n, NavigateToPose, "/navigate_to_pose")

    # ----- 구독
    def on_front(self, m):
        self.front = (float(m.data), time.time())
        self.fronts.append((round(time.time() - self.t0, 3), round(float(m.data), 3)))

    def on_out(self, m):
        self.out = (m.linear.x, m.linear.y, m.angular.z, time.time())
        self.n_out += 1

    def on_in(self, m):
        self.cin = (m.linear.x, m.linear.y, m.angular.z, time.time())

    def on_wd(self, m):
        k = m.width * m.height
        if k > 0 and self.wd[0] == 0:
            self.wd_trips.append(round(time.time() - self.t0, 2))
        self.wd = (k, time.time())

    def on_mem(self, m):
        self.mem = cloud_xy(m)

    # ----- 도우미
    def spin(self, secs):
        end = time.time() + secs
        while True:
            rclpy.spin_once(self.n, timeout_sec=max(0.0, min(0.02, end - time.time())))
            if time.time() >= end:
                return

    def call(self, cli, req, timeout=3.0):
        if not cli.wait_for_service(timeout_sec=timeout):
            return None
        fut = cli.call_async(req)
        end = time.time() + timeout
        while not fut.done() and time.time() < end:
            rclpy.spin_once(self.n, timeout_sec=0.05)
        return fut.result() if fut.done() else None

    def set_params(self, **kw):
        req = SetParameters.Request()
        req.parameters = [Parameter(k, value=v).to_parameter_msg() for k, v in kw.items()]
        res = self.call(self.cli_param, req)
        return res is not None and all(r.successful for r in res.results)

    def set_cond(self, cond, backend="local"):
        mute, expire = CONDS[cond]
        return self.set_params(mute=mute, memory_expire_s=float(expire), backend=backend)

    def clear_memory(self):
        res = self.call(self.cli_clear, Trigger.Request())
        return res.message if res else None

    def cm_state(self):
        res = self.call(self.cli_cm, GetState.Request(), 2.0)
        return res.current_state.label if res else "응답 없음"

    def pose(self, parent):
        try:
            t = self.tfb.lookup_transform(parent, "base_footprint", Time())
        except Exception:
            return None
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        age = (self.n.get_clock().now() - Time.from_msg(t.header.stamp)).nanoseconds * 1e-9
        return t.transform.translation.x, t.transform.translation.y, yaw, age

    def zero(self, k=10):
        for _ in range(k):
            self.pub_cmd.publish(Twist())
            self.spin(0.05)

    def reset_pose(self, sp):
        m = PoseWithCovarianceStamped()
        m.header.frame_id = "map"
        m.pose.pose.position.x, m.pose.pose.position.y = float(sp[0]), float(sp[1])
        m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(sp[2] / 2), math.cos(sp[2] / 2)
        cov = [0.0] * 36
        cov[0] = cov[7] = 0.01                             # 위치 표준편차 10cm, 방향 5도 (출발선 테이프 기준)
        cov[35] = 0.0076
        m.pose.covariance = cov
        end = time.time() + 2.0
        while self.pub_init.get_subscription_count() == 0 and time.time() < end:
            self.spin(0.1)
        # 보낸 뒤 AMCL 위치가 정말 바뀌었는지 확인하고, 아니면 다시 보냄(2026-10-31: 막 뜬 프로세스가 바로 보내면 AMCL 쪽 연결이
        # 덜 잡혀 버려지는 일이 있었음 — 목표 주행 12·13회차의 재설정이 AMCL에 안 들어가 13회차가 '이미 도착'으로 끝남)
        for _ in range(5):
            for _ in range(2):
                m.header.stamp = self.n.get_clock().now().to_msg()
                self.pub_init.publish(m)
                self.spin(0.2)
            self.spin(0.8)
            pm = self.pose("map")
            if pm and math.hypot(pm[0] - sp[0], pm[1] - sp[1]) < 0.25 and abs(wrap(pm[2] - sp[2])) < math.radians(10):
                return True
        return False

    def mem_front(self, po):
        """기억 점 중 지금 로봇 폭(±0.12m) 안에서 가장 가까운 앞 거리 (base 원점 기준, 사각지대 포함)"""
        if po is None or not len(self.mem):
            return 9.9
        c, s = math.cos(po[2]), math.sin(po[2])
        dx, dy = self.mem[:, 0] - po[0], self.mem[:, 1] - po[1]
        bx, by = c * dx + s * dy, -s * dx + c * dy
        m = (bx > 0) & (np.abs(by) <= 0.12)
        return float(bx[m].min()) if m.any() else 9.9

    def begin(self):
        self.t0 = time.time()
        self.cpu0 = cpu_sample()
        self.rows, self.fronts, self.wd_trips = [], [], []
        self.out, self.n_out = None, 0                      # 지난 회차의 Collision Monitor 출력은 버림
        if self.wd[0] > 0:                                  # 시작 순간 이미 켜져 있으면 그것도 한 번으로 셈
            self.wd_trips.append(0.0)

    def sample(self, ref, nav_dist=None):
        now = time.time()
        r = {"t": round(now - self.t0, 3)}
        po, pm = self.pose("odom"), self.pose("map")
        if po:
            c, s, k = math.cos(ref[2]), math.sin(ref[2]), self.scale
            dx, dy = po[0] - ref[0], po[1] - ref[1]
            r.update(along=round(k * (c * dx + s * dy), 4), lateral=round(k * (-s * dx + c * dy), 4), dyaw=round(wrap(po[2] - ref[2]), 4),
                     odom_x=round(po[0], 4), odom_y=round(po[1], 4), odom_yaw=round(po[2], 4))
        if pm:
            r.update(map_x=round(pm[0], 4), map_y=round(pm[1], 4), map_yaw=round(pm[2], 4))
        if self.cin:
            r["cmd_in_vx"] = round(self.cin[0], 4)
        if self.out:
            r.update(cmd_out_vx=round(self.out[0], 4), cmd_out_vy=round(self.out[1], 4), cmd_out_wz=round(self.out[2], 4),
                     cmd_out_age=round(now - self.out[3], 3))
        r.update(front_m=round(self.front[0], 3), front_age=round(now - self.front[1], 3), mem_front_m=round(self.mem_front(po), 3),
                 mem_n=len(self.mem), wd_n=self.wd[0])
        if nav_dist is not None:
            r["nav_dist"] = round(nav_dist, 3)
        self.rows.append(r)
        return r, po


# ---------- 기록·분석
def cpu_sample():
    """(전체 jiffies, 쉰 jiffies, 인식 노드 utime+stime jiffies 또는 None, 시각). 컨테이너의 /proc/stat = 로봇 전체"""
    try:
        v = list(map(int, open("/proc/stat").readline().split()[1:]))
        tot, idle = sum(v[:8]), v[3] + v[4]
    except (OSError, ValueError):
        return None
    node = None
    try:
        pid = open(os.path.join(FS_DIR, "node.pid")).read().strip()
        st = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
        node = int(st[11]) + int(st[12])                    # utime, stime (14·15번째 칸)
    except (OSError, ValueError, IndexError):
        pass
    return tot, idle, node, time.time()


def cpu_usage(c0, c1):
    if not c0 or not c1 or c1[0] == c0[0]:
        return {}
    out = {"cpu_pct": round(100 * (1 - (c1[1] - c0[1]) / (c1[0] - c0[0])), 1)}
    if c0[2] is not None and c1[2] is not None:
        out["node_cores"] = round((c1[2] - c0[2]) / os.sysconf("SC_CLK_TCK") / (c1[3] - c0[3]), 2)
    return out


def node_timing(t0, t1):
    """인식 노드 지연 기록(timing_*.csv)에서 이 회차 구간: 처리 fps, 촬영→발행 지연 분위수, 추론(또는 메인컴 왕복)·메인컴 처리 시간"""
    logs = glob.glob(os.path.join(FS_DIR, "logs", "timing_*.csv"))
    if not logs:
        return {}
    rows = [r for r in csv.DictReader(open(max(logs, key=os.path.getmtime))) if t0 <= int(r["stamp"]) * 1e-9 <= t1]
    if len(rows) < 2:
        return {}
    lag, inf = np.array([float(r["lag_ms"]) for r in rows]), np.array([float(r["inf_ms"]) for r in rows])
    out = {"fps": round(len(rows) / (t1 - t0), 2), "lag_p50_ms": round(float(np.median(lag)), 1),
           "lag_p95_ms": round(float(np.percentile(lag, 95)), 1), "lag_p99_ms": round(float(np.percentile(lag, 99)), 1),
           "lag_max_ms": round(float(lag.max()), 1), "inf_p50_ms": round(float(np.median(inf)), 1)}
    if "remote_fail" in rows[0]:
        srv = [float(r["server_ms"]) for r in rows if r["backend"] == "remote"]
        if srv:
            out["server_p50_ms"] = round(float(np.median(srv)), 1)
            out["up_kb"] = round(float(np.mean([int(r["up_bytes"]) for r in rows if r["backend"] == "remote"])) / 1024, 1)
        out["remote_fail"] = int(rows[-1]["remote_fail"]) - int(rows[0]["remote_fail"])
    return out


def session_dir(a):
    d = os.path.join(ROOT, a.session)
    os.makedirs(d, exist_ok=True)
    return d


def read_results(d):
    p = os.path.join(d, "results.csv")
    return list(csv.DictReader(open(p, encoding="utf-8"))) if os.path.exists(p) else []


def write_results(d, rows):
    with open(os.path.join(d, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def new_run(a, test):
    d = session_dir(a)
    k = 1 + max([int(r["run"]) for r in read_results(d)] + [int(x.split("_")[0]) for x in os.listdir(d) if x[:2].isdigit()] + [0])
    obj = getattr(a, "obj", "none")
    rd = os.path.join(d, f"{k:02d}_{test}_{a.cond}_{obj}" + (f"_{a.tag}" if getattr(a, "tag", "") else ""))
    os.makedirs(rd, exist_ok=True)
    return {"run": k, "dir": rd, "sess": d, "test": test, "time": time.strftime("%H:%M:%S")}


def nav_log_mark():
    logs = glob.glob(os.path.join(NAV_LOGS, "nav_*.log"))
    if not logs:
        return None
    p = max(logs, key=os.path.getmtime)
    return [p, os.path.getsize(p)]


def nav_log_events(mark, t0, t1):
    cm, probs = [], {k: 0 for k in PROBLEMS}
    if not mark or not os.path.exists(mark[0]):
        return cm, probs
    with open(mark[0], "rb") as f:
        f.seek(mark[1])
        seg = f.read().decode("utf-8", errors="ignore")
    for line in seg.splitlines():
        m = LOGRE.search(line)
        if not m or not (t0 - 0.5 <= float(m.group(2)) <= t1 + 0.5):
            continue
        msg = m.group(4)
        if msg.startswith("Robot to"):
            kind = ("stop" if msg.startswith("Robot to stop") else "slow" if msg.startswith("Robot to slow")
                    else "approach" if msg.startswith("Robot to approach") else "continue")
            cm.append((round(float(m.group(2)) - t0, 2), kind, msg))
        for k, p in PROBLEMS.items():
            if re.search(p, line):
                probs[k] += 1
    return cm, probs


def along_at(rows, t):
    best = None
    for r in rows:
        if "along" in r and (best is None or abs(r["t"] - t) < abs(best["t"] - t)):
            best = r
    return best["along"] if best else None


def first_detection(fronts):
    """정면 거리가 3번 연속 잡힌(9m 미만) 첫 순간 (실행 기준 초, base 앞 거리)"""
    for i in range(len(fronts) - 2):
        if all(fronts[j][1] < 9.0 for j in range(i, i + 3)):
            return fronts[i]
    return None


def last_visible(fronts):
    for t, v in reversed(fronts):
        if NEAR_VISIBLE < v < 9.0:
            return t, v
    return None


def seg_dist(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)))
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def overlap(P, Q):
    for poly in (P, Q):
        for i in range(len(poly)):
            (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
            nx, ny = y1 - y2, x2 - x1
            pa, qa = [nx * x + ny * y for x, y in P], [nx * x + ny * y for x, y in Q]
            if max(pa) < min(qa) or max(qa) < min(pa):
                return False
    return True


def clearance(rows, at, size):
    """로봇 외곽과 장애물(출발 base 기준 앞 FRONT_X+at부터 길이 L, 폭 W) 사이 최소 거리(m). 겹치면 음수"""
    L, W = size
    ox0, ox1 = FRONT_X + at, FRONT_X + at + L
    obs = [(ox0, -W / 2), (ox1, -W / 2), (ox1, W / 2), (ox0, W / 2)]
    best = None
    for r in rows:
        if "along" not in r or r["along"] + 0.4 < ox0 or r["along"] - 0.4 > ox1:
            continue
        c, s = math.cos(r["dyaw"]), math.sin(r["dyaw"])
        fp = [(r["along"] + c * px - s * py, r["lateral"] + s * px + c * py)
              for px, py in ((HALF_L, HALF_W), (HALF_L, -HALF_W), (-HALF_L, -HALF_W), (-HALF_L, HALF_W))]
        if overlap(fp, obs):
            d = -0.01
        else:
            d = min(min(seg_dist(p, obs[i], obs[(i + 1) % 4]) for p in fp for i in range(4)),
                    min(seg_dist(q, fp[i], fp[(i + 1) % 4]) for q in obs for i in range(4)))
        best = d if best is None else min(best, d)
    return best


def finish(T, a, run, result, extra, mark, t1):
    """기록 저장 + 결과 한 줄 추가 + 공통 요약 출력"""
    cm, probs = nav_log_events(mark, T.t0, t1)
    perf = {**node_timing(T.t0, t1), **cpu_usage(T.cpu0, cpu_sample())}
    extra = {**extra, **perf}
    n_frames = len(glob.glob(os.path.join(run["dir"], "frames", "*.jpg")))
    with open(os.path.join(run["dir"], "log.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(T.rows)
    ev = {"args": {k: v for k, v in vars(a).items() if k != "func"}, "t0_epoch": T.t0, "result": result, "extra": extra, "cm_log": cm,
          "problems": probs, "watchdog_trips_s": T.wd_trips, "front_distance": T.fronts}
    json.dump(ev, open(os.path.join(run["dir"], "events.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    cnt = {k: sum(1 for c in cm if c[1] == k) for k in ("stop", "slow", "approach")}
    prob_s = ", ".join(f"{k} {v}" for k, v in probs.items() if v)
    row = {"run": run["run"], "time": run["time"], "test": run["test"], "cond": a.cond, "backend": a.backend, "obj": getattr(a, "obj", "none"),
           "at_m": getattr(a, "at", ""), "result": result, "duration_s": round(t1 - T.t0, 1), "cm_stop": cnt["stop"],
           "cm_slow": cnt["slow"], "cm_approach": cnt["approach"], "wd_trips": len(T.wd_trips), "problems": prob_s,
           "frames": n_frames, "note": a.note or ""}
    row.update({k: v for k, v in extra.items() if k in RESULT_COLS})
    rows = read_results(run["sess"])
    rows.append(row)
    write_results(run["sess"], rows)
    if "lag_p50_ms" in perf:
        how = (f"메인컴 왕복 {perf['inf_p50_ms']:.0f} ms(메인컴 처리 {perf.get('server_p50_ms', 0):.0f}, JPEG {perf.get('up_kb', 0):.0f} KB)"
               if a.backend == "remote" else f"추론 {perf['inf_p50_ms']:.0f} ms")
        print(f"     지연(촬영→발행) 중앙값 {perf['lag_p50_ms']:.0f} ms · p95 {perf['lag_p95_ms']:.0f} · p99 {perf['lag_p99_ms']:.0f}"
              f" · 최대 {perf['lag_max_ms']:.0f} | {how} | {perf['fps']:.1f} fps"
              + (f" | 실패 {perf['remote_fail']}회" if perf.get("remote_fail") else "")
              + (f" | 로봇 CPU {perf['cpu_pct']:.0f}%" if "cpu_pct" in perf else "")
              + (f", 인식 노드 {perf['node_cores']:.2f}코어" if "node_cores" in perf else ""))
    print(f"     Collision Monitor 로그: 정지 {cnt['stop']} / 감속 {cnt['slow']} / 접근 {cnt['approach']}"
          f" · 감시 정지 신호 {len(T.wd_trips)}회" + (f"({', '.join(f'{t:.1f}초' for t in T.wd_trips)})" if T.wd_trips else "")
          + f" · Nav2 경고: {prob_s or '없음'} · 프레임 {n_frames}장")
    print(f"     기록: {run['dir']}")


def prepare(T, a, run, need_cm=True):
    """조건 설정 → 기억 비우기 → 프레임 기록 시작 → 점검. 문제 목록을 돌려줌(빈 목록 = 시작 가능)"""
    if not T.set_cond(a.cond, a.backend):
        return ["인식 노드 조건 설정 실패 — 노드가 켜져 있는지 확인(run_floor_seg.sh status)"]
    T.clear_memory()
    if not a.no_frames:
        T.set_params(record_dir=os.path.join(run["dir"], "frames"))
    T.spin(3.0)                                             # 새 조건으로 몇 프레임 받기 + 이 프로세스가 뜰 때 생기는 순간 부하가 출발 전에
                                                            # 지나가게(2026-10-31: 출발 3초 동안 추론 120→375ms, 카메라 점 드롭 14회)
    bad = []
    if need_cm and not getattr(a, "dry", False):
        st = T.cm_state()
        if st != "active":
            bad.append(f"Collision Monitor 상태: {st}")
    if time.time() - T.front[1] > 1.0:
        bad.append("인식 노드 데이터가 안 옴" + (" — 메인컴 offload_pc.py가 접속해 있는지 확인" if a.backend == "remote" else ""))
    if T.wd[0] > 0:
        bad.append("감시 노드 정지 신호가 켜져 있음")
    po = T.pose("odom")
    if po is None or po[3] > 0.5:                          # 순간 끊김일 수 있어 한 번 더 봄(2026-10-30 로봇에서 1/5회 나옴)
        T.spin(0.5)
        po = T.pose("odom")
        if po is None or po[3] > 0.5:
            bad.append("odom 위치(TF)가 없거나 늦음" + (f"({po[3]:.2f}초 전 값)" if po else ""))
    return bad


def head(a, run, more=""):
    obj = getattr(a, "obj", None)
    at = f" @{a.at:.2f} m" if obj not in (None, "none") and hasattr(a, "at") else ""
    what = f" | {OBJ_KO.get(obj, obj)}{at}" if obj else ""
    where = " | 메인컴 추론(오프로딩)" if a.backend == "remote" else ""
    print(f"[{run['run']:02d}] {TEST_KO[run['test']]} | {COND_KO[a.cond]}{where}{what}{more}", flush=True)


def stop_frames(T, a):
    if not a.no_frames:
        T.set_params(record_dir="")


# ---------- 명령
def cmd_straight(T, a):
    if abs(a.speed) > MAX_STRAIGHT_SPEED:
        sys.exit(f"안전: 직진 시험 속도는 {MAX_STRAIGHT_SPEED} m/s 이하로")
    if a.dry:
        a.session += "_dry"                                             # 점검 회차는 본 결과 표에 섞지 않음
    run = new_run(a, "straight")
    head(a, run, f" | {a.speed:.2f} m/s" + (" | 점검(--dry)" if a.dry else ""))
    bad = prepare(T, a, run)
    if bad:
        stop_frames(T, a)
        print("     시작 안 함: " + " / ".join(bad))
        return
    pub = T.n.create_publisher(Twist, "/drive_test/dry_cmd", 10) if a.dry else T.pub_cmd
    ref = T.pose("odom")[:3]
    limit = a.at + 0.03
    mark = nav_log_mark()
    T.begin()
    cmd = Twist()
    cmd.linear.x = a.speed
    reason, stop_since, last, hist = None, None, 0.0, []
    try:
        while reason is None:
            rclpy.spin_once(T.n, timeout_sec=0.01)
            now = time.time()
            if now - last < 0.05:
                continue
            last = now
            t = now - T.t0
            pub.publish(cmd)
            r, po = T.sample(ref)
            if po is None or po[3] > 1.0:                    # 1초 넘게 odom이 안 오면 멈춤(10cm/s면 최대 10cm)
                reason = "중단: odom 끊김" + (f"({po[3]:.2f}초 전 값)" if po else "")
                break
            hist.append((now, r["along"]))
            if r["along"] >= limit:
                reason = "충돌" if a.obj != "none" else "끝까지 감"
            elif a.dry and t > 3.0:
                reason = "점검 끝"
            elif not a.dry and t > 1.5 and T.n_out == 0:
                reason = "중단: Collision Monitor 출력 없음"
            elif t > a.timeout:
                reason = "시간 초과"
            elif T.out is not None and abs(T.out[0]) < 0.002 and (now - T.out[3] < 0.3 or stop_since is not None):
                stop_since = stop_since or t                 # Collision Monitor가 0을 내보냄 (stop_pub_timeout 뒤엔 조용해짐)
                old = [h for h in hist if h[0] <= now - 0.5]
                v = (r["along"] - old[-1][1]) / (now - old[-1][0]) if old else 1.0
                if t - stop_since >= 1.0 and abs(v) < 0.01:
                    reason = "정지"
            else:
                stop_since = None
    except KeyboardInterrupt:
        reason = "중단: 사용자"
    finally:
        T.zero()
    T.spin(0.5)
    t1 = time.time()
    stop_frames(T, a)
    end = T.rows[-1].get("along", 0.0) if T.rows else 0.0
    result = {"정지": "헛정지" if a.obj == "none" else "정지", "충돌": "충돌", "끝까지 감": "통과(장애물 없음)"}.get(reason, reason)
    ex = {"travel_m": round(end, 3), "reason": reason, "stop_t": stop_since}
    fd, lv = first_detection(T.fronts), last_visible(T.fronts)
    if a.obj != "none":
        ex["gap_odom_cm"] = round((a.at - end) * 100, 1)
        if lv:
            ex["gap_cam_cm"] = round(((lv[1] - FRONT_X) - (end - (along_at(T.rows, lv[0]) or 0.0))) * 100, 1)
        if fd:
            ex["first_det_m"] = round(fd[1] - FRONT_X, 3)
    slow = next((r for r in T.rows if r.get("cmd_out_age", 9) < 0.3 and 0.002 < r.get("cmd_out_vx", 0) < 0.9 * a.speed), None)
    if slow and a.obj != "none":
        ex["slow_gap_cm"] = round((a.at - slow["along"]) * 100, 1)
    print(f"  → {result}" + (f" ({reason})" if result != reason and not result.startswith(("정지", "충돌", "통과")) else "")
          + f" · 이동 {end:.2f} m(odom×{T.scale:g}) · {t1 - T.t0:.1f}초", flush=True)
    if a.obj != "none":
        g = f"     남은 거리(로봇 앞면~장애물): odom 계산 {ex['gap_odom_cm']:.0f} cm"
        g += f" / 카메라 마지막 관측 기준 {ex['gap_cam_cm']:.0f} cm" if "gap_cam_cm" in ex else ""
        print(g + " — A4로 잰 값은 note로 남겨 주세요")
        if fd:
            print(f"     카메라: 처음 잡음 {fd[1] - FRONT_X:.2f} m 앞({fd[0]:.1f}초)"
                  + (f" · 사각지대 들어가기 전 마지막 관측 {lv[1] - FRONT_X:.2f} m 앞({lv[0]:.1f}초)" if lv else ""))
        else:
            print("     카메라: 장애물을 3프레임 연속으로 잡은 적 없음")
        if slow:
            print(f"     감속 시작: 장애물까지 {ex['slow_gap_cm']:.0f} cm (odom)")
    finish(T, a, run, result, ex, mark, t1)


def cmd_goal(T, a):
    sp = load_start()
    if sp is None:
        print("출발 위치가 없습니다. RViz 2D Pose Estimate로 맞춘 뒤 'pose save'를 먼저 하세요.")
        return
    if a.to_click:                                         # RViz Publish Point로 찍은 목적지(2026-10-31 종합 데모)
        gp = load_click()
        if gp is None:
            print("찍은 목적지가 없습니다 — RViz 'Publish Point'로 찍은 뒤 'pose click'을 먼저 하세요.")
            return
        gx, gy = gp
        gyaw = math.atan2(gy - sp[1], gx - sp[0])
        where = f" | 목표 찍은 점 ({gx:.2f}, {gy:.2f}), 직선 {math.hypot(gx - sp[0], gy - sp[1]):.2f} m"
    else:
        gx, gy, gyaw = sp[0] + a.dist * math.cos(sp[2]), sp[1] + a.dist * math.sin(sp[2]), sp[2]
        where = f" | 목표 {a.dist:.2f} m"
    run = new_run(a, "goal")
    head(a, run, where)
    if not a.no_reset:
        if not T.reset_pose(sp):
            print("     시작 안 함: AMCL 위치를 출발 위치로 다시 맞추지 못함(5번 보냄) — 'pose reset'으로 확인")
            return
        T.spin(0.5)
    T.call(T.cli_lc, ClearEntireCostmap.Request(), 3.0)
    T.call(T.cli_gc, ClearEntireCostmap.Request(), 3.0)
    bad = prepare(T, a, run)
    if not bad and not T.nav.wait_for_server(timeout_sec=5.0):
        bad.append("Nav2 목표 서버(/navigate_to_pose) 없음")
    if bad:
        stop_frames(T, a)
        print("     시작 안 함: " + " / ".join(bad))
        return
    goal = NavigateToPose.Goal()
    goal.pose = PoseStamped()
    goal.pose.header.frame_id = "map"
    goal.pose.header.stamp = T.n.get_clock().now().to_msg()
    goal.pose.pose.position.x, goal.pose.pose.position.y = gx, gy
    goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = math.sin(gyaw / 2), math.cos(gyaw / 2)
    ref = T.pose("odom")[:3]
    mark = nav_log_mark()
    T.begin()
    fb = {"dist": None, "rec": 0}

    def on_fb(f):
        fb["dist"] = f.feedback.distance_remaining
        fb["rec"] = max(fb["rec"], f.feedback.number_of_recoveries)

    gh, res_fut, status, reason = None, None, None, None
    try:
        fut = T.nav.send_goal_async(goal, feedback_callback=on_fb)
        end = time.time() + 5.0
        while not fut.done() and time.time() < end:
            rclpy.spin_once(T.n, timeout_sec=0.05)
        gh = fut.result() if fut.done() else None
        if gh is None or not gh.accepted:
            reason = "목표 거절/응답 없음"
        else:
            res_fut, last = gh.get_result_async(), 0.0
            while reason is None:
                rclpy.spin_once(T.n, timeout_sec=0.01)
                now = time.time()
                if res_fut.done():
                    status = res_fut.result().status
                    reason = {GoalStatus.STATUS_SUCCEEDED: "도착", GoalStatus.STATUS_ABORTED: "Nav2가 포기(aborted)",
                              GoalStatus.STATUS_CANCELED: "취소됨"}.get(status, f"상태 {status}")
                elif now - T.t0 > a.timeout:
                    reason = "시간 초과"
                elif now - last >= 0.05:
                    last = now
                    T.sample(ref, fb["dist"])
    except KeyboardInterrupt:
        reason = "중단: 사용자"
    finally:
        if gh is not None and gh.accepted and (res_fut is None or not res_fut.done()):
            cf = gh.cancel_goal_async()
            end = time.time() + 3.0
            while not cf.done() and time.time() < end:
                rclpy.spin_once(T.n, timeout_sec=0.05)
            T.zero(5)
    T.spin(0.5)
    t1 = time.time()
    stop_frames(T, a)
    path = sum(math.hypot(b["along"] - p["along"], b["lateral"] - p["lateral"]) for p, b in zip(T.rows, T.rows[1:]) if "along" in p and "along" in b)
    pm = T.pose("map")
    left = math.hypot(gx - pm[0], gy - pm[1]) if pm else float("nan")
    lat = max([abs(r["lateral"]) for r in T.rows if "lateral" in r] or [0.0])
    ex = {"travel_m": round(path, 3), "reason": reason, "status": status, "goal_left_m": round(left, 3), "max_lateral_m": round(lat, 3),
          "recoveries": fb["rec"], "goal_map": [round(gx, 3), round(gy, 3)]}
    size = tuple(a.size) if a.size else OBJ[a.obj]
    c = clearance(T.rows, a.at, size) if a.obj not in ("none", "multi") else None
    if c is not None:
        ex["clear_cm"] = round(c * 100, 1)
    fd = first_detection(T.fronts)
    if fd and a.obj != "none":
        ex["first_det_m"] = round(fd[1] - FRONT_X, 3)
    print(f"  → {reason} · {t1 - T.t0:.1f}초 · 경로 {path:.2f} m · 복구 {fb['rec']}회 · 목표까지 남은 거리 {left:.2f} m"
          f" · 최대 옆 이동 {lat:.2f} m", flush=True)
    if c is not None:
        print(f"     장애물 옆 최소 여유: " + ("겹침 → 밟거나 밀었을 가능성 큼(영상으로 확인)" if c < 0 else f"{c * 100:.0f} cm (겹침 없음)")
              + f" · 장애물 크기 근사 {size[0]:.2f}×{size[1]:.2f} m")
    finish(T, a, run, reason, ex, mark, t1)


def cmd_idle(T, a):
    run = new_run(a, "idle")
    head(a, run, f" | {a.secs:.0f}초" + (f" | {a.tag}" if a.tag else ""))
    bad = prepare(T, a, run, need_cm=False)
    if bad:
        print("     주의: " + " / ".join(bad))
    po = T.pose("odom")
    ref = po[:3] if po else (0.0, 0.0, 0.0)
    mark = nav_log_mark()
    T.begin()
    try:
        while time.time() - T.t0 < a.secs:
            T.spin(0.1)
            T.sample(ref)
    except KeyboardInterrupt:
        pass
    t1 = time.time()
    stop_frames(T, a)
    v = np.array([f[1] for f in T.fronts])
    det = v[v < 9.0]
    ex = {"n_front": int(len(v)), "det_rate": round(float(len(det)) / max(1, len(v)), 3)}
    if len(det):
        ex.update(front_med_m=round(float(np.median(det)), 3), front_p10_m=round(float(np.percentile(det, 10)), 3),
                  front_p90_m=round(float(np.percentile(det, 90)), 3), front_std_cm=round(float(det.std()) * 100, 1))
        print(f"  → 정면 거리 중앙값 {ex['front_med_m']:.3f} m (p10 {ex['front_p10_m']:.3f}, p90 {ex['front_p90_m']:.3f}, "
              f"표준편차 {ex['front_std_cm']:.1f} cm) · 잡힌 비율 {ex['det_rate'] * 100:.0f}% ({len(v)}프레임)")
    else:
        print(f"  → 정면에 잡힌 장애물 없음 ({len(v)}프레임)")
    finish(T, a, run, "기록", ex, mark, t1)


def start_path():
    return os.path.join(ROOT, "start_pose.json")


def load_start():
    p = start_path()
    return json.load(open(p, encoding="utf-8"))["map"] if os.path.exists(p) else None


def click_path():
    return os.path.join(ROOT, "goal_point.json")


def load_click():
    p = click_path()
    return json.load(open(p, encoding="utf-8"))["xy"] if os.path.exists(p) else None


def cmd_pose(T, a):
    if a.what == "click":
        got = []
        T.n.create_subscription(PointStamped, "/clicked_point", lambda m: got.append(m), 10)
        print("RViz에서 'Publish Point'를 누르고 목적지 바닥을 찍어 주세요 (2분 기다림)", flush=True)
        end = time.time() + 120
        while not got and time.time() < end:
            T.spin(0.1)
        if not got:
            print("찍은 점이 안 왔습니다")
            return
        m = got[-1]
        os.makedirs(ROOT, exist_ok=True)
        json.dump({"xy": [m.point.x, m.point.y], "frame": m.header.frame_id, "saved": time.strftime("%Y-%m-%d %H:%M:%S")},
                  open(click_path(), "w"), indent=1)
        print(f"목적지 저장: {m.header.frame_id} ({m.point.x:.3f}, {m.point.y:.3f})"
              + ("" if m.header.frame_id == "map" else " — 주의: map 좌표가 아님(RViz Fixed Frame을 map으로)"))
        return
    if a.what == "save":
        T.spin(1.0)
        pm = T.pose("map")
        if pm is None:
            print("map 위치(TF)가 없습니다 — Nav2·AMCL이 켜져 있는지, RViz에서 2D Pose Estimate를 했는지 확인")
            return
        os.makedirs(ROOT, exist_ok=True)
        json.dump({"map": [pm[0], pm[1], pm[2]], "saved": time.strftime("%Y-%m-%d %H:%M:%S")}, open(start_path(), "w"), indent=1)
        print(f"출발 위치 저장: map ({pm[0]:.3f}, {pm[1]:.3f}), 방향 {math.degrees(pm[2]):.1f}도")
    else:
        sp = load_start()
        if sp is None:
            print("저장된 출발 위치가 없습니다 — 'pose save' 먼저")
            return
        ok = T.reset_pose(sp)
        T.spin(2.0)
        pm = T.pose("map")
        msg = (f"지금 map ({pm[0]:.3f}, {pm[1]:.3f}), 방향 {math.degrees(pm[2]):.1f}도 / 저장값과 차이 "
               f"{math.hypot(pm[0] - sp[0], pm[1] - sp[1]) * 100:.0f} cm, {math.degrees(wrap(pm[2] - sp[2])):.1f}도") if pm else "map 위치 없음"
        print(("AMCL 위치를 출발 위치로 다시 맞춤 — " if ok else "주의: AMCL 구독자가 안 보임 — ") + msg)


def cmd_cond(T, a):
    ok = T.set_cond(a.cond, a.backend)
    T.spin(1.0)
    print(f"조건 {COND_KO[a.cond]}" + (" + 메인컴 추론" if a.backend == "remote" else "") + ": "
          + ("설정됨" if ok else "실패 — 인식 노드가 켜져 있는지 확인")
          + f" · 정면 거리 {T.front[0]:.2f} m · 기억 점 {len(T.mem)}개 · 감시 신호 {'켜짐' if T.wd[0] else '꺼짐'}")


def cmd_stop(T, a):
    r = subprocess.run(["pkill", "-INT", "-f", "drive_test.py (straight|goal|idle)"])
    res = T.call(T.cli_cancel, CancelGoal.Request(), 2.0)              # 빈 요청 = 모든 목표 취소
    T.zero(10)
    print("비상 정지: 진행 중 시험 " + ("중단 신호 보냄" if r.returncode == 0 else "없음")
          + " · Nav2 목표 " + (f"취소 {len(res.goals_canceling)}개" if res else "취소 응답 없음") + " · 정지 명령 보냄")


def cmd_note(a):
    d = session_dir(a)
    rows = read_results(d)
    hit = [r for r in rows if int(r["run"]) == a.run]
    if not hit:
        print(f"{a.session}에 {a.run}회차가 없습니다")
        return
    text = " ".join(a.text)
    hit[0]["note"] = (hit[0]["note"] + " / " if hit[0].get("note") else "") + text
    write_results(d, rows)
    print(f"{a.run}회차 메모: {hit[0]['note']}")


def cmd_table(a):
    rows = read_results(session_dir(a))
    if not rows:
        print("결과 없음")
        return
    print(f"{'회차':>3} {'시험':<10} {'조건':<12} {'물체':<8} {'결과':<14} {'이동m':>6} {'남은cm':>7} {'여유cm':>7} {'첫검출m':>7} "
          f"{'CM정지/감속':>10} {'감시':>4} {'지연p50':>7}  메모")
    for r in rows:
        cond = COND_KO.get(r["cond"], r["cond"]) + ("·메인컴" if r.get("backend") == "remote" else "")
        print(f"{r['run']:>3} {TEST_KO.get(r['test'], r['test'])[:10]:<10} {cond:<12} "
              f"{OBJ_KO.get(r['obj'], r['obj']):<8} {r['result'][:14]:<14} {r['travel_m']:>6} {r['gap_odom_cm']:>7} {r['clear_cm']:>7} "
              f"{r['first_det_m']:>7} {r['cm_stop'] + '/' + r['cm_slow']:>10} {r['wd_trips']:>4} {r.get('lag_p50_ms', ''):>7}  {r['note']}")


def main():
    ap = argparse.ArgumentParser(description="MentorPi 낮은 장애물 주행 시험 (자세한 설명은 파일 첫머리)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, obj=True):
        p.add_argument("--cond", choices=list(CONDS), required=True)
        p.add_argument("--backend", choices=["local", "remote"], default="local")
        if obj:
            p.add_argument("--obj", choices=list(OBJ), default="none")
            p.add_argument("--size", type=float, nargs=2, metavar=("L", "W"), help="장애물 진행 방향 길이·폭(m)")
        p.add_argument("--note", default="")
        p.add_argument("--no-frames", action="store_true")
        p.add_argument("--session", default=time.strftime("%Y%m%d"))
        p.add_argument("--odom-scale", type=float, default=ODOM_SCALE, help="실제 이동 / odom 이동")

    p = sub.add_parser("straight")
    common(p)
    p.add_argument("--at", type=float, default=1.5)
    p.add_argument("--speed", type=float, default=0.10)
    p.add_argument("--timeout", type=float, default=40.0)               # 1.5 m: 10cm/s 12초 + 감속 구역 3cm/s 8초 + 여유
    p.add_argument("--dry", action="store_true")
    p = sub.add_parser("goal")
    common(p)
    p.add_argument("--at", type=float, default=1.2)
    p.add_argument("--dist", type=float, default=2.5)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--no-reset", action="store_true")
    p.add_argument("--to-click", action="store_true", help="'pose click'으로 저장한 점을 목표로")
    p = sub.add_parser("idle")
    common(p, obj=False)
    p.add_argument("--secs", type=float, default=10.0)
    p.add_argument("--tag", default="")
    p = sub.add_parser("cond")
    p.add_argument("cond", choices=list(CONDS))
    p.add_argument("--backend", choices=["local", "remote"], default="local")
    p = sub.add_parser("pose")
    p.add_argument("what", choices=["save", "reset", "click"])
    p = sub.add_parser("note")
    p.add_argument("run", type=int)
    p.add_argument("text", nargs="+")
    p.add_argument("--session", default=time.strftime("%Y%m%d"))
    p = sub.add_parser("table")
    p.add_argument("--session", default=time.strftime("%Y%m%d"))
    sub.add_parser("stop")
    a = ap.parse_args()
    if a.cmd == "note":
        return cmd_note(a)
    if a.cmd == "table":
        return cmd_table(a)

    signal.signal(signal.SIGTERM, signal.default_int_handler)           # kill(TERM)도 Ctrl-C처럼 → finally에서 정지 명령
    try:                                                                # rclpy 신호 처리기를 쓰지 않음: Ctrl-C 때 컨텍스트가 먼저
        from rclpy.signals import SignalHandlerOptions                  # 꺼지면 마지막 정지 명령을 못 보냄
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    except (ImportError, TypeError):
        rclpy.init()
    T = Tester()
    T.scale = getattr(a, "odom_scale", ODOM_SCALE)
    try:
        T.spin(0.5)                                                     # TF·토픽 받기
        {"straight": cmd_straight, "goal": cmd_goal, "idle": cmd_idle, "cond": cmd_cond, "pose": cmd_pose, "stop": cmd_stop}[a.cmd](T, a)
    finally:
        T.n.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
