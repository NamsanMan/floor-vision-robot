#!/usr/bin/env python3
"""2단계 확인 (로봇 호스트 pi, 2026-10-29): 카메라 연결 Nav2 + 조용한 실패 대책이 통하는지 정지 상태에서 확인.
  P0 인식 끔(60초): LiDAR 메시지 드롭·주기 경고 기준선 (공장 Nav2에 가까운 조건)
  P1 인식 켬(5Hz)+감시(120초): 인식 fps·추론·지연, 카메라 점 드롭·'갱신 안 됨' 경고, LiDAR 드롭, 코어 사용률, 코스트맵 표시율
  CM 시험: 감시 신호 꺼짐(인식 켬) → 1 mm/s 입력 / 인식 노드 끔 → 감시 신호 켜짐 → 1 mm/s 입력 (출력이 0이면 정지 성공)
변수에 명령을 담지 않고 subprocess 리스트로 실행(로봇 호스트 셸 zsh가 변수 속 명령을 안 쪼개는 문제, 2026-10-29).
usage: python3 session2.py"""
import csv
import glob
import os
import re
import subprocess
import time

H = "/home/pi/docker/tmp"
FS = ["docker", "exec", "-u", "ubuntu", "MentorPi", "bash", "/home/ubuntu/shared/floor_seg/run_floor_seg.sh"]
ROSPY = ["docker", "exec", "-u", "ubuntu", "MentorPi", "bash", "-c"]


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw).stdout


def ros(script):
    return run(ROSPY + [f"source /opt/ros/humble/setup.bash; {script}"])


def nav_log():
    return max(glob.glob(f"{H}/nav2_cam/logs/nav_*.log"), key=os.path.getmtime)


def cpu_times():
    out = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            f = line.split(); vals = list(map(int, f[1:]))
            out[f[0]] = (sum(vals), vals[3] + vals[4])
    return out


PATS = {"LiDAR 드롭": r"dropping message: frame .lidar_frame", "카메라 점 드롭": r"dropping message: frame .base_footprint",
        "카메라 층 갱신 안 됨": r"floor_obstacles/(memory|clearing) observation buffer has not been updated",
        "주기 못 맞춤": r"missed its desired rate|Failed to meet update rate", "TF 외삽 오류": r"Extrapolation Error"}


def window(tag, secs, with_node):
    log = nav_log(); s0 = os.path.getsize(log); c0 = cpu_times(); t0 = time.time()
    time.sleep(secs)
    c1 = cpu_times(); t1 = time.time()
    with open(log, "rb") as fh:
        fh.seek(s0); seg = fh.read().decode("utf-8", errors="ignore")
    cnt = {k: len(re.findall(p, seg)) for k, p in PATS.items()}
    use = " ".join(f"{k[3:]}:{100 * (1 - (c1[k][1] - c0[k][1]) / max(1, c1[k][0] - c0[k][0])):.0f}%" for k in sorted(c1))
    line = f"[{tag}] {secs:.0f}초 | " + ", ".join(f"{k} {v}회" for k, v in cnt.items()) + f" | 코어 {use}"
    if with_node:
        f = max(glob.glob(f"{H}/floor_seg/logs/timing_*.csv"), key=os.path.getmtime)
        rows = [r for r in csv.DictReader(open(f)) if t0 <= int(r["stamp"]) * 1e-9 <= t1]
        if rows:
            inf = sorted(float(r["inf_ms"]) for r in rows); lag = sorted(float(r["lag_ms"]) for r in rows)
            p95 = lambda a: a[min(len(a) - 1, int(0.95 * len(a)))]  # noqa: E731
            line += (f"\n      인식 {len(rows) / secs:.1f} fps, 추론 중앙값 {inf[len(inf) // 2]:.0f} ms (p95 {p95(inf):.0f}), "
                     f"영상 지연 중앙값 {lag[len(lag) // 2]:.0f} ms (p95 {p95(lag):.0f}, 최대 {lag[-1]:.0f})")
    print(line, flush=True)


print("Nav2 시작:", run(["bash", f"{H}/nav2_cam/nav_cam.sh", "start", "room_v2"]).strip(), flush=True)
for _ in range(30):                                         # 활성화 대기 (최대 90초)
    time.sleep(3)
    if "Managed nodes are active" in open(nav_log(), errors="ignore").read().split("lifecycle_manager_navigation")[-1]:
        break
time.sleep(5)
print(run(["bash", f"{H}/nav2_cam/nav_cam.sh", "status"]).strip().replace("\n", " / "), flush=True)
window("P0 인식 끔", 60, False)
print(run(FS + ["start"]).strip().replace("\n", " / "), flush=True)
time.sleep(20)
window("P1 인식 켬 5Hz + 감시", 120, True)
print(ros("python3 /home/ubuntu/shared/nav2_cam/costmap_check.py").strip(), flush=True)
print(ros("python3 /home/ubuntu/shared/nav2_cam/cm_test.py '감시 신호 꺼짐(인식 켬)' 2 0.001").strip(), flush=True)
run(FS + ["stop_node"]); time.sleep(1.5)
print(ros("python3 /home/ubuntu/shared/nav2_cam/cm_test.py '감시 신호 켜짐(인식 노드 끔)' 2 0.001").strip(), flush=True)
seg = open(nav_log(), errors="ignore").read().splitlines()[-400:]
print("Collision Monitor 로그:", [re.sub(r".*\]: ", "", l) for l in seg if "Robot to" in l][-4:], flush=True)
run(FS + ["stop"])
print("기본 구동으로:", run(["bash", f"{H}/nav2_cam/nav_cam.sh", "stop"]).strip(), flush=True)
