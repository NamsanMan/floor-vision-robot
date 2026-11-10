#!/usr/bin/env python3
"""CPU 나눠주기 실험 (로봇 호스트 pi, 2026-10-29). Nav2(nav_cam) + 바닥 분할 노드가 같이 도는 상태에서 실행.
조건마다 window초씩 재고 비교한다. 바꾸는 것은 실행 중 프로세스의 CPU 친화도(taskset)뿐이라 재시작하면 원래대로 돌아간다.
  A 그대로: 모든 프로세스가 코어 0~3 공유
  B 나눔  : 바닥 분할 노드 = 코어 2·3, 나머지 컨테이너 프로세스(Nav2·카메라·LiDAR·모터 등) = 코어 0·1
  C 나눔2 : 바닥 분할 노드 = 코어 1·2·3, 나머지 = 코어 0 (참고)
잴 것: 노드 지연 CSV(추론·전체 ms, 처리 fps), Nav2 로그의 '주기 못 맞춤' 경고 수, 코어별 사용률(/proc/stat)
usage: python3 cpu_split_test.py [window_s=30]"""
import csv
import glob
import os
import subprocess
import sys
import time

W = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
FS = "/home/pi/docker/tmp/floor_seg/logs"
NAV = "/home/pi/docker/tmp/nav2_cam/logs"


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def pids():
    node, others = [], []
    for line in sh("docker top MentorPi -eo pid,args").splitlines()[1:]:
        pid, _, args = line.strip().partition(" ")
        if "floor_seg_node.py" in args and "python3" in args:
            node.append(pid)
        elif args.strip():
            others.append(pid)
    return node, others


def pin(pidlist, cores):
    for p in pidlist:
        subprocess.run(f"sudo taskset -a -p -c {cores} {p}", shell=True, capture_output=True)


def cpu_times():
    out = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            f = line.split()
            vals = list(map(int, f[1:]))
            out[f[0]] = (sum(vals), vals[3] + vals[4])          # 전체, idle+iowait
    return out


def window(tag, node, others, plan):
    pin(node, plan[0]); pin(others, plan[1])
    time.sleep(3)                                              # 바뀐 배치가 자리 잡게
    log = max(glob.glob(f"{NAV}/nav_*.log"), key=os.path.getmtime)
    s0 = os.path.getsize(log)
    c0, t0 = cpu_times(), time.time()
    time.sleep(W)
    c1, t1 = cpu_times(), time.time()
    with open(log, "rb") as fh:                                # 재는 동안 새로 쌓인 로그만
        fh.seek(s0)
        seg = fh.read().decode("utf-8", errors="ignore")
    usage = {k: 100.0 * (1 - (c1[k][1] - c0[k][1]) / max(1, c1[k][0] - c0[k][0])) for k in sorted(c1)}
    rows = []
    f = max(glob.glob(f"{FS}/timing_*.csv"), key=os.path.getmtime)
    for r in csv.DictReader(open(f)):
        ts = int(r["stamp"]) * 1e-9
        if t0 <= ts <= t1:
            rows.append((float(r["inf_ms"]), float(r["total_ms"])))
    warn = sum(1 for line in seg.splitlines()
               if "missed its desired rate" in line or "Failed to meet update rate" in line or "Control loop missed" in line)
    rows.sort()
    inf = sorted(r[0] for r in rows)
    tot = sorted(r[1] for r in rows)
    p = lambda a, q: a[min(len(a) - 1, int(q * len(a)))] if a else float("nan")  # noqa: E731
    print(f"[{tag}] 노드 {plan[0]} / 나머지 {plan[1]}: 처리 {len(rows) / W:.1f} fps, 추론 평균 {sum(inf) / max(1, len(inf)):.0f} ms "
          f"(p95 {p(inf, 0.95):.0f}), 노드 전체 평균 {sum(tot) / max(1, len(tot)):.0f} ms (p95 {p(tot, 0.95):.0f}), "
          f"Nav2 주기 경고 {warn}회 | 코어 사용률 " + " ".join(f"{k[3:]}:{v:.0f}%" for k, v in usage.items()), flush=True)


node, others = pids()
print(f"바닥 분할 노드 pid {node}, 나머지 컨테이너 프로세스 {len(others)}개, 조건당 {W:.0f}초", flush=True)
if not node:
    sys.exit("바닥 분할 노드를 못 찾음")
try:
    if len(sys.argv) > 2:                                      # 한 조건만 (예: 노드 설정을 바꾼 뒤) python3 cpu_split_test.py 30 "D 스레드1"
        window(sys.argv[2], node, others, ("0-3", "0-3"))
    else:
        window("A 그대로", node, others, ("0-3", "0-3"))
        window("B 나눔 2+2", node, others, ("2,3", "0,1"))
        window("C 나눔 3+1", node, others, ("1-3", "0"))
finally:
    pin(node, "0-3"); pin(others, "0-3")
    print("친화도 원래대로(0~3) 되돌림", flush=True)
