#!/usr/bin/env python3
"""오프로딩 추론 — 메인컴 (2026-10-30). 로봇 인식 노드가 backend=remote일 때 보내는 프레임을 같은 INT8 모델·같은 ONNX Runtime
버전(1.23.2, conda env mot)으로 추론해 바닥-비바닥 로짓 차(float16, 60x80)를 돌려준다. 바뀌는 건 '어디서 추론하느냐'뿐.
메인컴 방화벽을 건드리지 않으려고 메인컴이 로봇(기본 mentorpi.local:5601, 환경변수 ROBOT_HOST)에 접속해 두는 방식. 끊기면 1초마다 다시 접속.
  요청: 'FSEG', 번호(uint32), JPEG 길이(uint32) + JPEG(240x320 BGR)
  응답: 'FSEG', 번호, 메인컴 처리 ms(float32), 높이, 너비(uint16), 데이터 길이(uint32) + float16 로짓 차
usage: python offload_pc.py [--model models\\ft_240x320_int8_gelu.onnx] [--host mentorpi.local]
       [--threads 4] [--cuda]   (--cuda는 FP32 모델 + onnxruntime-gpu가 있는 환경에서만)"""
import argparse
import os
import socket
import struct
import time

import cv2
import numpy as np
import onnxruntime as ort

REQ = struct.Struct("<4sII")
REP = struct.Struct("<4sIfHHI")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def recv_exact(c, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = c.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("로봇이 연결을 끊음")
        buf += chunk
    return bytes(buf)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(here, "models", "ft_240x320_int8_gelu.onnx"))
    ap.add_argument("--host", default=os.environ.get("ROBOT_HOST", "mentorpi.local"))
    ap.add_argument("--port", type=int, default=5601)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cuda", action="store_true")
    a = ap.parse_args()
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if a.cuda else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(a.model, so, providers=prov)
    inp = sess.get_inputs()[0].name
    sess.run(None, {inp: np.zeros((1, 3, 240, 320), np.float32)})              # 첫 실행 준비
    print(f"모델 {os.path.basename(a.model)} · ORT {ort.__version__} {sess.get_providers()[0]} · 스레드 {a.threads}", flush=True)
    while True:
        try:
            c = socket.create_connection((a.host, a.port), timeout=3)
        except OSError:
            time.sleep(1.0)
            continue
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        c.settimeout(None)
        print(f"로봇 {a.host}:{a.port} 연결됨", flush=True)
        n, ms, t_last = 0, [], time.time()
        try:
            while True:
                magic, rid, size = REQ.unpack(recv_exact(c, REQ.size))
                if magic != b"FSEG":
                    raise ConnectionError("요청 형식 오류")
                jpg = recv_exact(c, size)
                t0 = time.perf_counter()
                bgr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                x = ((cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
                lg = sess.run(None, {inp: x})[0][0]
                diff = (lg[1] - lg[0]).astype(np.float16)
                server_ms = (time.perf_counter() - t0) * 1e3
                body = diff.tobytes()
                c.sendall(REP.pack(b"FSEG", rid, server_ms, diff.shape[0], diff.shape[1], len(body)) + body)
                n += 1
                ms.append(server_ms)
                if time.time() - t_last > 10:
                    print(f"  {n}프레임 · 메인컴 처리 중앙값 {np.median(ms):.1f} ms (최대 {max(ms):.1f}) · 받은 JPEG {size / 1024:.1f} KB", flush=True)
                    ms, t_last = [], time.time()
        except (OSError, ConnectionError, struct.error) as e:
            print(f"연결 끊김({type(e).__name__}: {e}) — 1초 뒤 다시 접속", flush=True)
            c.close()
            time.sleep(1.0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("종료", flush=True)
