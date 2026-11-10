#!/bin/bash
# 미세조정 DS-Net Pi 지연 (2026-10-29): 20 warm-up + 50회, 배치 1. 참고로 10/07 CamVid 모델(같은 구조)도 같은 세션에서
V=/home/ubuntu/bench_venv/bin/python; B=/home/ubuntu/shared/bench
for m in ptq/A_240x320_int8_percentile_noup_gelu.onnx ft/ft_240x320_fp32.onnx ft/ft_240x320_int8.onnx ft/ft_240x320_int8_gelu.onnx; do
  for t in 2 4; do echo "$m threads=$t $($V $B/bench_ort.py $B/$m $t 2>/dev/null | tail -1)"; done
done
