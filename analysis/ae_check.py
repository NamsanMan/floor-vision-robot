#!/usr/bin/env python3
"""카메라 자동 노출 확인 분석 (메인컴, 2026-10-31): brightness_log.py가 남긴 프레임별 밝기에서 불 끔·켬 순간을 찾아,
그 뒤 몇 초 동안 밝기가 어떻게 변하는지 본다.
  고정 노출: 끈 직후 값 ≈ 몇 초 뒤 값(계단), 켠 직후 과노출 없음
  자동 노출: 끈 뒤 밝기가 다시 올라감, 켠 직후 과노출(포화 화소↑)됐다가 내려옴, 어두울 때 초당 장수가 줄기도 함
usage: python ae_check.py ae_test.csv [--out ae_check.png]"""
import argparse
import csv
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--ratio", type=float, default=3.0, help="0.5초 안에 밝기가 이 배수 넘게 바뀌면 불 끔/켬으로 봄")
    ap.add_argument("--out", default="ae_check.png")
    a = ap.parse_args()
    R = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    t = np.array([float(r["stamp"]) for r in R])
    t -= t[0]
    m = np.array([float(r["mean"]) for r in R])
    sat = np.array([float(r["sat_frac"]) for r in R])
    lm = np.log(np.maximum(m, 0.5))
    ev, i = [], 1
    while i < len(t):                                       # 0.5초 앞 프레임과 비교해 큰 변화 = 전환
        j = np.searchsorted(t, t[i] - 0.5)
        d = lm[i] - lm[j:i].min() if lm[i] > lm[j:i].min() + np.log(a.ratio) else (lm[i] - lm[j:i].max() if lm[i] < lm[j:i].max() - np.log(a.ratio) else 0)
        if d != 0 and (not ev or t[i] - ev[-1][0] > 2.0):
            ev.append((t[i], "on" if d > 0 else "off", i))
        i += 1
    bounds = [e[0] for e in ev] + [t[-1] + 1e-6]
    res = {"frames": len(t), "duration_s": round(float(t[-1]), 1),
           "fps_overall": round((len(t) - 1) / max(1e-6, t[-1]), 2), "events": []}
    for k, (te, kind, idx) in enumerate(ev):
        pre = m[(t >= te - 1.5) & (t < te - 0.2)]
        nxt = bounds[k + 1]
        seg = (t >= te) & (t < nxt)
        ts, ms_, ss = t[seg] - te, m[seg], sat[seg]
        settle = ms_[ts >= max(0.0, (nxt - te) - 2.5)]
        at = {f"+{d:g}s": round(float(ms_[np.searchsorted(ts, d)]), 2) for d in (0, 0.25, 0.5, 1, 2, 3, 5, 8) if d < ts[-1]}
        dt = np.diff(t[seg])
        e = {"t_s": round(float(te), 2), "kind": kind, "before": round(float(np.median(pre)), 2) if len(pre) else None,
             "first_after": round(float(ms_[0]), 2), "peak_or_min_in_first_1s": round(float((ms_[ts < 1].max() if kind == "on" else ms_[ts < 1].min())), 2),
             "settled": round(float(np.median(settle)), 2), "at": at, "max_saturated_frac_first_1s": round(float(ss[ts < 1].max()), 4),
             "fps_in_phase": round(float(1 / np.median(dt)), 1) if len(dt) else None, "phase_len_s": round(float(nxt - te), 1)}
        res["events"].append(e)
    js = os.path.splitext(a.out)[0] + ".json"
    json.dump(res, open(js, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    ax.semilogy(t, np.maximum(m, 0.5), color="black", lw=1.2, label="mean brightness (0-255)")
    ax.semilogy(t, np.maximum([float(r["bottom_half"]) for r in R], 0.5), color="tab:green", lw=0.8, alpha=0.7, label="bottom half (floor)")
    for k, (te, kind, idx) in enumerate(ev):
        if kind == "off":
            ax.axvspan(te, bounds[k + 1], color="gray", alpha=0.18, lw=0)
    ax.set_ylabel("brightness (log scale)")
    ax.set_ylim(0.5, 300)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("Camera auto-exposure check: room lights switched off (gray) and on, robot stationary", fontsize=11)
    dtt = np.diff(t)
    ax2.plot(t[1:], 1 / np.maximum(dtt, 1e-3), color="tab:blue", lw=0.8)
    ax2.set_ylabel("fps")
    ax2.set_ylim(0, 20)
    ax2.set_xlabel("time (s)")
    ax2.grid(alpha=0.3)
    fig.savefig(a.out, dpi=110, bbox_inches="tight")
    print(f"프레임 {len(t)}장, {t[-1]:.0f}초, 평균 {res['fps_overall']} fps, 전환 {len(ev)}번 → {a.out}, {js}")
    for e in res["events"]:
        name = "불 끔" if e["kind"] == "off" else "불 켬"
        print(f"  {e['t_s']:6.1f}초 {name}: 직전 {e['before']} → 직후 {e['first_after']} → 1초 안 {'최대' if e['kind'] == 'on' else '최소'} "
              f"{e['peak_or_min_in_first_1s']} → 안정 {e['settled']} | 시점별 {e['at']} | 처음 1초 포화 화소 최대 {e['max_saturated_frac_first_1s'] * 100:.1f}%"
              f" | 이 구간 {e['fps_in_phase']} fps, {e['phase_len_s']}초")


if __name__ == "__main__":
    main()
