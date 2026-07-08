#!/usr/bin/env python3
"""
SmolVLA vs π0.5(e6) 추론 자원 비교 시각화 (시스템 python3 + matplotlib).

각 모델을 '단독·동일 파라미터'로 측정한 infer_resource_result.json 2개를 읽어,
추론 1회당 일량(GPU util·s, CPU core·s) + latency + RAM% 를 나란히 막대로 그린다.
raw CPU% 는 추론 시간(창 길이)이 다르면 희석돼 착시를 주므로, 일량(=usage% × duration)
으로 정규화해 비교한다. per_run 배열에서 run별 일량을 계산해 평균/표준편차(에러바)를 낸다.

  사용:
    /usr/bin/python3 compare_smolvla_vs_pi05.py
    /usr/bin/python3 compare_smolvla_vs_pi05.py --smolvla A.json --pi05 B.json --out cmp.png
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    d = json.load(open(path))
    nc = d["meta"]["n_cpu"]
    pr = d["per_run"]
    lat = np.array([r["latency_ms"] for r in pr]) / 1000.0   # s
    gpu = np.array([r["gpu"] for r in pr])                    # %
    cpu = np.array([r["cpu"] for r in pr])                    # %
    ram = np.array([r["ram"] for r in pr])                    # %
    gpu_work = gpu / 100.0 * lat            # GPU util-seconds / inference
    cpu_work = cpu / 100.0 * nc * lat       # CPU core-seconds / inference
    return {
        "gpu_work": (gpu_work.mean(), gpu_work.std()),
        "cpu_work": (cpu_work.mean(), cpu_work.std()),
        "lat":      (lat.mean() * 1000, lat.std() * 1000),
        "ram":      (ram.mean(), ram.std()),
        "gpu_pct":  (gpu.mean(), gpu.std()),
        "cpu_pct":  (cpu.mean(), cpu.std()),
        "runs": d["meta"].get("runs"),
    }


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(__file__)
    ap.add_argument("--smolvla", default=os.path.expanduser("~/SmolVLA/SmolVLA-INFERENCE/scripts/infer_resource_result.json"))
    ap.add_argument("--pi05", default=os.path.join(here, "infer_resource_result.json"))
    ap.add_argument("--out", default=os.path.join(here, "compare_smolvla_vs_pi05.png"))
    ap.add_argument("--smolvla-name", default="SmolVLA exp5\n(~0.5B, torch in-proc)")
    ap.add_argument("--pi05-name", default="π0.5 e6 v23\n(~3B, openpi server)")
    args = ap.parse_args()

    s = load(args.smolvla)
    p = load(args.pi05)
    names = [args.smolvla_name, args.pi05_name]
    colors = ["#5FB37A", "#4C9BE8"]  # SmolVLA=green, π0.5=blue

    panels = [
        ("GPU work / inference (util·s)", "gpu_work", "{:.2f}"),
        ("CPU work / inference (core·s)", "cpu_work", "{:.2f}"),
        ("Latency (ms)", "lat", "{:.0f}"),
        ("RAM (%)", "ram", "{:.1f}"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
    for ax, (title, key, fmt) in zip(axes, panels):
        vals = [s[key][0], p[key][0]]
        errs = [s[key][1], p[key][1]]
        bars = ax.bar(names, vals, yerr=errs, capsize=7, color=colors,
                      edgecolor="black", linewidth=0.8, alpha=0.9)
        for b, v, e in zip(bars, vals, errs):
            ax.text(b.get_x() + b.get_width() / 2, v + e + max(vals) * 0.02,
                    fmt.format(v), ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, max(vals) * 1.28 + max(errs) + 1e-6)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8)
        if vals[0] > 0:
            ax.text(0.5, 0.95, f"π0.5/SmolVLA = {vals[1] / vals[0]:.2f}×",
                    transform=ax.transAxes, ha="center", fontsize=9, color="#444")

    fig.suptitle(
        f"SmolVLA (exp5) vs π0.5 (e6 v23) — per-inference resource cost   "
        f"|   Jetson AGX Orin, {p['runs']} runs each, measured standalone",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.text(0.5, 0.005,
             "work = mean(usage% x duration).  Each model measured standalone in its real deployment "
             "(SmolVLA in-process / π0.5 openpi server).  RAM is a system-wide level value.",
             ha="center", fontsize=8, color="#666")
    fig.savefig(args.out, dpi=150)
    print(f"saved -> {args.out}")

    # 콘솔 표
    print(f"\n{'metric':<28}{'SmolVLA exp5':>14}{'π0.5 e6 v23':>14}{'ratio':>8}")
    print("-" * 64)
    for title, key, _ in panels:
        a, b = s[key][0], p[key][0]
        print(f"{title:<28}{a:>14.2f}{b:>14.2f}{(b / a if a else 0):>8.2f}")
    print("-" * 64)
    print(f"[참고 raw %] GPU% {s['gpu_pct'][0]:.1f} vs {p['gpu_pct'][0]:.1f}  |  "
          f"CPU% {s['cpu_pct'][0]:.1f} vs {p['cpu_pct'][0]:.1f}  ← CPU%만 역전(창-길이 착시)")


if __name__ == "__main__":
    main()
