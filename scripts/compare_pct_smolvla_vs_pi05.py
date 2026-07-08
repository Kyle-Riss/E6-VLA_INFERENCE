#!/usr/bin/env python3
"""
SmolVLA vs π0.5(e6) — CPU/GPU/RAM(%) 그룹 막대 비교 (시스템 python3 + matplotlib).
각 모델 단독·동일 파라미터 측정 JSON 2개에서 means/stds 를 읽어 그룹 막대로 그린다.
(raw % 그대로. CPU% 는 추론 시간 차이로 희석됨 — 일량 비교는 compare_smolvla_vs_pi05.py)
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(__file__)
    ap.add_argument("--smolvla", default=os.path.expanduser("~/SmolVLA/SmolVLA-INFERENCE/scripts/infer_resource_result.json"))
    ap.add_argument("--pi05", default=os.path.join(here, "infer_resource_result.json"))
    ap.add_argument("--out", default=os.path.join(here, "compare_pct_smolvla_vs_pi05.png"))
    args = ap.parse_args()

    s = json.load(open(args.smolvla))
    p = json.load(open(args.pi05))

    metrics = ["gpu", "cpu", "ram"]
    labels = ["GPU", "CPU", "RAM"]
    s_vals = [s["means"][m] for m in metrics]
    s_err = [s["stds"][m] for m in metrics]
    p_vals = [p["means"][m] for m in metrics]
    p_err = [p["stds"][m] for m in metrics]

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5.2))
    b1 = ax.bar(x - w / 2, s_vals, w, yerr=s_err, capsize=6, label="SmolVLA exp5 (~0.5B)",
                color="#5FB37A", edgecolor="black", linewidth=0.8, alpha=0.9)
    b2 = ax.bar(x + w / 2, p_vals, w, yerr=p_err, capsize=6, label="π0.5 e6 v23 (~3B)",
                color="#4C9BE8", edgecolor="black", linewidth=0.8, alpha=0.9)

    for bars, vals, errs in ((b1, s_vals, s_err), (b2, p_vals, p_err)):
        for bar, v, e in zip(bars, vals, errs):
            ax.text(bar.get_x() + bar.get_width() / 2, v + e + 1.2,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Usage (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, loc="upper right")

    s_lat = s.get("latency_ms", {}).get("mean", 0)
    p_lat = p.get("latency_ms", {}).get("mean", 0)
    ax.set_title(
        "SmolVLA (exp5) vs π0.5 (e6 v23) — inference CPU/GPU/RAM usage\n"
        f"Jetson AGX Orin | {p['meta'].get('runs')} runs each, standalone | "
        f"latency {s_lat:.0f} vs {p_lat:.0f} ms",
        fontsize=11,
    )
    fig.text(0.5, 0.005,
             "Note: CPU% is diluted by inference duration (π0.5 1.6x longer) -- see work-normalized chart for fair CPU comparison.",
             ha="center", fontsize=8, color="#666")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(args.out, dpi=150)
    print(f"saved -> {args.out}")
    print(f"\n{'':6}{'GPU%':>8}{'CPU%':>8}{'RAM%':>8}{'lat(ms)':>10}")
    print(f"{'Smol':6}{s_vals[0]:>8.1f}{s_vals[1]:>8.1f}{s_vals[2]:>8.1f}{s_lat:>10.0f}")
    print(f"{'π0.5':6}{p_vals[0]:>8.1f}{p_vals[1]:>8.1f}{p_vals[2]:>8.1f}{p_lat:>10.0f}")


if __name__ == "__main__":
    main()
