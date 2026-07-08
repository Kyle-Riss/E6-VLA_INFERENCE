#!/usr/bin/env python3
"""forward_cpu_random_e6.py JSON -> 순수 부하(net = forward - idle) 그래프.
시스템 python3 (matplotlib) 로 실행. (한글 글리프 없어서 라벨은 ASCII)"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    here = os.path.dirname(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(here, "forward_cpu_random_e6.json"))
    ap.add_argument("--out", default=os.path.join(here, "forward_cpu_random_e6.png"))
    args = ap.parse_args()

    with open(args.json) as f:
        d = json.load(f)
    runs, meta, mean, std, idle = d["per_run"], d["meta"], d["means"], d["stds"], d["idle"]
    idx = [r["idx"] for r in runs]
    gpu_net = [r["gpu_net"] for r in runs]
    cpu_net = [r["cpu_net"] for r in runs]
    labels = [r["folder"] for r in runs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5),
                                   gridspec_kw={"width_ratios": [2.1, 1.15]})

    # ── 왼쪽: forward 회차별 net GPU(bar) + net CPU(line) ──
    bars = ax1.bar(idx, gpu_net, color="#4C9BE8", edgecolor="black", linewidth=0.8,
                   alpha=0.9, label="GPU net (per forward)")
    ax1.axhline(mean["gpu_net"], color="#1F5FA0", ls="--", lw=1.5,
                label=f"GPU net mean {mean['gpu_net']:.1f}%")
    for b, v in zip(bars, gpu_net):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.0f}",
                 ha="center", va="bottom", fontsize=8)
    ax1.set_ylim(0, max(gpu_net) * 1.18)
    ax1.set_xticks(idx)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_xlabel("forward #  (episode folder)")
    ax1.set_ylabel("GPU net load (%)  [forward - idle]")
    ax1.set_title(f"pi0.5 forward PURE load - {meta['runs']} random episodes (no ROS2)")
    ax1.grid(axis="y", alpha=0.3)

    axb = ax1.twinx()
    axb.plot(idx, cpu_net, "-o", color="#E8A23C", lw=1.8, ms=5, label="CPU net (per forward)")
    axb.set_ylabel("CPU net load (%)", color="#B8731B")
    axb.tick_params(axis="y", labelcolor="#B8731B")
    axb.set_ylim(0, max(max(cpu_net) * 2.2, 10))
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = axb.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper right", fontsize=8)

    # ── 오른쪽: idle(baseline) + net(pure forward) 분해 = absolute ──
    keys = ["gpu", "cpu", "ram"]
    klbl = ["GPU", "CPU", "RAM"]
    idle_v = [idle[k] for k in keys]
    net_v = [mean[k + "_net"] for k in keys]
    net_e = [std[k + "_net"] for k in keys]
    base = ax2.bar(klbl, idle_v, color="#CCCCCC", edgecolor="black", linewidth=0.8,
                   label="idle baseline (model loaded, no infer)")
    top = ax2.bar(klbl, net_v, bottom=idle_v, yerr=net_e, capsize=6,
                  color=["#4C9BE8", "#E8A23C", "#5FB37A"], edgecolor="black",
                  linewidth=0.8, label="forward PURE load (net)")
    for k, ib, nb, iv, nv in zip(keys, base, top, idle_v, net_v):
        # net 값 (윗 segment 중앙)
        ax2.text(nb.get_x() + nb.get_width() / 2, iv + nv / 2,
                 f"net\n{nv:.1f}", ha="center", va="center", fontsize=8, fontweight="bold")
        # total(absolute) 값 (막대 위)
        ax2.text(nb.get_x() + nb.get_width() / 2, iv + nv + 2.0,
                 f"={iv + nv:.0f}%", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("usage (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title("decompose: absolute = idle + net")
    ax2.legend(loc="upper center", fontsize=7.5)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"model={meta['model']}  config={meta['config']}  device={meta.get('device','?')}"
                 f"  |  net = forward - idle  (idle: GPU {idle['gpu']:.1f}%, CPU {idle['cpu']:.1f}%, "
                 f"RAM {idle['ram']:.1f}%)  |  latency {mean['latency_ms']:.0f}ms",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[plot] saved: {args.out}")


if __name__ == "__main__":
    main()
