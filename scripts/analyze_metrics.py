#!/usr/bin/env python3
"""
analyze_metrics.py — inference_metrics.txt 파싱 및 시각화

사용법:
  python3 analyze_metrics.py                         # ~/Desktop/inference_metrics.txt 자동 로드
  python3 analyze_metrics.py --file /path/to/metrics.txt
  python3 analyze_metrics.py --log /path/to/ros2.log  # inference latency 포함

출력:
  ~/Desktop/figures/01_grasp_time.png
  ~/Desktop/figures/02_joint_rmse.png
  ~/Desktop/figures/03_exec_stability.png
  ~/Desktop/figures/04_tcp_scatter.png
  ~/Desktop/figures/05_latency_hist.png  (--log 옵션 시)
  ~/Desktop/figures/summary_table.txt
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 디스플레이 없는 환경(Jetson headless) 대응
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 한글 폰트 (없으면 영문 fallback) ────────────────────────────────────────
try:
    import matplotlib.font_manager as fm
    _korean = [f for f in fm.findSystemFonts() if "Noto" in f and "CJK" in f]
    if _korean:
        plt.rcParams["font.family"] = fm.FontProperties(fname=_korean[0]).get_name()
except Exception:
    pass

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {"movj": "#4C72B0", "servoj": "#DD8452", "unknown": "#888888"}

# ── 파서 ─────────────────────────────────────────────────────────────────────

def _float(s: str) -> float:
    try:
        return float(s.strip())
    except Exception:
        return float("nan")


def parse_metrics(filepath: Path) -> list[dict]:
    """inference_metrics.txt → list of dict (구버전/신버전 모두 지원)"""
    text = filepath.read_text(encoding="utf-8")
    blocks = re.split(r"={40,}", text)

    records = []
    for block in blocks:
        block = block.strip()
        if "[Inference Metrics]" not in block:
            continue

        r: dict = {}

        def _get(pattern, group=1, default=None):
            m = re.search(pattern, block)
            return m.group(group) if m else default

        # ── 기본 필드 (구버전 호환) ──────────────────────────────────────────
        r["control_mode"] = (_get(r"control_mode\s*:\s*(\S+)") or "unknown").lower()
        r["elapsed"]      = _float(_get(r"이동 시작 → ON 시간\s*:\s*([\d.]+)") or "nan")
        r["infer_count"]  = int(_float(_get(r"궤적 계산 호출 횟수\s*:\s*([\d]+)") or "0"))

        # ── Joint RMSE ───────────────────────────────────────────────────────
        m_rmse = re.search(r"Joint RMSE j1~j6\s*:\s*([\d. ]+)deg", block)
        if m_rmse:
            vals = [float(v) for v in m_rmse.group(1).split()]
            r["rmse_joints"] = vals if len(vals) == 6 else [float("nan")] * 6
        else:
            r["rmse_joints"] = [float("nan")] * 6

        m_avg = re.search(r"Joint RMSE avg\s*:\s*([\d.]+)", block)
        r["rmse_avg"] = float(m_avg.group(1)) if m_avg else float("nan")

        # ── TCP at suction ON ────────────────────────────────────────────────
        m_tcp = re.search(r"TCP@suction ON.*?X=([-\d.]+)\s+Y=([-\d.]+)\s+Z=([-\d.]+)", block)
        if m_tcp:
            r["tcp_x"] = float(m_tcp.group(1))
            r["tcp_y"] = float(m_tcp.group(2))
            r["tcp_z"] = float(m_tcp.group(3))
        else:
            r["tcp_x"] = r["tcp_y"] = r["tcp_z"] = float("nan")

        # ── Executor tick jitter ─────────────────────────────────────────────
        m_tick = re.search(r"Executor tick\s*:\s*mean=([\d.]+)ms\s+std=([\d.]+)ms\s+min=([\d.]+)\s+max=([\d.]+)", block)
        if m_tick:
            r["tick_mean"] = float(m_tick.group(1))
            r["tick_std"]  = float(m_tick.group(2))
            r["tick_min"]  = float(m_tick.group(3))
            r["tick_max"]  = float(m_tick.group(4))
        else:
            r["tick_mean"] = r["tick_std"] = r["tick_min"] = r["tick_max"] = float("nan")

        # ── Chunk arrival interval ───────────────────────────────────────────
        m_chunk = re.search(r"Chunk interval\s*:\s*mean=([\d.]+)ms\s+std=([\d.]+)ms", block)
        if m_chunk:
            r["chunk_mean"] = float(m_chunk.group(1))
            r["chunk_std"]  = float(m_chunk.group(2))
        else:
            r["chunk_mean"] = r["chunk_std"] = float("nan")

        # ── Cmd RTT ──────────────────────────────────────────────────────────
        m_rtt = re.search(r"Cmd RTT\s*:\s*mean=([\d.]+)ms\s+std=([\d.]+)ms", block)
        if m_rtt:
            r["rtt_mean"] = float(m_rtt.group(1))
            r["rtt_std"]  = float(m_rtt.group(2))
        else:
            r["rtt_mean"] = r["rtt_std"] = float("nan")

        records.append(r)

    return records


def parse_latency_from_log(log_path: Path) -> dict[str, list[float]]:
    """ros2 log 파일에서 [INFER_LATENCY] XXXms 파싱 → mode별 latency list"""
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    latencies = re.findall(r"latency=([\d.]+)ms", text)
    return {"all": [float(v) for v in latencies]}


# ── 그룹화 헬퍼 ──────────────────────────────────────────────────────────────

def group_by_mode(records: list[dict]) -> dict[str, list[dict]]:
    g: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        g[r["control_mode"]].append(r)
    return dict(g)


def _vals(records: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in records if not np.isnan(r[key])])


# ── Figure 1: grasp time + infer count bar ────────────────────────────────────

def plot_grasp_time(groups: dict, out: Path):
    modes = sorted(groups.keys())
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Task Completion Performance", fontsize=14, fontweight="bold")

    for ax, key, label, unit in zip(
        axes,
        ["elapsed", "infer_count"],
        ["Time to Grasp", "Inference Calls"],
        ["seconds", "count"],
    ):
        means, stds, colors, xlabels = [], [], [], []
        for mode in modes:
            v = _vals(groups[mode], key)
            if len(v) == 0:
                continue
            means.append(v.mean())
            stds.append(v.std())
            colors.append(COLORS.get(mode, "#888888"))
            xlabels.append(f"{mode.upper()}\n(n={len(v)})")

        bars = ax.bar(xlabels, means, yerr=stds, capsize=6,
                      color=colors, alpha=0.85, width=0.5,
                      error_kw={"elinewidth": 2, "ecolor": "black"})
        ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=9)
        ax.set_title(label, fontsize=12)
        ax.set_ylabel(unit)

        # 개별 점 scatter
        x_pos = {m: i for i, m in enumerate(xlabels)}
        for i, mode in enumerate(modes):
            if mode not in groups:
                continue
            v = _vals(groups[mode], key)
            jitter = np.random.uniform(-0.08, 0.08, len(v))
            ax.scatter(np.full(len(v), i) + jitter, v,
                       color="black", alpha=0.5, s=25, zorder=3)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


# ── Figure 2: Joint RMSE bar (per joint + avg) ───────────────────────────────

def plot_joint_rmse(groups: dict, out: Path):
    modes = sorted(groups.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Joint Angle Tracking RMSE (target_deg vs actual_deg)", fontsize=13, fontweight="bold")

    # 왼쪽: per-joint RMSE
    ax = axes[0]
    joint_labels = [f"J{i+1}" for i in range(6)]
    x = np.arange(6)
    width = 0.35 / max(len(modes), 1)
    for i, mode in enumerate(modes):
        joint_rmse = np.array([
            [r["rmse_joints"][j] for j in range(6)]
            for r in groups[mode] if not any(np.isnan(r["rmse_joints"]))
        ])
        if joint_rmse.size == 0:
            continue
        means = joint_rmse.mean(axis=0)
        stds  = joint_rmse.std(axis=0)
        offset = (i - len(modes) / 2 + 0.5) * width * 2
        bars = ax.bar(x + offset, means, width * 1.8,
                      yerr=stds, capsize=4, label=mode.upper(),
                      color=COLORS.get(mode), alpha=0.85,
                      error_kw={"elinewidth": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels(joint_labels)
    ax.set_ylabel("RMSE (deg)")
    ax.set_title("Per-Joint RMSE")
    ax.legend()

    # 오른쪽: avg RMSE bar
    ax = axes[1]
    xlabels, means, stds, colors = [], [], [], []
    for mode in modes:
        v = _vals(groups[mode], "rmse_avg")
        if len(v) == 0:
            continue
        xlabels.append(f"{mode.upper()}\n(n={len(v)})")
        means.append(v.mean())
        stds.append(v.std())
        colors.append(COLORS.get(mode, "#888888"))
    bars = ax.bar(xlabels, means, yerr=stds, capsize=6,
                  color=colors, alpha=0.85, width=0.4,
                  error_kw={"elinewidth": 2})
    ax.bar_label(bars, fmt="%.3f°", padding=4, fontsize=9)
    ax.set_ylabel("RMSE avg (deg)")
    ax.set_title("Average Joint RMSE")

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


# ── Figure 3: Execution stability box plot ───────────────────────────────────

def plot_exec_stability(groups: dict, out: Path):
    modes = sorted(groups.keys())
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Execution Stability", fontsize=14, fontweight="bold")

    metrics = [
        ("tick_mean", "tick_std", "Executor Tick Interval", "ms", 62.5),
        ("chunk_mean", "chunk_std", "Chunk Arrival Interval", "ms", 500.0),
        ("rtt_mean",  "rtt_std",  "Robot Cmd RTT", "ms", None),
    ]

    for ax, (mean_key, std_key, title, unit, ideal) in zip(axes, metrics):
        data, labels, colors = [], [], []
        for mode in modes:
            recs = [r for r in groups[mode]
                    if not np.isnan(r[mean_key]) and not np.isnan(r[std_key])]
            if not recs:
                continue
            # box plot용: mean ± std로 가상 분포 대신 실측 mean values scatter
            v = np.array([r[mean_key] for r in recs])
            data.append(v)
            labels.append(f"{mode.upper()}\n(n={len(v)})")
            colors.append(COLORS.get(mode, "#888888"))

        if not data:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(data, patch_artist=True, widths=0.4)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(2)

        ax.set_xticklabels(labels)
        ax.set_ylabel(unit)
        ax.set_title(title)

        if ideal is not None:
            ax.axhline(ideal, color="red", linestyle="--", linewidth=1.2, alpha=0.7,
                       label=f"Ideal: {ideal}ms")
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


# ── Figure 4: TCP scatter (pick 반복 정확도) ─────────────────────────────────

def plot_tcp_scatter(groups: dict, out: Path):
    modes = sorted(groups.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("End-Effector Position at Suction ON (mm)", fontsize=13, fontweight="bold")

    ax_xy, ax_z = axes

    for mode in modes:
        recs = [r for r in groups[mode]
                if not any(np.isnan([r["tcp_x"], r["tcp_y"], r["tcp_z"]]))]
        if not recs:
            continue
        xs = np.array([r["tcp_x"] for r in recs])
        ys = np.array([r["tcp_y"] for r in recs])
        zs = np.array([r["tcp_z"] for r in recs])
        color = COLORS.get(mode, "#888888")
        label = f"{mode.upper()} (n={len(recs)})"

        ax_xy.scatter(xs, ys, s=60, alpha=0.8, color=color, label=label, zorder=3)
        # 중심 마커
        ax_xy.scatter([xs.mean()], [ys.mean()], s=150, marker="*",
                      color=color, edgecolors="black", linewidths=0.8, zorder=5)

        # Z분포
        ax_z.errorbar(
            x=[list(COLORS.keys()).index(mode) if mode in COLORS else 0],
            y=[zs.mean()],
            yerr=[zs.std()],
            fmt="o", markersize=8, capsize=8,
            color=color, label=label,
        )
        jitter = np.random.uniform(-0.08, 0.08, len(zs))
        ax_z.scatter(
            np.full(len(zs), list(COLORS.keys()).index(mode) if mode in COLORS else 0) + jitter,
            zs, s=25, alpha=0.5, color=color, zorder=3
        )

    ax_xy.set_xlabel("X (mm)")
    ax_xy.set_ylabel("Y (mm)")
    ax_xy.set_title("XY Position (top view)")
    ax_xy.legend(fontsize=9)
    ax_xy.set_aspect("equal", adjustable="datalim")

    ax_z.set_ylabel("Z (mm)")
    ax_z.set_title("Z Position (pick height)")
    ax_z.set_xticks(range(len(modes)))
    ax_z.set_xticklabels([m.upper() for m in modes])
    ax_z.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


# ── Figure 5: Inference latency histogram ────────────────────────────────────

def plot_latency(latencies: list[float], out: Path):
    if not latencies:
        print("  [SKIP] latency 데이터 없음")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(latencies, bins=20, color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.axvline(np.mean(latencies), color="red", linestyle="--", linewidth=1.5,
               label=f"mean={np.mean(latencies):.0f}ms")
    ax.axvline(np.median(latencies), color="orange", linestyle="--", linewidth=1.5,
               label=f"median={np.median(latencies):.0f}ms")
    ax.set_xlabel("Inference Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("π0.5 Inference Latency Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary_table(groups: dict, out: Path):
    lines = []
    lines.append("=" * 70)
    lines.append("Experimental Results Summary")
    lines.append("=" * 70)

    header = f"{'Metric':<30} {'MovJ':>16} {'ServoJ':>16}"
    lines.append(header)
    lines.append("-" * 70)

    def fmt(v: np.ndarray) -> str:
        if len(v) == 0 or np.all(np.isnan(v)):
            return "   N/A"
        v = v[~np.isnan(v)]
        return f"{v.mean():>7.2f} ± {v.std():>5.2f}"

    metrics_defs = [
        ("Time to Grasp (s)",       "elapsed"),
        ("Inference Calls (#)",     "infer_count"),
        ("Joint RMSE avg (deg)",    "rmse_avg"),
        ("Executor tick mean (ms)", "tick_mean"),
        ("Executor tick std (ms)",  "tick_std"),
        ("Chunk interval mean (ms)","chunk_mean"),
        ("Chunk interval std (ms)", "chunk_std"),
        ("Cmd RTT mean (ms)",       "rtt_mean"),
        ("TCP Z at suction (mm)",   "tcp_z"),
    ]

    for label, key in metrics_defs:
        movj   = _vals(groups.get("movj",   []), key)
        servoj = _vals(groups.get("servoj", []), key)
        lines.append(f"  {label:<28} {fmt(movj):>16} {fmt(servoj):>16}")

    lines.append("=" * 70)

    n_movj   = len(groups.get("movj",   []))
    n_servoj = len(groups.get("servoj", []))
    lines.append(f"  Trials: MovJ={n_movj}, ServoJ={n_servoj}")
    lines.append("=" * 70)

    table = "\n".join(lines)
    print(table)
    out.write_text(table, encoding="utf-8")
    print(f"  → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="inference_metrics.txt 시각화")
    parser.add_argument(
        "--file", default=str(Path.home() / "Desktop" / "inference_metrics.txt"),
        help="metrics 파일 경로 (기본: ~/Desktop/inference_metrics.txt)"
    )
    parser.add_argument(
        "--log", default=None,
        help="ros2 log 파일 경로 (inference latency 파싱용, optional)"
    )
    parser.add_argument(
        "--out", default=str(Path.home() / "Desktop" / "figures"),
        help="출력 디렉토리 (기본: ~/Desktop/figures)"
    )
    args = parser.parse_args()

    metrics_path = Path(args.file)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_path.exists():
        print(f"[ERROR] 파일 없음: {metrics_path}")
        sys.exit(1)

    print(f"파싱 중: {metrics_path}")
    records = parse_metrics(metrics_path)
    print(f"  총 {len(records)}개 에피소드 로드")

    if not records:
        print("[ERROR] 파싱된 데이터 없음")
        sys.exit(1)

    groups = group_by_mode(records)
    for mode, recs in groups.items():
        print(f"  {mode}: {len(recs)}회")

    print("\n그래프 생성 중...")
    plot_grasp_time(groups,     out_dir / "01_grasp_time.png")
    plot_joint_rmse(groups,     out_dir / "02_joint_rmse.png")
    plot_exec_stability(groups, out_dir / "03_exec_stability.png")
    plot_tcp_scatter(groups,    out_dir / "04_tcp_scatter.png")

    if args.log:
        log_path = Path(args.log)
        if log_path.exists():
            lat = parse_latency_from_log(log_path)
            plot_latency(lat["all"], out_dir / "05_latency_hist.png")
        else:
            print(f"  [WARN] log 파일 없음: {log_path}")

    print("\n요약 테이블:")
    print_summary_table(groups, out_dir / "summary_table.txt")
    print(f"\n완료 — 그래프 저장 위치: {out_dir}/")


if __name__ == "__main__":
    main()
