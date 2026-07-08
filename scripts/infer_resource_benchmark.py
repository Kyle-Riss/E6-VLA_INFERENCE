#!/usr/bin/env python3
"""
π0.5 (e6_vla) 추론 자원 사용량 벤치마크 (GPU / CPU / RAM %).

inference_bridge_node.py 와 "동일한 obs 스키마"로, 이미 실행 중인 openpi 정책 서버
(serve_policy.py, 기본 ws://127.0.0.1:8000) 에 더미 obs 를 N회(기본 10) 보내면서
매 추론 구간의 *시스템 전체* 자원 사용량을 샘플링한다.

  - GPU : tegrastats GR3D_FREQ (%)         ← 서버 프로세스 GPU 부하 포함(시스템 전체라 OK)
  - CPU : psutil.cpu_percent (%)           ← 시스템 전체 0~100%
  - RAM : psutil.virtual_memory().percent  ← 시스템 전체 0~100% (Orin 통합메모리)

각 추론 구간 [t0, t1] (= policy.infer 호출~반환) 안에 들어온 샘플을 평균 → 추론 1회 값.
그 값을 N회에 대해 평균/표준편차 내어 JSON 으로 저장한다(그래프는 plot 스크립트가 그림).

  주의:
    * 모델은 *서버 프로세스*에 있으므로 측정 전에 run_server_v23.sh 가 떠 있어야 한다.
      (모델 로딩은 서버 startup 때 끝나므로 측정에서 자동 제외됨.)
    * 첫 추론 cold-start(cudnn/flow-matching 초기화)는 --warmup 만큼 버린다.
    * 측정 창 [t0,t1] 은 websocket 왕복+직렬화까지 포함하는 "추론 요청 1회"이다
      (순수 GPU forward 가 아님 — 실제 배포 경로와 동일).

  smolvla 방법 B(infer_resource_benchmark.py) 와 1:1 대응. 차이는 in-process 모델 로드 대신
  websocket 클라이언트로 서버에 쏘는 것뿐. CUDA/torch 불필요(클라이언트라서).
"""

import argparse
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from openpi_client.websocket_client_policy import WebsocketClientPolicy  # type: ignore

# ── 노드와 동일 ──────────────────────────────────────────────────────────────
STATE_DIM = 7
# v16/v23 데이터셋 기준 프롬프트(방향형). per_frame 이라 단계별로 바뀌지만 자원량엔 영향 미미.
DEFAULT_PROMPT = "pick up the orange box from the left side and place it on the right side"


# ── tegrastats(GPU) 리더 (smolvla 와 동일) ───────────────────────────────────
class TegraReader(threading.Thread):
    """tegrastats 를 백그라운드로 돌리며 최신 GR3D_FREQ(%) 를 self.gpu 에 갱신."""

    _GR3D = re.compile(r"GR3D_FREQ\s+(\d+)%")

    def __init__(self, interval_ms: int = 50):
        super().__init__(daemon=True)
        self.gpu = 0.0
        self._stop = False
        self.proc = subprocess.Popen(
            ["tegrastats", "--interval", str(interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )

    def run(self):
        for line in self.proc.stdout:
            if self._stop:
                break
            m = self._GR3D.search(line)
            if m:
                self.gpu = float(m.group(1))

    def stop(self):
        self._stop = True
        try:
            self.proc.terminate()
        except Exception:
            pass


# ── 자원 샘플러 (GPU/CPU/RAM 을 일정 주기로 기록) (smolvla 와 동일) ──────────
class Sampler(threading.Thread):
    def __init__(self, tegra: TegraReader, interval: float = 0.025):
        super().__init__(daemon=True)
        self.tegra = tegra
        self.interval = interval
        self.samples = []  # (t, gpu, cpu, ram)
        self.lock = threading.Lock()
        self._stop = False

    def run(self):
        psutil.cpu_percent(None)  # cpu_percent prime (첫 호출은 0 반환)
        time.sleep(self.interval)
        while not self._stop:
            t = time.monotonic()
            cpu = psutil.cpu_percent(None)
            ram = psutil.virtual_memory().percent
            gpu = self.tegra.gpu
            with self.lock:
                self.samples.append((t, gpu, cpu, ram))
            time.sleep(self.interval)

    def stop(self):
        self._stop = True

    def window_mean(self, t0: float, t1: float):
        """[t0, t1] 안의 샘플 평균. 샘플이 없으면 t1 에 가장 가까운 샘플 1개."""
        with self.lock:
            arr = list(self.samples)
        inside = [s for s in arr if t0 <= s[0] <= t1]
        if not inside:
            if not arr:
                return None
            inside = [min(arr, key=lambda s: abs(s[0] - t1))]
        a = np.array([[s[1], s[2], s[3]] for s in inside], dtype=np.float64)
        return a.mean(axis=0), len(inside)  # [gpu,cpu,ram], n_samples


# ── 노드 _maybe_infer 의 obs 스키마와 동일 (더미 입력) ───────────────────────
def make_obs(img_size: int, prompt: str) -> dict:
    hik = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
    zed = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
    state = np.random.randn(STATE_DIM).astype(np.float32)
    return {
        "observation/exterior_image_1_left": hik,
        "observation/exterior_image_2_left": zed,
        "observation/state": state,
        "prompt": prompt,
    }


def jetson_model() -> str:
    try:
        return Path("/proc/device-tree/model").read_text(errors="ignore").strip("\x00").strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--img-size", type=int, default=224, help="더미 이미지 한 변(px). 노드 obs 기준 224")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--sample-interval", type=float, default=0.025,
                    help="샘플링 주기(초). e6 추론은 짧아서 smolvla(0.05)보다 촘촘히")
    ap.add_argument("--tegra-interval", type=int, default=50, help="tegrastats 주기(ms)")
    ap.add_argument("--gap", type=float, default=0.0, help="추론 사이 대기(초). 0=back-to-back")
    ap.add_argument("--label", default="pi05_e6_v23_lora")
    ap.add_argument("--out-json", default=os.path.join(os.path.dirname(__file__), "infer_resource_result.json"))
    args = ap.parse_args()

    print(f"[bench] connecting ws://{args.host}:{args.port} ...")
    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    try:
        meta_srv = policy.get_server_metadata()
        print(f"[bench] server metadata: {meta_srv}")
    except Exception as e:
        meta_srv = {}
        print(f"[bench] (server metadata 조회 실패: {e})")
    print(f"[bench] runs={args.runs} warmup={args.warmup} prompt={args.prompt!r}")

    # 샘플러 시작
    tegra = TegraReader(args.tegra_interval)
    tegra.start()
    sampler = Sampler(tegra, args.sample_interval)
    sampler.start()
    time.sleep(0.5)  # 샘플러/리더 안정화

    # 워밍업 (첫 추론 cold-start 제외)
    print(f"[bench] warmup x{args.warmup} ...")
    n_action_steps = None
    action_dim = None
    for _ in range(args.warmup):
        r = policy.infer(make_obs(args.img_size, args.prompt))
        a = np.asarray(r["actions"])
        n_action_steps, action_dim = int(a.shape[0]), int(a.shape[-1])

    # 측정
    print(f"[bench] measuring x{args.runs} ...")
    windows, latencies = [], []
    for i in range(args.runs):
        obs = make_obs(args.img_size, args.prompt)
        t0 = time.monotonic()
        policy.infer(obs)
        t1 = time.monotonic()
        windows.append((t0, t1))
        latencies.append((t1 - t0) * 1000.0)
        if args.gap > 0:
            time.sleep(args.gap)

    sampler.stop()
    tegra.stop()

    # 구간별 평균
    per_run = []
    for (t0, t1), lat in zip(windows, latencies):
        res = sampler.window_mean(t0, t1)
        if res is None:
            continue
        vals, n = res
        per_run.append({
            "gpu": float(vals[0]), "cpu": float(vals[1]), "ram": float(vals[2]),
            "latency_ms": float(lat), "n_samples": int(n),
        })

    def stat(key):
        a = np.array([r[key] for r in per_run], dtype=np.float64)
        return float(a.mean()), float(a.std())

    means = {k: stat(k)[0] for k in ("gpu", "cpu", "ram")}
    stds = {k: stat(k)[1] for k in ("gpu", "cpu", "ram")}
    lat_mean, lat_std = stat("latency_ms")

    result = {
        "meta": {
            "label": args.label, "device": jetson_model(),
            "host": args.host, "port": args.port, "prompt": args.prompt,
            "runs": args.runs, "warmup": args.warmup,
            "sample_interval_s": args.sample_interval, "tegra_interval_ms": args.tegra_interval,
            "img_size": args.img_size, "n_action_steps": n_action_steps, "action_dim": action_dim,
            "total_ram_mb": round(psutil.virtual_memory().total / (1024 ** 2), 1),
            "n_cpu": psutil.cpu_count(), "server_metadata": meta_srv,
            "note": "window=websocket infer round-trip (서버 GPU forward + 직렬화/IPC 포함)",
        },
        "means": means, "stds": stds,
        "latency_ms": {"mean": lat_mean, "std": lat_std},
        "per_run": per_run,
    }
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # 콘솔 표
    print("\n" + "=" * 60)
    print(f"{'run':>4} {'GPU%':>8} {'CPU%':>8} {'RAM%':>8} {'lat(ms)':>9} {'#smp':>5}")
    print("-" * 60)
    for i, r in enumerate(per_run, 1):
        print(f"{i:>4} {r['gpu']:>8.1f} {r['cpu']:>8.1f} {r['ram']:>8.1f} {r['latency_ms']:>9.1f} {r['n_samples']:>5}")
    print("-" * 60)
    print(f"{'mean':>4} {means['gpu']:>8.1f} {means['cpu']:>8.1f} {means['ram']:>8.1f} {lat_mean:>9.1f}")
    print(f"{'std':>4} {stds['gpu']:>8.1f} {stds['cpu']:>8.1f} {stds['ram']:>8.1f} {lat_std:>9.1f}")
    print("=" * 60)
    if per_run and min(r["n_samples"] for r in per_run) < 4:
        print(f"[warn] 창당 최소 샘플 {min(r['n_samples'] for r in per_run)}개 — 추론이 짧음. "
              f"--sample-interval/--tegra-interval 를 더 낮추거나 --runs 를 늘리세요.")
    print(f"[bench] JSON saved -> {args.out_json}")


if __name__ == "__main__":
    main()
