#!/usr/bin/env python3
"""
π0.5 (openpi) in-process forward CPU 벤치 — ROS2/서버 없이 모델만.

조건:
  1) ROS2 노드 없음. websocket 서버도 안 띄움. openpi 정책을 in-process 로 로드해 policy.infer() 직접 호출.
  2) forward 1회마다 그 구간의 CPU 사용률(%) 을 재서 JSON 으로 저장 → 그래프는 시스템 python3 로.
  3) 더미데이터 = 2CAM-Orange-init 의 1~199 폴더. 폴더 N 개를 랜덤 선택,
     각 폴더에서 hik/zed 프레임 1장씩 랜덤 선택해 obs 구성 (224x224, 리사이즈 불필요).

obs 스키마(e6_policy.make_e6_example 와 동일):
  observation/exterior_image_1_left = hik (224,224,3 uint8)
  observation/exterior_image_2_left = zed (224,224,3 uint8)
  observation/state                 = (7,) float
  prompt                            = str
"""
import argparse
import json
import os
import random
import re
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from PIL import Image

DATA_ROOT = "/media/billye6/새 볼륨1/Dobot/2CAM-Orange-init"
CKPT_DIR = "/media/billye6/새 볼륨1/e6_checkpoints/e6_v26_7500"
CONFIG_NAME = "pi05_e6_v26_lora"
PROMPT = "pick up the orange box from the left side and place it on the right side"
STATE_DIM = 7


# ── tegrastats(GPU) 리더 (있으면 GPU 부하도 같이 기록; 없으면 0) ──────────────
class TegraReader(threading.Thread):
    _GR3D = re.compile(r"GR3D_FREQ\s+(\d+)%")

    def __init__(self, interval_ms: int = 100):
        super().__init__(daemon=True)
        self.gpu = 0.0
        self.ok = False
        self._stop = False
        self.proc = None
        try:
            self.proc = subprocess.Popen(
                ["tegrastats", "--interval", str(interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
            )
            self.ok = True
        except Exception:
            self.ok = False

    def run(self):
        if not self.ok:
            return
        for line in self.proc.stdout:
            if self._stop:
                break
            m = self._GR3D.search(line)
            if m:
                self.gpu = float(m.group(1))

    def stop(self):
        self._stop = True
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass


# ── 자원 샘플러: CPU%/RAM%/GPU% 를 일정 주기로 기록 ──────────────────────────
class Sampler(threading.Thread):
    def __init__(self, tegra: TegraReader, interval: float = 0.03):
        super().__init__(daemon=True)
        self.tegra = tegra
        self.interval = interval
        self.samples = []  # (t, cpu, ram, gpu)
        self._stop = False

    def run(self):
        psutil.cpu_percent(None)  # prime (첫 호출은 0)
        while not self._stop:
            t = time.monotonic()
            cpu = psutil.cpu_percent(None)
            ram = psutil.virtual_memory().percent
            gpu = self.tegra.gpu
            self.samples.append((t, cpu, ram, gpu))
            time.sleep(self.interval)

    def stop(self):
        self._stop = True

    def window_mean(self, t0, t1):
        arr = [(c, r, g) for (t, c, r, g) in self.samples if t0 <= t <= t1]
        if not arr:
            # 너무 짧은 구간이면 가장 가까운 1개라도
            arr = [min(self.samples, key=lambda s: abs(s[0] - (t0 + t1) / 2))[1:]]
        a = np.array(arr, dtype=float)
        return a.mean(axis=0), len(arr)  # [cpu,ram,gpu], n


def list_episodes(root: Path):
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.isdigit()],
                  key=lambda p: int(p.name))


def load_obs(folder: Path, rng: random.Random):
    hik_dir, zed_dir = folder / "images" / "hik", folder / "images" / "zed"
    hik_names = {p.name for p in hik_dir.glob("*.jpg")}
    zed_names = {p.name for p in zed_dir.glob("*.jpg")}
    common = sorted(hik_names & zed_names)
    if not common:
        raise FileNotFoundError(f"{folder}: hik/zed 공통 프레임 없음")
    name = rng.choice(common)
    hik = np.asarray(Image.open(hik_dir / name).convert("RGB"))
    zed = np.asarray(Image.open(zed_dir / name).convert("RGB"))
    obs = {
        "observation/exterior_image_1_left": hik.astype(np.uint8),
        "observation/exterior_image_2_left": zed.astype(np.uint8),
        "observation/state": rng.random() * 0 + np.zeros(STATE_DIM, dtype=np.float32),
        "prompt": PROMPT,
    }
    return obs, name, hik.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--checkpoint", default=CKPT_DIR)
    ap.add_argument("--config", default=CONFIG_NAME)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2, help="콜드스타트 제거용(측정 제외)")
    ap.add_argument("--idle-sec", type=float, default=4.0,
                    help="idle baseline 측정 시간(모델 로드됨·추론 안 함). net=forward-idle")
    ap.add_argument("--sample-interval", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=20260627)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "forward_cpu_random_e6.json"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.data_root)
    eps = list_episodes(root)
    if len(eps) < args.runs:
        raise SystemExit(f"폴더 {len(eps)}개 < runs {args.runs}")
    chosen = rng.sample(eps, args.runs)
    print(f"[e6-bench] 선택된 폴더: {[p.name for p in chosen]}")

    # obs 미리 로드 (이미지 디스크 I/O 를 측정 구간에서 제외)
    obs_list, picks = [], []
    for f in chosen:
        obs, name, shp = load_obs(f, rng)
        obs_list.append(obs)
        picks.append({"folder": f.name, "frame": name, "img_shape": list(shp)})
    print(f"[e6-bench] 이미지 로드 완료 (shape {picks[0]['img_shape']})")

    # 정책 로드 (in-process, GPU)
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    t_load0 = time.monotonic()
    policy = _policy_config.create_trained_policy(
        _config.get_config(args.config), args.checkpoint,
        default_prompt=PROMPT, pytorch_device=args.device,
    )
    print(f"[e6-bench] 정책 로드 완료 ({time.monotonic()-t_load0:.1f}s) "
          f"config={args.config} ckpt={args.checkpoint}")

    tegra = TegraReader(100)
    tegra.start()
    sampler = Sampler(tegra, args.sample_interval)
    sampler.start()
    time.sleep(0.3)  # 샘플러 워밍업

    # 워밍업 (측정 제외): 더미 랜덤 이미지
    for i in range(args.warmup):
        dummy = {
            "observation/exterior_image_1_left": np.random.randint(0, 256, (224, 224, 3), np.uint8),
            "observation/exterior_image_2_left": np.random.randint(0, 256, (224, 224, 3), np.uint8),
            "observation/state": np.zeros(STATE_DIM, np.float32),
            "prompt": PROMPT,
        }
        policy.infer(dummy)
        print(f"[e6-bench] warmup {i+1}/{args.warmup} done")

    # ── idle baseline: 모델 로드됨, 추론 안 함 (GPU 가 가라앉도록 2s 대기 후 측정) ──
    time.sleep(2.0)
    ti0 = time.monotonic()
    time.sleep(args.idle_sec)
    ti1 = time.monotonic()
    im, in_ = sampler.window_mean(ti0, ti1)
    idle = {"cpu": float(im[0]), "ram": float(im[1]), "gpu": float(im[2]), "n_samples": int(in_)}
    print(f"[e6-bench] idle baseline (모델 로드·추론X): "
          f"CPU={idle['cpu']:.1f}%  RAM={idle['ram']:.1f}%  GPU={idle['gpu']:.1f}%  (n={in_})")

    # 측정: 폴더당 1장, 총 runs 회 forward. net = forward - idle (순수 부하)
    per_run = []
    for i, (obs, meta) in enumerate(zip(obs_list, picks), 1):
        t0 = time.monotonic()
        res = policy.infer(obs)
        t1 = time.monotonic()
        means, n = sampler.window_mean(t0, t1)
        lat_ms = (t1 - t0) * 1000.0
        act = np.asarray(res["actions"])
        rec = {
            "idx": i, "folder": meta["folder"], "frame": meta["frame"],
            "cpu": float(means[0]), "ram": float(means[1]), "gpu": float(means[2]),
            "cpu_net": max(0.0, float(means[0]) - idle["cpu"]),
            "ram_net": max(0.0, float(means[1]) - idle["ram"]),
            "gpu_net": max(0.0, float(means[2]) - idle["gpu"]),
            "latency_ms": float(lat_ms), "n_samples": int(n),
            "action_shape": list(act.shape),
        }
        per_run.append(rec)
        print(f"[e6-bench] {i}/{args.runs} folder={meta['folder']:>3} frame={meta['frame']} "
              f"CPU={rec['cpu']:5.1f}%(net {rec['cpu_net']:4.1f})  "
              f"GPU={rec['gpu']:5.1f}%(net {rec['gpu_net']:4.1f})  "
              f"RAM={rec['ram']:4.1f}%(net {rec['ram_net']:4.1f})  lat={lat_ms:7.1f}ms")

    sampler.stop()
    tegra.stop()
    time.sleep(0.2)

    keys = ("cpu", "ram", "gpu", "cpu_net", "ram_net", "gpu_net", "latency_ms")
    arr = {k: np.array([r[k] for r in per_run], float) for k in keys}
    out = {
        "meta": {
            "model": "pi0.5 (openpi)", "config": args.config, "checkpoint": args.checkpoint,
            "device": args.device, "runs": args.runs, "warmup": args.warmup,
            "idle_sec": args.idle_sec,
            "sample_interval_s": args.sample_interval, "seed": args.seed,
            "data_root": args.data_root, "prompt": PROMPT,
            "total_ram_mb": round(psutil.virtual_memory().total / (1024**2), 1),
            "n_cpu": psutil.cpu_count(), "no_ros2": True, "in_process": True,
        },
        "idle": idle,
        "means": {k: float(arr[k].mean()) for k in arr},
        "stds": {k: float(arr[k].std()) for k in arr},
        "per_run": per_run,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    m, s = out["means"], out["stds"]
    print(f"\n[e6-bench] DONE {args.runs}회 — 순수 부하(net = forward - idle):")
    print(f"           CPU net={m['cpu_net']:.1f}±{s['cpu_net']:.1f}%  "
          f"GPU net={m['gpu_net']:.1f}±{s['gpu_net']:.1f}%  RAM net={m['ram_net']:.1f}%")
    print(f"           (절대값 CPU={m['cpu']:.1f}% GPU={m['gpu']:.1f}% RAM={m['ram']:.1f}%  "
          f"| idle CPU={idle['cpu']:.1f}% GPU={idle['gpu']:.1f}% RAM={idle['ram']:.1f}%  "
          f"| lat={m['latency_ms']:.0f}ms)")
    print(f"[e6-bench] JSON 저장: {args.out}")


if __name__ == "__main__":
    main()
