#!/usr/bin/env python3
"""
더미 WebSocket 정책 서버 — 실제 모델 없이 파이프라인 테스트용

실제 serve_policy.py + 체크포인트 없이도 run_e6_client.py 전체 경로를 검증할 수 있습니다.
관측(obs)을 받으면 랜덤에 가까운 작은 액션(Δrad)을 반환합니다.

사용법:
  source ~/move-one/min-imum/move-one/bin/activate
  PYTHONPATH=~/e6-vla/src python ~/e6-vla/scripts/serve_dummy.py [--port 8000] [--action_dim 8] [--action_horizon 10]

클라이언트 (별도 터미널):
  python ~/e6-vla/examples/e6/run_e6_client.py \\
    --server_host 127.0.0.1 --port 8000 \\
    --dry_run --no_camera \\
    --prompt "approach red object"
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# ── openpi_client (서버 구현체) 경로 ─────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
_CLIENT_SRC = _REPO / "packages" / "openpi-client" / "src"
if _CLIENT_SRC.is_dir() and str(_CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(_CLIENT_SRC))

from openpi_client import base_policy as _base_policy  # noqa: E402
from openpi.serving import websocket_policy_server  # noqa: E402  # needs PYTHONPATH=src


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class DummyPolicy(_base_policy.BasePolicy):
    """obs를 받아 랜덤(미세) 액션을 반환하는 더미 정책."""

    def __init__(self, action_dim: int = 8, action_horizon: int = 10, scale: float = 0.005):
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.scale = scale
        self._call_count = 0

    def infer(self, obs: dict) -> dict:
        self._call_count += 1
        actions = np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)
        # 관절 0~5: 아주 작은 랜덤 delta (rad) — 로봇이 거의 안 움직임
        actions[:, :6] = np.random.randn(self.action_horizon, 6).astype(np.float32) * self.scale
        # 인덱스 7: 그리퍼 열림(0.0) 유지
        if self.action_dim > 7:
            actions[:, 7] = 0.0
        log.info(
            f"[DummyPolicy] infer #{self._call_count} | "
            f"obs keys={list(obs.keys())} | "
            f"actions shape={actions.shape} max_delta={np.abs(actions[:,:6]).max():.5f} rad"
        )
        return {"actions": actions}

    def reset(self) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="더미 WebSocket 정책 서버")
    parser.add_argument("--port", type=int, default=8000, help="서버 포트 (기본: 8000)")
    parser.add_argument("--host", default="0.0.0.0", help="서버 호스트 (기본: 0.0.0.0)")
    parser.add_argument("--action_dim", type=int, default=8, help="액션 차원 (기본: 8)")
    parser.add_argument("--action_horizon", type=int, default=10, help="액션 청크 길이 (기본: 10)")
    parser.add_argument("--scale", type=float, default=0.005, help="관절 delta 크기(rad) 기본: 0.005")
    args = parser.parse_args()

    policy = DummyPolicy(
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
        scale=args.scale,
    )
    metadata = {
        "dummy": True,
        "action_dim": args.action_dim,
        "action_horizon": args.action_horizon,
    }

    log.info(f"더미 서버 시작: ws://{args.host}:{args.port}")
    log.info(f"action_dim={args.action_dim}, action_horizon={args.action_horizon}, scale={args.scale}")
    log.info("Ctrl+C 로 종료")

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
