#!/usr/bin/env python3
"""
chunk_pacer — 청크 실행 페이싱(burst-then-freeze 완화)용 순수 수치 함수

executor_supervisor_node.py가 action_chunk를 8스텝(steps_per_inference)만 실행하고
다음 청크가 올 때까지(~2s) 완전히 정지하는 "burst-then-freeze" 패턴을, 이미 측정 중인
실측 청크 도착 간격(_chunk_arrival_intervals_ms)을 근거로 각 모델 스텝을 K틱 동안
유지(hold)해 고르게 펴는 데 쓰는 계산만 분리한 모듈. ROS/로봇 상태는 전혀 참조하지 않음
(executor_supervisor_node.py 쪽 게이팅 로직과 분리해 독립적으로 단위테스트 가능하게 함).

plan: silly-juggling-cat.md
"""
from __future__ import annotations


def update_interval_ema(prev_estimate: float | None, new_interval_ms: float, alpha: float) -> float:
    """실측 청크 도착 간격의 EMA 갱신. prev_estimate=None이면 new_interval_ms로 그대로 시드."""
    if prev_estimate is None:
        return float(new_interval_ms)
    return alpha * float(new_interval_ms) + (1.0 - alpha) * float(prev_estimate)


def compute_pace_k(
    interval_ms: float,
    n_steps: int,
    tick_ms: float,
    k_min_ms: float,
    k_max_ms: float,
    k_max: int,
) -> int:
    """
    EMA 추정 간격(interval_ms)을 n_steps개 모델 스텝에 고르게 나눴을 때, 스텝당 몇 틱(K)을
    유지해야 하는지 계산. interval_ms는 [k_min_ms, k_max_ms]로 먼저 클램프한다
    (k_min_ms=오늘의 burst 시간 이하로는 안 내려가게, k_max_ms=chunk_staleness_sec 대비 여유
    확보). 반환값은 항상 1 이상, k_max 이하.
    """
    span_ms = min(max(interval_ms, k_min_ms), k_max_ms)
    if n_steps <= 0 or tick_ms <= 0:
        return 1
    k = round(span_ms / (n_steps * tick_ms))
    return min(max(1, int(k)), k_max)
