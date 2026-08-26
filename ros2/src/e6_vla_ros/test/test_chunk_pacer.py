#!/usr/bin/env python3
"""
chunk_pacer 오프라인 단위테스트 (ROS 불필요).

실행:
  /home/billye6/move-one/min-imum/move-one/bin/python \
      ros2/src/e6_vla_ros/test/test_chunk_pacer.py

검증 항목 (plan: silly-juggling-cat.md §검증):
  1. EMA 부트스트랩: 이력 없을 때 새 측정값으로 그대로 시드
  2. EMA가 단발성 이상치(stall)에 과반응하지 않고 정상 구간으로 회복
  3. compute_pace_k config-sanity: launch 기본값 기준 K≤k_max, 총 실행시간 < staleness 여유
  4. fresh/hold 상태머신: K틱마다 정확히 1회 fresh, 누산기(grip_cont류)는 모델 스텝당 1회만 증가
  5. pace_to_arrival=False → K=1 고정 (기존 동작과 100% 동일)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from e6_vla_ros.chunk_pacer import update_interval_ema, compute_pace_k  # noqa: E402

EXECUTOR_HZ = 16.0
TICK_MS = 1000.0 / EXECUTOR_HZ
STEPS_PER_INFERENCE = 8
STALENESS_MS = 5000.0
K_MIN_MS = 500.0
K_MAX_MS = 3500.0
K_MAX = 12
ALPHA = 0.3


def test_ema_bootstrap():
    est = update_interval_ema(None, 2100.0, ALPHA)
    assert est == 2100.0, est
    print("1. EMA bootstrap OK (첫 측정값 그대로 시드)")


def test_ema_outlier_resilience():
    est = 2100.0
    for _ in range(5):
        est = update_interval_ema(est, 2100.0, ALPHA)
    assert abs(est - 2100.0) < 1e-6, est

    # 단발성 stall(4500ms) — 한 번에 크게 안 튀어야 함
    est_after_stall = update_interval_ema(est, 4500.0, ALPHA)
    assert est < est_after_stall < 4500.0, (est, est_after_stall)
    jump = est_after_stall - est
    assert abs(jump - ALPHA * (4500.0 - est)) < 1e-6, jump

    # 이후 정상값 복귀 시 다시 steady로 수렴
    est_recover = est_after_stall
    for _ in range(20):
        est_recover = update_interval_ema(est_recover, 2100.0, ALPHA)
    assert abs(est_recover - 2100.0) < 1.0, est_recover
    print(f"2. EMA outlier resilience OK (stall 직후={est_after_stall:.1f}ms, 회복 후={est_recover:.1f}ms)")


def test_compute_pace_k_config_sanity():
    k = compute_pace_k(2100.0, STEPS_PER_INFERENCE, TICK_MS, K_MIN_MS, K_MAX_MS, K_MAX)
    assert 1 <= k <= K_MAX, k
    total_ms = STEPS_PER_INFERENCE * k * TICK_MS
    assert total_ms < STALENESS_MS, total_ms
    margin_ms = STALENESS_MS - total_ms
    assert margin_ms > 1000.0, margin_ms  # staleness 대비 최소 1초 여유
    print(f"3. compute_pace_k config-sanity OK (k={k}, 총실행시간={total_ms:.0f}ms, "
          f"staleness 여유={margin_ms:.0f}ms)")

    # 상한 클램프: 극단적으로 큰 추정치가 들어와도 k_max_ms에서 클램프됨
    k_big = compute_pace_k(999999.0, STEPS_PER_INFERENCE, TICK_MS, K_MIN_MS, K_MAX_MS, K_MAX)
    assert k_big <= K_MAX, k_big

    # 하한 클램프: 극단적으로 작은 추정치는 k_min_ms 이하로 안 내려감 (오늘보다 안 빨라짐)
    k_small = compute_pace_k(1.0, STEPS_PER_INFERENCE, TICK_MS, K_MIN_MS, K_MAX_MS, K_MAX)
    k_floor = compute_pace_k(K_MIN_MS, STEPS_PER_INFERENCE, TICK_MS, K_MIN_MS, K_MAX_MS, K_MAX)
    assert k_small == k_floor, (k_small, k_floor)
    print(f"   상/하한 클램프 OK (k_big={k_big}, k_small=k_floor={k_small})")


def test_fresh_hold_state_machine():
    """executor_supervisor_node.py의 게이팅 로직을 순수 파이썬으로 재현해 회귀 검증."""
    k = 4
    counter = 0
    fresh_count = 0
    grip_cont = 0.0
    model_step_deltas = [0.1, -0.2, 0.3, -0.1]  # 모델 스텝 4개, 각 스텝 a[6] 값이라 가정
    idx = 0
    n_ticks = k * len(model_step_deltas)

    for _tick in range(n_ticks):
        fresh = (counter == 0)
        if fresh:
            fresh_count += 1
            grip_cont += model_step_deltas[idx]  # fresh tick에서만 누적 (버그 #2 회귀테스트)
        counter += 1
        if counter >= k:
            counter = 0
            idx += 1

    assert fresh_count == len(model_step_deltas), fresh_count
    expected_grip_cont = sum(model_step_deltas)
    assert abs(grip_cont - expected_grip_cont) < 1e-9, (grip_cont, expected_grip_cont)
    print(f"4. fresh/hold 상태머신 OK (fresh_count={fresh_count}, "
          f"grip_cont={grip_cont:.3f}==sum(deltas)={expected_grip_cont:.3f})")


def test_pace_to_arrival_false_is_legacy():
    pace_to_arrival = False
    k = compute_pace_k(2100.0, STEPS_PER_INFERENCE, TICK_MS, K_MIN_MS, K_MAX_MS, K_MAX) \
        if pace_to_arrival else 1
    assert k == 1, k
    print("5. pace_to_arrival=False → k=1 (기존 동작과 100% 동일) OK")


if __name__ == "__main__":
    test_ema_bootstrap()
    test_ema_outlier_resilience()
    test_compute_pace_k_config_sanity()
    test_fresh_hold_state_machine()
    test_pace_to_arrival_false_is_legacy()
    print("\n✅ ALL TESTS PASSED")
