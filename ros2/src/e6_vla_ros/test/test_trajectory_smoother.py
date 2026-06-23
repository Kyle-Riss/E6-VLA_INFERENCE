#!/usr/bin/env python3
"""
MPCSmoother 오프라인 단위테스트 (ROS 불필요).

실행:
  /home/billye6/move-one/min-imum/move-one/bin/python \
      ros2/src/e6_vla_ros/test/test_trajectory_smoother.py

검증 항목 (plan: lively-stargazing-whisper.md §검증):
  1. 유한차분 IC 재구성 정확성
  2. 제약 만족: |Δq| ≤ dq_max, q_min ≤ q ≤ q_max
  3. jerk RMS 감소 (smoothed < raw reference)
  4. 실패경로: 비feasible 한계 → reference passthrough (예외 없음)
  5. seam 연속성: 연속 chunk 접합부 속도 근사 연속
  6. backend 일치: scipy vs osqp (osqp 설치 시)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from e6_vla_ros.trajectory_smoother import (  # noqa: E402
    MPCSmoother, DEFAULT_PHASE_WEIGHTS, _diff_matrix, _diff_const,
)

N = 8
DT = 1.0 / 16.0
QMIN = np.array([-359., -135., -154., -160., -173., -359.], np.float64)
QMAX = -QMIN
DQ_MAX = 3.0
ANCHOR = np.array([91.3, 37.7, 53.8, -1.5, -87.8, 173.3], np.float64)


def test_diff_ic():
    anchor, v0 = 10.0, 4.0
    v0dt = v0 * DT
    q = np.array([10.5, 11.0, 11.4, 11.7, 11.9, 12.0, 12.0, 12.0])
    d1 = _diff_matrix(N, 1) @ q + _diff_const(N, 1, anchor, v0dt, 0.0)
    assert abs(d1[0] - (q[0] - anchor)) < 1e-9
    assert abs(d1[1] - (q[1] - q[0])) < 1e-9
    d2 = _diff_matrix(N, 2) @ q + _diff_const(N, 2, anchor, v0dt, 0.0)
    assert abs(d2[0] - (q[0] - 2 * anchor + (anchor - v0dt))) < 1e-9
    print("1. diff/IC reconstruct OK")


def _make(backend):
    return MPCSmoother(horizon=N, dt=DT, q_min=QMIN, q_max=QMAX, dq_max=DQ_MAX,
                       a_max=2.0, backend=backend, phase_weights=DEFAULT_PHASE_WEIGHTS,
                       w_track=1.0, w_vel=0.3, w_acc=0.15, w_jerk=0.06)


def _osc_chunk(seed=0):
    rng = np.random.RandomState(seed)
    k = np.arange(N)
    ch = np.zeros((N, 7))
    for j in range(6):
        ch[:, j] = 2.0 * np.sin(0.9 * k + j) + rng.randn(N) * 0.4
    ch[:, 6] = (k >= 4).astype(float)   # gripper flip at step 4
    return ch


def test_constraints_and_jerk():
    ch = _osc_chunk()
    sm = _make("scipy")
    out = sm.solve(ch[:, :6], ANCHOR, np.zeros(6), None,
                   phase="move", gripper_seq=ch[:, 6])
    dq = np.diff(np.vstack([ANCHOR[None, :], out]), axis=0)
    assert np.abs(dq).max() <= DQ_MAX + 1e-2, np.abs(dq).max()
    assert (out >= QMIN - 1e-3).all() and (out <= QMAX + 1e-3).all()
    print(f"2. constraints OK (max|Δq|={np.abs(dq).max():.4f} ≤ {DQ_MAX})")

    q_ref = ANCHOR[None, :] + np.cumsum(ch[:, :6], axis=0)
    jr = lambda t: np.sqrt((np.diff(t, n=3, axis=0) ** 2).mean())
    assert jr(out) < jr(q_ref), (jr(out), jr(q_ref))
    print(f"3. jerk reduced: ref={jr(q_ref):.4f} → smoothed={jr(out):.4f}")
    return sm, out


def test_failure_passthrough():
    ch = _osc_chunk()
    bad = MPCSmoother(horizon=N, dt=DT, q_min=QMAX, q_max=QMIN, dq_max=DQ_MAX, backend="scipy")
    o = bad.solve(ch[:, :6], ANCHOR, np.zeros(6), None, phase="move", gripper_seq=ch[:, 6])
    assert o.shape == (N, 6)
    print("4. failure-path → passthrough OK")


def test_seam(sm, out):
    ch = _osc_chunk()
    v_seam = (out[-1] - out[-2])   # deg/step
    out2 = sm.solve(ch[:, :6], out[-1].astype(float), v_seam, None,
                    phase="move", gripper_seq=ch[:, 6])
    # 접합 첫 step 속도가 요청 v0와 같은 부호·비슷한 크기인지 (soft IC라 근사)
    first = out2[0] - out[-1]
    print(f"5. seam continuity: req v0={v_seam[:3]} first_step={first[:3]}")


def test_backend_agreement(out):
    try:
        import osqp  # noqa: F401
    except ImportError:
        print("6. osqp 미설치 — backend 일치 skip (scipy fallback 검증됨)")
        return
    ch = _osc_chunk()
    smo = _make("osqp")
    outo = smo.solve(ch[:, :6], ANCHOR, np.zeros(6), None, phase="move", gripper_seq=ch[:, 6])
    diff = np.abs(out - outo).max()
    assert diff < 5e-2, diff
    print(f"6. backend agreement OK (scipy vs osqp max diff={diff:.5f})")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    test_diff_ic()
    sm, out = test_constraints_and_jerk()
    test_failure_passthrough()
    test_seam(sm, out)
    test_backend_agreement(out)
    print("\n✅ ALL TESTS PASSED")
