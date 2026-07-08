#!/usr/bin/env python3
"""
trajectory_smoother — π0.5 action chunk 후단 joint-space QP optimizer (MPC v1)

π0.5가 낸 16-step action chunk를 reference trajectory로 보고, 관절 위치/속도 한계와
smoothness(속도/가속/jerk) 비용 하에서 chunk 내부 관절 궤적을 재최적화한다.
정책 가중치는 건드리지 않는 execution-side(분리) 구조 — "chunk-level receding-horizon
trajectory optimization (kinematic)". 3-way ablation의 `π0.5 + QP execution optimizer` arm.

설계 요약 (plan: lively-stargazing-whisper.md):
- 결정변수 = 관절 궤적 q_1..q_N (6관절, joint-major x = [q^(0);...;q^(5)], 각 N차원)
- 비용(관절별):
    J = Σ wt_k (q_k - r_k)^2 + Σ wv_k (Δq)_k^2 + Σ wa_k (Δ²q)_k^2 + Σ wj_k (Δ³q)_k^2
  → ½xᵀPx + gᵀx (P는 관절 공통, g는 관절별 r/IC로 다름)
- 제약: box q_min≤q≤q_max (hard), velocity |Δq|≤dq_max (hard).
        accel은 기본 soft(비용 wa로만) → 항상 feasible. hard_accel=True면 |Δ²q|≤a_max·dt² 추가.
- seam 연속성: q_0=anchor, q_{-1}=anchor−v0·dt, q_{-2}=anchor−2v0·dt+a0·dt² 를 **상수**로 두어
  유한차분 IC를 선형항 g·제약 l/u 오프셋으로 흡수 (chunk 간 blending 없음, 속도 연속성만 주입).
- step-wise 가중치: gripper command가 바뀌는 step ±2 = high track / low jerk (grasp·place seam 보호),
  그 외 transport(stage==move)는 smoothing 강하게.
- backend: "scipy"(SLSQP, venv 기본·의존성0) | "osqp"(논문 메인). **동일 P,g,A,l,u** 사용.
- 실패(예외/non-optimal) → reference 그대로 반환(passthrough). 호출부 안전 클램프가 받쳐줌.
"""
from __future__ import annotations

import time

import numpy as np

try:
    from scipy import sparse as _sp
    from scipy.optimize import minimize as _minimize, LinearConstraint as _LinCon
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


# ── 유한차분 스텐실 (q_k - q_{k-1}, q_k-2q_{k-1}+q_{k-2}, ...) ──────────────────
# order o 의 계수: (-1)^i * C(o, i),  q_{k-i} 에 적용
def _stencil(order: int) -> np.ndarray:
    from math import comb
    return np.array([((-1) ** i) * comb(order, i) for i in range(order + 1)], dtype=np.float64)


def _diff_matrix(N: int, order: int) -> np.ndarray:
    """[q_1..q_N] 에 작용하는 변수항 유한차분 행렬 D_o (N×N). IC 상수는 별도(_diff_const)."""
    coeff = _stencil(order)
    D = np.zeros((N, N), dtype=np.float64)
    for k in range(1, N + 1):           # 1-indexed step
        for i in range(order + 1):
            j = k - i                   # q_j
            if j >= 1:                  # 변수
                D[k - 1, j - 1] += coeff[i]
    return D


def _diff_const(N: int, order: int, anchor: float, v0dt: float, a0dt2: float) -> np.ndarray:
    """유한차분의 IC 상수항 c_o (N,). known[0]=anchor, known[-1]=anchor−v0dt, known[-2]=anchor−2v0dt+a0dt2."""
    coeff = _stencil(order)
    known = {0: anchor, -1: anchor - v0dt, -2: anchor - 2.0 * v0dt + a0dt2}
    c = np.zeros(N, dtype=np.float64)
    for k in range(1, N + 1):
        for i in range(order + 1):
            j = k - i
            if j < 1:                   # 알려진 상수
                c[k - 1] += coeff[i] * known[j]
    return c


class MPCSmoother:
    """π0.5 action chunk → 제약·smoothness 만족 joint 궤적. solve()는 chunk 도착 시 1회 호출."""

    def __init__(self,
                 horizon: int,
                 n_joints: int = 6,
                 dt: float = 1.0 / 16.0,
                 w_track: float = 1.0,
                 w_vel: float = 0.1,
                 w_acc: float = 0.05,
                 w_jerk: float = 0.02,
                 q_min=None,
                 q_max=None,
                 dq_max: float = 3.0,
                 a_max: float = 2.0,
                 hard_accel: bool = False,
                 action_scale: float = 1.0,
                 backend: str = "scipy",
                 phase_weights: dict | None = None,
                 logger=None):
        self.N = int(horizon)
        self.nj = int(n_joints)
        self.dt = float(dt)
        self.w_base = (float(w_track), float(w_vel), float(w_acc), float(w_jerk))
        self.dq_max = float(dq_max)
        self.a_max = float(a_max)
        self.hard_accel = bool(hard_accel)
        self.action_scale = float(action_scale)
        self.backend = backend
        self._log = logger
        self._phase_weights = phase_weights or {}

        N = self.N
        self.q_min = (np.full(self.nj, -1e4, np.float64) if q_min is None
                      else np.asarray(q_min, np.float64)[:self.nj])
        self.q_max = (np.full(self.nj, +1e4, np.float64) if q_max is None
                      else np.asarray(q_max, np.float64)[:self.nj])

        # 정적 유한차분 행렬 (변수항만)
        self.D1 = _diff_matrix(N, 1)
        self.D2 = _diff_matrix(N, 2)
        self.D3 = _diff_matrix(N, 3)

        # backend 준비 (osqp 가용 시에만 osqp, 아니면 scipy fallback)
        self._osqp = None
        self._osqp_key = None           # P sparsity/weights 변경 감지용
        if backend == "osqp":
            try:
                import osqp  # noqa: F401
                self._osqp_mod = osqp
            except Exception as exc:
                self._warn(f"osqp import 실패({exc}) → scipy backend로 fallback")
                self.backend = "scipy"
        if self.backend == "scipy" and not _HAVE_SCIPY:
            raise RuntimeError("scipy 미설치 — MPCSmoother 사용 불가")

        self._warm_x = None             # scipy warm-start 상태 (6N,)

        # ── solve 상태 노출 (호출부 fallback 감지용; 반환값과 무관) ────────────
        # solve()가 성공/실패(passthrough) 모두 (N,6)을 반환하므로 호출부가 구분 불가.
        # 아래 멤버로 마지막 solve의 성공여부/소요시간/상태문자열을 노출한다.
        self.last_solve_ok = False      # True=최적해, False=passthrough(reference)
        self.last_solve_ms = 0.0        # 마지막 solve 소요(ms)
        self.last_status = ""           # "solved" | "passthrough:<exc>"

    # ── 로깅 헬퍼 ──────────────────────────────────────────────────────────────
    def _warn(self, msg: str):
        if self._log is not None:
            try:
                self._log.warn(msg)
                return
            except Exception:
                pass
        print(f"[MPCSmoother] WARN {msg}")

    # ── step-wise 가중치 벡터 (N,) 4종 ────────────────────────────────────────
    def _step_weights(self, gripper_seq: np.ndarray, phase: str):
        """gripper command 변화 ±2 step = high track/low jerk, 그 외 phase 기반."""
        N = self.N
        wt, wv, wa, wj = self._phase_weights.get(phase, self.w_base)
        wt_v = np.full(N, wt, np.float64)
        wv_v = np.full(N, wv, np.float64)
        wa_v = np.full(N, wa, np.float64)
        wj_v = np.full(N, wj, np.float64)
        if gripper_seq is not None and len(gripper_seq) >= 2:
            g = np.asarray(gripper_seq, np.float64)[:N]
            chg = np.where(np.abs(np.diff(g)) > 0.5)[0]   # command 전환 index
            for c in chg:
                lo, hi = max(0, c - 2), min(N - 1, c + 3)  # ±2 step
                wt_v[lo:hi] = max(wt, 8.0)                  # tracking 지배
                wj_v[lo:hi] = min(wj, 0.005)               # jerk smoothing 약화
                wa_v[lo:hi] = min(wa, 0.01)
                wv_v[lo:hi] = min(wv, 0.02)
        return wt_v, wv_v, wa_v, wj_v

    # ── 관절 공통 P_block (N×N) ───────────────────────────────────────────────
    def _build_Pblock(self, wt_v, wv_v, wa_v, wj_v) -> np.ndarray:
        D1, D2, D3 = self.D1, self.D2, self.D3
        P = (np.diag(wt_v)
             + D1.T @ np.diag(wv_v) @ D1
             + D2.T @ np.diag(wa_v) @ D2
             + D3.T @ np.diag(wj_v) @ D3)
        return 2.0 * P            # ½xᵀPx 규약

    # ── 메인 solve ────────────────────────────────────────────────────────────
    def solve(self,
              chunk_deltas: np.ndarray,   # (N,6) — delta(absolute=False) 또는 절대각(absolute=True)
              anchor_deg: np.ndarray,     # (6,) 현재 관절각
              v0: np.ndarray,             # (6,) deg/step (현재 속도)
              a0: np.ndarray | None = None,  # (6,) deg/step^2 (None→0)
              phase: str = "move",
              gripper_seq: np.ndarray | None = None,  # (N,) raw gripper command
              absolute: bool = False) -> np.ndarray:
        """smoothed joint targets (N,6) 반환. 실패 시 q_ref(reference) 그대로."""
        N, nj = self.N, self.nj
        anchor = np.asarray(anchor_deg, np.float64)[:nj]
        v0 = np.asarray(v0, np.float64)[:nj]
        a0 = (np.zeros(nj, np.float64) if a0 is None else np.asarray(a0, np.float64)[:nj])
        cd = np.asarray(chunk_deltas, np.float64)[:N, :nj]

        # reference trajectory (관절별 N) — q_ref[k,j]
        if absolute:
            q_ref = cd.copy()
        else:
            q_ref = anchor[None, :] + np.cumsum(cd * self.action_scale, axis=0)
        q_ref = np.clip(q_ref, self.q_min[None, :], self.q_max[None, :])

        if gripper_seq is None and chunk_deltas.shape[1] > 6:
            gripper_seq = np.asarray(chunk_deltas, np.float64)[:N, 6]

        _t0 = time.monotonic()
        try:
            wt_v, wv_v, wa_v, wj_v = self._step_weights(gripper_seq, phase)
            Pb = self._build_Pblock(wt_v, wv_v, wa_v, wj_v)          # (N,N) 관절 공통

            # 관절별 g, l, u 조립
            g_list, l_list, u_list = [], [], []
            for j in range(nj):
                v0dt = v0[j] * self.dt
                a0dt2 = a0[j] * self.dt * self.dt
                c1 = _diff_const(N, 1, anchor[j], v0dt, a0dt2)
                c2 = _diff_const(N, 2, anchor[j], v0dt, a0dt2)
                c3 = _diff_const(N, 3, anchor[j], v0dt, a0dt2)
                r = q_ref[:, j]
                g = 2.0 * (-(wt_v * r)
                           + self.D1.T @ (wv_v * c1)
                           + self.D2.T @ (wa_v * c2)
                           + self.D3.T @ (wj_v * c3))
                g_list.append(g)
                # box + velocity(+ accel hard 옵션)
                lj = [np.full(N, self.q_min[j]), -self.dq_max - c1]
                uj = [np.full(N, self.q_max[j]), +self.dq_max - c1]
                if self.hard_accel:
                    a_bound = self.a_max * self.dt * self.dt
                    lj.append(-a_bound - c2)
                    uj.append(+a_bound - c2)
                l_list.append(np.concatenate(lj))
                u_list.append(np.concatenate(uj))

            # A_block (rows × N): [I; D1] (+D2 if hard_accel)
            blocks = [np.eye(N), self.D1] + ([self.D2] if self.hard_accel else [])
            A_block = np.vstack(blocks)

            # 전체 6N 문제 (block-diagonal)
            P_full = _sp.block_diag([Pb] * nj, format="csc")
            g_full = np.concatenate(g_list)
            A_full = _sp.block_diag([A_block] * nj, format="csc")
            l_full = np.concatenate(l_list)
            u_full = np.concatenate(u_list)

            if self.backend == "osqp":
                x = self._solve_osqp(P_full, g_full, A_full, l_full, u_full, wt_v, wv_v, wa_v, wj_v)
            else:
                x = self._solve_scipy(P_full, g_full, A_full, l_full, u_full, q_ref)

            sm = x.reshape(nj, N).T                  # (N,6)
            # 최종 안전: reference 근처를 벗어난 NaN/inf 방어
            if not np.all(np.isfinite(sm)):
                raise ValueError("solver 결과에 비유한값")
            self.last_solve_ok = True
            self.last_status = "solved"
            self.last_solve_ms = (time.monotonic() - _t0) * 1000.0
            return sm.astype(np.float32)
        except Exception as exc:
            self.last_solve_ok = False
            self.last_status = f"passthrough:{exc}"
            self.last_solve_ms = (time.monotonic() - _t0) * 1000.0
            self._warn(f"solve 실패({exc}) → reference passthrough")
            return q_ref.astype(np.float32)

    # ── backend: OSQP ─────────────────────────────────────────────────────────
    def _solve_osqp(self, P, g, A, l, u, wt_v, wv_v, wa_v, wj_v):
        key = (tuple(np.round(wt_v, 6)), tuple(np.round(wv_v, 6)),
               tuple(np.round(wa_v, 6)), tuple(np.round(wj_v, 6)), A.shape)
        if self._osqp is None or self._osqp_key != key:
            self._osqp = self._osqp_mod.OSQP()
            self._osqp.setup(P=P.tocsc(), q=g, A=A.tocsc(), l=l, u=u,
                             warm_start=True, verbose=False, polish=True,
                             eps_abs=1e-6, eps_rel=1e-6, max_iter=8000)
            self._osqp_key = key
        else:
            self._osqp.update(q=g, l=l, u=u)
        res = self._osqp.solve()
        status = str(res.info.status)
        if "solved" not in status:
            raise RuntimeError(f"osqp status={status}")
        return np.asarray(res.x, np.float64)

    # ── backend: scipy SLSQP (동일 P,g,A,l,u) ─────────────────────────────────
    def _solve_scipy(self, P, g, A, l, u, q_ref):
        Pd = P.toarray() if _sp.issparse(P) else np.asarray(P)
        Ad = A.toarray() if _sp.issparse(A) else np.asarray(A)

        def fun(x):
            return 0.5 * float(x @ (Pd @ x)) + float(g @ x)

        def jac(x):
            return Pd @ x + g

        x0 = self._warm_x if self._warm_x is not None and self._warm_x.size == Pd.shape[0] \
            else q_ref.T.reshape(-1)        # joint-major 초기값 = reference
        con = _LinCon(Ad, l, u)
        res = _minimize(fun, x0, jac=jac, method="SLSQP",
                        constraints=[con], options={"maxiter": 200, "ftol": 1e-6})
        if not res.success:
            raise RuntimeError(f"slsqp: {res.message}")
        self._warm_x = res.x
        return np.asarray(res.x, np.float64)


# 기본 phase 가중치 (executor에서 주입; stage → (w_track, w_vel, w_acc, w_jerk))
DEFAULT_PHASE_WEIGHTS = {
    "approach": (1.0, 0.10, 0.05, 0.02),
    "pick":     (8.0, 0.02, 0.01, 0.005),   # grasp seam: track 지배
    "lift":     (1.0, 0.20, 0.10, 0.04),
    "move":     (1.0, 0.30, 0.15, 0.06),    # transport: smoothing 강
    "place":    (8.0, 0.02, 0.01, 0.005),   # release seam: track 지배
    "release":  (4.0, 0.05, 0.02, 0.01),
}
