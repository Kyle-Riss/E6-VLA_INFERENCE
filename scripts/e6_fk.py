#!/usr/bin/env python3
"""
e6_fk.py — Dobot Magician E6 forward kinematics (URDF 유도, 로봇 불필요)

상수 출처: ros2/src/e6_description/urdf/me6_robot.xacro 를 `xacro` 로 전개해 추출.
전 관절 axis 가 (0,0,1) 이고 프레임 회전은 origin rpy 가 담당하는 표준 URDF 구성이다.

용도
  - 저장된 관절 궤적을 EEF 공간으로 환산 (paper_docx_prompt.md 의 "FK-based analysis is
    future work" 를 채우는 데 필요)
  - 관절공간 RMSE(deg) 를 mm 로 해석할 때의 변환

⚠️ 한계 (반드시 읽을 것)
  1. FK 는 **flange(Link6)** 까지다. Dobot 이 보고하는 TCP 는 툴 오프셋이 포함될 수 있다
     (xArm 에서 실제로 172mm 오프셋 문제를 겪었다). `--tool-z` 로 보정한다.
  2. URDF 는 공칭 기하다. xArm 실측에서 관절 오프셋 0.2~0.6° 때문에 FK 와 컨트롤러 TCP 가
     최대 7.95mm 어긋났다. **컨트롤러 TCP 를 진실로 삼고 FK 는 변환용으로만 쓸 것.**
  3. 검증에 쓸 (관절, TCP) 동시 표본이 필요하다 — 저장 에피소드 CSV 또는 MCAP.

사용
    python3 scripts/e6_fk.py --selftest
    python3 scripts/e6_fk.py --joints-deg 0 0 0 0 0 0
    python3 scripts/e6_fk.py --approach-logs approach_logs/            # 흡착 순간 TCP 추정
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np

# ── URDF 에서 추출한 상수 (parent→child, origin xyz [m], origin rpy [rad], axis) ──────
# joint1  base_link→Link1   0 0 0.1268          0 0 0                z
# joint2  Link1→Link2      -0.046 0 0.04        1.5708 0 -1.5708     z
# joint3  Link2→Link3       0 0.18906 0.003     0 0 0                z
# joint4  Link3→Link4       0 0.16 0.005        0 0 0                z
# joint5  Link4→Link5       0 0.067 0.032      -1.5708 1.5708 0      z
# joint6  Link5→Link6      -0.047 0 0.034       1.5708 0 -1.5708     z
JOINTS = (
    ("joint1", (0.0,     0.0,     0.1268), (0.0,     0.0,    0.0)),
    ("joint2", (-0.046,  0.0,     0.04),   (1.5708,  0.0,   -1.5708)),
    ("joint3", (0.0,     0.18906, 0.003),  (0.0,     0.0,    0.0)),
    ("joint4", (0.0,     0.16,    0.005),  (0.0,     0.0,    0.0)),
    ("joint5", (0.0,     0.067,   0.032),  (-1.5708, 1.5708, 0.0)),
    ("joint6", (-0.047,  0.0,     0.034),  (1.5708,  0.0,   -1.5708)),
)
# base_link 는 world 로부터 z+0.03 (world_joint, fixed). 로봇 좌표계는 base_link 기준이므로
# 기본적으로 포함하지 않는다. --include-world 로 켠다.
WORLD_TO_BASE_Z = 0.03

JOINT_LIMITS_DEG = (
    (-359.2, 359.2), (-135.0, 135.0), (-154.0, 154.0),
    (-160.0, 160.0), (-173.0, 173.0), (-359.2, 359.2),
)


def rpy_to_R(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def rot_z(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _T(R: np.ndarray, p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(p, dtype=float)
    return T


def fk(q_rad, *, tool_z: float = 0.0, include_world: bool = False) -> np.ndarray:
    """관절각(rad, 6개) → base_link←flange 4x4 동차변환. tool_z 는 flange z 방향 오프셋[m]."""
    q = np.asarray(q_rad, dtype=float).reshape(6)
    T = np.eye(4)
    if include_world:
        T = T @ _T(np.eye(3), (0.0, 0.0, WORLD_TO_BASE_Z))
    for qi, (_, xyz, rpy) in zip(q, JOINTS):
        T = T @ _T(rpy_to_R(*rpy), xyz) @ _T(rot_z(qi), (0.0, 0.0, 0.0))
    if tool_z:
        T = T @ _T(np.eye(3), (0.0, 0.0, tool_z))
    return T


def fk_all_frames(q_rad, *, include_world: bool = False) -> list[np.ndarray]:
    q = np.asarray(q_rad, dtype=float).reshape(6)
    T = np.eye(4)
    if include_world:
        T = T @ _T(np.eye(3), (0.0, 0.0, WORLD_TO_BASE_Z))
    out = [T.copy()]
    for qi, (_, xyz, rpy) in zip(q, JOINTS):
        T = T @ _T(rpy_to_R(*rpy), xyz) @ _T(rot_z(qi), (0.0, 0.0, 0.0))
        out.append(T.copy())
    return out


def tcp_mm(q_rad, **kw) -> np.ndarray:
    return fk(q_rad, **kw)[:3, 3] * 1000.0


def max_reach_mm() -> float:
    """축 방향 링크 길이 합 — 도달 반경의 상한 근사(자료값 대조용)."""
    return 1000.0 * sum(float(np.linalg.norm(xyz)) for _, xyz, _ in JOINTS)


# ── 자기검증 ────────────────────────────────────────────────────────────────────
def selftest() -> int:
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK ' if cond else 'FAIL'}] {name} {detail}")

    print("E6 FK self-test")

    T0 = fk([0] * 6)
    chk("zero pose 유한", np.isfinite(T0).all())
    chk("회전행렬 직교 (zero)", np.allclose(T0[:3, :3] @ T0[:3, :3].T, np.eye(3), atol=1e-9))
    chk("det(R)=+1 (zero)", abs(np.linalg.det(T0[:3, :3]) - 1.0) < 1e-9,
        f"det={np.linalg.det(T0[:3,:3]):+.12f}")

    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.uniform(-2.0, 2.0, 6)
        R = fk(q)[:3, :3]
        if not (np.allclose(R @ R.T, np.eye(3), atol=1e-9) and abs(np.linalg.det(R) - 1) < 1e-9):
            ok = False
    chk("무작위 200 자세 회전행렬 유효", ok)

    # joint1 은 base z 축 회전이므로 TCP 반경(√(x²+y²))이 j1 에 불변이어야 한다
    q = np.array([0.0, 0.3, -0.5, 0.2, 0.4, 0.1])
    radii = []
    for a in np.linspace(-math.pi, math.pi, 13):
        qq = q.copy(); qq[0] = a
        p = tcp_mm(qq)
        radii.append(math.hypot(p[0], p[1]))
    chk("joint1 회전에 TCP 반경 불변", np.ptp(radii) < 1e-6, f"ptp={np.ptp(radii):.3e} mm")

    # joint1 은 z 를 바꾸지 않아야 한다
    zs = []
    for a in np.linspace(-math.pi, math.pi, 13):
        qq = q.copy(); qq[0] = a
        zs.append(tcp_mm(qq)[2])
    chk("joint1 회전에 TCP z 불변", np.ptp(zs) < 1e-6, f"ptp={np.ptp(zs):.3e} mm")

    # joint6 은 flange 자전이므로 tool_z=0 이면 위치가 안 바뀐다
    ps = []
    for a in np.linspace(-math.pi, math.pi, 13):
        qq = q.copy(); qq[5] = a
        ps.append(tcp_mm(qq))
    chk("joint6 회전에 flange 위치 불변 (tool_z=0)",
        np.ptp(np.asarray(ps), axis=0).max() < 1e-6,
        f"max ptp={np.ptp(np.asarray(ps),axis=0).max():.3e} mm")

    # 🔴 축상(z) 툴 오프셋은 j6 회전에 **불변**이어야 한다 — 축 위의 점은 축 회전으로 안 움직인다.
    #    (첫 작성 때 반대로 단정했다가 self-test 가 잡아냈다. FK 가 아니라 테스트가 틀렸던 것.)
    p_a = tcp_mm(q, tool_z=0.10)
    qq = q.copy(); qq[5] += 1.0
    p_b = tcp_mm(qq, tool_z=0.10)
    chk("축상 tool_z 는 j6 회전에 불변", np.linalg.norm(p_a - p_b) < 1e-9,
        f"Δ={np.linalg.norm(p_a-p_b):.3e} mm")
    # 대신 tool_z 는 위치를 축 방향으로 정확히 그만큼 옮겨야 한다
    d = np.linalg.norm(tcp_mm(q, tool_z=0.10) - tcp_mm(q, tool_z=0.0))
    chk("tool_z=100mm 가 정확히 100mm 이동", abs(d - 100.0) < 1e-6, f"Δ={d:.6f} mm")
    # ⇒ 실용적 함의: 흡착컵이 flange 축상에 달려 있으면 **j6 는 TCP 위치를 전혀 바꾸지 않는다.**

    print(f"\n  축 방향 링크길이 합(도달 상한 근사) = {max_reach_mm():.1f} mm")
    print(f"  zero pose flange = X {T0[0,3]*1000:.1f}  Y {T0[1,3]*1000:.1f}  Z {T0[2,3]*1000:.1f} mm")
    print(f"\n{'통과' if ok else '실패'}")
    return 0 if ok else 1


def scan_approach_logs(d: str, tool_z: float) -> int:
    """approach_logs 의 관절 궤적에 FK 를 적용해 흡착 순간 TCP 를 추정한다.

    ⚠️ approach_logs 는 이 Jetson 산출물이 아니다(checkpoint 경로가 /media/billy/).
       FK 구현의 타당성 점검용으로만 쓰고 연구 근거로 인용하지 않는다.
    """
    files = sorted(glob.glob(os.path.join(d, "*.jsonl")))
    if not files:
        print(f"{d} 에 jsonl 없음")
        return 2
    print(f"approach_logs {len(files)}개, tool_z={tool_z*1000:.0f}mm")
    print("⚠️ 타 머신 산출물 — FK 타당성 점검 전용, 연구 근거 아님\n")
    print(f"{'file':44s} {'n':>5s} {'grip@step':>9s}   TCP@grip (mm)            Z range (mm)")
    for f in files:
        rows = []
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
        qs = [(r.get("step"), r.get("q_current_rad_6"), r.get("tool_on_cmd")) for r in rows]
        qs = [(s, q, g) for s, q, g in qs if isinstance(q, list) and len(q) == 6]
        if not qs:
            print(f"{os.path.basename(f):44s} {'-':>5s}  (q_current_rad_6 없음)")
            continue
        zs = [tcp_mm(q, tool_z=tool_z)[2] for _, q, _ in qs]
        grip = next((i for i, (_, _, g) in enumerate(qs) if g), None)
        if grip is None:
            print(f"{os.path.basename(f)[:44]:44s} {len(qs):5d} {'없음':>9s}   "
                  f"{'-':24s}  {min(zs):7.1f} ~ {max(zs):7.1f}")
        else:
            p = tcp_mm(qs[grip][1], tool_z=tool_z)
            print(f"{os.path.basename(f)[:44]:44s} {len(qs):5d} {qs[grip][0]!s:>9s}   "
                  f"X{p[0]:8.1f} Y{p[1]:8.1f} Z{p[2]:7.1f}  {min(zs):7.1f} ~ {max(zs):7.1f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Dobot E6 forward kinematics (URDF 유도)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--joints-deg", nargs=6, type=float)
    ap.add_argument("--approach-logs", metavar="DIR")
    ap.add_argument("--tool-z", type=float, default=0.0, help="flange z 툴 오프셋 [m]")
    ap.add_argument("--include-world", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if a.approach_logs:
        return scan_approach_logs(a.approach_logs, a.tool_z)
    if a.joints_deg:
        q = np.radians(a.joints_deg)
        for i, (lo, hi) in enumerate(JOINT_LIMITS_DEG):
            if not (lo <= a.joints_deg[i] <= hi):
                print(f"⚠️ joint{i+1}={a.joints_deg[i]}° 가 URDF 리밋 [{lo},{hi}] 밖")
        T = fk(q, tool_z=a.tool_z, include_world=a.include_world)
        p = T[:3, 3] * 1000.0
        print(f"flange  X {p[0]:.2f}  Y {p[1]:.2f}  Z {p[2]:.2f}  mm")
        print("R =\n", np.array2string(T[:3, :3], precision=6, suppress_small=True))
        for i, F in enumerate(fk_all_frames(q, include_world=a.include_world)):
            q_ = F[:3, 3] * 1000.0
            print(f"  frame{i}  X {q_[0]:8.2f}  Y {q_[1]:8.2f}  Z {q_[2]:8.2f}")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
