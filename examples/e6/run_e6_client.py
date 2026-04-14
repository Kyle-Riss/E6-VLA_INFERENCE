#!/usr/bin/env python3
"""
Dobot E6 추론 클라이언트 — 6DOF-VLA WebSocket 서버 연결 버전

아키텍처:
  [serve_policy.py] ← WebSocket → [이 스크립트] → Dobot E6

관측 계약 (DroidInputs 기준, pi0_e6_freeze_vlm* 체크포인트):
  observation/exterior_image_1_left : (224, 224, 3) uint8  RGB  — 탑뷰 카메라
  observation/wrist_image_left      : (224, 224, 3) uint8  RGB  — 손목(ur5_style=zeros)
  observation/joint_position        : (7,) float32          rad  — [j1..j6, 0]
  observation/gripper_position      : (1,) float32          0/1  — ToolDO(1) 직전값
  prompt                            : str

액션 계약 (DroidOutputs, 8-D):
  actions[:, 0:6]  Δ관절각 (rad) — 현재 관절에 누산
  actions[:, 6]    패딩 (무시)
  actions[:, 7]    그리퍼 (0/1 연속값, hysteresis 처리)

환경: 6DOF-VLA 전용 .venv 없이 move-one의 기존 venv 재사용 가능.
  serve_policy.py 는 두 레포 완전 동일 파일이고, openpi_client 는 move-one venv 에 이미 설치됨.

서버 실행 예:
  source ~/move-one/min-imum/move-one/bin/activate
  PYTHONPATH=~/move-one/openpi/src \\
  python ~/move-one/openpi/scripts/serve_policy.py \\
    --port 8000 \\
    policy:checkpoint \\
    --policy.config pi0_e6_freeze_vlm_primitive_176_local \\
    --policy.dir /home/billy/26kp/RoboVLA/checkpoints/pi0_e6_freeze_vlm_primitive_176_local/e6_primitive176_baseinit_20260405_2052/35000

  # UR5-style 체크포인트:
  PYTHONPATH=~/move-one/openpi/src \\
  python ~/move-one/openpi/scripts/serve_policy.py \\
    --port 8000 \\
    policy:checkpoint \\
    --policy.config pi0_e6_freeze_vlm_primitive_176_local_ur5 \\
    --policy.dir /home/billy/26kp/RoboVLA/checkpoints/pi0_e6_freeze_vlm_primitive_176_local_ur5/.../35000

클라이언트 실행 예 (approach red object, 자동 return↔approach 순환):
  source ~/move-one/min-imum/move-one/bin/activate
  python ~/6DOF-VLA/examples/e6/run_e6_client.py \\
    --server_host 127.0.0.1 \\
    --robot_ip 192.168.5.1 \\
    --auto_cycle_return_approach \\
    --approach_prompt "approach red object" \\
    --return_prompt "return" \\
    --approach_cycle_sec 8 \\
    --return_cycle_sec 2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# ── openpi_client 경로 추가 ─────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[2]
_CLIENT_SRC = _REPO / "packages" / "openpi-client" / "src"
if _CLIENT_SRC.is_dir() and str(_CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(_CLIENT_SRC))

# ── Dobot SDK 경로 추가 ──────────────────────────────────────────────────────
_HARDWARE = _REPO / "hardware"
_DOBOT_SDK = _HARDWARE / "dobot"
if _DOBOT_SDK.is_dir() and str(_DOBOT_SDK) not in sys.path:
    sys.path.insert(0, str(_DOBOT_SDK))

# ── camera_capture ───────────────────────────────────────────────────────────
_camera_capture_mod = None
if _HARDWARE.is_dir() and str(_HARDWARE) not in sys.path:
    sys.path.insert(0, str(_HARDWARE))
try:
    import camera_capture as _camera_capture_mod  # type: ignore[import]
except ImportError:
    _camera_capture_mod = None


# ── 프롬프트 프리셋 (e6_v1_task_contract 기반) ──────────────────────────────
TASK_PRESETS: dict[str, str] = {
    "approach":    "approach red object",
    "pick":        "pick red object",
    "move_left":   "move object to left",
    "move_right":  "move object to right",
    "move_middle": "move object to middle",
    "place_left":  "place object to left",
    "place_right": "place object to right",
    "place_middle":"place object to middle",
    "return":      "return",
    "init_hold":   "init_hold",
}


def _stage_from_prompt(prompt: str) -> str:
    p = prompt.lower().strip()
    for tag in ("approach", "pick", "move", "place", "return", "init_hold"):
        if p.startswith(tag) or f"[{tag}]" in p:
            return tag
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dobot E6 → 6DOF-VLA WebSocket 클라이언트"
    )

    # ── 서버 ─────────────────────────────────────────────────────────────────
    parser.add_argument("--server_host", default="127.0.0.1", help="serve_policy.py 호스트")
    parser.add_argument("--server_port", type=int, default=8000, help="serve_policy.py 포트")

    # ── 로봇 ─────────────────────────────────────────────────────────────────
    parser.add_argument("--robot_ip", default="192.168.5.1", help="Dobot E6 IP")
    parser.add_argument("--dry_run", action="store_true", help="로봇 전송 없이 추론만")
    parser.add_argument("--no_camera", action="store_true", help="카메라 미사용 (더미 이미지)")

    # ── 프롬프트 ─────────────────────────────────────────────────────────────
    parser.add_argument("--prompt", default="approach red object", help="작업 지시 문구")
    parser.add_argument(
        "--task_preset", choices=sorted(TASK_PRESETS), default=None,
        help=f"TASK_PRESETS 중 하나로 --prompt 덮어씀. 선택지: {list(TASK_PRESETS)}"
    )

    # ── 관측/액션 계약 ────────────────────────────────────────────────────────
    parser.add_argument(
        "--input_layout", choices=["baseline", "ur5_style"], default="baseline",
        help="baseline=wrist=top복제, ur5_style=wrist=zeros"
    )

    # ── 제어 루프 ─────────────────────────────────────────────────────────────
    parser.add_argument("--hz", type=float, default=20.0, help="제어 주파수 (기본 20 Hz)")
    parser.add_argument(
        "--steps_per_inference", type=int, default=None,
        help="청크에서 몇 스텝 실행 후 재추론. None=action_horizon 전체"
    )
    parser.add_argument("--max_runtime_sec", type=float, default=0.0,
                        help="최대 실행 시간 (0=무제한)")
    parser.add_argument("--max_staleness_ms", type=float, default=5000.0,
                        help="청크 재사용 최대 경과 시간(ms)")

    # ── 액션 스케일/안전 ──────────────────────────────────────────────────────
    parser.add_argument("--action_scale", type=float, default=1.0, help="관절 델타 배율")
    parser.add_argument("--max_delta_deg", type=float, default=3.0,
                        help="스텝당 최대 관절 이동(도). 0=무제한")
    parser.add_argument("--min_tool_z", type=float, default=101.0,
                        help="안전: TCP Z(mm) 이 값 이하면 루프 중단")
    parser.add_argument("--safety_hold_pose", action="store_true",
                        help="min_tool_z 도달 시 루프 중단 대신 현재 포즈 유지")
    parser.add_argument("--movj_velocity", type=int, default=70, help="MovJ 속도 0~100")
    parser.add_argument("--movj_accel", type=int, default=60, help="MovJ 가속 0~100")
    parser.add_argument("--control_mode", choices=["movj", "servoj", "reljoint"],
                        default="movj", help="로봇 제어 모드")

    # ── 그리퍼 ────────────────────────────────────────────────────────────────
    parser.add_argument("--grip_open_threshold", type=float, default=0.45)
    parser.add_argument("--grip_close_threshold", type=float, default=0.55)
    parser.add_argument("--grip_close_latch_steps", type=int, default=0)
    parser.add_argument("--z_grip_trigger", type=float, default=None,
                        help="TCP Z(mm) ≤ 이 값 && grip>0 이면 강제 ON")

    # ── 카메라 안전 ────────────────────────────────────────────────────────────
    parser.add_argument("--hold_on_bad_camera", action="store_true", default=True)
    parser.add_argument("--no_hold_on_bad_camera", action="store_false",
                        dest="hold_on_bad_camera")
    parser.add_argument("--camera_black_mean", type=float, default=8.0,
                        help="frame 평균 이 값 미만이면 bad camera")
    parser.add_argument("--bad_camera_consecutive", type=int, default=10,
                        help="연속 bad camera 이 횟수 초과 시 FAIL_SAFETY")

    # ── 초기 자세 ─────────────────────────────────────────────────────────────
    parser.add_argument("--no_init_pose", action="store_true",
                        help="초기 자세 이동 스킵")
    parser.add_argument("--init_pose_version", choices=["ver1", "ver2"], default="ver1",
                        help="초기 자세 버전")

    # ── auto cycle return ↔ approach ─────────────────────────────────────────
    parser.add_argument("--auto_cycle_return_approach", action="store_true",
                        help="return↔approach 자동 순환 (재시작 불필요)")
    parser.add_argument("--approach_prompt", default="approach red object")
    parser.add_argument("--return_prompt", default="return")
    parser.add_argument("--approach_cycle_sec", type=float, default=8.0,
                        help="approach 단계 유지 시간(초)")
    parser.add_argument("--return_cycle_sec", type=float, default=2.0,
                        help="return 단계 유지 시간(초)")

    # ── 로깅 ──────────────────────────────────────────────────────────────────
    parser.add_argument("--exec_log_jsonl", default=None, help="실행 로그 JSONL 경로")
    parser.add_argument("--save_frames_dir", default=None,
                        help="카메라 프레임 저장 폴더 (디버깅용)")

    args = parser.parse_args()

    if args.task_preset:
        args.prompt = TASK_PRESETS[args.task_preset]
        print(f"[PRESET] prompt = {args.prompt!r}")

    # ── 1) WebSocket 정책 연결 ────────────────────────────────────────────────
    from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: PLC0415
    print(f"[1/3] 정책 서버 연결 중: ws://{args.server_host}:{args.server_port}")
    policy = WebsocketClientPolicy(host=args.server_host, port=args.server_port)
    print(f"      연결 완료. 서버 메타데이터: {policy.get_server_metadata()}")

    # ── 2) 로봇 & 카메라 연결 ────────────────────────────────────────────────
    dashboard = move = feed = None
    if not args.dry_run:
        try:
            from DobotApis.DobotApis import DobotApis  # type: ignore[import]  # noqa: PLC0415
            print(f"[2/3] 로봇 연결: {args.robot_ip}")
            dashboard = DobotApis.DashboardApis(args.robot_ip, 29999)
            move = DobotApis.MoveApis(args.robot_ip, 30004)
            feed = DobotApis.FeedBackApis(args.robot_ip, 30005)
            dashboard.EnableRobot()
            time.sleep(0.5)
            print("      EnableRobot 완료")
        except Exception as exc:
            print(f"[WARN] 로봇 연결 실패 ({exc}). dry_run 모드로 계속.")
            dashboard = move = feed = None

    camera = None
    if not args.no_camera:
        if _camera_capture_mod is not None:
            camera = _camera_capture_mod.CameraCapture()
            print(f"[2/3] 카메라: {camera._name}")
        else:
            print("[WARN] camera_capture 모듈 없음 → 더미 이미지")

    # ── 2.5) 초기 자세 ───────────────────────────────────────────────────────
    if not args.no_init_pose and dashboard is not None:
        INIT_POSES: dict[str, list[float]] = {
            "ver1": [0.0, 0.0, 90.0, 0.0, 90.0, 0.0],
            "ver2": [-0.16, -43.88, 79.66, -2.49, 54.22, -0.15],
        }
        pose = INIT_POSES[args.init_pose_version]
        print(f"[2.5/3] 초기 자세 이동 ({args.init_pose_version}): {pose}")
        try:
            j1, j2, j3, j4, j5, j6 = pose
            dashboard.MovJ(j1, j2, j3, j4, j5, j6, 1,
                           v=args.movj_velocity, a=args.movj_accel)
            time.sleep(1.0)
        except Exception as exc:
            print(f"[WARN] 초기 자세 이동 실패: {exc}")

    # ── 3) 추론 루프 ─────────────────────────────────────────────────────────
    print("[3/3] 추론 루프 시작 (Ctrl+C 종료)")
    dt = 1.0 / args.hz
    step = 0
    current_chunk: np.ndarray | None = None
    chunk_index = 0
    chunk_infer_t0: float | None = None
    last_tool_on = 0
    grip_latch_remaining = 0
    bad_camera_streak = 0
    save_frame_count = 0
    save_frames_max = 30
    ref_tool_vec: np.ndarray | None = None
    ref_joints_deg: np.ndarray | None = None

    stage_name = _stage_from_prompt(args.prompt)
    loop_start_mono = time.monotonic()
    stage_start_mono = loop_start_mono
    task_result = "RUNNING"
    task_done_reason = ""

    steps_per_inference = args.steps_per_inference  # None → will use chunk length

    if args.save_frames_dir:
        os.makedirs(args.save_frames_dir, exist_ok=True)

    try:
        while True:
            t0 = time.monotonic()
            elapsed_runtime = t0 - loop_start_mono
            stage_elapsed = t0 - stage_start_mono

            # ── 최대 실행 시간 ───────────────────────────────────────────────
            if args.max_runtime_sec > 0 and elapsed_runtime > args.max_runtime_sec:
                task_result = "FAIL_TIMEOUT"
                task_done_reason = f"runtime>{args.max_runtime_sec}s"
                print(f"[TASK_DONE] {task_result} {task_done_reason}")
                break

            # ── auto cycle return ↔ approach ────────────────────────────────
            if args.auto_cycle_return_approach and stage_name in {"approach", "return"}:
                limit = args.approach_cycle_sec if stage_name == "approach" else args.return_cycle_sec
                if limit > 0 and stage_elapsed > limit:
                    if stage_name == "approach":
                        next_stage, next_prompt = "return", args.return_prompt
                    else:
                        next_stage, next_prompt = "approach", args.approach_prompt
                    print(
                        f"[STAGE_SWITCH] {stage_name}→{next_stage} "
                        f"elapsed={stage_elapsed:.2f}s limit={limit:.2f}s step={step}"
                    )
                    args.prompt = next_prompt
                    stage_name = next_stage
                    stage_start_mono = time.monotonic()
                    stage_elapsed = 0.0
                    current_chunk = None
                    chunk_index = 0
                    task_result = "RUNNING"
                    task_done_reason = ""
                    continue

            # ── 추론 필요 여부 ───────────────────────────────────────────────
            chunk_len = current_chunk.shape[0] if current_chunk is not None else 0
            spi = steps_per_inference if steps_per_inference is not None else chunk_len
            need_infer = current_chunk is None or chunk_index >= spi or chunk_index >= chunk_len

            infer_time_ms: float | None = None
            if need_infer:
                # ── 관절 상태 읽기 ───────────────────────────────────────────
                current_gripper = float(last_tool_on)
                if dashboard is not None:
                    try:
                        res = dashboard.GetToolDO(1)
                        if res:
                            parts = res.split(",")
                            if len(parts) >= 3:
                                current_gripper = float(int(parts[2]))
                            elif parts[0].strip().isdigit():
                                current_gripper = float(int(parts[0].strip()))
                    except Exception:
                        pass

                joint_7 = np.zeros(7, dtype=np.float32)
                if feed is not None:
                    try:
                        fb = feed.feedBackData()
                        if isinstance(fb, dict):
                            q = fb["QActual"][0]
                        else:
                            arr = np.asarray(fb, dtype=object)
                            q = arr[0][17] if arr.size else [0] * 7
                        deg = np.asarray(q, dtype=np.float32).ravel()
                        n = min(6, deg.size)
                        joint_7[:n] = np.deg2rad(deg[:n])
                    except Exception as exc:
                        print(f"  피드백 읽기 실패: {exc}")

                gripper_pos = np.array([current_gripper], dtype=np.float32)

                # ── 이미지 수집 ──────────────────────────────────────────────
                H = W = 224
                if camera is not None:
                    frame = camera.get_frame()
                    obs_img = np.asarray(frame, dtype=np.uint8) if frame is not None else np.zeros((H, W, 3), dtype=np.uint8)
                else:
                    obs_img = np.zeros((H, W, 3), dtype=np.uint8)

                # ── bad camera 안전 홀드 ─────────────────────────────────────
                camera_hold = False
                if args.hold_on_bad_camera and float(obs_img.mean()) < args.camera_black_mean:
                    camera_hold = True
                    bad_camera_streak += 1
                else:
                    bad_camera_streak = 0
                if bad_camera_streak > args.bad_camera_consecutive:
                    task_result = "FAIL_SAFETY"
                    task_done_reason = f"bad_camera>{args.bad_camera_consecutive}"
                    print(f"[TASK_DONE] {task_result} {task_done_reason}")
                    break

                # ── 프레임 저장 ──────────────────────────────────────────────
                if args.save_frames_dir and save_frame_count < save_frames_max and step % 20 == 0:
                    try:
                        import cv2  # noqa: PLC0415
                        path = os.path.join(args.save_frames_dir, f"frame_{save_frame_count:03d}_step{step}.png")
                        cv2.imwrite(path, cv2.cvtColor(obs_img, cv2.COLOR_RGB2BGR))
                        save_frame_count += 1
                    except Exception:
                        pass

                wrist_img = np.zeros_like(obs_img) if args.input_layout == "ur5_style" else obs_img.copy()

                obs = {
                    "observation/exterior_image_1_left": obs_img,
                    "observation/wrist_image_left": wrist_img,
                    "observation/joint_position": joint_7,
                    "observation/gripper_position": gripper_pos,
                    "prompt": args.prompt,
                }

                t_infer0 = time.monotonic()
                result = policy.infer(obs)
                infer_time_ms = (time.monotonic() - t_infer0) * 1000.0
                if step == 0 or step % 10 == 0:
                    print(f"  [추론] step={step} {infer_time_ms:.1f}ms prompt={args.prompt!r}")

                actions = np.asarray(result["actions"])
                if step == 0:
                    print(f"  [ACTION_SHAPE] {actions.shape}")

                spi = steps_per_inference if steps_per_inference is not None else actions.shape[0]
                current_chunk = actions[:spi]
                chunk_index = 0
                chunk_len = current_chunk.shape[0]
                chunk_infer_t0 = time.monotonic()

            # ── 청크 staleness 체크 ──────────────────────────────────────────
            if current_chunk is not None and chunk_infer_t0 is not None:
                stale_ms = (time.monotonic() - chunk_infer_t0) * 1000.0
                if stale_ms > args.max_staleness_ms:
                    print(f"  [STALE_DROP] {stale_ms:.0f}ms → chunk 폐기")
                    current_chunk = None
                    chunk_index = 0
                    time.sleep(dt)
                    continue

            if current_chunk is None or chunk_index >= chunk_len:
                time.sleep(dt)
                continue

            a = current_chunk[chunk_index]

            # ── 현재 관절/TCP 피드백 ─────────────────────────────────────────
            current_joints_deg: np.ndarray | None = None
            current_joints_rad: np.ndarray | None = None
            current_tool_z: float | None = None
            if feed is not None:
                try:
                    fb = feed.feedBackData()
                    if isinstance(fb, dict):
                        current_joints_deg = np.asarray(fb["QActual"][0], dtype=np.float32).ravel()
                        tv = fb.get("ToolVectorActual")
                        if tv is not None:
                            tv = np.asarray(tv[0], dtype=np.float32)
                            current_tool_z = float(tv[2])
                            if ref_tool_vec is None:
                                ref_tool_vec = tv.copy()
                    else:
                        arr = np.asarray(fb, dtype=object)
                        if arr.size:
                            tup = arr[0]
                            try:
                                current_joints_deg = np.asarray(tup[17], dtype=np.float32).ravel()
                            except Exception:
                                pass
                            try:
                                tv = np.asarray(tup[26], dtype=np.float32).ravel()
                                current_tool_z = float(tv[2])
                                if ref_tool_vec is None:
                                    ref_tool_vec = tv.copy()
                            except Exception:
                                pass
                    if current_joints_deg is not None:
                        current_joints_rad = np.deg2rad(current_joints_deg)
                        if ref_joints_deg is None:
                            ref_joints_deg = current_joints_deg.copy()
                except Exception:
                    pass

            # ── 관절 델타 계산 ───────────────────────────────────────────────
            delta_rad = np.asarray([float(a[i]) for i in range(6)], dtype=np.float32)
            if args.action_scale != 1.0:
                delta_rad *= args.action_scale
            if camera_hold:
                delta_rad[:] = 0.0

            delta_deg = np.rad2deg(delta_rad)
            if args.max_delta_deg > 0:
                delta_deg = np.clip(delta_deg, -args.max_delta_deg, args.max_delta_deg)

            execute_motion = True
            if current_joints_deg is None:
                execute_motion = False
                target_joints_deg = np.zeros(6, dtype=np.float32)
            else:
                target_joints_deg = current_joints_deg[:6] + delta_deg

            # ── Z 안전 한계 ──────────────────────────────────────────────────
            if current_tool_z is not None and current_tool_z <= args.min_tool_z:
                print(f"  [SAFETY] Tool Z={current_tool_z:.1f}mm ≤ {args.min_tool_z:.1f}mm")
                if args.safety_hold_pose:
                    if current_joints_deg is not None:
                        target_joints_deg = current_joints_deg[:6].copy()
                    execute_motion = bool(current_joints_deg is not None)
                else:
                    task_result = "FAIL_SAFETY"
                    task_done_reason = f"tool_z<={args.min_tool_z}"
                    print(f"[TASK_DONE] {task_result} {task_done_reason}")
                    break

            # ── 그리퍼 ──────────────────────────────────────────────────────
            action_dim = len(a) if hasattr(a, "__len__") else 0
            if action_dim >= 8:
                grip_raw = float(a[7])
            elif action_dim >= 7:
                grip_raw = float(a[6])
            else:
                grip_raw = 0.0

            if grip_raw >= args.grip_close_threshold:
                hys = 1
            elif grip_raw <= args.grip_open_threshold:
                hys = 0
            else:
                hys = last_tool_on
            tool_on = int(hys)

            if args.grip_close_latch_steps > 0:
                if tool_on == 1:
                    grip_latch_remaining = max(grip_latch_remaining, args.grip_close_latch_steps)
                if grip_latch_remaining > 0:
                    tool_on = 1
                    grip_latch_remaining -= 1

            if args.z_grip_trigger is not None and current_tool_z is not None:
                if current_tool_z <= args.z_grip_trigger and grip_raw > 0.0:
                    tool_on = 1

            if camera_hold:
                tool_on = last_tool_on

            # ── 디버그 출력 (첫 10스텝 + 이후 50마다) ────────────────────────
            dbg = step < 10 or step % 50 == 0
            if dbg:
                j_str = [f"{x:.2f}" for x in target_joints_deg] if execute_motion else ["N/A"] * 6
                print(
                    f"  [step={step}] Δ(deg)=[{','.join(f'{x:.3f}' for x in delta_deg)}] "
                    f"target=[{','.join(j_str)}] grip={grip_raw:.3f}→{tool_on} "
                    f"z={current_tool_z:.1f}mm" if current_tool_z else ""
                )

            # ── 로봇 명령 전송 ───────────────────────────────────────────────
            if dashboard is not None:
                try:
                    j1, j2, j3, j4, j5, j6 = (float(x) for x in target_joints_deg)
                    if not execute_motion:
                        dashboard.ToolDO(1, tool_on)
                    elif args.control_mode == "reljoint":
                        d1, d2, d3, d4, d5, d6 = (float(x) for x in delta_deg)
                        if max(abs(d1), abs(d2), abs(d3), abs(d4), abs(d5), abs(d6)) > 1e-6:
                            dashboard.RelJointMovJ(d1, d2, d3, d4, d5, d6,
                                                   a=args.movj_accel, v=args.movj_velocity)
                        dashboard.ToolDO(1, tool_on)
                    elif args.control_mode == "servoj":
                        dashboard.ServoJ(j1, j2, j3, j4, j5, j6)
                        dashboard.ToolDO(1, tool_on)
                    else:
                        if (current_joints_deg is None or
                                np.max(np.abs(current_joints_deg[:6] - target_joints_deg)) > 1e-6):
                            dashboard.MovJ(j1, j2, j3, j4, j5, j6, 1,
                                           v=args.movj_velocity, a=args.movj_accel)
                        dashboard.ToolDO(1, tool_on)
                    last_tool_on = tool_on
                except Exception as exc:
                    print(f"  [ROBOT_ERR] {exc}")

            # ── JSONL 로그 ────────────────────────────────────────────────────
            if args.exec_log_jsonl:
                try:
                    log_path = os.path.expanduser(args.exec_log_jsonl)
                    os.makedirs(os.path.dirname(log_path), exist_ok=True)
                    entry = {
                        "t": time.time(), "step": step,
                        "stage": stage_name, "prompt": args.prompt,
                        "task_result": task_result,
                        "infer_ms": infer_time_ms,
                        "delta_deg": delta_deg.tolist() if execute_motion else None,
                        "target_deg": target_joints_deg.tolist() if execute_motion else None,
                        "tool_z": current_tool_z,
                        "grip_raw": grip_raw, "tool_on": tool_on,
                    }
                    with open(log_path, "a") as f:
                        f.write(json.dumps(entry) + "\n")
                except Exception:
                    pass

            chunk_index += 1
            step += 1

            # ── Hz 대기 ──────────────────────────────────────────────────────
            elapsed = time.monotonic() - t0
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        task_result = "INTERRUPTED"
        print("\n[TASK_DONE] Ctrl+C 종료")
    finally:
        print(f"[DONE] task_result={task_result} reason={task_done_reason!r} total_steps={step}")
        if camera is not None:
            try:
                camera.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
