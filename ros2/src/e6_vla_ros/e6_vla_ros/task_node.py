#!/usr/bin/env python3
"""
task_node — task_sequence 상태머신 (episode 모드) / per-frame phase 감지 (per_frame 모드) /
            단일 고정 프롬프트 (single 모드, v13).

구독 토픽:
  /e6/supervisor/status   std_msgs/String
  /e6/robot/state         std_msgs/Float32MultiArray  [j1..j6 deg, gripper]  (per_frame 모드)
  /e6/robot/tcp_z         std_msgs/Float32                                    (per_frame 모드)

발행 토픽:
  /e6/task/prompt         std_msgs/String  (QoS: transient_local)
  /e6/task/status         std_msgs/String  10

파라미터 (공통):
  prompt_mode    (str,   default "episode")  "episode" (v6) | "per_frame" (v8) | "single" (v13)
  task_sequence  (str,   default "pick_from_left")
  stage_timeout_sec (float, default 0.0)
  loop_sequence  (bool,  default False)

파라미터 (per_frame 모드 전용):
  source_side    (str,   default "left")   "left" | "right"
  target_side    (str,   default "right")  "right" | "left"
  z_lift         (float, default 180.0)    phase 구분 기준 TCP Z (mm)
  grip_threshold (float, default 0.5)      gripper ON 판단 임계값
  phase_hz       (float, default 16.0)     phase 감지 + 프롬프트 발행 주파수
  return_z_done  (float, default 180.0)    return 완료 기준 TCP Z (mm, 이 값 이상)
  return_done_steps (int, default 5)       return 완료 조건 연속 만족 프레임 수

파라미터 (single 모드 전용):
  source_side    (str,   default "left")   "left" | "right"
  prompt_variant (int,   default -1)       0~2 고정 선택, -1이면 랜덤
"""
from __future__ import annotations

import random

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Float32MultiArray, Float32, String

V8_PHASE_PROMPTS: dict[str, str] = {
    "approach":  "move the arm down to approach the orange box on the {source}",
    "grasp":     "grasp the orange box on the {source}",
    "lift":      "lift the orange box from the {source}",
    "transport": "lift and carry the orange box to the {target}",
    "place":     "lower the orange box onto the {target}",
    "release":   "release the orange box on the {target}",
    "return":    "return the arm to the ready position",
}


class PhaseTracker:
    """gripper 상태 + TCP Z → 7-phase 실시간 분류 (v8 contract)."""

    Z_LIFT = 180.0
    TRANSITION_FRAMES = 5

    def __init__(self, z_lift: float = 180.0, transition_frames: int = 5, grip_threshold: float = 0.5):
        self.z_lift = z_lift
        self.transition_frames = transition_frames
        self.grip_threshold = grip_threshold

        self._phase = "approach"
        self._prev_gripper = 0
        self._crossed_lift = False  # gripper=1 상태에서 z>z_lift 통과 여부
        self._released = False      # 이 에피소드에서 gripper 1→0 전환 발생 여부
        self._trans_counter = 0     # gripper 전환 후 남은 window 프레임 수
        self._trans_type: str | None = None  # "close" | "open"

    def reset(self):
        self._phase = "approach"
        self._prev_gripper = 0
        self._crossed_lift = False
        self._released = False
        self._trans_counter = 0
        self._trans_type = None

    def update(self, gripper_raw: float, tcp_z: float) -> str:
        gripper = 1 if gripper_raw >= self.grip_threshold else 0

        # gripper 전환 감지
        if gripper != self._prev_gripper:
            self._trans_counter = self.transition_frames
            self._trans_type = "close" if gripper == 1 else "open"
            if gripper == 0:
                self._released = True

        # crossed_lift 플래그 갱신
        if gripper == 1 and tcp_z > self.z_lift:
            self._crossed_lift = True
        if gripper == 0:
            self._crossed_lift = False  # 다음 pick cycle을 위해 리셋

        # phase 결정 (우선순위 순)
        if self._trans_counter > 0 and self._trans_type == "close":
            phase = "grasp"
        elif self._trans_counter > 0 and self._trans_type == "open":
            phase = "release"
        elif gripper == 0 and self._released:
            phase = "return"
        elif gripper == 0:
            phase = "approach"
        elif gripper == 1 and tcp_z > self.z_lift:
            phase = "transport"
        elif gripper == 1 and self._crossed_lift:
            phase = "place"
        else:
            phase = "lift"

        if self._trans_counter > 0:
            self._trans_counter -= 1

        self._prev_gripper = gripper
        self._phase = phase
        return phase

    @property
    def phase(self) -> str:
        return self._phase


V13_PROMPTS: dict[str, list[str]] = {
    "left": [
        "pick up the orange box from the left side and place it on the right side",
        "move the orange box from the left to the right",
        "grasp the orange box on the left and put it down on the right",
    ],
    "right": [
        "pick up the orange box from the right side and place it on the left side",
        "move the orange box from the right to the left",
        "grasp the orange box on the right and put it down on the left",
    ],
}

TASK_PRESETS: dict[str, str] = {
    # v2 episode-level prompts (orange box, A↔B)
    "pick_from_left":  "pick up the orange box from the left side and place it on the right side",
    "pick_from_right": "pick up the orange box from the right side and place it on the left side",
    # v1 segment-level prompts (red block) — kept for backward compatibility
    "approach":    "approach red object",
    "pick":        "pick red object",
    "move_left":   "move object to left",
    "move_right":  "move object to right",
    "move_middle": "move object to middle",
    "place_left":  "place object to left",
    "place_right": "place object to right",
    "place_middle": "place object to middle",
    "return":      "return",
    "init_hold":   "init_hold",
}


class TaskNode(Node):

    def __init__(self):
        super().__init__("task_node")

        # 공통 파라미터
        self.declare_parameter("prompt_mode", "episode")  # "episode" | "per_frame" | "single"
        self.declare_parameter("task_sequence", "pick_from_left")
        self.declare_parameter("stage_timeout_sec", 0.0)
        self.declare_parameter("loop_sequence", False)

        # per_frame / single 모드 공용
        self.declare_parameter("source_side", "left")
        self.declare_parameter("target_side", "right")

        # per_frame 모드 전용 파라미터
        self.declare_parameter("z_lift", 180.0)
        self.declare_parameter("grip_threshold", 0.5)
        self.declare_parameter("phase_hz", 16.0)
        self.declare_parameter("return_z_done", 180.0)
        self.declare_parameter("return_done_steps", 5)

        # single 모드 전용
        self.declare_parameter("prompt_variant", -1)  # 0~2 고정 선택, -1이면 랜덤

        self._prompt_mode = self.get_parameter("prompt_mode").value
        seq_str = self.get_parameter("task_sequence").value
        self._timeout = self.get_parameter("stage_timeout_sec").value
        self._loop = self.get_parameter("loop_sequence").value

        self._seq = [s.strip() for s in seq_str.split(",") if s.strip()]
        self._idx = 0
        self._stage_start = self.get_clock().now()
        self._done = False

        # transient_local: 나중에 구독해도 최신값 받음
        qos = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        self._prompt_pub  = self.create_publisher(String, "/e6/task/prompt",  qos)
        self._status_pub  = self.create_publisher(String, "/e6/task/status",  10)

        # supervisor status 구독 (episode 모드에서만 stage 전환에 사용)
        self.create_subscription(String, "/e6/supervisor/status", self._cb_status, 10)

        if self._prompt_mode == "per_frame":
            self._source_side = self.get_parameter("source_side").value
            self._target_side = self.get_parameter("target_side").value
            self._return_z_done = self.get_parameter("return_z_done").value
            self._return_done_steps = self.get_parameter("return_done_steps").value

            self._phase_tracker = PhaseTracker(
                z_lift=self.get_parameter("z_lift").value,
                transition_frames=5,
                grip_threshold=self.get_parameter("grip_threshold").value,
            )
            self._latest_gripper: float = 0.0
            self._latest_tcp_z: float = 200.0
            self._return_streak: int = 0

            self.create_subscription(Float32MultiArray, "/e6/robot/state", self._cb_state,  10)
            self.create_subscription(Float32,           "/e6/robot/tcp_z", self._cb_tcpz,   10)

            phase_hz = self.get_parameter("phase_hz").value
            self.create_timer(1.0 / phase_hz, self._phase_tick)

            self.get_logger().info(
                f"task_node 시작 (per_frame) — "
                f"source={self._source_side} target={self._target_side} "
                f"z_lift={self.get_parameter('z_lift').value}mm "
                f"phase_hz={phase_hz}"
            )

        elif self._prompt_mode == "single":
            source = self.get_parameter("source_side").value
            variants = V13_PROMPTS.get(source)
            if variants is None:
                self.get_logger().error(
                    f"source_side='{source}' 는 'left' 또는 'right' 여야 합니다."
                )
                raise ValueError(f"invalid source_side: {source!r}")
            variant_idx = self.get_parameter("prompt_variant").value
            if variant_idx < 0 or variant_idx >= len(variants):
                chosen = random.choice(variants)
            else:
                chosen = variants[variant_idx]
            self._prompt_pub.publish(String(data=chosen))
            self.get_logger().info(
                f"task_node 시작 (single) — source={source} prompt={chosen!r}"
            )

        else:
            # episode 모드: 기존 stage-based 동작
            if self._timeout > 0:
                self.create_timer(0.5, self._check_timeout)
            self._publish_current()
            self.get_logger().info(
                f"task_node 시작 (episode) — sequence={self._seq} "
                f"timeout={self._timeout}s loop={self._loop}"
            )

    # ── 구독 콜백 (per_frame 모드) ────────────────────────────────────────────

    def _cb_state(self, msg: Float32MultiArray):
        d = msg.data
        if len(d) >= 7:
            self._latest_gripper = float(d[6])

    def _cb_tcpz(self, msg: Float32):
        self._latest_tcp_z = msg.data

    # ── per_frame phase 감지 타이머 ───────────────────────────────────────────

    def _phase_tick(self):
        if self._done:
            return

        phase = self._phase_tracker.update(self._latest_gripper, self._latest_tcp_z)

        # return 완료 감지 → TASK_COMPLETE
        if phase == "return" and self._latest_tcp_z >= self._return_z_done:
            self._return_streak += 1
            if self._return_streak >= self._return_done_steps:
                self.get_logger().info("=" * 60)
                self.get_logger().info("TASK_COMPLETE (return phase 완료)")
                self.get_logger().info("=" * 60)
                self._status_pub.publish(String(data="TASK_COMPLETE"))
                self._done = True
                return
        else:
            self._return_streak = 0

        prompt = V8_PHASE_PROMPTS[phase].format(
            source=self._source_side,
            target=self._target_side,
        )
        self._prompt_pub.publish(String(data=prompt))

    # ── supervisor status 콜백 ────────────────────────────────────────────────

    def _cb_status(self, msg: String):
        if self._done:
            return
        status = msg.data

        if status.startswith("STAGE_DONE:") and self._prompt_mode == "episode":
            self.get_logger().info(f"supervisor STAGE_DONE 수신: {status}")
            self._advance_stage()

        elif status == "TASK_COMPLETE" and self._prompt_mode == "single":
            # single 모드: executor가 B+C 종료를 감지 → /e6/supervisor/status 로 발행
            # → 여기서 /e6/task/status 로 중계 (MCAP 기록 + executor 수신 용)
            self.get_logger().info("=" * 60)
            self.get_logger().info("TASK_COMPLETE (executor 종료 감지)")
            self.get_logger().info("=" * 60)
            self._status_pub.publish(String(data="TASK_COMPLETE"))
            self._done = True

        elif status.startswith("FAIL_SAFETY"):
            self.get_logger().error(f"안전 이상 감지 — task 중단: {status}")
            self._status_pub.publish(String(data=status))
            self._done = True

    # ── timeout 체크 타이머 ───────────────────────────────────────────────────

    def _check_timeout(self):
        if self._done:
            return
        elapsed = (self.get_clock().now() - self._stage_start).nanoseconds / 1e9
        if elapsed >= self._timeout:
            self.get_logger().info(
                f"stage '{self._seq[self._idx]}' timeout ({elapsed:.1f}s >= {self._timeout}s)"
            )
            self._advance_stage()

    # ── stage 전환 ────────────────────────────────────────────────────────────

    def _advance_stage(self):
        self._idx += 1
        if self._idx >= len(self._seq):
            if self._loop:
                self._idx = 0
                self.get_logger().info("전체 sequence 완료 → 처음으로 루프")
            else:
                self.get_logger().info("=" * 60)
                self.get_logger().info("TASK_COMPLETE: 모든 stage 완료! 모션 정지")
                self.get_logger().info("=" * 60)
                self._status_pub.publish(String(data="TASK_COMPLETE"))
                self._done = True
                return
        self._stage_start = self.get_clock().now()
        self._publish_current()

    # ── 현재 stage 발행 ───────────────────────────────────────────────────────

    def _publish_current(self):
        if self._idx >= len(self._seq):
            return
        key = self._seq[self._idx]
        prompt = TASK_PRESETS.get(key, key)  # 프리셋에 없으면 key 자체를 prompt로
        self._prompt_pub.publish(String(data=prompt))
        self.get_logger().info(f"[stage {self._idx}/{len(self._seq)-1}] prompt={prompt!r}")


def main(args=None):
    rclpy.init(args=args)
    node = TaskNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
