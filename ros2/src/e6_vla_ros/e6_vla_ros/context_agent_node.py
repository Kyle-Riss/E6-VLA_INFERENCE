#!/usr/bin/env python3
"""
context_agent_node — Claude API 기반 Context Agent (ablation 실험용, PhaseTracker 대체)

구독 토픽:
  /e6/robot/state   std_msgs/Float32MultiArray  [j1..j6 deg, gripper]
  /e6/robot/tcp_z   std_msgs/Float32

발행 토픽:
  /e6/task/prompt   std_msgs/String  (QoS: transient_local)

파라미터:
  source_side        (str,   default "left")                "left" | "right"
  target_side        (str,   default "right")               "right" | "left"
  grasp_z_max        (float, default 130.0)                 PhaseTracker 파라미터 (task_node와 동일하게)
  min_hold_frames    (int,   default 16)                    PhaseTracker 파라미터
  pick_prearm_z      (float, default 159.0)                 PhaseTracker 파라미터
  phase_hz           (float, default 16.0)                  phase 감지 주파수
  anthropic_api_key  (str,   default "")                    비어있으면 ANTHROPIC_API_KEY 환경변수 사용
  agent_model        (str,   default "claude-haiku-4-5-20251001")  API 호출 모델
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Float32MultiArray, Float32, String

# PhaseTracker는 task_node.py에서 공유 (동일 패키지 내)
from e6_vla_ros.task_node import PhaseTracker, V16_PHASE_PROMPTS, _V16_PHASE_KEY

# fallback prompts: API 호출 실패 시 PhaseTracker baseline과 동일한 고정 프롬프트 사용
_FALLBACK_PROMPTS = V16_PHASE_PROMPTS

_SYSTEM_PROMPT = """You are a robot instruction generator for a 6-DOF arm performing pick-and-place tasks.
Generate a SHORT (under 12 words), action-oriented instruction for the robot's current phase.
The instruction will be fed directly to a vision-language-action policy (π0.5).
Respond with ONLY the instruction text — no explanation, no punctuation at the end."""


def _build_user_message(phase: str, source_side: str, target_side: str,
                        tcp_z: float, gripper_raw: float) -> str:
    side_map = {
        "approach":  f"arm is descending toward the orange box on the {source_side}",
        "grasp":     f"arm is at grasping height (Z={tcp_z:.0f}mm), gripper closing",
        "lift":      f"arm is lifting the orange box upward (Z={tcp_z:.0f}mm)",
        "transport": f"arm is carrying the orange box toward the {target_side} (Z={tcp_z:.0f}mm)",
        "place":     f"arm is descending to place the orange box on the {target_side} (Z={tcp_z:.0f}mm)",
        "release":   f"arm is releasing the orange box on the {target_side}",
        "return":    f"arm is returning to home position after release (Z={tcp_z:.0f}mm)",
    }
    context = side_map.get(phase, f"phase={phase}, Z={tcp_z:.0f}mm")
    return (
        f"Current phase: {phase}\n"
        f"Context: {context}\n"
        f"Source side: {source_side}, Target side: {target_side}\n"
        f"Gripper: {gripper_raw:.2f}\n"
        f"Generate instruction:"
    )


class ContextAgentNode(Node):

    def __init__(self):
        super().__init__("context_agent_node")

        self.declare_parameter("source_side",       "left")
        self.declare_parameter("target_side",       "right")
        self.declare_parameter("grasp_z_max",       130.0)
        self.declare_parameter("min_hold_frames",   16)
        self.declare_parameter("pick_prearm_z",     159.0)
        self.declare_parameter("phase_hz",          16.0)
        self.declare_parameter("anthropic_api_key", "")
        self.declare_parameter("agent_model",       "claude-haiku-4-5-20251001")

        self._source  = self.get_parameter("source_side").value
        self._target  = self.get_parameter("target_side").value
        model_name    = self.get_parameter("agent_model").value
        api_key       = self.get_parameter("anthropic_api_key").value or os.environ.get("ANTHROPIC_API_KEY", "")

        self._tracker = PhaseTracker(
            grasp_z_max=self.get_parameter("grasp_z_max").value,
            min_hold_frames=self.get_parameter("min_hold_frames").value,
            pick_prearm_z=self.get_parameter("pick_prearm_z").value,
        )

        # Claude API 클라이언트 초기화 (anthropic 패키지)
        try:
            import anthropic  # type: ignore
            if not api_key:
                self.get_logger().warn("anthropic_api_key 미설정 — ANTHROPIC_API_KEY 환경변수 확인 필요")
            self._client = anthropic.Anthropic(api_key=api_key)
            self._model  = model_name
            self._api_ok = True
        except ImportError:
            self.get_logger().error("anthropic 패키지 미설치 → fallback(고정 프롬프트) 모드로 동작")
            self._client = None
            self._api_ok = False

        qos = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        self._prompt_pub = self.create_publisher(String, "/e6/task/prompt", qos)

        self.create_subscription(Float32MultiArray, "/e6/robot/state", self._cb_state, 10)
        self.create_subscription(Float32,           "/e6/robot/tcp_z", self._cb_tcp_z, 10)

        self._tcp_z:     float | None = None
        self._gripper:   float        = 0.0
        self._last_phase: str         = ""
        self._pending_api: bool       = False   # API 호출 진행 중 플래그

        self._executor_pool = ThreadPoolExecutor(max_workers=1)

        hz = self.get_parameter("phase_hz").value
        self.create_timer(1.0 / hz, self._tick)

        self.get_logger().info(
            f"context_agent_node 시작: source={self._source} target={self._target} "
            f"model={model_name} api_ok={self._api_ok}"
        )

    def _cb_state(self, msg: Float32MultiArray):
        if len(msg.data) >= 7:
            self._gripper = float(msg.data[6])

    def _cb_tcp_z(self, msg: Float32):
        self._tcp_z = float(msg.data)

    def _tick(self):
        if self._tcp_z is None:
            return

        phase = self._tracker.update(self._gripper, self._tcp_z)

        if phase == self._last_phase:
            return  # phase 변화 없음 → API 호출 불필요

        self._last_phase = phase
        self.get_logger().info(f"[PHASE] {phase} (z={self._tcp_z:.1f}mm grip={self._gripper:.2f})")

        # API 호출 중 중복 요청 방지
        if self._pending_api:
            self.get_logger().warn(f"[AGENT] 이전 API 호출 진행 중 — phase={phase} fallback 사용")
            self._publish_fallback(phase)
            return

        if self._api_ok:
            self._pending_api = True
            tcp_z_snap = self._tcp_z
            grip_snap  = self._gripper
            self._executor_pool.submit(self._call_api, phase, tcp_z_snap, grip_snap)
        else:
            self._publish_fallback(phase)

    def _call_api(self, phase: str, tcp_z: float, gripper_raw: float):
        try:
            user_msg = _build_user_message(phase, self._source, self._target, tcp_z, gripper_raw)
            response = self._client.messages.create(
                model=self._model,
                max_tokens=64,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = response.content[0].text.strip()
            self.get_logger().info(f"[AGENT] phase={phase} → \"{text}\"")
            self._publish_prompt(text)
        except Exception as e:
            self.get_logger().error(f"[AGENT] API 오류: {e} → fallback 사용")
            self._publish_fallback(phase)
        finally:
            self._pending_api = False

    def _publish_prompt(self, text: str):
        msg = String()
        msg.data = text
        self._prompt_pub.publish(msg)

    def _publish_fallback(self, phase: str):
        phase_key = _V16_PHASE_KEY.get(phase, "approach")
        text = _FALLBACK_PROMPTS.get(self._source, _FALLBACK_PROMPTS["left"]).get(
            phase_key, "perform the current task"
        )
        self.get_logger().info(f"[FALLBACK] phase={phase} → \"{text}\"")
        self._publish_prompt(text)

    def destroy_node(self):
        self._executor_pool.shutdown(wait=False)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ContextAgentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
