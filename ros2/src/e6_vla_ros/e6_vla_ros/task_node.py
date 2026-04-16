#!/usr/bin/env python3
"""
task_node — task_sequence 상태머신. 하드웨어 I/O 없음.

구독 토픽:
  /e6/supervisor/status   std_msgs/String

발행 토픽:
  /e6/task/prompt         std_msgs/String  (QoS: transient_local)

파라미터:
  task_sequence  (str,   default "approach")
    쉼표 구분 stage 키 목록. 예: "approach,pick,move_left,place_left"
  stage_timeout_sec (float, default 0.0)
    > 0 이면 timeout 후 자동으로 다음 stage 전환 (0=수동/supervisor_done 전환)
  loop_sequence  (bool,  default False)
    True이면 마지막 stage 완료 후 처음으로 되돌아감
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String

TASK_PRESETS: dict[str, str] = {
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

        # 파라미터
        self.declare_parameter("task_sequence", "approach")
        self.declare_parameter("stage_timeout_sec", 0.0)
        self.declare_parameter("loop_sequence", False)

        seq_str = self.get_parameter("task_sequence").value
        self._timeout = self.get_parameter("stage_timeout_sec").value
        self._loop = self.get_parameter("loop_sequence").value

        self._seq = [s.strip() for s in seq_str.split(",") if s.strip()]
        self._idx = 0
        self._stage_start = self.get_clock().now()
        self._done = False

        # transient_local: 나중에 구독해도 최신값 받음
        qos = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        self._prompt_pub = self.create_publisher(String, "/e6/task/prompt", qos)

        # supervisor status 구독
        self.create_subscription(String, "/e6/supervisor/status", self._cb_status, 10)

        # stage_timeout 타이머 (0이면 무시)
        if self._timeout > 0:
            self.create_timer(0.5, self._check_timeout)

        self._publish_current()
        self.get_logger().info(
            f"task_node 시작 — sequence={self._seq} "
            f"timeout={self._timeout}s loop={self._loop}"
        )

    # ── supervisor status 콜백 ────────────────────────────────────────────────

    def _cb_status(self, msg: String):
        if self._done:
            return
        status = msg.data

        if status.startswith("STAGE_DONE:"):
            self.get_logger().info(f"supervisor STAGE_DONE 수신: {status}")
            self._advance_stage()

        elif status.startswith("FAIL_SAFETY"):
            self.get_logger().error(f"안전 이상 감지 — task 중단: {status}")
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
                self.get_logger().info("전체 task_sequence 완료")
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
