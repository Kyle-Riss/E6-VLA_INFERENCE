#!/usr/bin/env python3
"""
executor_supervisor_node — 액션 실행 + 안전 감시

구독 토픽:
  /e6/policy/action_chunk std_msgs/Float32MultiArray  (16*7 flatten)
  /e6/camera/image        sensor_msgs/Image           (bad_camera 감시용)
  /e6/robot/state         std_msgs/Float32MultiArray  [j1..j6 deg, gripper]
  /e6/robot/tcp_z         std_msgs/Float32

발행 토픽:
  /e6/supervisor/status   std_msgs/String  10Hz
    값: "RUNNING" | "STAGE_DONE:<stage>" | "FAIL_SAFETY:<reason>"

서비스:
  /e6/emergency_stop      std_srvs/Trigger

파라미터:
  robot_ip              (str,   default "192.168.5.1")
  dry_run               (bool,  default False)
  max_delta_deg         (float, default 3.0)
  min_tool_z            (float, default 101.0)   mm
  grip_close_threshold  (float, default 0.5)
  grip_open_threshold   (float, default 0.4)
  grip_close_latch_steps(int,   default 0)
  movj_velocity         (int,   default 70)
  movj_accel            (int,   default 60)
  chunk_staleness_sec   (float, default 5.0)
  steps_per_inference   (int,   default 8)    — 청크에서 실행할 스텝 수 (0=전체)
  executor_hz           (float, default 18.0) — MovJ 전송 주파수 (낮출수록 느려짐)
  approach_z_done       (float, default 85.0) — approach 완료: TCP Z ≤ 이 값 (mm)
  lift_z_done           (float, default 200.0)— pick/place 완료: TCP Z ≥ 이 값 (mm)
  stage_done_steps      (int,   default 3)    — 완료 조건 연속 만족 스텝 수
  bad_camera_consecutive(int,   default 10)
  camera_black_mean     (float, default 8.0)
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Float32, String
from std_srvs.srv import Trigger

# ── Dobot SDK 경로 ───────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[4]
_HARDWARE = _REPO / "hardware"
_DOBOT_SDK = _HARDWARE / "dobot"
for _p in [str(_HARDWARE), str(_DOBOT_SDK)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

ACTION_DIM = 7
ACTION_HORIZON = 16


class ExecutorSupervisorNode(Node):

    def __init__(self):
        super().__init__("executor_supervisor_node")

        # 파라미터
        self.declare_parameter("robot_ip", "192.168.5.1")
        self.declare_parameter("dry_run", False)
        self.declare_parameter("no_camera", False)
        self.declare_parameter("max_delta_deg", 3.0)
        self.declare_parameter("min_tool_z", 101.0)
        self.declare_parameter("grip_close_threshold", 0.5)
        self.declare_parameter("grip_open_threshold", 0.4)
        self.declare_parameter("grip_close_latch_steps", 0)
        self.declare_parameter("movj_velocity", 70)
        self.declare_parameter("movj_accel", 60)
        self.declare_parameter("chunk_staleness_sec", 5.0)
        self.declare_parameter("steps_per_inference", 8)
        self.declare_parameter("executor_hz", 18.0)
        self.declare_parameter("approach_z_done", 85.0)
        self.declare_parameter("lift_z_done", 200.0)
        self.declare_parameter("stage_done_steps", 3)
        self.declare_parameter("bad_camera_consecutive", 10)
        self.declare_parameter("camera_black_mean", 8.0)

        self._dry_run = self.get_parameter("dry_run").value
        self._no_camera = self.get_parameter("no_camera").value
        self._max_delta = self.get_parameter("max_delta_deg").value
        self._min_tool_z = self.get_parameter("min_tool_z").value
        self._grip_close = self.get_parameter("grip_close_threshold").value
        self._grip_open = self.get_parameter("grip_open_threshold").value
        self._grip_latch_steps = self.get_parameter("grip_close_latch_steps").value
        self._movj_v = self.get_parameter("movj_velocity").value
        self._movj_a = self.get_parameter("movj_accel").value
        self._staleness_sec = self.get_parameter("chunk_staleness_sec").value
        _spi = self.get_parameter("steps_per_inference").value
        self._steps_per_inference = _spi if _spi > 0 else ACTION_HORIZON
        _executor_hz = self.get_parameter("executor_hz").value
        self._approach_z_done = self.get_parameter("approach_z_done").value
        self._lift_z_done = self.get_parameter("lift_z_done").value
        self._stage_done_steps = self.get_parameter("stage_done_steps").value
        self._bad_cam_limit = self.get_parameter("bad_camera_consecutive").value
        self._black_mean = self.get_parameter("camera_black_mean").value

        # 청크 상태
        self._chunk: np.ndarray | None = None       # (16, 7)
        self._chunk_idx = 0
        self._chunk_t = 0.0                          # 수신 시각
        self._chunk_lock = threading.Lock()

        # 로봇 상태 (camera_state_node에서 구독)
        self._current_deg = np.zeros(6, dtype=np.float32)
        self._tcp_z: float | None = None
        self._last_gripper = 0
        self._grip_latch_remaining = 0

        # 안전 감시 상태
        self._bad_streak = 0
        self._emergency_stop = False
        self._status = "RUNNING"

        # stage 완료 감지
        self._current_stage = "approach"   # 현재 stage (prompt로 갱신)
        self._done_streak = 0              # 완료 조건 연속 만족 카운트
        self._stage_done_published = False # 같은 stage에서 중복 발행 방지

        # task 완료 플래그
        self._task_complete = False

        # 구독
        self.create_subscription(Float32MultiArray, "/e6/policy/action_chunk", self._cb_chunk,       10)
        self.create_subscription(Image,             "/e6/camera/image",         self._cb_img,         10)
        self.create_subscription(Float32MultiArray, "/e6/robot/state",          self._cb_state,       10)
        self.create_subscription(Float32,           "/e6/robot/tcp_z",          self._cb_tcpz,        10)
        self.create_subscription(String,            "/e6/task/prompt",          self._cb_prompt,      10)
        self.create_subscription(String,            "/e6/task/status",          self._cb_task_status, 10)

        # 발행
        self._status_pub = self.create_publisher(String, "/e6/supervisor/status", 10)

        # 서비스
        self.create_service(Trigger, "/e6/emergency_stop", self._cb_estop)

        # 로봇 연결
        self._dashboard = None
        robot_ip = self.get_parameter("robot_ip").value
        if not self._dry_run:
            self._init_robot(robot_ip)

        # 타이머
        self.create_timer(1.0 / _executor_hz, self._executor_tick)
        self.create_timer(0.10,  self._supervisor_tick) # 10Hz

        self.get_logger().info(
            f"executor_supervisor_node 시작 — "
            f"robot={'연결됨' if self._dashboard else 'dry_run'} "
            f"executor_hz={_executor_hz} max_delta={self._max_delta}° "
            f"min_tool_z={self._min_tool_z}mm steps_per_inference={self._steps_per_inference}/{ACTION_HORIZON}"
        )

    # ── 초기화 ──────────────────────────────────────────────────────────────

    # e6_v1 학습 데이터 기준 초기 자세 (degree)
    INIT_POSE_DEG = [90.128, 42.907, 59.355, -11.702, -87.582, 177.813]

    def _init_robot(self, robot_ip: str):
        try:
            from dobot_api import DobotApiDashboard  # type: ignore
            self._dashboard = DobotApiDashboard(robot_ip, 29999)
            self._dashboard.EnableRobot()
            time.sleep(0.5)
            j1, j2, j3, j4, j5, j6 = self.INIT_POSE_DEG
            self._dashboard.MovJ(j1, j2, j3, j4, j5, j6, 1, v=50, a=40)
            self.get_logger().info(f"초기 자세 이동 중: {self.INIT_POSE_DEG}")
            time.sleep(3.0)
            self.get_logger().info(f"Dobot dashboard 연결: {robot_ip}:29999")
        except Exception as exc:
            self.get_logger().warn(f"Dobot 연결 실패 ({exc}) → dry_run 모드")
            self._dashboard = None

    # ── 구독 콜백 ────────────────────────────────────────────────────────────

    def _cb_chunk(self, msg: Float32MultiArray):
        data = np.array(msg.data, dtype=np.float32)
        if data.size != ACTION_HORIZON * ACTION_DIM:
            self.get_logger().warn(f"chunk 크기 이상: {data.size}")
            return
        with self._chunk_lock:
            self._chunk = data.reshape(ACTION_HORIZON, ACTION_DIM)[:self._steps_per_inference]
            self._chunk_idx = 0
            self._chunk_t = time.monotonic()

    def _cb_img(self, msg: Image):
        frame = np.frombuffer(msg.data, dtype=np.uint8)
        self._frame_mean = float(frame.mean()) if frame.size else 255.0

    def _cb_state(self, msg: Float32MultiArray):
        d = np.array(msg.data, dtype=np.float32)
        if d.size >= 6:
            self._current_deg = d[:6]

    def _cb_tcpz(self, msg: Float32):
        self._tcp_z = msg.data

    def _cb_task_status(self, msg: String):
        if msg.data == "TASK_COMPLETE" and not self._task_complete:
            self._task_complete = True
            self.get_logger().info("TASK_COMPLETE 수신 → 모션 정지")
            with self._chunk_lock:
                self._chunk = None
                self._chunk_idx = 0

    def _cb_prompt(self, msg: String):
        prompt = msg.data
        # prompt → stage 이름 추출
        if "approach" in prompt:
            stage = "approach"
        elif "pick" in prompt:
            stage = "pick"
        elif "move" in prompt:
            stage = "move"
        elif "place" in prompt:
            stage = "place"
        else:
            stage = prompt
        if stage != self._current_stage:
            self.get_logger().info(f"stage 변경: {self._current_stage} → {stage}")
            self._current_stage = stage
            self._done_streak = 0
            self._stage_done_published = False

    # ── 20Hz 실행 루프 ───────────────────────────────────────────────────────

    def _executor_tick(self):
        if self._emergency_stop or self._task_complete:
            return

        with self._chunk_lock:
            chunk = self._chunk
            idx = self._chunk_idx
            chunk_t = self._chunk_t

        if chunk is None:
            return

        # staleness 체크
        if time.monotonic() - chunk_t > self._staleness_sec:
            with self._chunk_lock:
                self._chunk = None
            self.get_logger().warn("chunk staleness 초과 → 폐기")
            return

        if idx >= len(chunk):
            return  # 청크 소진, 다음 chunk 대기

        a = chunk[idx]  # (7,)

        # 절대 목표 관절각 (e6_v1: action = 절대 degree next-position)
        target_deg = np.asarray(a[:6], dtype=np.float32)

        # 그리퍼 hysteresis
        grip_raw = float(a[6]) if len(a) > 6 else 0.0
        if grip_raw >= self._grip_close:
            hys = 1
        elif grip_raw <= self._grip_open:
            hys = 0
        else:
            hys = self._last_gripper

        if self._grip_latch_steps > 0:
            if hys == 1:
                self._grip_latch_remaining = max(self._grip_latch_remaining, self._grip_latch_steps)
            if self._grip_latch_remaining > 0:
                hys = 1
                self._grip_latch_remaining -= 1

        tool_on = int(hys)

        # bad camera → MovJ 스킵 (현재 위치 유지)
        camera_hold = hasattr(self, "_frame_mean") and self._frame_mean < self._black_mean

        # 로봇 명령 전송
        if self._dashboard is not None:
            try:
                if not camera_hold:
                    j1, j2, j3, j4, j5, j6 = (float(x) for x in target_deg)
                    self._dashboard.MovJ(j1, j2, j3, j4, j5, j6, 1,
                                         v=self._movj_v, a=self._movj_a)
                self._dashboard.ToolDO(1, tool_on)
            except Exception as exc:
                self.get_logger().warn(f"MovJ 실패: {exc}", throttle_duration_sec=2.0)

        self._last_gripper = tool_on
        with self._chunk_lock:
            self._chunk_idx += 1

    # ── 10Hz 감시 루프 ───────────────────────────────────────────────────────

    def _supervisor_tick(self):
        if self._emergency_stop:
            self._status_pub.publish(String(data="FAIL_SAFETY:emergency_stop"))
            return

        status = "RUNNING"

        # bad camera 스트릭 (no_camera 또는 dry_run이면 스킵)
        if not self._no_camera and not self._dry_run:
            frame_mean = getattr(self, "_frame_mean", 255.0)
            if frame_mean < self._black_mean:
                self._bad_streak += 1
                if self._bad_streak > self._bad_cam_limit:
                    status = "FAIL_SAFETY:bad_camera"
                    self.get_logger().warn(f"bad_camera streak={self._bad_streak}")
            else:
                self._bad_streak = 0

        # min_tool_z 안전 한계
        if (not self._dry_run
                and self._dashboard is not None
                and self._tcp_z is not None
                and self._tcp_z <= self._min_tool_z):
            status = f"FAIL_SAFETY:min_tool_z({self._tcp_z:.1f}mm)"
            self.get_logger().warn(f"Tool Z={self._tcp_z:.1f}mm ≤ {self._min_tool_z:.1f}mm")

        # STAGE_DONE 판정 (FAIL이 없을 때만)
        if status == "RUNNING" and not self._stage_done_published and self._tcp_z is not None:
            done = False
            stage = self._current_stage
            z = self._tcp_z
            g = self._last_gripper

            if stage == "approach":
                done = z <= self._approach_z_done
            elif stage == "pick":
                done = z >= self._lift_z_done and g == 1
            elif stage == "place":
                done = z >= self._lift_z_done and g == 0
            # move: timeout에 맡김 (센서 조건 없음)

            if done:
                self._done_streak += 1
                if self._done_streak >= self._stage_done_steps:
                    status = f"STAGE_DONE:{stage}"
                    self._stage_done_published = True
                    self.get_logger().info(
                        f"[STAGE_DONE] {stage} | TCP_Z={z:.1f}mm gripper={g}"
                    )
            else:
                self._done_streak = 0

        self._status = status
        self._status_pub.publish(String(data=status))

    # ── 긴급 정지 서비스 ─────────────────────────────────────────────────────

    def _cb_estop(self, request, response):
        self._emergency_stop = True
        self.get_logger().error("긴급 정지 호출!")
        if self._dashboard is not None:
            try:
                self._dashboard.EmergencyStop(0)
            except Exception:
                pass
        response.success = True
        response.message = "emergency stop activated"
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ExecutorSupervisorNode()
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
