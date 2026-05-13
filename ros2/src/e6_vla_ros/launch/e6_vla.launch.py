import datetime
import os

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression


def generate_launch_description():
    return LaunchDescription([
        # ── 인자 선언 ──────────────────────────────────────────────────────
        DeclareLaunchArgument("robot_ip",       default_value="192.168.5.1"),
        DeclareLaunchArgument("server_host",    default_value="127.0.0.1"),
        DeclareLaunchArgument("server_port",    default_value="8000"),
        DeclareLaunchArgument("task_sequence",  default_value="pick_from_left"),
        DeclareLaunchArgument("stage_timeout_sec", default_value="0.0"),
        DeclareLaunchArgument("loop_sequence",  default_value="false"),
        DeclareLaunchArgument("dry_run",             default_value="false"),
        DeclareLaunchArgument("no_camera",           default_value="false"),
        DeclareLaunchArgument("max_delta_deg",       default_value="3.0"),
        DeclareLaunchArgument("min_tool_z",          default_value="101.0"),
        DeclareLaunchArgument("steps_per_inference", default_value="8"),
        DeclareLaunchArgument("executor_hz",         default_value="16.0"),
        DeclareLaunchArgument("approach_z_done",     default_value="85.0"),
        DeclareLaunchArgument("lift_z_done",         default_value="200.0"),
        DeclareLaunchArgument("stage_done_steps",    default_value="3"),
        DeclareLaunchArgument("save_debug_images",   default_value="false"),
        DeclareLaunchArgument("movj_velocity",       default_value="70"),
        DeclareLaunchArgument("movj_accel",          default_value="60"),
        DeclareLaunchArgument("record_mcap",         default_value="false"),
        DeclareLaunchArgument("mcap_output_dir",     default_value="/media/billye6/새 볼륨/Dobot/inference_mcap"),
        DeclareLaunchArgument("mcap_session_id",     default_value=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")),
        DeclareLaunchArgument("foxglove",            default_value="false"),
        DeclareLaunchArgument("foxglove_port",       default_value="8765"),
        DeclareLaunchArgument("action_mode",         default_value="absolute"),  # "absolute" (v6) | "delta" (v8/v13)
        DeclareLaunchArgument("action_scale",        default_value="1.0"),
        DeclareLaunchArgument("prompt_mode",         default_value="episode"),   # "episode" (v6) | "per_frame" (v8) | "single" (v13)
        DeclareLaunchArgument("source_side",         default_value="left"),      # v8 per_frame / v13 single 모드 필수
        DeclareLaunchArgument("target_side",         default_value="right"),
        DeclareLaunchArgument("prompt_variant",      default_value="-1"),        # v13: 0~2 고정, -1이면 랜덤
        DeclareLaunchArgument("max_steps",           default_value="500"),       # v13 종료 B: 최대 step
        DeclareLaunchArgument("min_steps",           default_value="100"),       # v13 종료 가드: 시작 후 최소 step
        DeclareLaunchArgument("home_tol_deg",        default_value="5.0"),       # v13 종료 C: j1..j3 허용 오차
        DeclareLaunchArgument("home_consec_req",     default_value="16"),        # v13 종료 C: 연속 만족 프레임 수
        DeclareLaunchArgument("vacuum_check_enabled",        default_value="false"),
        DeclareLaunchArgument("vacuum_check_z",              default_value="85.0"),
        DeclareLaunchArgument("vacuum_timeout_sec",          default_value="1.0"),
        DeclareLaunchArgument("place_force_release_enabled", default_value="false"),
        DeclareLaunchArgument("place_z_threshold",           default_value="120.0"),

        # ── 노드 1: camera_state_node ──────────────────────────────────────
        Node(
            package="e6_vla_ros",
            executable="camera_state_node",
            name="camera_state_node",
            output="screen",
            parameters=[{
                "robot_ip":  LaunchConfiguration("robot_ip"),
                "dry_run":   LaunchConfiguration("dry_run"),
                "no_camera": LaunchConfiguration("no_camera"),
            }],
        ),

        # ── 노드 2: inference_bridge_node ──────────────────────────────────
        Node(
            package="e6_vla_ros",
            executable="inference_bridge_node",
            name="inference_bridge_node",
            output="screen",
            parameters=[{
                "server_host":       LaunchConfiguration("server_host"),
                "server_port":       LaunchConfiguration("server_port"),
                "save_debug_images": LaunchConfiguration("save_debug_images"),
                "action_mode":       LaunchConfiguration("action_mode"),
            }],
            additional_env={
                "PYTHONPATH": "/home/billye6/E6-VLA_INFERENCE/packages/openpi-client/src"
                              + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
            },
        ),

        # ── 노드 3: executor_supervisor_node ───────────────────────────────
        Node(
            package="e6_vla_ros",
            executable="executor_supervisor_node",
            name="executor_supervisor_node",
            output="screen",
            parameters=[{
                "robot_ip":      LaunchConfiguration("robot_ip"),
                "dry_run":       LaunchConfiguration("dry_run"),
                "no_camera":     LaunchConfiguration("no_camera"),
                "max_delta_deg":       LaunchConfiguration("max_delta_deg"),
                "min_tool_z":          LaunchConfiguration("min_tool_z"),
                "steps_per_inference": LaunchConfiguration("steps_per_inference"),
                "executor_hz":         LaunchConfiguration("executor_hz"),
                "movj_velocity":       LaunchConfiguration("movj_velocity"),
                "movj_accel":          LaunchConfiguration("movj_accel"),
                "approach_z_done":     LaunchConfiguration("approach_z_done"),
                "lift_z_done":         LaunchConfiguration("lift_z_done"),
                "stage_done_steps":    LaunchConfiguration("stage_done_steps"),
                "action_mode":         LaunchConfiguration("action_mode"),
                "action_scale":        LaunchConfiguration("action_scale"),
                "vacuum_check_enabled":        LaunchConfiguration("vacuum_check_enabled"),
                "vacuum_check_z":              LaunchConfiguration("vacuum_check_z"),
                "vacuum_timeout_sec":          LaunchConfiguration("vacuum_timeout_sec"),
                "place_force_release_enabled": LaunchConfiguration("place_force_release_enabled"),
                "place_z_threshold":           LaunchConfiguration("place_z_threshold"),
                "max_steps":                   LaunchConfiguration("max_steps"),
                "min_steps":                   LaunchConfiguration("min_steps"),
                "home_tol_deg":                LaunchConfiguration("home_tol_deg"),
                "home_consec_req":             LaunchConfiguration("home_consec_req"),
            }],
        ),

        # ── 노드 4: task_node ──────────────────────────────────────────────
        Node(
            package="e6_vla_ros",
            executable="task_node",
            name="task_node",
            output="screen",
            parameters=[{
                "task_sequence":     LaunchConfiguration("task_sequence"),
                "stage_timeout_sec": LaunchConfiguration("stage_timeout_sec"),
                "loop_sequence":     LaunchConfiguration("loop_sequence"),
                "prompt_mode":       LaunchConfiguration("prompt_mode"),
                "source_side":       LaunchConfiguration("source_side"),
                "target_side":       LaunchConfiguration("target_side"),
                "prompt_variant":    LaunchConfiguration("prompt_variant"),
            }],
        ),

        # ── 노드 5: foxglove_bridge (foxglove:=true 일 때만 실행) ─────────────
        # 사용: ros2 launch e6_vla_ros e6_vla.launch.py foxglove:=true
        # Foxglove Studio에서 ws://<jetson-ip>:8765 로 연결하면 실시간 시각화
        Node(
            package="foxglove_bridge",
            executable="foxglove_bridge",
            name="foxglove_bridge",
            output="screen",
            parameters=[{
                "port": LaunchConfiguration("foxglove_port"),
                "address": "0.0.0.0",
                "tls": False,
                "topic_whitelist": [".*"],
                "send_buffer_limit": 10000000,
            }],
            condition=IfCondition(LaunchConfiguration("foxglove")),
        ),

        # ── MCAP 레코더 (record_mcap:=true 일 때만 실행) ───────────────────
        # 기록 토픽: 카메라 입력 2개 + 로봇 상태 + 프롬프트 + AI 출력 + 태스크 상태
        # 사용: ros2 launch e6_vla_ros e6_vla.launch.py record_mcap:=true
        # Foxglove Studio에서 .mcap 파일 열면 타임라인·카메라·관절값 동시 재생 가능
        ExecuteProcess(
            cmd=[
                "ros2", "bag", "record",
                "--storage", "mcap",
                "--output", PythonExpression([
                    "'", LaunchConfiguration("mcap_output_dir"), "/' + '",
                    LaunchConfiguration("mcap_session_id"), "'"
                ]),
                "/e6/camera/image",
                "/e6/camera/zed_image",
                "/e6/robot/state",
                "/e6/task/prompt",
                "/e6/policy/action_chunk",
                "/e6/task/status",
            ],
            output="screen",
            condition=IfCondition(LaunchConfiguration("record_mcap")),
        ),
    ])
