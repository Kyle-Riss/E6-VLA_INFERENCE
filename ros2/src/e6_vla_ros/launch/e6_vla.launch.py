from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        # ── 인자 선언 ──────────────────────────────────────────────────────
        DeclareLaunchArgument("robot_ip",       default_value="192.168.5.1"),
        DeclareLaunchArgument("server_host",    default_value="127.0.0.1"),
        DeclareLaunchArgument("server_port",    default_value="8000"),
        DeclareLaunchArgument("task_sequence",  default_value="approach"),
        DeclareLaunchArgument("stage_timeout_sec", default_value="0.0"),
        DeclareLaunchArgument("loop_sequence",  default_value="false"),
        DeclareLaunchArgument("dry_run",             default_value="false"),
        DeclareLaunchArgument("no_camera",           default_value="false"),
        DeclareLaunchArgument("max_delta_deg",       default_value="3.0"),
        DeclareLaunchArgument("min_tool_z",          default_value="101.0"),
        DeclareLaunchArgument("steps_per_inference", default_value="8"),
        DeclareLaunchArgument("executor_hz",         default_value="10.0"),
        DeclareLaunchArgument("approach_z_done",     default_value="85.0"),
        DeclareLaunchArgument("lift_z_done",         default_value="200.0"),
        DeclareLaunchArgument("stage_done_steps",    default_value="3"),
        DeclareLaunchArgument("save_debug_images",   default_value="false"),
        DeclareLaunchArgument("movj_velocity",       default_value="70"),
        DeclareLaunchArgument("movj_accel",          default_value="60"),
        DeclareLaunchArgument("record_mcap",         default_value="false"),
        DeclareLaunchArgument("mcap_output_dir",     default_value="/media/billye6/새 볼륨/Dobot/inference_mcap"),

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
            }],
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
            }],
        ),

        # ── 노드 4: task_node ──────────────────────────────────────────────
        Node(
            package="e6_vla_ros",
            executable="task_node",
            name="task_node",
            output="screen",
            parameters=[{
                "task_sequence":    LaunchConfiguration("task_sequence"),
                "stage_timeout_sec": LaunchConfiguration("stage_timeout_sec"),
                "loop_sequence":    LaunchConfiguration("loop_sequence"),
            }],
        ),

        # ── MCAP 레코더 (record_mcap:=true 일 때만 실행) ───────────────────
        # 기록 토픽: 카메라 입력 2개 + 로봇 상태 + 프롬프트 + AI 출력 + 태스크 상태
        # 사용: ros2 launch e6_vla_ros e6_vla.launch.py record_mcap:=true
        # Foxglove Studio에서 .mcap 파일 열면 타임라인·카메라·관절값 동시 재생 가능
        ExecuteProcess(
            cmd=[
                "ros2", "bag", "record",
                "--storage", "mcap",
                "--output", LaunchConfiguration("mcap_output_dir"),
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
