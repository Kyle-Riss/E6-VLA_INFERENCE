from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
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
                "server_host": LaunchConfiguration("server_host"),
                "server_port": LaunchConfiguration("server_port"),
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
    ])
