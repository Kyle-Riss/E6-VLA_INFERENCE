#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  v14 ROS2 추론 클라이언트 실행 스크립트
#
#  사용법:
#    bash run_ros_v14.sh [left|right]   # left = A section, right = B section
#
#  예시:
#    bash run_ros_v14.sh left            # left→right 집기
#    bash run_ros_v14.sh right           # right→left 집기
#    bash run_ros_v14.sh left record_mcap:=true foxglove:=true
# ══════════════════════════════════════════════════════════════════════════════

set -e

SECTION="${1:-left}"
shift 2>/dev/null || true

if [ "$SECTION" = "left" ]; then
    SOURCE_SIDE="left"
elif [ "$SECTION" = "right" ]; then
    SOURCE_SIDE="right"
else
    echo "[오류] 섹션은 'left' 또는 'right'만 허용됩니다."
    echo "  사용법: bash run_ros_v14.sh [left|right]"
    exit 1
fi

REPO="$(cd "$(dirname "$0")" && pwd)"
ROS2_WS="$REPO/ros2"

source /opt/ros/humble/setup.bash
source "$ROS2_WS/install/setup.bash"

echo "=============================="
echo " v14 ROS2 추론 클라이언트"
echo " section     : $SECTION section"
echo " source_side : $SOURCE_SIDE"
echo " action_mode : delta (velocity deg/frame)"
echo " prompt_mode : single / dataset=v14 (anchor 고정)"
echo " gripper     : Δ 누산, ±0.5 threshold"
echo " fps         : 16Hz | steps_per_inference: 8"
echo "=============================="
echo ""

ros2 launch e6_vla_ros e6_vla.launch.py \
    action_mode:=delta \
    prompt_mode:=single \
    prompt_dataset:=v14 \
    source_side:="$SOURCE_SIDE" \
    steps_per_inference:=8 \
    "$@"
