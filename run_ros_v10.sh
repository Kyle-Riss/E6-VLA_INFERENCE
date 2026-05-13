#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  v10 ROS2 추론 클라이언트 실행 스크립트
#
#  사용법:
#    bash run_ros_v10.sh [left|right]   # left = A section, right = B section
#
#  예시:
#    bash run_ros_v10.sh left    # A section: left→right 집기
#    bash run_ros_v10.sh right   # B section: right→left 집기
#
#  옵션 (뒤에 ros2 launch 인자 추가 가능):
#    bash run_ros_v10.sh left record_mcap:=true foxglove:=true
# ══════════════════════════════════════════════════════════════════════════════

set -e

SECTION="${1:-left}"
shift 2>/dev/null || true   # 첫 인자 제거, 나머지는 launch에 전달

if [ "$SECTION" = "left" ]; then
    SOURCE_SIDE="left"
    TARGET_SIDE="right"
elif [ "$SECTION" = "right" ]; then
    SOURCE_SIDE="right"
    TARGET_SIDE="left"
else
    echo "[오류] 섹션은 'left' 또는 'right'만 허용됩니다."
    echo "  사용법: bash run_ros_v10.sh [left|right]"
    exit 1
fi

REPO="$(cd "$(dirname "$0")" && pwd)"
ROS2_WS="$REPO/ros2"

# ── 환경 설정 ─────────────────────────────────────────────────────────────────
source /opt/ros/humble/setup.bash
source "$ROS2_WS/install/setup.bash"

echo "=============================="
echo " v10 ROS2 추론 클라이언트"
echo " section    : $SECTION section"
echo " source_side: $SOURCE_SIDE → target_side: $TARGET_SIDE"
echo " action_mode: delta (velocity deg/frame)"
echo " prompt_mode: per_frame (7-phase)"
echo " fps        : 16Hz | Z_LIFT: 180mm"
echo " j3 시작    : 53.8° (A/B 공통)"
echo "=============================="
echo ""

ros2 launch e6_vla_ros e6_vla.launch.py \
    action_mode:=delta \
    prompt_mode:=per_frame \
    source_side:="$SOURCE_SIDE" \
    target_side:="$TARGET_SIDE" \
    steps_per_inference:=16 \
    action_scale:=1.2 \
    "$@"
