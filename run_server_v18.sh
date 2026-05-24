#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  v18 정책 서버 실행 스크립트
#
#  state     : 7D [j1..j6, gripper] (dummy joint 없음)
#  action[0:6]: velocity delta (deg/frame)
#  action[6]  : absolute (0.0 or 1.0) → suction = ON if > 0.5
#  model      : pi05_e6_v17_lora 구조 동일, 20k steps 학습
#  norm_stats : 체크포인트 내 assets/ 자동 로딩
#               (Kyle-Riss/dobot_e6_pick_place_orange_v16)
#
#  사용법:
#    bash run_server_v18.sh [체크포인트경로]
# ══════════════════════════════════════════════════════════════════════════════

set -e

REPO="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_CKPT="/media/billye6/새 볼륨/e6_checkpoints/e6_v18_20k"
CHECKPOINT_DIR="${1:-$DEFAULT_CKPT}"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "[오류] 체크포인트 폴더 없음: $CHECKPOINT_DIR"
    exit 1
fi

source ~/move-one/min-imum/move-one/bin/activate

export MVCAM_COMMON_RUNENV=/opt/MVS/lib
export PYTHONPATH="$REPO/src:$PYTHONPATH"
export LD_LIBRARY_PATH="$HOME/DobotControl/min-imum/move-one/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH"
export TORCHDYNAMO_DISABLE=1

echo "=============================="
echo " v18 정책 서버"
echo " config : pi05_e6_v18_lora (vision LoRA 0~25 전체)"
echo " ckpt   : $CHECKPOINT_DIR"
echo " port   : 8000"
echo " state  : 7D (dummy joint 없음)"
echo " action : delta joints / absolute gripper"
echo " steps  : 20k (v17=15k 대비 +5k)"
echo " fps    : 16Hz"
echo "=============================="
echo ""

python "$REPO/scripts/serve_policy.py" \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_e6_v18_lora \
    --policy.dir "$CHECKPOINT_DIR"
