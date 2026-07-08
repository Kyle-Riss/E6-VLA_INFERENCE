#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  v10 정책 서버 실행 스크립트
#
#  체크포인트 (학습 서버 기준):
#    checkpoints/pi05_e6_v10_lora/e6_2cam_lora_v10/49999_pytorch_lora_merged
#
#  norm stats (수동 패치 완료, gripper q01=-1.0 / q99=+1.0):
#    assets/pi05_e6_v10_lora/Kyle-Riss/dobot_e6_pick_place_orange_v10/norm_stats.json
#
#  사용법:
#    bash run_server_v10.sh [체크포인트경로]
#
#  예시 (HDD에 복사된 경우):
#    bash run_server_v10.sh "/media/billye6/새 볼륨1/e6_checkpoints/e6_v10_50k"
#
#  체크포인트 경로 생략 시 기본값 사용.
# ══════════════════════════════════════════════════════════════════════════════

set -e

DEFAULT_CKPT="/media/billye6/새 볼륨1/e6_checkpoints/e6_v10_50k"
CHECKPOINT_DIR="${1:-$DEFAULT_CKPT}"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "[오류] 체크포인트 폴더 없음: $CHECKPOINT_DIR"
    echo "  HDD 마운트 확인 또는 경로를 인자로 직접 지정하세요."
    exit 1
fi

REPO="$(cd "$(dirname "$0")" && pwd)"
source ~/move-one/min-imum/move-one/bin/activate

export MVCAM_COMMON_RUNENV=/opt/MVS/lib
export PYTHONPATH="$REPO/src:$PYTHONPATH"
_NV="$HOME/move-one/min-imum/move-one/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="$_NV/cusparselt/lib:$_NV/nccl/lib:$_NV/nvshmem/lib:$_NV/cu12/lib:$LD_LIBRARY_PATH"
export TORCHDYNAMO_DISABLE=1

echo "=============================="
echo " v10 정책 서버"
echo " config : pi05_e6_v10_lora"
echo " ckpt   : $CHECKPOINT_DIR"
echo " port   : 8000"
echo " action : velocity delta (deg/frame)"
echo " prompt : per_frame 7-phase"
echo " fps    : 16Hz"
echo "=============================="
echo ""

python "$REPO/scripts/serve_policy.py" \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_e6_v10_lora \
    --policy.dir "$CHECKPOINT_DIR"
