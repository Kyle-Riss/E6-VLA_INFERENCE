#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  v12 정책 서버 실행 스크립트
#
#  체크포인트: /media/billye6/새 볼륨/e6_checkpoints/e6_v12_30k
#  action expert: gemma_300m_lora_r16 (rank=16, scope 전체 18 layer)
#  norm stats: Kyle-Riss/dobot_e6_pick_place_orange_v10 (gripper 패치 적용 완료)
#
#  사용법:
#    bash run_server_v12.sh [체크포인트경로]
#
#  체크포인트 경로 생략 시 기본값 사용.
# ══════════════════════════════════════════════════════════════════════════════

set -e

DEFAULT_CKPT="/media/billye6/새 볼륨/e6_checkpoints/e6_v12_30k"
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
export LD_LIBRARY_PATH="$HOME/DobotControl/min-imum/move-one/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH"
export TORCHDYNAMO_DISABLE=1

echo "=============================="
echo " v12 정책 서버"
echo " config : pi05_e6_v12_lora"
echo " ckpt   : $CHECKPOINT_DIR"
echo " port   : 8000"
echo " action : velocity delta (deg/frame)"
echo " prompt : per_frame 7-phase"
echo " fps    : 16Hz"
echo " rank   : 16 (전체 18 layer)"
echo "=============================="
echo ""

python "$REPO/scripts/serve_policy.py" \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_e6_v12_lora \
    --policy.dir "$CHECKPOINT_DIR"
