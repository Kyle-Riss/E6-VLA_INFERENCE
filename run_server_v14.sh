#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  v14 정책 서버 실행 스크립트
#
#  체크포인트: checkpoints/pi05_e6_v14_lora/e6_2cam_lora_v14/22500_pytorch
#  state     : 8D (dummy index 6 자동 삽입 — 서버 내부 처리)
#  action    : velocity delta (deg/frame), gripper Δ (q01=-1 / q99=+1)
#  norm_stats: 자동 로딩 (패치 불필요)
#
#  사용법:
#    bash run_server_v14.sh [체크포인트경로]
#
#  체크포인트 경로 생략 시 기본값 사용.
# ══════════════════════════════════════════════════════════════════════════════

set -e

REPO="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_CKPT="/media/billye6/새 볼륨/e6_checkpoints/e6_v14_22500"
CHECKPOINT_DIR="${1:-$DEFAULT_CKPT}"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "[오류] 체크포인트 폴더 없음: $CHECKPOINT_DIR"
    echo "  경로를 인자로 직접 지정하세요."
    exit 1
fi

source ~/move-one/min-imum/move-one/bin/activate

export MVCAM_COMMON_RUNENV=/opt/MVS/lib
export PYTHONPATH="$REPO/src:$PYTHONPATH"
export LD_LIBRARY_PATH="$HOME/DobotControl/min-imum/move-one/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH"
export TORCHDYNAMO_DISABLE=1

echo "=============================="
echo " v14 정책 서버"
echo " config : pi05_e6_v14_lora"
echo " ckpt   : $CHECKPOINT_DIR (HDD 직접 참조)"
echo " port   : 8000"
echo " state  : 8D (dummy index 6 자동 삽입)"
echo " action : velocity delta (deg/frame)"
echo " gripper: Δ 누산, ±0.5 threshold"
echo " prompt : single anchor (2종)"
echo " fps    : 16Hz"
echo "=============================="
echo ""

python "$REPO/scripts/serve_policy.py" \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_e6_v14_lora \
    --policy.dir "$CHECKPOINT_DIR"
