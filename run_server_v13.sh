#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  v13 정책 서버 실행 스크립트
#
#  체크포인트: /media/billy/새 볼륨2/e6_v13_22k  (22,500 step)
#  action expert: gemma_300m_lora_r16 (rank=16, scope 전체 18 layer)
#  norm stats: assets/pi05_e6_v13_lora/ 자동 로딩 (config 이름 기준)
#
#  사용법:
#    bash run_server_v13.sh [체크포인트경로]
#
#  체크포인트 경로 생략 시 기본값 사용.
# ══════════════════════════════════════════════════════════════════════════════

set -e

DEFAULT_CKPT="/media/billy/새 볼륨2/e6_v13_22k"
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
echo " v13 정책 서버"
echo " config : pi05_e6_v13_lora"
echo " ckpt   : $CHECKPOINT_DIR"
echo " port   : 8000"
echo " action : velocity delta (deg/frame)"
echo " prompt : single (고정 1개, episode 전체 유지)"
echo " fps    : 16Hz"
echo " rank   : 16 (전체 18 layer)"
echo "=============================="
echo ""

python "$REPO/scripts/serve_policy.py" \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_e6_v13_lora \
    --policy.dir "$CHECKPOINT_DIR"
