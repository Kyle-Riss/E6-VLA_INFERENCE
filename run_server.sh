#!/bin/bash
# 정책 서버 실행 스크립트 — pi05_e6_v1_lora (E6Inputs, 7D degree)
#
# 사용법:
#   bash ~/e6-vla/run_server.sh /path/to/checkpoint/XXXXX
#
# 예:
#   bash ~/e6-vla/run_server.sh /media/billy/새\ 볼륨/pi05_e6_v1_lora/35000

set -e

CHECKPOINT_DIR="${1:-}"
if [ -z "$CHECKPOINT_DIR" ]; then
  echo "[오류] 체크포인트 경로를 인자로 주세요."
  echo "  사용법: bash run_server.sh /path/to/checkpoint/XXXXX"
  exit 1
fi

if [ ! -d "$CHECKPOINT_DIR" ]; then
  echo "[오류] 체크포인트 폴더가 없습니다: $CHECKPOINT_DIR"
  exit 1
fi

REPO="$(cd "$(dirname "$0")" && pwd)"
source ~/move-one/min-imum/move-one/bin/activate

# HIKRobot MVS SDK (카메라 없어도 무방하지만 경로는 설정)
export MVCAM_COMMON_RUNENV=/opt/MVS/lib64

echo "=============================="
echo " e6-vla 정책 서버"
echo " config : pi05_e6_v1_lora"
echo " checkpoint : $CHECKPOINT_DIR"
echo " port : 8000"
echo "=============================="
echo ""

PYTHONPATH="$REPO/src" python "$REPO/scripts/serve_policy.py" \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_e6_v1_lora \
  --policy.dir "$CHECKPOINT_DIR"
