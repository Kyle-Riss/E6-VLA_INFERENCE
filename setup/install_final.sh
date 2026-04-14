#!/bin/bash
# Jetson AGX Orin 전용 최종 설치 스크립트
# e6-vla 가상환경 기준: ~/e6-vla/.venv/
#
# 사용법:
#   cd ~/e6-vla
#   bash setup/install_final.sh
#
# 가상환경이 없으면 먼저 생성:
#   python3 -m venv .venv
#   source .venv/bin/activate
#   pip install uv
# 또는 move-one의 기존 venv를 그대로 사용해도 됨:
#   source ~/move-one/min-imum/move-one/bin/activate

set -e

# 현재 활성화된 venv 확인
if [ -z "$VIRTUAL_ENV" ]; then
  echo "[경고] 가상환경이 활성화되지 않았습니다."
  echo "  move-one venv 재사용: source ~/move-one/min-imum/move-one/bin/activate"
  echo "  또는 e6-vla 전용 venv: source ~/e6-vla/.venv/bin/activate"
  exit 1
fi

echo "가상환경: $VIRTUAL_ENV"
echo ""

echo "1. Installing PyTorch (Jetson aarch64) ..."
# WHL 파일이 현재 디렉터리 또는 setup/ 에 있어야 함
WHL="torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl"
if [ -f "$WHL" ]; then
  uv pip install "$WHL" --force-reinstall --no-deps
elif [ -f "setup/$WHL" ]; then
  uv pip install "setup/$WHL" --force-reinstall --no-deps
else
  echo "[건너뜀] $WHL 파일이 없습니다. PyTorch는 이미 설치되어 있다고 가정합니다."
fi

echo "2. Ensuring NumPy compatibility ..."
uv pip install "numpy<2,>=1.26"

echo "3. Final Verification ..."
python -c "import torch; print(f'Torch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
echo "완료."
