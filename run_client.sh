#!/bin/bash
# 로봇 제어 클라이언트 실행 스크립트 — pi05_e6_v1_lora (E6Inputs, 7D degree)
#
# 사용법:
#   bash ~/e6-vla/run_client.sh [--prompt "approach red object"] [추가 인자...]
#
# 예) 기본 실행:
#   bash ~/e6-vla/run_client.sh
#
# 예) 프롬프트 지정:
#   bash ~/e6-vla/run_client.sh --prompt "pick red object"
#
# 예) approach↔return 자동 순환:
#   bash ~/e6-vla/run_client.sh \
#     --auto_cycle_return_approach \
#     --approach_prompt "approach red object" \
#     --return_prompt "return" \
#     --approach_cycle_sec 8 \
#     --return_cycle_sec 2
#
# 예) 파이프라인 테스트 (로봇·카메라 없이):
#   bash ~/e6-vla/run_client.sh --dry_run --no_camera --no_init_pose --max_runtime_sec 10
#
# 인자 기본값 (아래에서 수정 가능)
#   --hz 20 --steps_per_inference 5 --max_delta_deg 3.0
#   --movj_velocity 70 --movj_accel 60

REPO="$(cd "$(dirname "$0")" && pwd)"
source ~/move-one/min-imum/move-one/bin/activate

export MVCAM_COMMON_RUNENV=/opt/MVS/lib64

echo "=============================="
echo " e6-vla 로봇 클라이언트"
echo " layout : e6_v1 (state=deg 7D)"
echo " server : 127.0.0.1:8000"
echo " robot  : 192.168.5.1"
echo "=============================="
echo ""

python "$REPO/examples/e6/run_e6_client.py" \
  --server_host 127.0.0.1 \
  --server_port 8000 \
  --robot_ip 192.168.5.1 \
  --input_layout e6_v1 \
  --prompt "approach red object" \
  --hz 20 \
  --steps_per_inference 5 \
  --max_delta_deg 3.0 \
  --movj_velocity 70 \
  --movj_accel 60 \
  "$@"
