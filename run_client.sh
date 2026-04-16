#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  e6-vla 로봇 클라이언트 — pi05_e6_v1_lora (E6Inputs, 7D degree)
#
#  V1 공식 프롬프트 (학습 데이터 포함, 이 문자열만 사용할 것):
#    "approach red object"
#    "pick red object"
#    "move object to left"   "move object to right"   "move object to middle"
#    "place object to left"  "place object to right"  "place object to middle"
#
# ── 단일 stage 실행 ───────────────────────────────────────────────────────────
#
#   bash ~/e6-vla/run_client.sh --no_init_pose --prompt "approach red object"
#   bash ~/e6-vla/run_client.sh --no_init_pose --prompt "pick red object"
#   bash ~/e6-vla/run_client.sh --no_init_pose --prompt "move object to left"
#   bash ~/e6-vla/run_client.sh --no_init_pose --prompt "move object to right"
#   bash ~/e6-vla/run_client.sh --no_init_pose --prompt "move object to middle"
#   bash ~/e6-vla/run_client.sh --no_init_pose --prompt "place object to left"
#   bash ~/e6-vla/run_client.sh --no_init_pose --prompt "place object to right"
#   bash ~/e6-vla/run_client.sh --no_init_pose --prompt "place object to middle"
#
# ── 센서 기반 전체 파이프라인 (권장) ─────────────────────────────────────────
#
#   왼쪽으로 이동:
#   bash ~/e6-vla/run_client.sh --no_init_pose \
#     --task_sequence "approach,pick,move_left,place_left"
#
#   오른쪽으로 이동:
#   bash ~/e6-vla/run_client.sh --no_init_pose \
#     --task_sequence "approach,pick,move_right,place_right"
#
#   가운데로 이동:
#   bash ~/e6-vla/run_client.sh --no_init_pose \
#     --task_sequence "approach,pick,move_middle,place_middle"
#
#   stage 완료 threshold 조정 (기본값):
#     --approach_z_done 100   # approach 완료: TCP Z ≤ 100mm
#     --lift_z_done 200       # pick/place 완료: TCP Z ≥ 200mm
#     --stage_done_steps 5    # 조건 연속 만족 스텝 수
#     --stage_timeout_sec 15  # stage별 최대 시간(초)
#
#   approach가 내려가지 않을 때 (모델이 j3를 잘못된 방향으로 출력할 때):
#     --approach_j3_assist_deg 0.5   # 매 스텝 Δj3 += 0.5° 강제 추가 (0.3~1.0 조정)
#
# ── 디버그 / 테스트 ───────────────────────────────────────────────────────────
#
#   로봇·카메라 없이 추론만 확인:
#   bash ~/e6-vla/run_client.sh --dry_run --no_camera --no_init_pose \
#     --max_runtime_sec 10
#
#   TCP 명령 전체 출력 (DOBOT_DEBUG):
#   DOBOT_DEBUG=1 bash ~/e6-vla/run_client.sh --no_init_pose \
#     --prompt "approach red object"
#
# ══════════════════════════════════════════════════════════════════════════════

REPO="$(cd "$(dirname "$0")" && pwd)"
source ~/move-one/min-imum/move-one/bin/activate

export MVCAM_COMMON_RUNENV=/opt/MVS/lib

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
