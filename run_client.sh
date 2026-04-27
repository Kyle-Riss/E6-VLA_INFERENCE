#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  e6-vla 로봇 클라이언트 — pi05_e6_v1_lora (E6Inputs, absolute degree)
#
#  V1 공식 프롬프트 (학습 데이터 포함, 이 문자열만 사용할 것):
#    "approach red object"
#    "pick red object"
#    "move object to left"   "move object to right"   "move object to middle"
#    "place object to left"  "place object to right"  "place object to middle"
#
# ── 단일 stage 실행 (완료 시 자동 종료) ──────────────────────────────────────
#
#   bash run_client.sh --task_sequence "approach"
#   bash run_client.sh --task_sequence "pick"
#   bash run_client.sh --task_sequence "move_left"
#   bash run_client.sh --task_sequence "place_left"
#
#   또는 프롬프트 직접 지정 (완료 감지 없이 timeout 후 종료):
#   bash run_client.sh --prompt "approach red object" --stage_timeout_sec 20
#
# ── 시퀀스 실행 (각 stage 완료 → 다음 → 최종 완료 시 자동 종료) ──────────────
#
#   bash run_client.sh --task_sequence "approach,pick,move_left,place_left"
#   bash run_client.sh --task_sequence "approach,pick,move_right,place_right"
#   bash run_client.sh --task_sequence "approach,pick,move_middle,place_middle"
#
# ── threshold 조정 ──────────────────────────────────────────────────────────
#
#   --approach_z_done 85    # approach 완료: TCP Z(mm) ≤ 이 값
#   --lift_z_done 200       # pick/place 완료: TCP Z(mm) ≥ 이 값
#   --stage_done_steps 3    # 조건 연속 만족 스텝 수
#   --stage_timeout_sec 30  # stage별 최대 시간(초)
#   --hz 10                 # 추론 빈도 (낮을수록 느리게)
#   --movj_velocity 30      # 로봇 속도 0~100
#   --movj_accel 20         # 로봇 가속 0~100
#   --no_zed                # ZED 카메라 비활성화 (HIK만 사용)
#
# ── 디버그 ──────────────────────────────────────────────────────────────────
#
#   bash run_client.sh --dry_run --no_camera --no_zed --max_runtime_sec 10
#   bash run_client.sh --task_sequence "approach" --save_frames_dir ~/debug_frames
#
# ══════════════════════════════════════════════════════════════════════════════

REPO="$(cd "$(dirname "$0")" && pwd)"
source ~/move-one/min-imum/move-one/bin/activate

export MVCAM_COMMON_RUNENV=/opt/MVS/lib
export PYTHONPATH="$REPO/src:$PYTHONPATH"
export LD_LIBRARY_PATH="$HOME/DobotControl/min-imum/move-one/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH"

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
  --task_sequence "approach" \
  --hz 10 \
  --steps_per_inference 8 \
  --max_delta_deg 0 \
  --movj_velocity 30 \
  --movj_accel 20 \
  --approach_z_done 85 \
  --stage_done_steps 3 \
  --stage_timeout_sec 30 \
  "$@"
