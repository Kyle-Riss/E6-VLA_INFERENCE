# e6-vla 사용 가이드

Dobot E6 + π0/π0.5 VLA 추론 파이프라인 전체 사용법입니다.

---

## 아키텍처 개요

```
[GPU 서버 or Jetson]          [Jetson AGX Orin]
  serve_policy.py   ←── WebSocket ──→  run_e6_client.py  →  Dobot E6
  (모델 추론)                            (로봇 제어)          (192.168.5.1)
```

- **serve_policy.py** — 체크포인트를 로드해 WebSocket으로 관측(obs) → 액션(actions) 서빙
- **run_e6_client.py** — 카메라+로봇 상태 수집 → obs 전송 → actions 수신 → Dobot MovJ/ToolDO 실행

---

## 1. 환경 준비

### 1.1 가상환경 (move-one venv 재사용)

```bash
source ~/move-one/min-imum/move-one/bin/activate
# 프롬프트에 (move-one) 이 보이면 OK
```

### 1.2 Dobot 네트워크 설정

```bash
# Jetson eno1에 Dobot과 같은 대역 IP 고정 (최초 1회)
sudo nmcli con add type ethernet ifname eno1 con-name dobot-static \
  ipv4.method manual ipv4.addresses 192.168.5.100/24 autoconnect yes
sudo nmcli con up dobot-static
ping -c 2 192.168.5.1
```

### 1.3 HIKRobot 카메라 (MVS SDK)

```bash
# 실행 전 SDK 경로 설정 (카메라 사용 시)
export MVCAM_COMMON_RUNENV=/opt/MVS/lib
```

---

## 2. 정책 서버 실행 (serve_policy.py)

### 2.1 실제 체크포인트로 서버 시작

```bash
source ~/move-one/min-imum/move-one/bin/activate
export MVCAM_COMMON_RUNENV=/opt/MVS/lib

PYTHONPATH=~/e6-vla/src \
python ~/e6-vla/scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi0_e6_freeze_vlm_primitive_176_local \
  --policy.dir /path/to/checkpoint/35000
```

**config 선택 기준:**

| config | 용도 | 액션 계약 |
|--------|------|-----------|
| `pi0_e6_freeze_vlm` | 기본 E6 fine-tune | DroidInputs, 8D |
| `pi0_e6_freeze_vlm_primitive_176_local` | Primitive-176 태스크 | DroidInputs, 8D |
| `pi0_e6_freeze_vlm_primitive_176_local_ur5` | UR5-style (wrist=zeros) | DroidInputs, 8D |
| `pi05_e6_v1` | E6Inputs (exterior only) | E6Inputs, 7D |
| `pi05_e6_v1_lora` | E6Inputs LoRA | E6Inputs, 7D |

### 2.2 더미 서버 (파이프라인 테스트용, 체크포인트 불필요)

```bash
PYTHONPATH=~/e6-vla/src \
python ~/e6-vla/scripts/serve_dummy.py \
  --port 8000 \
  --action_dim 8 \
  --action_horizon 10
```

---

## 3. 추론 클라이언트 실행 (run_e6_client.py)

서버가 실행 중인 상태에서 **별도 터미널**에서 실행합니다.

### 3.1 기본 실행

```bash
source ~/move-one/min-imum/move-one/bin/activate
export MVCAM_COMMON_RUNENV=/opt/MVS/lib

python ~/e6-vla/examples/e6/run_e6_client.py \
  --server_host 127.0.0.1 \
  --server_port 8000 \
  --robot_ip 192.168.5.1 \
  --prompt "approach red object" \
  --hz 20 \
  --steps_per_inference 5
```

### 3.2 자동 approach ↔ return 순환

```bash
python ~/e6-vla/examples/e6/run_e6_client.py \
  --server_host 127.0.0.1 \
  --robot_ip 192.168.5.1 \
  --auto_cycle_return_approach \
  --approach_prompt "approach red object" \
  --return_prompt "return" \
  --approach_cycle_sec 8 \
  --return_cycle_sec 2
```

### 3.3 파이프라인 전체 테스트 (로봇·카메라 없이)

```bash
python ~/e6-vla/examples/e6/run_e6_client.py \
  --server_host 127.0.0.1 \
  --dry_run \
  --no_camera \
  --no_init_pose \
  --prompt "approach red object" \
  --hz 5 \
  --steps_per_inference 3 \
  --max_runtime_sec 10
```

### 3.4 주요 인자 요약

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--server_host` | 127.0.0.1 | serve_policy.py 호스트 |
| `--server_port` | 8000 | serve_policy.py 포트 |
| `--robot_ip` | 192.168.5.1 | Dobot E6 IP |
| `--prompt` | approach red object | 작업 지시 문구 |
| `--task_preset` | None | 프리셋 프롬프트 선택 (approach/pick/return 등) |
| `--input_layout` | baseline | `ur5_style`: wrist_image를 zeros로 전송 |
| `--hz` | 20 | 제어/카메라 주기(Hz) |
| `--steps_per_inference` | None(=chunk 전체) | 청크에서 몇 스텝 후 재추론 |
| `--action_scale` | 1.0 | 관절 delta 배율 |
| `--max_delta_deg` | None | 안전 상한: 스텝당 최대 관절 이동(deg) |
| `--movj_velocity` | 70 | MovJ 속도 0~100 |
| `--movj_accel` | 60 | MovJ 가속도 0~100 |
| `--grip_open_threshold` | 0.03 | 그리퍼 열림 임계값 |
| `--grip_close_threshold` | 0.07 | 그리퍼 닫힘 임계값 |
| `--z_grip_trigger` | None | TCP Z(mm) 이하 + grip>0 이면 강제 닫힘 (예: 170) |
| `--dry_run` | False | 로봇 전송 없이 추론만 |
| `--no_camera` | False | 카메라 미사용 (더미 이미지) |
| `--show_actions` | False | 매 스텝 예측 액션값 출력 (관절 delta + 그리퍼) |
| `--no_init_pose` | False | 초기 자세 이동 스킵 |
| `--max_runtime_sec` | None | 최대 실행 시간(초) |
| `--save_frames_dir` | None | 프레임 저장 디렉터리 |
| `--exec_log_jsonl` | None | 추론 로그 저장 경로 |

---

## 4. 파이프라인 전체 테스트 절차 (체크포인트 없는 경우)

```bash
# 터미널 1: 더미 서버 시작
source ~/move-one/min-imum/move-one/bin/activate
PYTHONPATH=~/e6-vla/src python ~/e6-vla/scripts/serve_dummy.py --port 8000

# 터미널 2: 클라이언트 테스트 (로봇·카메라 없이)
source ~/move-one/min-imum/move-one/bin/activate
python ~/e6-vla/examples/e6/run_e6_client.py \
  --dry_run --no_camera --no_init_pose \
  --prompt "approach red object" \
  --hz 5 --steps_per_inference 3 --max_runtime_sec 10
```

**정상 출력 예시:**
```
[1/3] 정책 서버 연결 중: ws://127.0.0.1:8000
      연결 완료. 서버 메타데이터: {'dummy': True, 'action_dim': 8, 'action_horizon': 10}
[3/3] 추론 루프 시작 (Ctrl+C 종료)
  [추론] step=0 3.2ms prompt='approach red object'
  [ACTION_SHAPE] (10, 8)
...
[TASK_DONE] FAIL_TIMEOUT runtime>10.0s
[DONE] task_result=FAIL_TIMEOUT reason='runtime>10.0s' total_steps=50
```

---

## 5. 관측(obs) 계약

서버로 전송되는 obs dict:

| 키 | shape | dtype | 설명 |
|----|-------|-------|------|
| `observation/exterior_image_1_left` | (224, 224, 3) | uint8 | HIKRobot 탑뷰 RGB |
| `observation/exterior_image_2_left` | (224, 224, 3) | uint8 | ZED X left eye RGB |
| `observation/state` | (7,) | float32 | [j1..j6 deg, gripper 0~1] |
| `prompt` | str | — | 작업 지시 문구 |

---

## 6. 액션 계약

서버에서 받는 actions ndarray (DroidOutputs 기준):

| 인덱스 | 의미 |
|--------|------|
| 0~5 | 관절 6개 Δ각(rad) — 현재 관절에 누산 |
| 6 | 패딩 (무시) |
| 7 | 그리퍼 (0~1 연속값, hysteresis 처리) |

---

## 7. 트러블슈팅

| 현상 | 원인 | 조치 |
|------|------|------|
| `Connection refused` | 서버 미실행 | serve_policy.py 또는 serve_dummy.py 먼저 실행 |
| `ModuleNotFoundError: etils` | 가상환경 미사용 | `source ~/move-one/min-imum/move-one/bin/activate` |
| `Dobot 소켓 연결 실패` | eno1 IP 없음 또는 케이블 미연결 | `ping 192.168.5.1`, nmcli 설정 확인 |
| 카메라 `ImportError: MvCameraControl` | MVS SDK 경로 없음 | `export MVCAM_COMMON_RUNENV=/opt/MVS/lib` |
| `norm_stats.json not found` | 체크포인트 폴더 구조 불완전 | `assets/droid/norm_stats.json` 존재 여부 확인 |
| 로봇이 거의 안 움직임 | action_scale 작음 | `--action_scale 2.0` 또는 `--action_scale 3.0` 시도 |
| 그리퍼가 안 닫힘 | grip 출력 낮음 | `--grip_close_threshold 0.05 --z_grip_trigger 170` |

---

## 8. 주요 파일 위치

| 파일 | 경로 | 역할 |
|------|------|------|
| 추론 클라이언트 | `examples/e6/run_e6_client.py` | 로봇 제어 메인 |
| 정책 서버 | `scripts/serve_policy.py` | 실제 모델 서빙 |
| 더미 서버 | `scripts/serve_dummy.py` | 파이프라인 테스트 |
| E6 정책 정의 | `src/openpi/policies/e6_policy.py` | E6Inputs/E6Outputs |
| 학습 config | `src/openpi/training/config.py` | 모든 config 정의 |
| 카메라 캡처 | `hardware/camera_capture.py` | HIKRobot MVS |
| Dobot API | `hardware/dobot/dobot_api.py` | TCP/IP 제어 |
| 로봇 연결 테스트 | `hardware/utils/test_robot_connection.py` | 연결 확인 유틸 |
