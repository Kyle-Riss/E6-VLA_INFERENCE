# E6-VLA Inference

Dobot E6 로봇 팔을 위한 π0.5 VLA(Vision-Language-Action) 추론 파이프라인입니다.  
[Physical Intelligence의 openpi](https://github.com/physical-intelligence/openpi) 기반으로, Jetson AGX Orin 환경에 맞게 구성되어 있습니다.

## 아키텍처

```
[serve_policy.py]  ←── WebSocket ──→  [run_e6_client.py]  ──→  Dobot E6
  정책 서버 (π0.5 추론)                  로봇 제어 클라이언트       192.168.5.1
```

- **serve_policy.py** — 체크포인트를 로드해 WebSocket으로 obs → actions 서빙
- **run_e6_client.py** — 카메라 + 로봇 상태 수집 → 서버로 전송 → MovJ / ToolDO 실행

## 요구 환경

- Jetson AGX Orin (aarch64, JetPack 6)
- Python 3.10
- HIKRobot MVS SDK (`/opt/MVS/`)
- Dobot E6 (TCP/IP, 192.168.5.1)
- 가상환경: move-one venv (`~/move-one/min-imum/move-one/bin/activate`)

## 빠른 시작

### 1. 가상환경 활성화

```bash
source ~/move-one/min-imum/move-one/bin/activate
export MVCAM_COMMON_RUNENV=/opt/MVS/lib64
```

### 2. 정책 서버 실행 (터미널 1)

```bash
bash ~/e6-vla/run_server.sh /path/to/checkpoint/XXXXX
```

### 3. 로봇 클라이언트 실행 (터미널 2)

```bash
bash ~/e6-vla/run_client.sh --prompt "approach red object"
```

### 파이프라인 테스트 (로봇·카메라 없이)

```bash
# 터미널 1
PYTHONPATH=~/e6-vla/src python ~/e6-vla/scripts/serve_dummy.py --port 8000

# 터미널 2
bash ~/e6-vla/run_client.sh --dry_run --no_camera --no_init_pose --max_runtime_sec 10
```

## 관측 / 액션 계약 (pi05_e6_v1_lora 기준)

### 관측 (obs)

| 키 | Shape | 설명 |
|----|-------|------|
| `observation/exterior_image_1_left` | (224, 224, 3) uint8 | 탑뷰 카메라 RGB |
| `observation/state` | (7,) float32 | [j1..j6 deg, gripper 0~1] |
| `prompt` | str | 태스크 지시 문구 |

### 액션 (actions)

| 인덱스 | 의미 |
|--------|------|
| `[:, 0:6]` | 관절 Δ각 (degree) |
| `[:, 6]` | 그리퍼 절대값 (0=열림, 1=닫힘) |

- action_horizon: 16 / 제어 주기: 20Hz 권장

### 태스크 프롬프트 목록

```
"approach red object"
"pick red object"
"move object to left / right / middle"
"place object to left / right / middle"
"return"
```

## 지원 Config

| Config | 입력 계약 | 액션 | 비고 |
|--------|-----------|------|------|
| `pi05_e6_v1_lora` | E6Inputs (state deg 7D) | (16, 7) Δdeg | **메인** |
| `pi05_e6_v1` | E6Inputs | (16, 7) Δdeg | LoRA 미적용 |
| `pi0_e6_freeze_vlm_primitive_176_local` | DroidInputs (joint_pos rad 8D) | (10, 8) Δrad | 이전 버전 |
| `pi0_e6_freeze_vlm_primitive_176_local_ur5` | DroidInputs (wrist=zeros) | (10, 8) Δrad | UR5-style |

## 파일 구조

```
e6-vla/
├── run_server.sh                      # 서버 실행 (체크포인트 경로 인자)
├── run_client.sh                      # 클라이언트 실행
├── scripts/
│   ├── serve_policy.py                # 실제 모델 서버
│   └── serve_dummy.py                 # 파이프라인 테스트용 더미 서버
├── examples/e6/
│   ├── run_e6_client.py               # 로봇 제어 클라이언트
│   └── e6_v1_task_contract.py         # 태스크 프롬프트 정의
├── hardware/
│   ├── camera_capture.py              # HIKRobot MVS 카메라
│   ├── dobot/dobot_api.py             # Dobot E6 TCP 제어
│   └── utils/                         # 연결 테스트 유틸
├── src/openpi/                        # 모델 아키텍처 (serve_policy 의존)
├── packages/openpi-client/            # WebSocket 클라이언트
├── setup/                             # Jetson 환경 설정 스크립트
└── docs/                              # USAGE.md 등 상세 문서
```

## 상세 문서

- [전체 사용 가이드](docs/USAGE.md)
- [추론 파이프라인](docs/INFERENCE.md)
- [로봇 추론 가이드](docs/ROBOT_INFERENCE.md)

## 관련 레포

- **[6DOF-VLA](https://github.com/Kyle-Riss/6DOF-VLA)** — 학습(fine-tuning) 파이프라인
