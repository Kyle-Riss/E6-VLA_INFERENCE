# E6-VLA Inference

Dobot E6 로봇 팔을 위한 π0.5 VLA(Vision-Language-Action) 추론 파이프라인입니다.  
[Physical Intelligence의 openpi](https://github.com/physical-intelligence/openpi) 기반으로, Jetson AGX Orin 환경에 맞게 구성되어 있습니다.

## 아키텍처

```
[serve_policy.py]  ←── WebSocket ──→  [inference_bridge_node]
  정책 서버 (π0.5 추론)                [camera_state_node]      ──→  Dobot E6
                                       [executor_supervisor_node]    192.168.5.1
                                       [task_node]
```

ROS2 노드 4개로 구성:

| 노드 | 역할 |
|------|------|
| `camera_state_node` | HIK(top) + ZED(scene) 카메라 → `/e6/camera/image`, `/e6/camera/zed_image` |
| `inference_bridge_node` | obs 조립 → WebSocket 정책 서버 → `/e6/policy/action_chunk` |
| `executor_supervisor_node` | action_chunk 수신 → MovJ/ToolDO 실행 + 종료 조건 감시 |
| `task_node` | 프롬프트 발행, TASK_COMPLETE 중계 |

## 요구 환경

- Jetson AGX Orin (aarch64, JetPack 6)
- Python 3.10
- HIKRobot MVS SDK (`/opt/MVS/`)
- ZED SDK (ZED 카메라 사용 시)
- Dobot E6 (TCP/IP, 192.168.5.1)
- 가상환경: `~/move-one/min-imum/move-one/bin/activate`

## 빠른 시작 (v13)

### 터미널 1 — 정책 서버

```bash
cd ~/E6-VLA_INFERENCE
bash run_server_v13.sh /media/billy/새\ 볼륨2/e6_v13_22k
```

### 터미널 2 — ROS2 추론

```bash
cd ~/E6-VLA_INFERENCE/ros2
source install/setup.bash

ros2 launch e6_vla_ros e6_vla.launch.py \
  prompt_mode:=single \
  source_side:=left \
  action_mode:=delta \
  max_delta_deg:=5.0
```

### 주요 launch 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `action_mode` | `delta` | `delta` / `absolute` |
| `prompt_mode` | `single` | `single` (v13) / `per_frame` (v8~v12) |
| `source_side` | `left` | 오렌지 박스 시작 위치 (`left` / `right`) |
| `prompt_variant` | `-1` | 0~2 고정 선택, -1이면 랜덤 |
| `max_delta_deg` | `5.0` | delta 클램핑 상한 (degree) |
| `max_steps` | `500` | 강제 종료 스텝 수 (안전망) |
| `min_steps` | `100` | 정상 종료 감지 시작 스텝 |
| `home_tol_deg` | `5.0` | 초기 자세 복귀 허용 오차 (degree) |
| `home_consec_req` | `16` | 복귀 판정 연속 프레임 수 |
| `record_mcap` | `false` | MCAP 기록 켜기 |
| `foxglove` | `false` | Foxglove Bridge 실시간 스트리밍 |
| `task_sequence` | `approach` | 실행할 stage (쉼표 구분) |

## 관측 / 액션 계약

### 관측 (obs)

| 키 | Shape | 설명 |
|----|-------|------|
| `observation/exterior_image_1_left` | (224, 224, 3) uint8 | HIK 탑뷰 카메라 RGB |
| `observation/exterior_image_2_left` | (224, 224, 3) uint8 | ZED 씬 카메라 RGB |
| `observation/state` | (7,) float32 | [j1..j6 deg, gripper 0~1] |
| `prompt` | str | 태스크 지시 문구 |

### 카메라 전처리

| 카메라 | 파이프라인 |
|--------|-----------|
| HIK | 640×480 → 320×240 → crop[16:240, 55:279] → **224×224** |
| ZED | HD1080 → 640×480 → crop[120:480, 150:510] → 360×360 → **224×224** |

### 액션 (v8 이후)

| 인덱스 | 의미 |
|--------|------|
| `[:, 0:6]` | 관절 velocity delta (deg/frame) |
| `[:, 6]` | 그리퍼 delta (누산: `clip(grip_cont + Δ, 0, 1)`) |

- action_horizon: **16** / 실행: 앞 8개 / 제어 주기: **16Hz**
- state 입력: 7D 절대값 [j1..j6 deg, gripper]

## 종료 조건 (v13)

| 조건 | 설명 |
|------|------|
| **B (안전망)** | `step_count > max_steps(500)` → 강제 종료 |
| **C (정상)** | `step_count > min_steps(100)` 이후 j1~j3이 INIT_POSE `[91.3, 37.7, 53.8]°` ±5° 이내 16프레임 연속 |

## 지원 모델

| Config | 데이터셋 | Action | Prompt | 체크포인트 |
|--------|---------|--------|--------|-----------|
| `pi05_e6_v8_lora` | v8 | delta | per_frame | `pytorch_from_jax_v8_lora_merged` |
| `pi05_e6_v9_lora` | v8 | delta | per_frame | `pytorch_from_jax_v9_lora_merged` |
| `pi05_e6_v10_lora` | v10 | delta | per_frame | `e6_checkpoints/e6_v10_50k` |
| `pi05_e6_v11_lora` | v10 | delta | per_frame | `e6_checkpoints/e6_v11_30k` |
| `pi05_e6_v12_lora` | v10 | delta | per_frame | `e6_checkpoints/e6_v12_30k` |
| `pi05_e6_v13_lora` | v13 | delta | single | `e6_v13_22k` |

> 체크포인트 기본 경로: `/media/billy/새 볼륨2/` (v13) / `/media/billye6/새 볼륨/e6_checkpoints/` (v8~v12)

## 파일 구조

```
E6-VLA_INFERENCE/
├── run_server_v8.sh ~ run_server_v13.sh   # 버전별 정책 서버 실행 스크립트
├── scripts/
│   ├── serve_policy.py                    # 정책 서버 (WebSocket)
│   ├── test_train_image_infer.py          # 학습 이미지 기반 추론 테스트
│   └── test_grounding.py                  # 카메라/state grounding 검증
├── examples/e6/
│   └── run_e6_client.py                   # 단일 스크립트 모드 클라이언트
├── ros2/src/e6_vla_ros/e6_vla_ros/
│   ├── camera_state_node.py               # HIK + ZED 카메라 퍼블리셔
│   ├── inference_bridge_node.py           # obs 조립 + WebSocket 추론
│   ├── executor_supervisor_node.py        # MovJ/ToolDO + 종료 감시
│   └── task_node.py                       # 프롬프트 발행 + TASK_COMPLETE 중계
├── ros2/src/e6_vla_ros/launch/
│   └── e6_vla.launch.py                   # 전체 런치 파일
└── src/openpi/
    ├── models/                            # 모델 아키텍처
    └── training/config.py                 # 버전별 TrainConfig 정의
```

## Jetson 환경 주의사항

- **HIKRobot SDK**: `MVCAM_COMMON_RUNENV=/opt/MVS/lib` (lib64 아님 — aarch64 `.so`는 `/opt/MVS/lib/aarch64/`)
- **torch.compile 비활성화**: Jetson aarch64는 Triton 미지원 → `TORCHDYNAMO_DISABLE=1`
- **cusparseLt**: `LD_LIBRARY_PATH`에 `nvidia/cusparselt/lib` 추가 필요 (torch GPU 초기화)

## Foxglove 실시간 모니터링

```bash
ros2 launch e6_vla_ros e6_vla.launch.py foxglove:=true record_mcap:=true
```

Foxglove Studio → `ws://100.76.114.107:8765` (Tailscale IP)

## 관련 레포

- **[6DOF-VLA](https://github.com/Kyle-Riss/6DOF-VLA)** — 학습(fine-tuning) 파이프라인
