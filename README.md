# E6-VLA Inference

Dobot E6 로봇 팔을 위한 π0.5 VLA(Vision-Language-Action) 추론 파이프라인입니다.  
[Physical Intelligence의 openpi](https://github.com/physical-intelligence/openpi) 기반으로, Jetson AGX Orin 환경에 맞게 구성되어 있습니다.

## 아키텍처

```
                        ┌──────────────────────────────────────────┐
                        │          입력 레이어                       │
  [HIK 카메라]  [ZED 카메라]  [Dobot E6]          [마이크 / 텍스트]  │
       └──────────────────────┘    │                    │           │
                  │                │                    │           │
                  ▼                │                    ▼           │
       [camera_state_node]◄────────┘      [voice_command_node] ★   │
            18Hz                                 STT + 명령 분류    │
         │  │  │                                   │        │       │
  image  │  │  │ state/tcp_z          voice_command│        │STOP   │
  zed    │  │  │                                   ▼        ▼       │
         │  │  └──────────────────►  [task_node] ◄────  [executor]  │
         │  │                           PhaseTracker       │        │
         │  └──────────────────────►  /e6/task/prompt      │        │
         │                                │                │        │
         └──────────────────►  [inference_bridge_node]     │        │
                                    obs 조립                │        │
                                    WebSocket 추론           │        │
                                          │                 │        │
                               [Policy Server (π0 + LoRA)]  │        │
                                    action_chunk (16×7)     │        │
                                          │                 │        │
                                          └────────────────►│        │
                                                    MovJ/ServoJ      │
                                                    ToolDO           │
                                                    → Dobot E6       │
                        └──────────────────────────────────────────┘
```

### ROS2 노드

| 노드 | 역할 |
|------|------|
| `camera_state_node` | HIK(top) + ZED(scene) 카메라 → `/e6/camera/image`, `/e6/camera/zed_image`, `/e6/robot/state` |
| `inference_bridge_node` | obs 조립 → WebSocket 정책 서버 → `/e6/policy/action_chunk` |
| `executor_supervisor_node` | action_chunk 수신 → MovJ/ServoJ/ToolDO 실행 + 안전 감시 |
| `task_node` | PhaseTracker 기반 per-frame 프롬프트 발행 + TASK_COMPLETE 중계 |
| `voice_command_node` ★ | 음성/텍스트 → prompt 변환 → task_node·executor로 전달 |

### 토픽 목록

| 토픽 | 타입 | 발행 | 구독 |
|------|------|------|------|
| `/e6/camera/image` | Image | camera_state | inference_bridge, executor |
| `/e6/camera/zed_image` | Image | camera_state | inference_bridge |
| `/e6/robot/state` | Float32MultiArray | camera_state | inference_bridge, executor, task |
| `/e6/robot/tcp_z` | Float32 | camera_state | executor, task |
| `/e6/gripper/commanded` | Float32 | executor | camera_state |
| `/e6/task/prompt` | String | task | inference_bridge, executor |
| `/e6/task/status` | String | task | inference_bridge, executor |
| `/e6/supervisor/status` | String | executor | task |
| `/e6/policy/action_chunk` | Float32MultiArray | inference_bridge | executor |
| `/e6/task/voice_command` ★ | String | voice_command | task |
| `/e6/supervisor/voice_override` ★ | String | voice_command | executor |
| `/e6/voice/text_input` ★ | String | (외부 CLI) | voice_command |

---

## 요구 환경

- Jetson AGX Orin (aarch64, JetPack 6)
- Python 3.10
- HIKRobot MVS SDK (`/opt/MVS/`)
- ZED SDK (ZED 카메라 사용 시)
- Dobot E6 (TCP/IP, 192.168.5.1)
- 가상환경: `~/move-one/min-imum/move-one/bin/activate`

---

## 빠른 시작 (v17 — 현재 권장)

### 터미널 1 — 정책 서버

```bash
cd ~/E6-VLA_INFERENCE
bash run_server_v17.sh /media/billye6/새\ 볼륨/e6_checkpoints/e6_v17_15000
```

### 터미널 2 — ROS2 추론

```bash
cd ~/E6-VLA_INFERENCE/ros2
source install/setup.bash
ros2 launch e6_vla_ros e6_vla.launch.py
```

> v17 기본값이 모두 설정되어 있어 인자 없이 실행 가능  
> (action_mode=delta, gripper_mode=absolute, prompt_mode=per_frame_v16)

---

## 음성 명령 사용법 ★

### 마이크 사용 (STT 포함)

```bash
# 의존 패키지 설치 (최초 1회)
pip3 install faster-whisper sounddevice

# 마이크 활성화 launch
ros2 launch e6_vla_ros e6_vla.launch.py use_voice:=true
```

말할 수 있는 명령 예시:

| 발화 | 변환되는 prompt |
|------|----------------|
| "왼쪽 박스 집어줘" | `pick up the orange box from the left side and place it on the right side` |
| "오른쪽 박스 집어줘" | `pick up the orange box from the right side and place it on the left side` |
| "집어줘" | `pick up the orange box` |
| "멈춰" / "정지" / "그만" / "stop" | → executor 즉시 긴급 정지 |

### 마이크 없이 텍스트로만 사용

```bash
# 마이크 없이 노드 실행 (텍스트 입력 전용)
ros2 launch e6_vla_ros e6_vla.launch.py use_voice:=true use_mic:=false

# 별도 터미널에서 텍스트 명령 주입
ros2 topic pub --once /e6/voice/text_input std_msgs/msg/String \
  "data: '왼쪽 박스 집어줘'"

# 영어 prompt 직접 주입
ros2 topic pub --once /e6/voice/text_input std_msgs/msg/String \
  "data: 'pick up the orange box from the left side'"

# STOP 명령
ros2 topic pub --once /e6/voice/text_input std_msgs/msg/String \
  "data: '멈춰'"
```

### voice_command_node 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `use_mic` | `true` | 마이크 캡처 활성화 |
| `model_size` | `base` | Whisper 모델 크기 (tiny/base/small/medium) |
| `language` | `ko` | STT 언어 |
| `vad_min_amplitude` | `0.02` | 음성 감지 최소 RMS 진폭 |
| `silence_duration_sec` | `1.5` | 발화 종료 판정 침묵 시간 (초) |
| `device_index` | `-1` | 마이크 장치 인덱스 (-1=시스템 기본값) |

---

## 주요 launch 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `action_mode` | `delta` | `delta` (v8+) / `absolute` (v6) |
| `gripper_mode` | `absolute` | `absolute` (v16/v17) / `delta` (v13/v14) |
| `prompt_mode` | `per_frame_v16` | `per_frame_v16` / `single` (v13) / `per_frame` (v8) |
| `source_side` | `left` | 오렌지 박스 시작 위치 |
| `max_delta_deg` | `3.0` | delta 클램핑 상한 (degree) |
| `max_steps` | `3000` | 강제 종료 스텝 수 (안전망) |
| `min_tool_z` | `75.0` | 최소 TCP Z (mm) — 이하 시 FAIL_SAFETY |
| `grip_enable_z` | `125.0` | 이 Z 이하에서만 흡착 허용 (조기 흡착 방지) |
| `grasp_z_max` | `130.0` | grasp phase 진입 최대 Z (mm) |
| `scripted_lift_enabled` | `true` | lift stall 시 RelMovLUser 강제 상승 |
| `place_force_release_enabled` | `true` | place Z ≤ 110mm 시 강제 release |
| `control_mode` | `movj` | `movj` / `servoj` (실시간 갱신) |
| `use_voice` | `false` | voice_command_node 활성화 ★ |
| `use_mic` | `true` | 마이크 캡처 (false=텍스트 전용) ★ |
| `record_mcap` | `false` | MCAP 기록 |
| `foxglove` | `true` | Foxglove Bridge 실시간 스트리밍 |

---

## 관측 / 액션 계약 (v16/v17)

### 관측 (obs)

| 키 | Shape | 설명 |
|----|-------|------|
| `observation/exterior_image_1_left` | (224, 224, 3) uint8 | HIK 탑뷰 |
| `observation/exterior_image_2_left` | (224, 224, 3) uint8 | ZED 씬 |
| `observation/state` | (7,) float32 | [j1..j6 deg, gripper 0~1] |
| `prompt` | str | 태스크 지시 문구 |

### 카메라 전처리

| 카메라 | 파이프라인 |
|--------|-----------|
| HIK | 640×480 → 320×240 → crop[16:240, 55:279] → **224×224** |
| ZED | HD1080 → 640×480 → crop[120:480, 150:510] → 360×360 → **224×224** |

### 액션 (v16/v17)

| 인덱스 | 의미 |
|--------|------|
| `[:, 0:6]` | 관절 velocity delta (deg/frame) |
| `[:, 6]` | 그리퍼 absolute (0.0 or 1.0, threshold 0.5) |

- action_horizon: **16** / 실행: 앞 8개 / 제어 주기: **16Hz**
- state 입력: 7D 절대값 [j1..j6 deg, gripper]

---

## 지원 모델

| Config | Action | Gripper | Prompt | 체크포인트 |
|--------|--------|---------|--------|-----------|
| `pi05_e6_v8_lora` | delta | delta 누산 | per_frame | `pytorch_from_jax_v8_lora_merged` |
| `pi05_e6_v13_lora` | delta | delta 누산 | single | `e6_checkpoints/e6_v13_30k` |
| `pi05_e6_v16_lora` | delta | absolute | per_frame_v16 | `e6_checkpoints/e6_v16_*` |
| `pi05_e6_v17_lora` | delta | absolute | per_frame_v16 | `e6_checkpoints/e6_v17_15000` |
| `pi05_e6_v18_lora` | delta | absolute | per_frame_v16 | `e6_checkpoints/e6_v18_20k` |
| `pi05_e6_v19_lora` | delta | absolute | per_frame_v16 | `e6_checkpoints/e6_v19_20k` |

> 체크포인트 기본 경로: `/media/billye6/새 볼륨/e6_checkpoints/`

---

## 파일 구조

```
E6-VLA_INFERENCE/
├── run_server_v17.sh ~ run_server_v19.sh  # 버전별 정책 서버 실행 스크립트
├── scripts/
│   ├── serve_policy.py                    # 정책 서버 (WebSocket)
│   ├── test_train_image_infer.py          # 학습 이미지 기반 추론 테스트
│   └── test_grounding.py                  # 카메라/state grounding 검증
├── ros2/src/e6_vla_ros/e6_vla_ros/
│   ├── camera_state_node.py               # HIK + ZED 카메라 퍼블리셔
│   ├── inference_bridge_node.py           # obs 조립 + WebSocket 추론
│   ├── executor_supervisor_node.py        # MovJ/ServoJ/ToolDO + 안전 감시
│   ├── task_node.py                       # PhaseTracker + 프롬프트 발행
│   └── voice_command_node.py             # 음성/텍스트 → prompt 변환 ★
├── ros2/src/e6_vla_ros/launch/
│   └── e6_vla.launch.py                   # 전체 런치 파일
└── src/openpi/
    ├── models/                            # 모델 아키텍처
    └── training/config.py                 # 버전별 TrainConfig 정의
```

---

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
