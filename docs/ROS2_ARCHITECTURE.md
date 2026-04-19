# E6-VLA ROS2 아키텍처 설계

## 전체 구조

```
┌──────────────────────────────┐
│        serve_policy.py       │
│   π0.5 policy server         │
│   checkpoint / norm / infer  │
└──────────────┬───────────────┘
               │ WebSocket
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                           ROS2 영역 (e6_vla_ros)                     │
│                                                                      │
│  [camera_state_node] ── /e6/camera/image ─────────────────┐
                    └── /e6/camera/zed_image ──────────────┤         │
│                                                           │         │
│  [camera_state_node] ─ /e6/robot/state ───────────────────┤         │
│                       /e6/robot/tcp_z                     ▼         │
│  [task_node] ──────── /e6/task/prompt ──────> [inference_bridge_node]│
│                                          obs 조립 → WebSocket 질의  │
│                                          action_chunk publish        │
│                                                           │         │
│                                                           ▼         │
│                              /e6/policy/action_chunk   [executor_node]
│                                               MovJ / ToolDO 실행    │
│                                                           │         │
│                                                           ▼         │
│                                                       Dobot E6      │
│                                                                      │
│  [supervisor_node]  bad camera / tcp_z / timeout / emergency stop   │
│                     /e6/supervisor/status publish                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 패키지 디렉토리

```
e6-vla/
└── ros2/
    └── src/
        └── e6_vla_ros/
            ├── package.xml
            ├── setup.py
            ├── setup.cfg
            ├── resource/e6_vla_ros
            ├── launch/
            │   └── e6_vla.launch.py
            └── e6_vla_ros/
                ├── __init__.py
                ├── camera_state_node.py
                ├── inference_bridge_node.py
                ├── executor_supervisor_node.py
                ├── task_node.py
                └── __init__.py
```

---

## 토픽 / 서비스 목록

| 토픽 | 타입 | 발행 노드 | Hz | 설명 |
|------|------|----------|----|------|
| `/e6/camera/image` | `sensor_msgs/Image` | camera_state_node | 20 | 224×224 RGB uint8 — HIKRobot |
| `/e6/camera/zed_image` | `sensor_msgs/Image` | camera_state_node | 20 | 224×224 RGB uint8 — ZED left |
| `/e6/robot/state` | `std_msgs/Float32MultiArray` | camera_state_node | 20 | [j1..j6 deg, gripper 0\~1] |
| `/e6/robot/tcp_z` | `std_msgs/Float32` | camera_state_node | 20 | TCP Z(mm) |
| `/e6/task/prompt` | `std_msgs/String` | task_node | 이벤트 | QoS transient_local |
| `/e6/policy/action_chunk` | `std_msgs/Float32MultiArray` | inference_bridge_node | ~1 | 16×7 flatten |
| `/e6/supervisor/status` | `std_msgs/String` | supervisor_node | 10 | RUNNING / STAGE_DONE:approach / FAIL_SAFETY / … |

| 서비스 | 타입 | 서버 노드 | 설명 |
|--------|------|----------|------|
| `/e6/task/set_prompt` | `std_srvs/SetBool` | task_node | 수동 프롬프트 변경 |
| `/e6/emergency_stop` | `std_srvs/Trigger` | supervisor_node | 즉시 정지 |

---

## 노드별 핵심 책임

### camera_state_node
- HIKRobot (`hardware/camera_capture.py`) → `/e6/camera/image`
- ZED X (`pyzed.sl`) → `/e6/camera/zed_image` (left eye, 224×224 RGB)
- Dobot `feedBackData()` → `/e6/robot/state` [j1..j6 deg, gripper], `/e6/robot/tcp_z`
- 18Hz 타이머, 이미지 encoding `rgb8`
- `no_camera=true` 시 두 카메라 모두 zeros 발행 (graceful degradation)
- `dry_run=true` 시 로봇 상태 zeros 발행

### task_node
- `--task_sequence "approach,pick,move_left,place_left"` 파라미터
- `/e6/supervisor/status` 구독 → `STAGE_DONE:*` 수신 시 다음 stage로 전환
- 프롬프트 전환 후 `/e6/task/prompt` re-publish

### inference_bridge_node
- 각 토픽 latest 캐시 (ApproximateTimeSynchronizer 미사용)
- `policy.infer()` → **ThreadPoolExecutor** 별도 스레드 실행 (~2초 소요)
- 결과 수신 시 `/e6/policy/action_chunk` publish

### executor_node
- `/e6/policy/action_chunk` 수신 시 청크 갱신
- 18Hz 타이머 → `chunk[index]` 꺼내서 MovJ / ToolDO 실행
- `max_delta_deg` 클리핑, staleness 체크

### supervisor_node
- bad_camera (frame_mean 임계값)
- tcp_z ≤ min_tool_z → FAIL_SAFETY
- stage_timeout 초과 → FAIL_STAGE_TIMEOUT
- stage 완료 조건 (approach_z, lift_z, gripper) → STAGE_DONE:{stage}

---

## 빌드 / 실행

```bash
# 1. ROS2 소싱
source /opt/ros/humble/setup.bash

# 2. 빌드
cd ~/e6-vla/ros2
colcon build --symlink-install

# 3. 패키지 소싱
source install/setup.bash

# 4. 전체 런치
ros2 launch e6_vla_ros e6_vla.launch.py \
  checkpoint:=... \
  task_sequence:="approach,pick,move_left,place_left"

# 또는 단일 노드 개별 실행
ros2 run e6_vla_ros camera_node
ros2 run e6_vla_ros robot_state_node
```

---

## ✅ ZED X 스테레오 카메라 추가 (완료)

2채널 카메라 파이프라인 구현 완료.

### obs 구조

| 키 | 카메라 | ROS2 토픽 | 모델 슬롯 |
|----|--------|----------|----------|
| `observation/exterior_image_1_left` | HIKRobot 탑뷰 | `/e6/camera/image` | `base_0_rgb` |
| `observation/exterior_image_2_left` | ZED X (left eye) | `/e6/camera/zed_image` | `left_wrist_0_rgb` |
| `observation/state` | Dobot feedBack | `/e6/robot/state` | — |

### 완료된 변경

- `camera_state_node.py` — ZED 초기화 + `/e6/camera/zed_image` 18Hz 발행
- `inference_bridge_node.py` — `/e6/camera/zed_image` 구독, obs에 `exterior_image_2_left` 포함
- `e6_policy.py` — `E6Inputs` 2채널 처리, `image_mask` 둘 다 `True`
- `config.py` — `LeRobotE6DataConfig` `exterior_image_2_left` 매핑 추가

---

*현재 구현 브랜치: `feature/ros2-integration`*  
*관련 레포: [6DOF-VLA](https://github.com/Kyle-Riss/E6-VLA_INFERENCE) (학습 파이프라인)*
