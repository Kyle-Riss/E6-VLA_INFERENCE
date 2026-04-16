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
│  [camera_node] ──── /e6/camera/image ─────────────────────┐         │
│                                                           │         │
│  [robot_state_node] ─ /e6/robot/state ────────────────────┤         │
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
                ├── camera_node.py
                ├── robot_state_node.py
                ├── task_node.py
                ├── inference_bridge_node.py
                ├── executor_node.py
                └── supervisor_node.py
```

---

## 토픽 / 서비스 목록

| 토픽 | 타입 | 발행 노드 | Hz | 설명 |
|------|------|----------|----|------|
| `/e6/camera/image` | `sensor_msgs/Image` | camera_node | 20 | 224×224 RGB uint8 |
| `/e6/robot/state` | `std_msgs/Float32MultiArray` | robot_state_node | 20 | [j1..j6 deg, gripper 0\~1] |
| `/e6/robot/tcp_z` | `std_msgs/Float32` | robot_state_node | 20 | TCP Z(mm) |
| `/e6/task/prompt` | `std_msgs/String` | task_node | 이벤트 | QoS transient_local |
| `/e6/policy/action_chunk` | `std_msgs/Float32MultiArray` | inference_bridge_node | ~1 | 16×7 flatten |
| `/e6/supervisor/status` | `std_msgs/String` | supervisor_node | 10 | RUNNING / STAGE_DONE:approach / FAIL_SAFETY / … |

| 서비스 | 타입 | 서버 노드 | 설명 |
|--------|------|----------|------|
| `/e6/task/set_prompt` | `std_srvs/SetBool` | task_node | 수동 프롬프트 변경 |
| `/e6/emergency_stop` | `std_srvs/Trigger` | supervisor_node | 즉시 정지 |

---

## 노드별 핵심 책임

### camera_node
- `hardware/camera_capture.py` 재사용 (HIKRobot / OpenCV fallback)
- 20Hz 타이머 → `sensor_msgs/Image` publish
- 이미지 224×224 RGB uint8 → encoding `rgb8`

### robot_state_node
- Dobot `feedBackData()` → joint deg 6D
- `GetToolDO(1)` → gripper 0/1
- state = `[j1, j2, j3, j4, j5, j6, gripper]` float32 7D

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
- 20Hz 타이머 → `chunk[index]` 꺼내서 MovJ / ToolDO 실행
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

## ⚠️ 필수 확장 계획: ZED X 스테레오 카메라 추가

> **이 항목은 반드시 구현해야 합니다.**

현재 파이프라인은 탑뷰 카메라 **1채널**만 사용합니다 (`observation/exterior_image_1_left`).  
추후 **ZED X 카메라를 추가하여 이미지를 2채널로 확장**할 계획입니다.

### 목표 obs 구조 (ZED X 추가 후)

| 키 | 현재 | ZED X 추가 후 |
|----|------|--------------|
| `observation/exterior_image_1_left` | HIKRobot 탑뷰 | HIKRobot 탑뷰 (유지) |
| `observation/exterior_image_1_right` | ❌ 없음 | **ZED X 좌/우 중 하나** |
| `observation/state` | (7,) deg | (7,) deg (동일) |

또는 ZED X를 탑뷰 단독으로 교체하거나 와이드/클로즈업 시점 2채널로 구성할 수 있음.  
최종 채널 배치는 **학습 데이터 수집 시 사용한 카메라 배치와 반드시 일치**해야 합니다.

### camera_node 확장 포인트

```
/e6/camera/image          ← 현재 (HIKRobot 탑뷰)
/e6/camera/image_zed      ← 추가 예정 (ZED X)
```

`inference_bridge_node`의 obs 조립 시 두 이미지를 모두 사용:

```python
# ZED X 추가 후 obs 예시
obs = {
    "observation/exterior_image_1_left":  hik_img,   # HIKRobot
    "observation/exterior_image_1_right": zed_img,   # ZED X
    "observation/state":                  state_7,
    "prompt":                             prompt,
}
```

### 체크리스트

- [ ] ZED X SDK (`pyzed`) Jetson 설치 확인
- [ ] `camera_node.py`에 ZED X 토픽 추가 (`/e6/camera/image_zed`)
- [ ] 학습 시 2채널 obs 계약 확정 (`e6_policy.py` `E6Inputs` 수정)
- [ ] `inference_bridge_node.py` obs 조립 2채널로 업데이트
- [ ] 2채널 기준으로 데이터 재수집 및 재학습

---

*현재 구현 브랜치: `feature/ros2-integration`*  
*관련 레포: [6DOF-VLA](https://github.com/Kyle-Riss/E6-VLA_INFERENCE) (학습 파이프라인)*
