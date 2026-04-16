# E6-VLA 개발 로드맵

> 이 파일은 개발 우선순위와 현재 상태를 기록합니다.
> 새 세션이 시작되면 이 파일을 먼저 읽고 작업을 이어가세요.

---

## 현재 브랜치 상태

| 브랜치 | 상태 | 설명 |
|--------|------|------|
| `main` | 안정 | 단일 스크립트 파이프라인 (run_server.sh + run_client.sh) |
| `feature/ros2-integration` | **진행 중** | ROS2 4노드 파이프라인 — 카메라·로봇 연결 테스트 남음 |
| `feature/web-bridge` | 예정 | FastAPI web_bridge_node 추가 |

---

## 완료된 작업

### ✅ 단일 스크립트 파이프라인 (main)
- `run_server.sh` + `run_client.sh` 정상 동작
- Jetson aarch64 호환성 수정 (`MVCAM_COMMON_RUNENV=lib`, `TORCHDYNAMO_DISABLE=1`)
- `pi05_e6_v1_lora` config (E6Inputs, 7D degree, action_horizon=16)

### ✅ ROS2 4노드 구현 (feature/ros2-integration)
- **camera_state_node** — HIKRobot + Dobot feedBack 20Hz 발행
- **inference_bridge_node** — obs 조립 + WebSocket 비블로킹 추론
- **executor_supervisor_node** — MovJ/ToolDO 실행 + 안전 감시
- **task_node** — task_sequence 상태머신 + prompt 관리
- `dry_run + no_camera` 테스트 통과
- `serve_policy.py` WebSocket 연동 및 추론 (16×7) 확인

---

## 우선순위별 다음 작업

### 🔴 P1 — 즉시 (feature/ros2-integration에서)

#### 1. 카메라 연결 테스트
```bash
# HIKRobot 연결 후
ros2 launch e6_vla_ros e6_vla.launch.py dry_run:=true task_sequence:="approach"
# → /e6/camera/image 정상 발행 확인
ros2 topic hz /e6/camera/image   # 20Hz 나와야 함
```

#### 2. 로봇 연결 테스트
```bash
# Dobot E6 연결 후
ros2 launch e6_vla_ros e6_vla.launch.py no_camera:=true task_sequence:="approach"
# → /e6/robot/state, /e6/robot/tcp_z 정상 발행 확인
ros2 topic echo /e6/robot/state
```

#### 3. 전체 파이프라인 실제 동작 테스트
```bash
# 터미널 1
bash ~/e6-vla/run_server.sh ~/checkpoints/pi05_e6_v1_lora/e6_primitive_v1_bs8_nowb_20260408_1607/pytorch_from_jax

# 터미널 2
ros2 launch e6_vla_ros e6_vla.launch.py task_sequence:="approach"

# 확인
ros2 topic echo /e6/supervisor/status   # RUNNING 유지되는지
ros2 topic hz /e6/policy/action_chunk   # ~0.6Hz
```

#### 4. stage_timeout 기반 자동 전환 테스트
```bash
ros2 launch e6_vla_ros e6_vla.launch.py \
  task_sequence:="approach,pick,move_left,place_left" \
  stage_timeout_sec:=8.0
```

---

### 🟡 P2 — ROS2 안정화 후 (feature/web-bridge 브랜치)

#### FastAPI web_bridge_node 추가

**목표:** 브라우저에서 로봇 상태 모니터링 + 제어

**구현 파일:** `ros2/src/e6_vla_ros/e6_vla_ros/web_bridge_node.py`

**제공 엔드포인트:**
```
GET  /status          supervisor/status, 노드 생존 여부
GET  /camera/latest   최신 프레임 (JPEG)
GET  /camera/stream   MJPEG 스트림
WS   /ws/live         실시간 상태 (robot/state, action_chunk Hz)
POST /prompt          task_prompt 변경
POST /emergency_stop  긴급 정지
POST /stage_next      수동 stage 전환
```

**핵심 패턴 (rclpy + asyncio 공존):**
```python
# rclpy는 별도 스레드
threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

# 공유 상태 (thread-safe dict)
shared = {"status": "RUNNING", "robot_state": [], "latest_frame": None}

# FastAPI uvicorn (포트 9000)
uvicorn.run(app, host="0.0.0.0", port=9000)
```

**브랜치 작업 순서:**
```bash
git checkout feature/ros2-integration
git checkout -b feature/web-bridge
# web_bridge_node.py 작성
# setup.py entry_points에 추가
# launch 파일에 선택적 노드로 추가 (use_web_bridge:=true 파라미터)
```

---

### 🟢 P3 — 추후 (장기)

#### 브라우저 대시보드 UI
- 카메라 피드 실시간 표시
- 로봇 관절 상태 그래프
- supervisor status 표시
- prompt / stage 제어 버튼

#### main 브랜치 병합
- ROS2 파이프라인 안정화 후 main에 머지
- `feature/web-bridge` → `feature/ros2-integration` → `main` 순서

---

## 파일 구조 (현재)

```
e6-vla/
├── ROADMAP.md                         ← 이 파일 (우선순위 기준)
├── run_server.sh                      # 정책 서버 실행
├── run_client.sh                      # 단일 스크립트 클라이언트
├── ros2/
│   └── src/e6_vla_ros/
│       ├── e6_vla_ros/
│       │   ├── camera_state_node.py       # HIKRobot + feedBack 20Hz
│       │   ├── inference_bridge_node.py   # obs 조립 + WebSocket 추론
│       │   ├── executor_supervisor_node.py# MovJ/ToolDO + 안전 감시
│       │   ├── task_node.py               # task_sequence 상태머신
│       │   └── web_bridge_node.py         # (P2) FastAPI 브리지 — 미구현
│       └── launch/e6_vla.launch.py
├── hardware/
│   ├── camera_capture.py              # HIKRobot MVS
│   └── dobot/dobot_api.py             # DobotApiDashboard / DobotApiFeedBack
├── docs/
│   ├── ROS2_FLOWCHART.md
│   ├── ROS2_ARCHITECTURE.md
│   └── ROS2_IMPLEMENTATION_PLAN.md
└── checkpoints/
    └── pi05_e6_v1_lora/
        └── e6_primitive_v1_bs8_nowb_20260408_1607/pytorch_from_jax
```

---

## 주요 경로 / 설정값

| 항목 | 값 |
|------|-----|
| 체크포인트 | `~/checkpoints/pi05_e6_v1_lora/e6_primitive_v1_bs8_nowb_20260408_1607/pytorch_from_jax` |
| 가상환경 | `~/move-one/min-imum/move-one/bin/activate` |
| 로봇 IP | `192.168.5.1` |
| 정책 서버 포트 | `8000` |
| FastAPI 포트 (예정) | `9000` |
| Dobot Dashboard | `192.168.5.1:29999` → `DobotApiDashboard` |
| Dobot FeedBack | `192.168.5.1:30005` → `DobotApiFeedBack` |
| 카메라 SDK | `MVCAM_COMMON_RUNENV=/opt/MVS/lib` |

---

## 빠른 실행 참조

```bash
# ROS2 빌드
source /opt/ros/humble/setup.bash
cd ~/e6-vla/ros2 && colcon build --symlink-install && source install/setup.bash

# dry_run 테스트
ros2 launch e6_vla_ros e6_vla.launch.py dry_run:=true no_camera:=true task_sequence:="approach"

# 실제 실행 (서버 먼저 띄운 후)
ros2 launch e6_vla_ros e6_vla.launch.py task_sequence:="approach,pick,move_left,place_left"

# 모니터링
ros2 topic echo /e6/supervisor/status
ros2 topic hz   /e6/policy/action_chunk
ros2 topic list

# 긴급 정지
ros2 service call /e6/emergency_stop std_srvs/srv/Trigger
```
