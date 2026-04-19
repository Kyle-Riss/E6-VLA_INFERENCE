# E6-VLA ROS2 파이프라인 플로우차트

## 노드 구조

```
┌─────────────────────────────────────────┐
│           serve_policy.py               │
│        π0.5 모델 (GPU 점유)              │
│        WebSocket :8000                  │
└────────────────┬────────────────────────┘
                 │ WebSocket (localhost)
                 │
─────────────────┼──────── ROS2 영역 ─────────────────────────────
                 │
                 ▼
┌──────────────────────────────┐      /e6/policy/action_chunk
│    inference_bridge_node     │ ───────────────────────────────────┐
│  - obs 조립                  │                                    │
│  - WebSocket client          │                                    ▼
│  - 추론 스레드 분리           │        ┌──────────────────────────────┐
└───┬──────────────────────────┘        │  executor_supervisor_node    │
    │                                   │  - MovJ / ToolDO 실행        │
    │ 구독                              │  - bad_camera 감시           │
    │                                   │  - min_tool_z 감시           │
    │ /e6/camera/image     (20Hz)       │  - emergency_stop 서비스     │
    │ /e6/camera/zed_image (20Hz)       └──────────────┬───────────────┘
    │ /e6/robot/state      (20Hz)                       │
    │ /e6/task/prompt      (이벤트)                     │ /e6/supervisor/status
    │                                                  │ (RUNNING / STAGE_DONE / FAIL_*)
    │         ┌────────────────────────┐               │
    │         │   camera_state_node    │               ▼
    │         │  - HIKRobot 224x224    │   ┌────────────────────────────┐
    │         │  - ZED 224x224 (left)  │   │         task_node          │
    │         │  - Dobot feedBack      │   │  - task_sequence 상태머신  │
    │         │  - 20Hz 타이머         │   │  - STAGE_DONE 받으면       │
    │         └──────────┬─────────────┘   │    다음 stage로 전환        │
    │                    │                 │  - /e6/task/prompt 발행    │
    │ /e6/camera/image   │                 └────────────────────────────┘
    │ /e6/camera/zed_image│
    │ /e6/robot/state    │
    │ /e6/robot/tcp_z    │
    └────────────────────┘

                          Dobot E6 로봇
                    (192.168.5.1:29999 / 30005)
```

## 단순 흐름

```
[카메라 + 로봇상태]
        │
        ▼
[camera_state_node]
        │
        ├── /e6/camera/image       (HIK)
        ├── /e6/camera/zed_image   (ZED left)
        ├── /e6/robot/state
        └── /e6/robot/tcp_z
                │
                ▼
      [inference_bridge_node]
      obs 조립 + WebSocket 요청
                │
                ▼
         [serve_policy.py]
           π0.5 추론 서버
                │
         action_chunk 반환
                ▼
   [executor_supervisor_node]
   MovJ / ToolDO / 안전감시
                │
                ▼
            [Dobot E6]

[task_node]
  └─ prompt 관리 / stage 전환
     └─ /e6/task/prompt 발행
```

## 데이터 흐름

```
카메라/로봇 상태
→ camera_state_node
→ inference_bridge_node
→ serve_policy.py
→ action_chunk
→ executor_supervisor_node
→ Dobot E6 실행
→ supervisor status
→ task_node가 다음 prompt 결정
```

## 타이밍

```
t=0ms      camera_state_node 20Hz 타이머
           → /e6/camera/image     발행  (HIK)
           → /e6/camera/zed_image 발행  (ZED left)
           → /e6/robot/state      발행
           → /e6/robot/tcp_z      발행

t=0ms      inference_bridge_node obs 캐시 업데이트
           → inference_running=False 이면 추론 스레드 시작

t=0~1600ms serve_policy.py 추론 (GPU, Jetson ~1.5초)

t=1600ms   inference_bridge_node 결과 수신
           → /e6/policy/action_chunk 발행 (16×7)

t=1600ms   executor_supervisor_node chunk 교체, index=0 리셋

t=1650ms   executor_timer (20Hz) → chunk[0] → MovJ
t=1700ms                          → chunk[1] → MovJ
...
t=2400ms                          → chunk[15] → MovJ (16/20Hz = 0.8초)

           → 다음 추론 트리거 (최신 obs 사용)
```

## 토픽 / 서비스

| 토픽 | 타입 | 발행 노드 | Hz |
|------|------|----------|----|
| `/e6/camera/image` | `sensor_msgs/Image` | camera_state_node | 20 |
| `/e6/camera/zed_image` | `sensor_msgs/Image` | camera_state_node | 20 |
| `/e6/robot/state` | `std_msgs/Float32MultiArray` | camera_state_node | 20 |
| `/e6/robot/tcp_z` | `std_msgs/Float32` | camera_state_node | 20 |
| `/e6/task/prompt` | `std_msgs/String` | task_node | 이벤트 |
| `/e6/policy/action_chunk` | `std_msgs/Float32MultiArray` | inference_bridge_node | ~0.6 |
| `/e6/supervisor/status` | `std_msgs/String` | executor_supervisor_node | 10 |

| 서비스 | 타입 | 설명 |
|--------|------|------|
| `/e6/emergency_stop` | `std_srvs/Trigger` | 즉시 정지 |

## 실행 순서

```
# 터미널 1 — 정책 서버
bash ~/e6-vla/run_server.sh <체크포인트 경로>

# 터미널 2 — ROS2 런치
source /opt/ros/humble/setup.bash
cd ~/e6-vla/ros2 && source install/setup.bash
ros2 launch e6_vla_ros e6_vla.launch.py \
  task_sequence:="approach,pick,move_left,place_left"

# 터미널 3 — 모니터링
ros2 topic echo /e6/supervisor/status
ros2 topic hz   /e6/policy/action_chunk
ros2 topic echo /e6/policy/action_chunk

# 긴급 정지
ros2 service call /e6/emergency_stop std_srvs/srv/Trigger
```

## 파라미터 (launch 인자)

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `robot_ip` | `192.168.5.1` | Dobot E6 IP |
| `server_host` | `127.0.0.1` | serve_policy 호스트 |
| `server_port` | `8000` | serve_policy 포트 |
| `task_sequence` | `approach` | 쉼표 구분 stage 목록 |
| `stage_timeout_sec` | `0.0` | stage 자동 전환 시간 (0=수동) |
| `loop_sequence` | `false` | 마지막 stage 후 처음으로 |
| `dry_run` | `false` | 로봇 없이 실행 |
| `no_camera` | `false` | 카메라 없이 실행 |
| `max_delta_deg` | `3.0` | 관절 최대 이동량 (도) |
| `min_tool_z` | `101.0` | TCP Z 안전 하한 (mm) |
