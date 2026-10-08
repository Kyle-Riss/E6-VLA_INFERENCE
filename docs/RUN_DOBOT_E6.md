# Dobot E6 실행 가이드

Dobot E6 를 VLA 정책으로 움직이는 데 필요한 폴더와 실행 순서.
(2026-10-08 기준, Jetson AGX Orin)

> 문서 전체의 읽는 순서는 [README](../README.md#-문서-읽는-순서) 참고. 이 문서는 **2번**이다.

---

## 폴더 구성

| 위치 | 역할 |
|---|---|
| `run_server_v*.sh` | 정책 서버 (터미널 1). 버전별로 하나씩 |
| `ros2/src/e6_vla_ros/launch/e6_vla.launch.py` | ROS2 전체 실행 (터미널 2) |
| `ros2/src/e6_vla_ros/e6_vla_ros/` | 노드 소스 — **편집은 여기서만** |
| `hardware/camera_capture.py` | HIK 카메라 설정 |
| `scripts/serve_policy.py` | 서버 스크립트가 호출하는 정책 서버 본체 |
| `/mnt/robotdata/e6_checkpoints/` | 체크포인트 (외장 HDD, v1~v26) |
| `~/move-one/min-imum/move-one/` | Python venv (서버 스크립트가 자동 활성화) |

> ⚠️ `ros2/build/e6_vla_ros/e6_vla_ros/` 는 `src/` 를 가리키는 **디렉토리 심볼릭 링크**다.
> build/ 경로로 파일을 지우면 원본이 지워지고, 개별 파일에 `ln -s` 를 만들면 순환 링크가 생긴다.
> 소스는 `src/` 에서만 고칠 것. `colcon build` 재실행은 필요 없다.

### ROS2 노드 5개

| 노드 | 역할 |
|---|---|
| `camera_state_node` | HIK + ZED 영상, 로봇 상태 발행 (피드백 포트 30005, 읽기 전용) |
| `inference_bridge_node` | 관측 조립 → WebSocket(:8000) → `/e6/policy/action_chunk` |
| `executor_supervisor_node` | **실제로 로봇을 움직이는 노드** (대시보드 포트 29999) + 안전 감시 |
| `task_node` | phase 별 프롬프트 발행 |
| `voice_command_node` | 음성/텍스트 명령 (`use_voice:=true` 일 때만) |

---

## 실행

```bash
# 터미널 1 — 정책 서버 (체크포인트 경로를 반드시 직접 지정)
cd ~/E6-VLA_INFERENCE
bash run_server_v17.sh /mnt/robotdata/e6_checkpoints/e6_v17_15000

# 터미널 2 — ROS2
cd ~/E6-VLA_INFERENCE/ros2 && source install/setup.bash
ros2 launch e6_vla_ros e6_vla.launch.py
```

- 추천 모델 **v17**. launch 기본값이 v16/v17 에 맞춰져 있어 인자 없이 동작한다.
- 처음엔 `dry_run:=true` 로 로봇을 움직이지 않고 배선만 확인할 것.
- 로봇 IP 기본값 `192.168.5.1` (`robot_ip:=...` 로 변경).

### v23 + ServoJ

```bash
bash run_server_v23.sh /mnt/robotdata/e6_checkpoints/e6_v23_19999
ros2 launch e6_vla_ros e6_vla.launch.py control_mode:=servoj servoj_t:=0.0625
```

### 주요 launch 인자 (기본값)

| 인자 | 기본값 | 비고 |
|---|---|---|
| `action_mode` | `delta` | v6 만 `absolute` |
| `gripper_mode` | `absolute` | v13/v14 는 `delta` |
| `prompt_mode` | `per_frame_v16` | |
| `control_mode` | `movj` | `servoj` 가능 |
| `infer_hz` / `steps_per_inference` | `2.0` / `8` | 16개 청크 중 8개 실행 |
| `executor_hz` | `16.0` | |
| `max_delta_deg` | `3.0` | 프레임당 관절 변화 상한 |
| `min_tool_z` | `75.0` | |
| `record_mcap` | `false` | |
| `foxglove` | `true` | 포트 8765 |

### 버전별 action 규약

| 버전 | joint | gripper | launch |
|---|---|---|---|
| v6 | absolute | absolute | `action_mode:=absolute` |
| v8~v13 | delta | delta(누산) | `gripper_mode:=delta` |
| v14 | delta | delta(누산), state 8D | 서버가 자동 변환 |
| **v16/v17~** | delta | absolute | 기본값 |

---

## ⚠️ 실행 전 확인 3가지

**1. 체크포인트 기본 경로가 깨져 있다.**
여러 `run_server_v*.sh` 의 기본값이 옛 마운트 경로 `/media/billye6/새 볼륨1/...` 이다.
HDD 는 이제 `/mnt/robotdata` 로 고정 마운트되므로 **경로를 인자로 직접 넣어야** 한다.
MCAP 도 마찬가지: `record_mcap:=true mcap_output_dir:=/mnt/robotdata/Dobot/inference_mcap`

**2. HIK 카메라 화각이 학습 때와 다를 수 있다.**
`camera_capture.py` 는 노출(20ms)·게인(10dB)만 설정하고 **AOI 와 gamma 는 건드리지 않는다.**
다른 프로젝트(xArm)에서 설정한 AOI·gamma 는 **카메라 펌웨어에 저장돼 남는다.**
그러면 에러 없이 학습과 다른 이미지가 정책에 들어간다.
→ 실행 전 HIK 영상이 학습 때와 같은 전체 화면인지 눈으로 확인할 것.

**3. HIK 카메라가 두 대면 다른 카메라를 열 수 있다.**
`camera_capture.py` 가 처음 발견한 장치(`pDeviceInfo[0]`)를 연다.
Dobot 을 쓸 때는 다른 HIK 카메라를 빼 둘 것.

---

## 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| `[오류] 체크포인트 폴더 없음` | 위 1번 — 경로를 인자로 지정 |
| `EnableRobot` 에서 Broken pipe | 포트 29999 를 이전 프로세스가 점유. 이전 실행을 종료 |
| norm_stats 로드 실패 | 데이터셋 이름 대소문자 (`Kyle-Riss`, v6 이후) |
| v10~v12 그리퍼 이상 | norm_stats gripper q01=-1.0 / q99=+1.0 수동 패치 필요 |
| 정책 서버 `libcudss.so.0` 에러 | 서버 스크립트를 쓰지 않고 직접 실행한 경우 — `LD_LIBRARY_PATH` 누락 |
