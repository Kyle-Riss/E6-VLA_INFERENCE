# ROS2 노드 구현 계획 — 내일 아침 작업 가이드

## 전제 조건 확인

```bash
# 서버 실행 (터미널 1, 이미 실행 중이면 스킵)
bash ~/e6-vla/run_server.sh ~/checkpoints/pi05_e6_v1_lora/e6_primitive_v1_bs8_nowb_20260408_1607/pytorch_from_jax

# ROS2 소싱 확인
source /opt/ros/humble/setup.bash
python3 -c "import rclpy; print('rclpy OK')"
which colcon
```

---

## 싱글 Jetson 프로세스 구조

```
Jetson AGX Orin (단일 머신)
│
├─ [Process 1] serve_policy.py         GPU 점유 (π0.5 추론)
│                                       WebSocket :8000 (localhost)
│
├─ [Process 2] ros2 launch             ROS2 노드들 (CPU)
│   ├─ camera_state_node               HIKRobot + Dobot feedBack
│   ├─ inference_bridge_node           WebSocket 클라이언트 + 스레드
│   ├─ executor_supervisor_node        MovJ/ToolDO + 안전 감시
│   └─ task_node                       task_sequence 상태머신
│
└─ [DDS] rmw_fastrtps — 노드 간 통신 (로컬 메모리, 오버헤드 무시 수준)
```

---

## 토픽 / 서비스 정의

```
/e6/camera/image          sensor_msgs/Image          20Hz  (224×224 RGB uint8)
/e6/robot/state           std_msgs/Float32MultiArray 20Hz  [j1..j6 deg, gripper 0~1]
/e6/robot/tcp_z           std_msgs/Float32           20Hz  TCP Z(mm)
/e6/task/prompt           std_msgs/String            이벤트 (QoS: transient_local)
/e6/policy/action_chunk   std_msgs/Float32MultiArray ~1Hz  (16×7 flatten)
/e6/supervisor/status     std_msgs/String            10Hz  RUNNING / STAGE_DONE:* / FAIL_*

서비스:
/e6/task/set_prompt       std_srvs/SetBool           수동 프롬프트 변경
/e6/emergency_stop        std_srvs/Trigger           즉시 정지
```

---

## 타이밍 전체 그림

```
t=0ms      camera_state_node 타이머 (50ms 주기)
           → /e6/camera/image publish
           → /e6/robot/state publish

t=0ms      inference_bridge_node 캐시 업데이트
           → inference_running=False 이면 추론 스레드 시작

t=0~2000ms serve_policy.py 추론 (GPU, ~2초)

t=2000ms   inference_bridge_node 결과 수신
           → /e6/policy/action_chunk publish (16개 액션)

t=2000ms   executor_supervisor_node 수신
           → chunk 교체, index=0 리셋

t=2050ms   executor_timer (20Hz) → chunk[0] → MovJ
t=2100ms                          → chunk[1] → MovJ
...
t=2800ms                          → chunk[15] → MovJ  (16/20Hz = 0.8초)

           → 다음 inference 트리거 (최신 obs 사용)
```

---

## 디렉토리 구조 (만들 것)

```
e6-vla/
└── ros2/
    └── src/
        └── e6_vla_ros/
            ├── package.xml
            ├── setup.py
            ├── setup.cfg
            ├── resource/
            │   └── e6_vla_ros          (빈 파일)
            ├── launch/
            │   └── e6_vla.launch.py
            └── e6_vla_ros/
                ├── __init__.py
                ├── camera_state_node.py
                ├── inference_bridge_node.py
                ├── executor_supervisor_node.py
                └── task_node.py
```

---

## 구현 순서

### Step 1 — 패키지 뼈대

**`package.xml`**
```xml
<?xml version="1.0"?>
<package format="3">
  <name>e6_vla_ros</name>
  <version>0.1.0</version>
  <description>E6-VLA ROS2 inference pipeline</description>
  <maintainer email="todo@todo.com">billy</maintainer>
  <license>Apache-2.0</license>
  <depend>rclpy</depend>
  <depend>sensor_msgs</depend>
  <depend>std_msgs</depend>
  <depend>std_srvs</depend>
  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

**`setup.py`**
```python
from setuptools import setup

setup(
    name='e6_vla_ros',
    version='0.1.0',
    packages=['e6_vla_ros'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/e6_vla_ros']),
        ('share/e6_vla_ros', ['package.xml']),
        ('share/e6_vla_ros/launch', ['launch/e6_vla.launch.py']),
    ],
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'camera_state_node          = e6_vla_ros.camera_state_node:main',
            'inference_bridge_node      = e6_vla_ros.inference_bridge_node:main',
            'executor_supervisor_node   = e6_vla_ros.executor_supervisor_node:main',
            'task_node                  = e6_vla_ros.task_node:main',
        ],
    },
)
```

**`setup.cfg`**
```ini
[develop]
script_dir=$base/lib/e6_vla_ros
[install]
install_scripts=$base/lib/e6_vla_ros
```

빌드:
```bash
source /opt/ros/humble/setup.bash
cd ~/e6-vla/ros2
colcon build --symlink-install
source install/setup.bash
```

---

### Step 2 — `camera_state_node.py`

**이식 출처:** `run_e6_client.py` → `camera.get_frame()` + `feedBackData()` + `GetToolDO`

**핵심 구조:**
```python
class CameraStateNode(Node):
    def __init__(self):
        super().__init__('camera_state_node')
        # 퍼블리셔
        self.img_pub   = self.create_publisher(Image, '/e6/camera/image', 10)
        self.state_pub = self.create_publisher(Float32MultiArray, '/e6/robot/state', 10)
        self.tcpz_pub  = self.create_publisher(Float32, '/e6/robot/tcp_z', 10)

        # 하드웨어 초기화
        self.camera  = CameraCapture()       # hardware/camera_capture.py
        self.feed    = FeedBackApis(ROBOT_IP, 30005)
        self.dashboard = DashboardApis(ROBOT_IP, 29999)

        # 20Hz 타이머
        self.timer = self.create_timer(0.05, self.tick)
        self._last_state = np.zeros(7, dtype=np.float32)

    def tick(self):
        # 1) 이미지
        frame = self.camera.get_frame()   # (224,224,3) uint8 RGB
        self.img_pub.publish(numpy_to_image(frame))

        # 2) 로봇 상태
        fb = self.feed.feedBackData()
        deg6 = fb["QActual"][0][:6]
        gripper = read_gripper(self.dashboard)
        state = np.array([*deg6, gripper], dtype=np.float32)
        self._last_state = state

        msg = Float32MultiArray(data=state.tolist())
        self.state_pub.publish(msg)

        # 3) TCP Z
        tcp_z = fb["ToolVectorActual"][0][2]
        self.tcpz_pub.publish(Float32(data=float(tcp_z)))
```

---

### Step 3 — `inference_bridge_node.py`

**이식 출처:** `run_e6_client.py` → `policy.infer(obs)` 블록

**핵심: 추론 스레드 분리**
```python
class InferenceBridgeNode(Node):
    def __init__(self):
        super().__init__('inference_bridge_node')

        # 캐시 (latest 유지)
        self._latest_img    = None   # np.ndarray (224,224,3) uint8
        self._latest_state  = None   # np.ndarray (7,) float32
        self._latest_prompt = "approach red object"
        self._inference_running = False
        self._executor = ThreadPoolExecutor(max_workers=1)

        # 구독
        self.create_subscription(Image,              '/e6/camera/image',  self.cb_img,    10)
        self.create_subscription(Float32MultiArray,  '/e6/robot/state',   self.cb_state,  10)
        self.create_subscription(String,             '/e6/task/prompt',   self.cb_prompt, qos_transient)

        # 발행
        self.chunk_pub = self.create_publisher(Float32MultiArray, '/e6/policy/action_chunk', 10)

        # WebSocket 연결 (서버가 뜰 때까지 대기)
        self.policy = WebsocketClientPolicy(host='127.0.0.1', port=8000)

        # 추론 트리거 타이머 (action_horizon/hz = 16/20 = 0.8초)
        self.create_timer(0.8, self.maybe_infer)

    def maybe_infer(self):
        if self._inference_running:
            return      # 이전 추론 아직 진행 중 → 스킵
        if self._latest_img is None or self._latest_state is None:
            return      # obs 아직 없음

        # obs 스냅샷 (추론 중 덮어써도 안전)
        obs = {
            "observation/exterior_image_1_left": self._latest_img.copy(),
            "observation/state":                 self._latest_state.copy(),
            "prompt":                            self._latest_prompt,
        }
        self._inference_running = True
        self._executor.submit(self._run_infer, obs)

    def _run_infer(self, obs):
        try:
            result  = self.policy.infer(obs)          # ~2초 블로킹
            actions = np.asarray(result["actions"])   # (16, 7)
            msg = Float32MultiArray(data=actions.flatten().tolist())
            self.chunk_pub.publish(msg)
        finally:
            self._inference_running = False
```

---

### Step 4 — `executor_supervisor_node.py`

**이식 출처:** `run_e6_client.py` → MovJ/ToolDO + bad_camera + min_tool_z + _stage_complete

**핵심 구조:**
```python
class ExecutorSupervisorNode(Node):
    def __init__(self):
        super().__init__('executor_supervisor_node')

        # 청크 상태
        self._chunk   = None          # np.ndarray (16, 7)
        self._chunk_i = 0
        self._chunk_t = 0.0           # 수신 시각 (staleness 체크)
        self._last_gripper = 0

        # 구독
        self.create_subscription(Float32MultiArray, '/e6/policy/action_chunk', self.cb_chunk,  10)
        self.create_subscription(Image,             '/e6/camera/image',        self.cb_img,    10)
        self.create_subscription(Float32,           '/e6/robot/tcp_z',         self.cb_tcpz,   10)
        self.create_subscription(Float32MultiArray, '/e6/robot/state',         self.cb_state,  10)

        # 발행
        self.status_pub = self.create_publisher(String, '/e6/supervisor/status', 10)

        # 서비스
        self.create_service(Trigger, '/e6/emergency_stop', self.cb_estop)

        # 하드웨어
        self.dashboard = DashboardApis(ROBOT_IP, 29999)

        # executor_timer  20Hz
        self.create_timer(0.05,  self.executor_tick)
        # supervisor_timer 10Hz
        self.create_timer(0.10,  self.supervisor_tick)

    def executor_tick(self):
        if self._chunk is None:
            return
        # staleness 체크 (5초 초과 시 폐기)
        if time.monotonic() - self._chunk_t > 5.0:
            self._chunk = None
            return
        if self._chunk_i >= len(self._chunk):
            return

        a = self._chunk[self._chunk_i]        # (7,) float32
        delta_deg = np.clip(a[:6], -MAX_DELTA_DEG, MAX_DELTA_DEG)
        gripper   = int(a[6] >= GRIP_THRESHOLD)

        # MovJ
        target = self._current_deg[:6] + delta_deg
        j1,j2,j3,j4,j5,j6 = target
        self.dashboard.MovJ(j1,j2,j3,j4,j5,j6, 1, v=70, a=60)
        self.dashboard.ToolDO(1, gripper)
        self._last_gripper = gripper
        self._chunk_i += 1

    def supervisor_tick(self):
        status = "RUNNING"

        # bad camera
        if self._frame_mean < 8.0:
            self._bad_streak += 1
            if self._bad_streak > 10:
                status = "FAIL_SAFETY:bad_camera"
        else:
            self._bad_streak = 0

        # min_tool_z
        if self._tcp_z is not None and self._tcp_z <= MIN_TOOL_Z:
            status = "FAIL_SAFETY:min_tool_z"

        # stage 완료 판정
        done = _stage_complete(self._stage, self._tcp_z, self._last_gripper,
                               APPROACH_Z_DONE, LIFT_Z_DONE, HOME_Z_DONE)
        if done:
            status = f"STAGE_DONE:{self._stage}"

        self.status_pub.publish(String(data=status))
```

---

### Step 5 — `task_node.py`

**역할:** task_sequence 상태머신. 하드웨어 I/O 없음.

```python
TASK_PRESETS = {
    "approach":    "approach red object",
    "pick":        "pick red object",
    "move_left":   "move object to left",
    "move_right":  "move object to right",
    "move_middle": "move object to middle",
    "place_left":  "place object to left",
    "place_right": "place object to right",
    "place_middle":"place object to middle",
}

class TaskNode(Node):
    def __init__(self):
        super().__init__('task_node')
        seq_str = self.declare_parameter('task_sequence', 'approach').value
        self._seq   = [s.strip() for s in seq_str.split(',')]
        self._idx   = 0

        # transient_local: 늦게 구독해도 최신값 받음
        qos = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        self.prompt_pub = self.create_publisher(String, '/e6/task/prompt', qos)

        self.create_subscription(String, '/e6/supervisor/status', self.cb_status, 10)

        self._publish_current()

    def cb_status(self, msg):
        if msg.data.startswith('STAGE_DONE:'):
            self._idx += 1
            if self._idx < len(self._seq):
                self._publish_current()
            else:
                self.get_logger().info('전체 task_sequence 완료')

    def _publish_current(self):
        key    = self._seq[self._idx]
        prompt = TASK_PRESETS.get(key, key)
        self.prompt_pub.publish(String(data=prompt))
        self.get_logger().info(f'stage {self._idx}: {prompt}')
```

---

### Step 6 — `launch/e6_vla.launch.py`

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('task_sequence', default_value='approach,pick,move_left,place_left'),
        DeclareLaunchArgument('robot_ip',      default_value='192.168.5.1'),
        DeclareLaunchArgument('server_host',   default_value='127.0.0.1'),
        DeclareLaunchArgument('server_port',   default_value='8000'),

        Node(package='e6_vla_ros', executable='camera_state_node',
             parameters=[{'robot_ip': LaunchConfiguration('robot_ip')}]),

        Node(package='e6_vla_ros', executable='inference_bridge_node',
             parameters=[{'server_host': LaunchConfiguration('server_host'),
                          'server_port': LaunchConfiguration('server_port')}]),

        Node(package='e6_vla_ros', executable='executor_supervisor_node',
             parameters=[{'robot_ip':      LaunchConfiguration('robot_ip'),
                          'max_delta_deg': 3.0,
                          'min_tool_z':    80.0}]),

        Node(package='e6_vla_ros', executable='task_node',
             parameters=[{'task_sequence': LaunchConfiguration('task_sequence')}]),
    ])
```

---

### Step 7 — 빌드 & 실행

```bash
# 1. ROS2 소싱
source /opt/ros/humble/setup.bash

# 2. 빌드
cd ~/e6-vla/ros2
colcon build --symlink-install
source install/setup.bash

# 3. 전체 런치 (터미널 2, 서버는 터미널 1에서 이미 실행 중)
ros2 launch e6_vla_ros e6_vla.launch.py \
  task_sequence:="approach,pick,move_left,place_left"

# 수동 프롬프트 override (디버깅용)
ros2 topic pub /e6/task/prompt std_msgs/msg/String "data: 'pick red object'" --once

# 상태 모니터링
ros2 topic echo /e6/supervisor/status
ros2 topic hz /e6/policy/action_chunk

# 긴급 정지
ros2 service call /e6/emergency_stop std_srvs/srv/Trigger
```

---

## 기존 코드 → 노드 이식 맵

| `run_e6_client.py` 블록 | 이식할 노드 |
|---|---|
| `camera.get_frame()` | `camera_state_node` |
| `feedBackData()` + `GetToolDO` | `camera_state_node` |
| `policy.infer(obs)` + obs 조립 | `inference_bridge_node` |
| `MovJ / ToolDO` 실행 | `executor_supervisor_node` |
| `bad_camera_streak` 체크 | `executor_supervisor_node` |
| `min_tool_z` 체크 | `executor_supervisor_node` |
| `_stage_complete()` | `executor_supervisor_node` |
| `task_sequence` 상태머신 | `task_node` |

---

## ⚠️ ZED X 카메라 추가 (필수 — 별도 작업)

2채널 이미지 파이프라인 확장 계획은 `docs/ROS2_ARCHITECTURE.md` 참고.  
`camera_state_node`에 `/e6/camera/image_zed` 토픽 추가 필요.

---

*브랜치: `feature/ros2-integration`*
