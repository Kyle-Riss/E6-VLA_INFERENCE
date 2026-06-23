# E6-VLA Layer Architecture — Context / Policy / Execution

`ROS2_ARCHITECTURE.md`가 "토픽이 어떻게 흐르는가"를 다룬다면, 이 문서는 "이 파이프라인을
개념적으로 어떤 3계층으로 볼 것인가"를 다룬다.

```
Context Structuring Layer  →  π0.5 VLA Policy  →  Rule-Based Constrained Execution Layer
   (task_node.py)              (pi05_e6_v*_lora)        (executor_supervisor_node.py)
```

## 배경

외부 자문(2026-06-16, Claude.ai 대화)에서 π0.5를 중심으로 앞단을 "MCP"(Model Context
Protocol), 뒷단을 "MPC"(Model Predictive Control)로 부르는 3계층 프레이밍을 제안받았다.
개념은 유용하지만 두 용어 모두 **현재 구현과 정확히 일치하지 않는다** — 코드를 직접 확인한
결과는 아래와 같다.

## 용어 교정 매핑표

| 제안된 용어 | 실제 구현 | 정확한 명칭 |
|---|---|---|
| MCP context interface | `task_node.py` `PhaseTracker.update()` (L120-188) + `V16_PHASE_PROMPTS` (L44-61) | **Context Structuring Layer** |
| π0.5 VLA policy | `pi05_e6_v*_lora` (v16~v26), 16-step action chunk, 앞 8개만 실행 | π0.5 기반 VLA policy (그대로 정확함) |
| MPC execution layer | `executor_supervisor_node.py` `_executor_tick()` (L449-950) | **Rule-Based Constrained Execution Layer** |

### Context Structuring Layer 상세 (`task_node.py`)

입력은 `gripper_raw`(0~1)와 `tcp_z`(mm) 두 개뿐이며, 하드코딩된 if/elif 우선순위 규칙으로
phase(`approach/grasp/lift/transport/place/release/return`)를 분류한 뒤, 고정 딕셔너리에서
문자열 프롬프트를 골라 `/e6/task/prompt`에 발행한다 (`_phase_tick_v16`, L466-484).
`voice_command_node`도 동일하게 변환된 문자열을 그대로 발행한다.

→ scene_objects, robot_state, tool 목록 등을 구조화된 리소스로 노출하는 프로토콜이 아니라
**(센서값 2개 → 규칙 기반 분류 → 문자열 1개)** 변환기에 가깝다.

### Rule-Based Constrained Execution Layer 상세 (`executor_supervisor_node.py`)

action chunk 한 스텝마다 순서대로: ① 속도 클램프 `max_delta_deg` (L507-530) → ② URDF
위치 클램프 (L532-542) → ③ gripper hysteresis (L562-580) → ④ 상황별 가드들
(`grip_enable_z`, transport OFF 차단, `place_force_release`, lift grip hold; L591-648) →
⑤ 막히면 `RelMovLUser` 기반 scripted lift/return fallback (L659-767) → ⑥ `ServoJ`/`MovJ`로
명령 전송 (L786-796).

→ 비용함수도 예측 호라이즌도 없다. **클램프 + 상황별 가드 + 막히면 스크립트 동작으로
대체**하는 구조다.

## 진짜 MCP/MPC와의 격차

- **진짜 MCP**: scene_objects / robot_state / available_tools를 구조화된 리소스로 노출하는
  프로토콜 서버 필요. 현재는 prompt 문자열 1개만 정책에 전달됨.
- **진짜 MPC**: action chunk를 따라가되 joint limit·속도·smoothness를 비용함수로 최적화하는
  예측 호라이즌 컨트롤러 필요. 현재는 hard clamp + 가드 규칙 나열.

## 다음 실험 단계 — 2×2 ablation plan (2026-06-16 확정)

### MCP는 π0.5에 직접 못 붙는다는 점 먼저 확인

MCP(Model Context Protocol)는 "tool/resource를 스스로 호출할지 판단하는 에이전트"가
있어야 의미가 있는 패턴인데, π0.5는 고정 입력(image+prompt+state)을 받아 action chunk를
내는 정책일 뿐 스스로 tool 호출을 결정할 능력이 없다. 따라서 "MCP 실험"은 **π0.5 앞에
Context Agent를 하나 추가**하는 것으로 구현한다 (PhaseTracker를 대체).

```
camera/robot state
        ↓
[MCP tool server]  ← get_scene_objects / get_robot_state / get_task_phase 등 노출
        ↓ 호출
[Context Agent = Claude API]   ← PhaseTracker 대체
        ↓ 자연어 prompt 생성
π0.5 policy
```

### 4-condition 2×2 ablation

| Condition | Context Layer | Execution Layer |
|---|---|---|
| Baseline | PhaseTracker (고정 prompt, 현재) | Rule-based clamp/guard (현재) |
| Context-only | Claude API Context Agent (MCP tool 호출) | Rule-based clamp/guard |
| Execution-only | PhaseTracker | 1-step lookahead smoothing (MPC-lite) |
| Full | Claude API Context Agent | 1-step lookahead smoothing |

- n=3 trials × 4 condition = 총 12 trials
- 모델/제어 모드 등 나머지 조건은 기존 paper 실험(v23, per_frame_v16)과 동일하게 고정 —
  control_mode는 ServoJ로 고정할지 별도 확인 필요

### MCP Context Agent 설계 (Claude API)

- `task_node.py`의 `PhaseTracker` 분류 결과(phase) + gripper/tcp_z/source·target side를
  MCP tool로 노출
- Claude API가 이 tool들을 호출해 현재 상황을 파악한 뒤 π0.5에 줄 자연어 prompt를 생성
- 미정: 매 추론(16Hz)마다 호출할지, phase 전환 시점에만 호출할지 — 네트워크 latency 때문에
  16Hz 그대로는 무리, phase 전환 시점 트리거가 현실적

### MPC-lite 설계 (1-step lookahead smoothing)

- 기존 클램프(`max_delta_deg`, URDF limit)는 유지
- 다음 target_deg 결정 시 현재 청크 스텝 + 다음 스텝 예상값을 함께 고려해 smoothing
  (예: exponential moving average / 가중 평균으로 jerk 감소)
- "진짜 QP 기반 MPC"는 아니므로 논문에는 "MPC" 대신 "lookahead-smoothed execution" 등
  정확한 명칭 사용 — Section 4 매핑표에 동일하게 적용

### 측정 지표

기존 paper metric(pick time, inference calls, chunk interval, cmd RTT, tracking RMSE,
joint delta RMS) 그대로 유지 + MPC-lite 효과 확인을 위한 jerk/smoothness 지표 추가 검토

### 논문 포지셔닝 후보

"Phase-Conditioned π0.5 VLA Policy with Rule-Based Constrained Execution for Robust
Robotic Pick-and-Place" (MPC 표현을 정확한 용어로 교체, [[project_vla_layer_architecture]])

## 연관 메모리 (Claude Code 세션 간 기억)

`project_vla_layer_architecture.md`, `project_e6_vla_inference.md`,
`reference_action_semantics.md`, `project_moveit2_integration.md`,
`project_paper_trajectory_prediction.md`
