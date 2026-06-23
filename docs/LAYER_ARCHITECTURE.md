<div align="center">

# 🧱 E6-VLA Layer Architecture

### Context · Policy · Execution — 3계층 프레이밍

![branch](https://img.shields.io/badge/branch-mpc%2B-6f42c1)
![ros2](https://img.shields.io/badge/ROS2-Humble-22314E?logo=ros)
![policy](https://img.shields.io/badge/policy-%CF%800.5%20LoRA-0a7e8c)
![robot](https://img.shields.io/badge/robot-Dobot%20E6-ff6a00)
![platform](https://img.shields.io/badge/platform-Jetson%20AGX%20Orin-76b900?logo=nvidia)
![phase](https://img.shields.io/badge/research-Phase%202-blue)

</div>

> `ROS2_ARCHITECTURE.md`가 **"토픽이 어떻게 흐르는가"**를 다룬다면,
> 이 문서는 **"이 파이프라인을 개념적으로 어떤 3계층으로 볼 것인가"**를 다룬다.
> `mpc+` 브랜치는 π0.5 정책의 **앞단(Context)·뒷단(Execution)** 에
> 재학습 없는 deployment 계층을 추가한 **Phase 2** 작업이다.

```
   ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────────────┐
   │  [LAYER 1]       │     │  [LAYER 2]       │     │  [LAYER 3]               │
   │  Context         │ ──▶ │  π0.5 VLA Policy │ ──▶ │  Constrained Execution   │
   │  Structuring     │     │                  │     │                          │
   │  task_node  /    │     │  pi05_e6_v*_lora │     │  executor_supervisor +   │
   │  context_agent   │     │  16-step chunk   │     │  trajectory_smoother(QP) │
   └──────────────────┘     └──────────────────┘     └──────────────────────────┘
       prompt 생성              action chunk              joint/gripper 명령
```

---

## 📌 한눈에 보기 (TL;DR)

| 계층 | 노드 | 역할 | `mpc+`에서 바뀐 점 |
|---|---|---|---|
| **1. Context** | `task_node` ↔ `context_agent_node` | π0.5에 줄 자연어 prompt 생성 | 🆕 Claude API 기반 `context_agent_node` 추가 (택1) |
| **2. Policy** | `inference_bridge_node` | obs 조립 → 정책 서버 → action chunk | 변경 없음 |
| **3. Execution** | `executor_supervisor_node` (+ `trajectory_smoother`) | chunk → 안전·매끄러운 joint 명령 | 🆕 joint-space **QP optimizer** + 3-way 스위치 |

> ⚠️ **이름 주의**: "MCP"·"MPC"는 정식 프로토콜/최적화 컨트롤러가 **아닌** 비유였다.
> 실제 구현에 맞춰 **Context Structuring Layer** / **(Rule-based or QP) Constrained Execution Layer** 로 부른다.
> QP도 dynamics 모델 없는 **kinematic receding-horizon trajectory optimization** 이다.

---

## 🗺️ 런타임 구조 (ROS2 노드 그래프)

```
                          DOBOT E6 (실로봇)
            ┌────────────────────┴─────────────────────┐
            │ FeedBack :30005                Dashboard :29999
            │ (QActual / QDActual / TCP)     (MovJ / ServoJ + EnableRobot)
            ▼                                            ▲
   ┌───────────────────────┐                            │
   │   camera_state_node    │  HIK(top) + ZED(scene)     │
   │  센서·로봇상태 허브      │  + Dobot 피드백 읽기         │
   └───────────┬───────────┘                            │
               │ /e6/camera/image · /e6/camera/zed_image │
               │ /e6/robot/state · /e6/robot/tcp(_z)     │
               │ /e6/robot/state_vel  🆕 (QDActual, QP seam IC)
               ▼                                         │
   ╔═══════════════════════════════════════════════╗    │
   ║ [LAYER 1] Context Structuring                  ║    │
   ║   task_node ──(기본)──┐   ┌── context_agent 🆕  ║    │
   ║   PhaseTracker        │택1│   Claude API        ║    │
   ║   (고정 prompt)        └───┘   (자연어 prompt)    ║    │
   ║            └────▶ /e6/task/prompt · /e6/task/status
   ╚════════════════════════╪══════════════════════╝    │
                            ▼                            │
   ╔═══════════════════════════════════════════════╗    │
   ║ [LAYER 2] π0.5 VLA Policy                      ║    │
   ║   inference_bridge_node                        ║    │
   ║   image+zed+state+prompt ─▶ openpi server (ws) ║    │
   ║   ─▶ /e6/policy/action_chunk (16×7, 앞 8개 실행) ║    │
   ╚════════════════════════╪══════════════════════╝    │
                            ▼                            │
   ╔═══════════════════════════════════════════════╗    │
   ║ [LAYER 3] Constrained Execution                ║    │
   ║   executor_supervisor_node  (16Hz tick)        ║    │
   ║   ┌─ 3-way (launch arg로 택1) ───────────────┐ ║    │
   ║   │ (a) raw + safety clamp     기본          │ ║    │
   ║   │ (b) MPC-lite EMA           use_mpc_lite   │ ║    │
   ║   │ (c) QP optimizer 🆕         use_mpc        │ ║    │
   ║   │      └ trajectory_smoother.py (MPCSmoother)│║    │
   ║   └────────────────────────────────────────────┘║   │
   ║   + URDF/max_delta hard clamp (3군 공통 바닥)     ║   │
   ║   + grip guard / scripted lift / phase 판정      ║   │
   ║        /e6/gripper/commanded ─────────────────────────┘ (→ camera_state)
   ║        /e6/supervisor/status ── MovJ/ServoJ ──────────┘
   ╚═══════════════════════════════════════════════╝
                  ▲ /e6/supervisor/voice_override
   ┌──────────────┴───────────────┐
   │   voice_command_node (옵션)    │  use_voice:=true
   │   /e6/voice/text_input ─▶ STT  │  → /e6/task/voice_command (task_node)
   └───────────────────────────────┘  → /e6/supervisor/voice_override (executor STOP)
```

### ROS2 노드

| 노드 | 계층 | 역할 |
|------|------|------|
| `camera_state_node` | 센서 | HIK·ZED 카메라 + Dobot FeedBack(:30005) → 이미지·state·`state_vel`🆕·tcp |
| `task_node` | 1 | PhaseTracker 기반 per-frame 프롬프트 발행 (기본) |
| `context_agent_node` 🆕 | 1 | Claude API가 robot state 보고 자연어 prompt 생성 (`use_context_agent:=true`) |
| `inference_bridge_node` | 2 | obs 조립 → WebSocket 정책 서버 → `action_chunk` |
| `executor_supervisor_node` | 3 | chunk 수신 → (raw/EMA/QP) → MovJ/ServoJ + 안전 감시 |
| `voice_command_node` | — | 음성/텍스트 → prompt·STOP override (옵션) |

### 토픽 (🆕 = `mpc+`에서 추가)

| 토픽 | 타입 | 발행 | 구독 |
|------|------|------|------|
| `/e6/camera/image` | Image | camera_state | inference_bridge, executor |
| `/e6/camera/zed_image` | Image | camera_state | inference_bridge |
| `/e6/robot/state` | Float32MultiArray | camera_state | inference_bridge, executor, task, context_agent |
| `/e6/robot/state_vel` 🆕 | Float32MultiArray | camera_state | executor (QP seam 초기조건) |
| `/e6/robot/tcp_z` | Float32 | camera_state | executor, task, context_agent |
| `/e6/task/prompt` | String | task **또는** context_agent | inference_bridge, executor |
| `/e6/policy/action_chunk` | Float32MultiArray | inference_bridge | executor |
| `/e6/gripper/commanded` | Float32 | executor | camera_state |
| `/e6/supervisor/status` | String | executor | task |
| `/e6/supervisor/voice_override` | String | voice_command | executor |

---

## 🎛️ Ablation 스위치 (launch 인자)

`mpc+`의 핵심은 **"한 줄 launch 인자로 계층을 갈아끼우는"** 구조다.

| 인자 | 기본 | 효과 |
|------|------|------|
| `use_context_agent` | `false` | `false`=task_node(PhaseTracker) / `true`=context_agent_node(Claude API) |
| `use_mpc_lite` | `false` | `true`=EMA lookahead smoothing |
| `use_mpc` | `false` | `true`=**joint-space QP optimizer** (use_mpc_lite보다 우선) |
| `control_mode` | `movj` | `movj`(내부 가감속) / `servoj`(즉시 추종) |

```bash
# (a) baseline — raw + safety clamp
ros2 launch e6_vla_ros e6_vla.launch.py

# (b) lookahead smoothing
ros2 launch e6_vla_ros e6_vla.launch.py use_mpc_lite:=true

# (c) QP-MPC + servoj 부드럽게
ros2 launch e6_vla_ros e6_vla.launch.py \
    use_mpc:=true control_mode:=servoj \
    servoj_t:=0.1 servoj_aheadtime:=50 servoj_gain:=300

# Context Agent (Claude API) — ANTHROPIC_API_KEY 필요
ros2 launch e6_vla_ros e6_vla.launch.py use_context_agent:=true
```

> **2×2 ablation** = `use_context_agent` × `{use_mpc_lite | use_mpc}`.
> **Execution 3-way** = raw / MPC-lite / QP. 모델·제어 모드는 paper 실험(v23, per_frame_v16)과 동일 고정.

---

## 🔬 Layer 상세 + 용어 교정

| 제안된 용어 | 실제 구현 | 정확한 명칭 |
|---|---|---|
| MCP context interface | `task_node` PhaseTracker + `context_agent_node`(Claude API) | **Context Structuring Layer** |
| π0.5 VLA policy | `pi05_e6_v*_lora` (v16~v26), 16-step chunk, 앞 8개 실행 | π0.5 VLA policy *(그대로 정확)* |
| MPC execution layer | `executor_supervisor_node` + `trajectory_smoother.py` | **(Rule-based / QP) Constrained Execution Layer** |

### LAYER 1 — Context Structuring

- **task_node** (기본): `gripper_raw`·`tcp_z` 2개 입력 → 하드코딩 if/elif로 phase
  (`approach/grasp/lift/transport/place/release/return`) 분류 → 고정 딕셔너리에서
  문자열 prompt 선택 → `/e6/task/prompt` 발행. **(센서 2개 → 규칙 → 문자열 1개) 변환기.**
- **context_agent_node** 🆕: PhaseTracker를 대체하는 Claude API 에이전트. robot state·tcp_z를
  읽고 상황을 해석해 π0.5용 자연어 prompt를 생성. (16Hz 전부는 latency상 무리 → phase 전환 트리거 권장)

### LAYER 3 — Constrained Execution

**(a) Rule-based** — chunk 스텝마다: ① `max_delta_deg` 속도 클램프 → ② URDF 위치 클램프 →
③ gripper hysteresis → ④ 상황 가드(`grip_enable_z`, transport OFF, `place_force_release`,
lift grip hold) → ⑤ 막히면 `RelMovLUser` scripted lift/return → ⑥ `ServoJ`/`MovJ` 전송.

**(c) QP-MPC** 🆕 — `trajectory_smoother.py`의 `MPCSmoother`. chunk를 reference로 두고
joint-space receding-horizon QP를 푼다:

```
reference:  q_ref_{t+k} = q_t + Σ Δq      (chunk delta 누적, anchor=현재각)

   min   Σ ‖q_k − q_ref_k‖²_Q  +  λ_v‖Δq‖²  +  λ_a‖Δ²q‖²  +  λ_j‖Δ³q‖²
   s.t.  q_min ≤ q_k ≤ q_max         (box, hard)
         |Δq_k| ≤ Δq_max             (velocity, hard)
         |Δ²q_k| ≤ a_max             (accel, optional hard)
```

- **seam IC**: `/e6/robot/state_vel`(QDActual)로 현재 속도·가속을 초기조건으로 흡수 → chunk 이음매 jerk 억제
- **phase-aware λ**: grasp/place 전환부는 λ 낮춰 reference 그대로(타이밍 보존), transport만 강하게 smoothing
- **per-step incremental 실행**: QP는 절대 target을 반환하지만 tick에선 `current + clip(Δ, ±max_delta)`로 실행 → catch-up 질주 방지
- **backend**: scipy(SLSQP) / osqp 동일 P,q,A,l,u → 결과 0.00008° 일치. solve 8~21ms
- 실패 시 reference passthrough (노드 안 죽음)

---

## 🧪 온로봇 결과 & 알려진 이슈 (2026-06-22, arm3 QP)

| 항목 | 상태 |
|---|---|
| QDActual seam IC 수신 (`have_qd=True`) | ✅ 정상 |
| QP solve latency (8~21ms vs chunk ~2s) | ✅ 여유 충분 |
| 절대좌표 catch-up → 안전장치 trip | ✅ per-step incremental로 해결 |
| servoj judder (고주파 떨림) | 🟡 `servoj_t/aheadtime/gain` 튜닝 or `control_mode:=movj`로 완화 |
| ROS 노드 = 시스템 python3 (venv 아님) | ⚠️ osqp 미가시 → scipy backend fallback (수학적 동일, 실험 유효) |

> 📍 **검증 완료**: `test/test_trajectory_smoother.py` — 제약(|Δq|≤3.0)·jerk 감소·실패 passthrough·seam·backend 일치.

---

## 🌱 브랜치 & 커밋 이력

```
* (mpc+) 5d13004  feat(mpc): Constrained Execution Layer (QP) + Context Structuring Layer
│           ├─ trajectory_smoother.py   (MPCSmoother, joint-space QP)
│           ├─ context_agent_node.py    (Claude API context agent)
│           ├─ executor_supervisor_node (3-way: raw / MPC-lite / QP, incremental, jerk metric)
│           ├─ camera_state_node        (/e6/robot/state_vel 발행)
│           ├─ launch / setup.py        (use_mpc · use_context_agent 스위치)
│           └─ test / analyze_metrics / 이 문서
│
* c26d7c4  fix: replace DobotControl path with move-one in run_server scripts
* 7ea9f26  cleanup: remove outdated scripts and SmolVLA files
* fbe5693  feat: URDF joint limits, v18~v26 configs, inference metrics
* f6a708a  feat: voice_command_node 추가 — 음성/텍스트 입력으로 robot prompt 제어
* 74c6c1e  feat: ServoJ control_mode 추가
*  …       (feature/v13-inference 계보)
```

| 항목 | 값 |
|---|---|
| 분기 기준 | `feature/v13-inference` @ `c26d7c4` |
| 브랜치 | `mpc+` → `origin/mpc+` |
| 커밋 | `5d13004` (9 files, +1757 / −46) |
| 리모트 | `git@github.com:Kyle-Riss/E6-VLA_INFERENCE.git` |

---

## 🔗 연관 문서

- [`ROS2_ARCHITECTURE.md`](ROS2_ARCHITECTURE.md) — 토픽 흐름 상세
- [`ROS2_FLOWCHART.md`](ROS2_FLOWCHART.md) — 노드 플로우차트
- [`INFERENCE.md`](INFERENCE.md) · [`ROBOT_INFERENCE.md`](ROBOT_INFERENCE.md) — 추론·실행
- [`ACTION_CHUNKING_20HZ.md`](ACTION_CHUNKING_20HZ.md) — chunk/delta 의미

> 🧭 **Phase 1** = Trajectory Prediction 논문(ServoJ vs MovJ 실측).
> **Phase 2** = 본 문서(Context + Execution 계층). 두 축 모두 **정책 재학습 없는 deployment 계층**이다.
