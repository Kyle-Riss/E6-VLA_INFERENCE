# Stage 1 논문 근거 정리 — E6 VLA (FlowBridge)

> 작성 2026-08-07. 이 Jetson(AGX Orin)에서 직접 확인·재계산한 것만 담는다.
> 목적: 논문에 쓸 수 있는 것 / 조건부로 쓸 것 / 절대 쓰면 안 되는 것을 근거 등급으로 고정.
> 공용 프로젝트 지식은 `~/SHARED_MEMORY.md`, 이 문서는 **논문 서술용 근거 대장**이다.

Stage 1 연구 질문 (공개 사이트 `research.html` 기준):

> *Can a pretrained flow-matched VLA policy be adapted with limited local data
> to produce reliable real-robot joint trajectories?*

⇒ 주장 단위는 "적은 로컬 데이터로 적응시킨 π0.5가 실로봇 관절 궤적을 안정적으로 내는가"이고,
실행 계층(제어 주기 안정성·궤적 품질)이 그 근거가 된다.

---

## 0. 원자료 위치 (전부 이 Jetson)

| 근거 | 경로 | 성격 |
|---|---|---|
| 온디바이스 추론 비용 A | `~/E6-VLA_INFERENCE/scripts/infer_resource_result.json` | WebSocket 왕복, n=10 |
| 온디바이스 추론 비용 B | `~/E6-VLA_INFERENCE/scripts/forward_cpu_random_e6.json` | in-process forward, n=10 |
| 실행 계층 런타임 로그 | `~/Desktop/inference_metrics.txt` | 26 엔트리 (2026-06-22 ~ 07-15) |
| **vision LoRA band ablation + ServoJ/MovJ 원본** | `~/Desktop/metrics_arm1.txt` (사본: `docs/metrics_arm1_band_ablation_20260529.txt`, md5 `571888650f5243a40f8e7ce6ff0538cb`) | **59 레코드, band 라벨 포함** (2026-05-29 ~ 06-22) — 🔴 **2026-07-24 감사가 이 파일을 보지 않았다**(감사 산출물 전체에서 `metrics_arm1` 참조 0건) |
| 그림 생성 스크립트 | `~/Desktop/plot_metrics.py`, `plot_results_summary.py` | trial 원시값 하드코딩 |
| 그림 (n=3 / 성공률) | `~/Desktop/fig1~fig5, figA~figD` | 아래 §4 인벤토리 |
| 노드 소스 | `~/E6-VLA_INFERENCE/ros2/src/e6_vla_ros/e6_vla_ros/` | 5 노드 + 2 helper 모듈 |

---

## 0.5 기여 요약 — "무엇을 밝혔는가" (2026-08-11 신설)

> 이 문서는 원래 Tier(A/B/C)별로만 정렬돼 있어 **읽어도 무엇을 밝혔는지 보이지 않았다.**
> 아래는 같은 근거를 **기여 단위**로 다시 묶은 색인이다. 각 항목은 상세 절을 가리킨다.
> 순서는 논문 서술 순서(시스템 → 측정 → 발견 → 감사)를 따른다.
> ⚠️ 범위는 **E6/π0.5 전용**이다. xArm(Stage 2A)·SmolVLA는 이 문서에 넣지 않는다.

### I. 만든 것 — 작동하는 시스템 (1차 기여)

로보틱스에서 돌아가는 시스템은 그 자체가 결과다. 아래 III의 발견 3개는 전부
**이 시스템을 실기로 돌리다 실패한 지점**에서 나왔으므로, I 없이는 III가 존재할 수 없다.

| # | 만든 것 | 근거 |
|---|---|---|
| **I-1** | **per-frame 6-phase prompting** — 에피소드 단위 프롬프트(v2)는 실기 실패, per-frame phase(v16~v23)는 작동. π0.5는 히스토리가 없어 관측만으로 "넣는 중"과 "빼는 중"을 구분 못 하는데 그것을 **언어로** 해결했다 | `task_node.py:44-61` `V16_PHASE_PROMPTS` + 실기 실적 |
| **I-2** | **5노드 분리와 그 근거** — "기능마다 노드"가 아니라 **I/O 경계와 주기마다 노드, 순수 함수는 모듈** | §5 |
| **I-3** | **Rule-Based Constrained Execution Layer** — 속도 클램프 + URDF 관절 절대 리밋 자동 파싱 + Z 가드 + scripted lift + 강제 릴리즈 백스톱. 외부에서 "MPC"라 부른 것을 구현에 맞게 **용어 교정** | `executor_supervisor_node.py` |
| **I-4** | **efference-copy gripper state** — DigitalOutputs가 그리퍼 상태를 주지 않으므로 executor가 명령 시 `/e6/gripper/commanded`를 발행하고 PhaseTracker가 구독 | `executor_supervisor_node.py:411` → `camera_state_node.py:104` |

⚠️ **I-4가 III-1의 원인이다.** 하드웨어 제약을 우회한 설계가 phase 검출을 사후적으로 만들었다.
설계 선택이 나중에 실패 모드가 된 사례로 본문에 그대로 쓸 수 있다.

### II. 잰 것 — 정량 특성 (전부 이 Jetson, Tier A)

| 측정 | 값 | n | 절 |
|---|---|---|---|
| 모델 forward (WebSocket 왕복) | **2087.9 ± 14.0 ms** | 10 | A-1 |
| 모델 forward (in-process, ROS2 없음) | **2122.8 ± 28.5 ms** | 10 | A-1 |
| **Executor tick** | **62.52 ± 0.07 ms** | 25 | A-2 |
| Chunk 도착 간격 | 2500.96 ± 13.22 ms | 25 | A-2 · A-3 |
| Cmd RTT (baseline) | 9.10 ± 1.01 ms | 19 | A-2 |
| Tracking RMSE (baseline) | 0.324 ± 0.115° | 19 | A-2 |
| ServoJ vs MovJ pick time | 36.5 ± 8.5 s vs 67.9 ± 2.4 s | 3 | B-1 |

🔴 **가장 강한 한 쌍은 `2087.9 ms ↔ 62.52 ms`다**(§A-4). 두 측정이 **서로 다른 경로인데**
2.09/2.12 s로 일치하므로 **비용은 직렬화가 아니라 forward 자체**이고, 그것이 62.5 ms tick과
33배 차이라는 사실이 §5의 구조 전체를 정당화한다. 표 둘에 흩지 말고 한 문장 안에 병치할 것.

### III. 밝혀낸 것 — 발견 3개

**공통 형태: "X가 안 되는 건 모델 탓이라 믿었는데, 데이터·배포 파이프라인이었다."**
셋 다 이 리그를 넘어 일반화된다.

| # | 믿었던 것 | 실제 | 절 |
|---|---|---|---|
| **III-1** | 릴리즈가 안 된 건 프롬프트가 없었거나 프레임워크 제약 | 프롬프트는 **있었고 정확했다.** 학습 라벨은 그리퍼 이벤트를 **16프레임 앞당겨** 붙는데(198/198) 배포는 pick만 재현하고 release는 안 했다. 게다가 배포의 release는 **executor 자신의 명령**을 보고 켜지므로 **lead가 양수일 수 없는 인과 루프** | **A-6** |
| **III-2** | 접근 각도가 안 변한 건 scripted lift 후 OOD | **가르친 적이 없다.** 흡착은 축대칭이라 J6는 접근각에 **0.0000°** 기여하고 J5가 마지막인데, 수집 자동화가 그것을 **Δj5 q01–q99 = 0.258°**로 붕괴시켰다. quantile 정규화가 그 죽은 축을 **~13배 증폭** | A-4b |
| **III-3** | 마지막 레이어가 공간 정보를 덮어쓰니 끝단을 빼면 좋다(v17 설계 근거) | **끝점 무효 · 폭 무효 · 위치가 전부.** late 9층(65.2 s) ≈ all 27층(65.8 s) ≪ early(125.4)·mid(146.2). 이유는 **proprioception이 아예 없어**(`state_proj` 미생성 + `discrete_state_input=False`) **vision이 공간 정보의 유일한 출처**이기 때문 | A-5 |

⇒ 일반화 문장 3개 (그대로 본문 후보):
1. *Phase-conditioned VLA에서 학습 라벨은 미래를 당겨 쓰는데 배포 phase 검출기는 인과적이다.
   로봇 자신의 행동으로 검출되는 phase는 원리적으로 재현 불가능하다.*
2. *수집 자동화의 편의로 고정한 DOF가 축별 정규화를 거쳐 학습 그래디언트를 잠식한다.
   데이터에 분산이 없으면 모델 문제가 아니다.*
3. *Frozen VLM + LoRA에서 adaptation의 **폭**은 이득이 없고 **위치**가 전부다.
   그 이유는 아키텍처가 공간 정보를 어디서 얻는지에 달려 있다.*

### IV. 뒤집은 것 — 감사

같은 기준을 **남의 것과 내 것에 함께** 적용했다는 것이 이 축의 내용이다.

**IV-1. 이전 기록·외부 판정** — v22 "없음" → **존재**(range `(9,17)`) / v26 **복구**
(`norm_stats.json`이 truncated PNG → v25 byte-identical 사본) / v20은 **orphan**
(TrainConfig 정의도 run script도 없음 ⇒ `--policy.config` 해석 불가로 서빙 자체 불가) /
**band ablation "원본 재현 불가"(2026-07-24 감사 결론) → `metrics_arm1.txt`에 있었다**(§0·§A-5) /
layer 26 설계 근거 실기 무효(§B-4)

🔴 **IV-1에서 가장 값어치 있는 것은 band ablation이다.** "데이터가 없다"로 종결된 실험이
사실은 **파일 하나를 안 열어서**였고(감사 산출물 전체에서 `metrics_arm1` 참조 0건),
찾으니 **Tier B → Tier A로 올라가며 논문 핵심 ablation이 근거를 얻었다.**
ServoJ/MovJ 원본도 같은 파일에 있어 함께 승격됐다.

**IV-2. 내 주장을 내가 뒤집은 것** — RMSE 0.35°(n=26)가 **arm 혼합**(→ baseline n=19,
0.324 ± 0.115°) / n=26 → **25** / 2.5 s "합산" 설명 → **폴링 양자화**(§A-3) /
j5 부호 규약 불일치 → **population mixing**(§A-4b) / "사이트 본문에 그 설명 없다" → **있었음**

⚠️ RMSE와 j5는 **같은 실수(집단 혼합)를 하루 안에 두 번** 한 것이다.
우연이 아니라 **실수 유형**이므로, 파생 통계를 인용할 때는 n을 새로 세지 말고
**기존 게시값이 재현되는 집단을 찾아 맞출 것.**

**IV-3. 근거로 쓸 뻔했다가 배제한 것** — §3 Tier C 참조
(`approach_logs/` 타 머신 / `~/Desktop/figures/` 빈 그림 / 지연 최적화 3단계 표 검증 불가)

---

## 1. Tier A — 원자료가 있고 재계산으로 검증한 것 (논문에 그대로 쓸 수 있음)

### A-1. 온디바이스 추론 비용 — 두 경로가 일치한다

| 측정 | 경로 | latency | GPU | CPU | RAM |
|---|---|---|---|---|---|
| WebSocket 왕복 (n=10, warmup 3) | 로컬 정책서버 `127.0.0.1:8000` | **2087.9 ± 14.0 ms** | 79.7 ± 0.8 % | 18.3 ± 1.5 % | 26.8 % |
| In-process forward (n=10, warmup 2) | `cuda:0`, ROS2 없음 | **2122.8 ± 28.5 ms** | 77.5 ± 2.0 % (net 68.9) | 17.2 ± 1.2 % (net 3.7) | 45.7 % |

**핵심 논거**: 두 경로가 서로 다른데 2.09 s / 2.12 s로 일치한다 ⇒ 비용은 WebSocket 직렬화·
ROS2 오버헤드가 아니라 **모델 forward 자체**다. "지연이 통신 때문 아니냐"는 질문을 닫는 유일한 근거.

device 필드는 `/proc/device-tree/model` 에서 읽은 `NVIDIA Jetson AGX Orin Developer Kit`.
GPU는 `tegrastats GR3D_FREQ`(시스템 전체), 창은 추론 왕복 구간 평균.

⚠️ **반드시 병기할 provenance 한계**
- 두 측정의 **체크포인트가 다르다.** WebSocket 쪽 `label: pi05_e6_v23_lora` 는 **CLI 인자**이고
  `server_metadata` 가 `{}` 로 비어 있어 서버가 확인해 준 값이 아니다. in-process 쪽은 파일에
  경로가 남아 `pi05_e6_v16_lora` 로 확정된다.
- 입력은 **합성 224 더미 이미지**다. 자원·지연 측정이며 태스크 성공률과 무관하다.

### A-2. 실행 계층 안정성 — `inference_metrics.txt`

**집단 정의**: 로그 26 엔트리 중 25개가 확장 포맷(`resolved_mode`/`pace_to_arrival` 보유).
record 0만 구버전 포맷이라 감사 파서가 25로 셌고, **공개 사이트의 기존 p95 값
(tick 62.66 / chunk 2520.9)이 n=25 에서만 정확히 재현된다.** ⇒ 사이트·그림의 정본 집단은 **n=25**.
그리고 `19 (safety-clamp) + 6 (QP-optimizer) = 25` 로 arm 합계도 맞는다.

SD는 사이트 관례대로 **표본 SD(n−1)**.

| 지표 | 전체 n=25 | safety-clamp n=19 | QP-optimizer n=6 | arm별로 갈리는가 |
|---|---|---|---|---|
| Executor tick | 62.520 ± 0.071 ms | 62.511 ± 0.046 | 62.543 ± 0.113 | ❌ (1 SD 이내) |
| Chunk interval | 2500.96 ± 13.22 ms | 2498.4 ± 8.96 | 2506.0 ± 20.9 | ❌ (1 SD 이내) |
| Cmd RTT | 8.96 ± 0.93 ms | **9.10 ± 1.01** | 8.51 ± 0.32 | ⭕ |
| Tracking RMSE avg | 0.350 ± 0.133° | **0.324 ± 0.115°** | 0.431 ± 0.164° | ⭕ |
| Joint delta RMS avg | — | **0.378 ± 0.133°** | 0.496 ± 0.178° | ⭕ |

🔴 **논문 baseline 은 safety-clamp n=19 를 쓴다.** `Tracking RMSE = 0.324 ± 0.115°`.
n=25 혼합값(0.350)은 Stage 1 이 서술하지 않는 후속 연구 arm 을 섞은 값이라 쓰지 않는다.
tick·chunk interval 은 arm 무관이 실측으로 확인됐으므로 통합값을 써도 된다.

**Stage 1 에서 가장 강한 숫자**: `Executor tick 62.52 ± 0.07 ms` — 목표 62.5 ms 대비 편차 0.1 %.

### A-3. ★ 2.5 s 청크 갱신 주기의 메커니즘 (2026-08-07 신규)

지금까지 "약 2.5초마다 갱신"이라고만 썼는데, **원인이 코드+실측으로 분해된다. 두 층은 별개다.**

**(가) 2.5 s 는 forward 시간이 폴링 타이머에 양자화된 결과다**

```
inference_bridge_node.py:56          infer_hz = 2.0          → 타이머 주기 0.5 s
e6_vla.launch.py:31                  infer_hz 기본값 "2.0"    (launch 도 동일)
inference_bridge_node.py:_maybe_infer
        if self._inference_running: return       ← 이전 추론 중이면 그 tick 을 버린다
```

추론이 2.09 s 걸리므로 완료 후 **다음 0.5 s 경계**에서 재시작한다:

```
ceil(2.09 / 0.5) × 0.5 = 2.5 s          실측 2500.96 ± 13.22 ms  (n=25)
```

뒷받침: SD 13 ms 는 **forward 자체의 변동(±14~29 ms)보다 작다.** 격자에 붙어 있다는 뜻이다.
26개 전부 `[2469.3, 2544.9]` 안에 있고 2.5 s 격자에서 벗어난 엔트리가 없다.

⚠️ `inference_bridge_node.py:17` docstring 은 아직 `infer_hz (default 1.25)` 로 **stale** 하다.
실제 코드 기본값은 `:56` 의 `2.0` 이다. 논문에 인용할 값은 후자.

**(나) pacer 는 이 간격의 원인이 아니라, 실행을 그 간격에 맞춘 장치다**

`chunk_pacer` 를 끈 실행에서도 간격이 같다 — 이게 (가)와 (나)를 분리하는 증거다:

| record | `pace_to_arrival` | Chunk interval |
|---|---|---|
| 1 | **False (k=1)** = 페이싱 없음 | **2496.0 ms** |
| 3 | **False (k=1)** = 페이싱 없음 | **2501.9 ms** |
| 나머지 22 | True (k=5) | 2469 ~ 2545 ms |

페이싱이 하는 일은 8 스텝을 그 2.5 s 에 고르게 펴서 burst-then-freeze 를 없애는 것이다:

```
steps_per_inference 8  ×  k 5 ticks  ×  62.5 ms  =  2500 ms      ← 도착 간격과 일치
```

⇒ **인과 서술**: forward 2.09 s → 0.5 s 폴링 양자화로 도착 간격 2.5 s 확정 → pacer(k=5)가
8 스텝을 그 2.5 s 에 균등 분배. **간격을 만든 것은 (가), 간격을 채운 것은 (나).**
`chunk_pacer.py` docstring 에 이 목적("burst-then-freeze 완화")이 명시돼 있다.

이 절이 필요한 이유: Stage 1 의 질문이 "신뢰할 만한 궤적이 나오는가"인데, action chunking 이
**왜 성립하는지**를 설명하지 못하면 그 신뢰성이 우연으로 읽힌다.

### A-4b. ★ J5 저분산의 정체 — 수집 설계가 자세를 고정했다 (2026-08-09)

학습서버가 정규화 쪽에서 본 **"j5 = loss 의 25.7 %"** 의 **원인**이 수집 코드와 데이터에 있다.

**원인 — 자세 3축 고정**
`dobot-xarm-datacollect/Dobot_E6_Moveit2/config/robot_config.json` 의 모든 pick/place 좌표가
**`rx=180, ry=0, rz=0`** 이다. 툴이 항상 수직 아래를 보도록 자세를 완전히 고정한 태스크다.

**결과 — 실제 수집 데이터** (`vla_dataset/vla_auto_2026012*/robot_data.csv`, 3 에피소드 450 프레임)

| joint | mean | **std** | 범위폭 | \|Δ\| mean (action 라벨과 같은 양) |
|---|---|---|---|---|
| j1 | −81.6 | 8.24 | 28.4° | 0.470 |
| j2 | −46.9 | **26.38** ← 최대 | 106.4° | 3.307 |
| j3 | −43.7 | 14.06 | 61.9° | 0.948 |
| j4 | +21.5 | 12.28 | 53.0° | 1.840 |
| **j5** | **+90.2** | **4.46** ← **최소** | **21.2°** ← 최소 | **0.535** |
| j6 | +97.7 | 10.10 | 47.8° | 0.528 |

**j5 는 +90° 에 눌러앉아 "툴을 수직 아래로 유지하는" 기계적 상수 역할**을 하고, 나머지 5축이
위치를 만든다. 태스크가 그렇게 설계돼 있으므로 j5 는 **움직일 이유가 없다.**

**이것이 세 가지를 한 번에 설명한다**
1. **quantile normalization 왜곡** — `use_quantile_norm = model_type != PI0`(`config.py:188`)라
   축마다 자기 분위수로 정규화한다. j5 의 원본 Δ 분산이 작으니 **분모가 작아지고 노이즈가 다른
   축과 같은 크기로 부풀려진다.** 실제 기여가 없는 축이 loss 의 1/4 을 먹는 구조다
2. **`fig2` 에서 J5 tracking RMSE 가 유독 낮은 것**(ServoJ 0.14° vs 다른 축 0.37~0.48°) — 정확도가
   좋은 게 아니라 **과제가 쉬운 것**이다. `paper_docx_prompt.md` 가 "J5 는 수직 방향 운동이 적어
   기여가 낮다"고 이미 적었는데 **왜 그런지가 이제 수집 설계로 설명된다**
3. **xArm 의 dj4 저분산과 같은 병** — 자세를 고정하면 손목 하나가 죽는다. E6 에서 실제로
   일어났음이 확인된 셈이고, 이것이 xArm 에서 회전 텔레옵을 여는 이유가 된다

⇒ 논문 문장: *the low variance of J5 is a property of the task design (fixed tool orientation),
not of the policy; quantile normalization then amplifies its residual noise.*

🔴 **CSV 의 `rx/ry/rz` 를 그대로 통계 내지 말 것**
같은 파일에서 `rx mean 65.45 / std 158.46 / 범위 −180.0 ~ +180.0` 이 나오는데 자세가 흔들린 게
아니라 **오일러각 wrap** 이다(`+179.99` 와 `−179.99` 는 같은 자세). 설정은 `rx=180` 고정이다.
xArm 세션에도 같은 함정이 기록돼 있다("rx≈±180° 근처 특이점에서 축간 커플링, 수학적으로 정상").

⚠️ **한계**: 이 데이터는 **2026-01-26 자동수집 3 에피소드**이고 학습에 쓴
`2CAM-Orange-init` 549 에피소드가 아니다. 경향은 같을 가능성이 크지만 **학습 데이터로 재확인
필요**(HDD 마운트 후). `|Δj5| max = 8.44°` 라 완전 고정은 아니고 미세 보정은 한다.

⚠️ **FK 기반 IK 로 이걸 확인하려던 시도는 실패했다** — 관절 리밋을 걸지 않아 해가 여러 바퀴
감긴 채 수렴했다(j3 범위 ±1400°). 실제 수집 데이터가 추정보다 강한 증거이므로 IK 결과는 쓰지 않는다.

### A-4. 논문에서 병치할 한 쌍

> forward **2087.9 ± 14.0 ms** ↔ executor tick **62.52 ± 0.07 ms**

2초 걸리는 추론을 물고 있는데 제어 주기 편차가 0.1 %. 아키텍처가 작동했다는 증거를 두 숫자로
끝낸다. **표 두 개에 흩어 놓지 말고 한 문장 안에 붙일 것.**

### A-5. ★ vision LoRA band ablation — 원본 발견 (2026-08-07)

근거 `~/Desktop/metrics_arm1.txt` 1·50·93·129 행에 band 라벨(`0-8`, `9-17`, `18-26`, `full`)이
그대로 있다. **2026-05-29 단일 세션**이라 조명·리그 조건이 동일하다.
🔴 **2026-07-24 감사가 이 파일을 보지 않았다** — 그래서 v21 이 `NOT FOUND` 로 판정됐다.

| band | range | 버전 | 시도 | 로그 | mean | SD | min–max |
|---|---|---|---|---|---|---|---|
| early | 0–8 | v21 | 20 | 6 | **125.4 s** | 28.7 | 96.3–175.6 |
| mid | 9–17 | v22 | 20 | 5 | **146.2 s** | 29.1 | 105.8–187.2 |
| **late** | 18–26 | v23 | 20 | 4 | **65.2 s** | **5.1** | 58.1–69.9 |
| all | 0–26 | v24 | 20 | 5 | **65.8 s** | 10.8 | 57.6–84.1 |

`0–8 + 9–17 + 18–26 = 0–26` 이므로 **의도된 분할 설계**이고 `full` = **v24** 다(v18 은 이 파일에 없다).

**결과 3개**
1. **late 가 2배 이상 빠르다** — 65 s vs 125/146 s
2. **late(9레이어) ≈ all(27레이어)** — 65.2 vs 65.8 s, 차이 0.6 초로 서로의 SD 안.
   **앞쪽 18개 레이어를 더 붙여도 이득이 없다**
3. **late 가 가장 일관적** — SD 5.1 vs 28.7 / 29.1 / 10.8

⇒ 논문 문장: **"어느 band 를 적응시키는지가 전부다. 범위를 넓히는 것도, 끝점 한 칸도 도움이 안 된다."**
(끝점 = v18 vs v24, 사용자 실기 관찰로 차이 없음 — B-4 참조)

🔴 **로그 개수는 성공 횟수가 아니다 — 성공률로 환산 금지**
각 조건 **20회 시도**, **대체로 집었고**(성공), 로그를 다 남기지 않은 것뿐이다(사용자 확인 2026-08-07).
4~6개는 **보존된 표본**이다. ⚠️ `6/20 = 30%` 는 **틀린 계산**이다.
오독하기 쉬운 이유: 이 지표는 흡착 전환 순간에만 출력된다
(`executor_supervisor_node.py:1081` — `tool_on==1 and _last_gripper==0 and not _metrics_saved`),
즉 실패는 구조적으로 레코드가 안 생긴다. 그래서 "없음"이 "실패"로 읽히지만 여기서는 **미기록**이다.
⇒ 논문 표기: **`20 attempts per condition; N records retained in the log`**. 성공률은 주장하지 않는다.
⚠️ 보존 표본은 무작위가 아니라 **편의 표본** — mean 은 n 과 함께, 모평균처럼 쓰지 말 것.
⚠️ 이 지표는 **"집기 성공"까지**다(흡착 ON 순간 기록 → 이후 낙하도 성공으로 남음). 완주 지표가 아니다.

⚠️ **`full` 라벨이 06-13·06-17 세션까지 삼킨다** — 순진하게 파싱하면 44개로 잡힌다.
band 비교는 **05-29 의 5개만** 쓸 것.

---

### A-6. ★ release 라벨 타이밍 비대칭 — 배포가 재현할 수 없는 phase (2026-08-11 신규)

**증상**: 실기에서 흡착 후 상승·횡이동·하강까지 되는데 **릴리즈가 안 됐다.**
(처음엔 "옆으로 가라는 프롬프트를 안 줬거나 프레임워크 제약"으로 의심했으나
`task_node.py:49` 에 `transport` 프롬프트가 **있고 정확하다** — 프롬프트는 원인이 아니다.)

#### 학습 쪽 (학습서버 측정)

| | 값 |
|---|---|
| 라벨 생성 | 규칙 생성 `release_start = max(carry_start, open_idx − ACTION_HORIZON)` |
| lead | 그리퍼 open 전환보다 **+16 프레임 앞** |
| 적용 범위 | **198/198 에피소드** (예외 없음) |

⇒ 학습에서 `release` 프롬프트는 **그리퍼가 열리기 1초 전부터** 붙어 있다.

#### 배포 쪽 (이 Jetson 소스, 전수 확인)

| phase | 트리거 | lead |
|---|---|---|
| **pick** | `tcp_z <= pick_prearm_z` (기본 159.0) — Z 기반 **선진입** (`task_node.py:154`) | **양수** ✅ |
| **release** | `_trans_counter > 0 and _trans_type == "open"` — 그리퍼 1→0 **전환 후** (`:151`) | **음수** ❌ |

- 🔴 `task_node.py:154` 주석이 그대로 **`# Z 기반 미리 pick_up (학습 레이블 타이밍 맞춤)`** 이다.
  ⇒ **비대칭이 pick에서는 인지되어 처리됐고 release에서는 안 됐다.** 몰라서가 아니라 빠뜨린 것이다.
- `prearm` 전수 검색: `ros2/src/e6_vla_ros/` 전체 **9건이 모두 `pick_prearm_z`**,
  `place_prearm_z` / `release_prearm_z` 는 **존재하지 않는다.**

#### 🔴 파라미터로는 고칠 수 없다 — 인과 루프

배포에서 그리퍼 상태는 **executor 자신이 낸 ToolDO 명령의 efference copy** 로 돌아온다:
`executor_supervisor_node.py:411` → `/e6/gripper/commanded` → `camera_state_node.py:104`
→ `state[6]` → PhaseTracker.
따라서 `release` phase 는 **정의상 자기 명령보다 뒤**이며, 어떤 임계값을 주더라도 **lead ≤ 0** 이다.
이것이 §0.5 I-4(하드웨어 우회)의 대가다.

#### 2차 실패 — 백스톱도 조건부

`executor_supervisor_node.py:898-906` 의 강제 릴리즈는 `tcp_z <= place_z_threshold`
(launch 기본 110 mm) 조건부다. 팔이 ~115 mm 에서 멈추면 **백스톱도 발동하지 않는다.**
⇒ 두 요인은 **병렬이 아니라 직렬**이다: 선진입 없음 → 정책이 못 냄 → 백스톱도 못 잡음.

#### 논문 서술 (초안)

> Training-time phase labels were generated with a fixed lookahead of one action horizon
> (16 frames) relative to the gripper transition, applied uniformly across all 198 episodes.
> At deployment the pick phase reproduced this lead through a height-triggered pre-arm,
> whereas the release phase was detected from the executor's own gripper command and
> therefore could not attain a positive lead under any parameter setting.

⚠️ **"배포 release 프롬프트가 5프레임이라 짧다"는 서술은 쓰지 말 것.**
`_V16_PHASE_KEY["return"] = "release"`(`task_node.py:71`) 이므로 전환 5프레임 이후에도
**같은 문자열이 유지**된다. 결손은 오직 **open 이전 16프레임**이다.
"짧다"고 쓰면 반박 지점이 생기고, 실제 발견(인과 루프)이 더 강한데 약하게 말하는 셈이 된다.

⚠️ **근거 등급**: 학습 쪽(+16 프레임 · 198/198)은 **학습서버 측정**이고 이 Jetson에서 재현하지
않았다. 배포 쪽은 전부 이 Jetson 소스 전수 확인이다. 본문에 쓸 때 출처를 나눠 적을 것.

---

## 2. Tier B — 집계값만 남아 원자료 재현이 안 되는 것 (조건부 사용)

### B-1. ServoJ vs MovJ (n=3, 2026-06-13)

| Metric | ServoJ | MovJ |
|---|---|---|
| Pick time (s) | 36.5 ± 8.5 | 67.9 ± 2.4 |
| Inference calls | 18 ± 5 | 33 ± 1 |
| Chunk interval (ms) | **2054 ± 24** | **2049 ± 48** |
| Command RTT (ms) | 9.15 ± 0.16 | 11.16 ± 0.29 |
| Tracking RMSE avg (deg) | 0.385 ± 0.111 | 0.347 ± 0.022 |
| Joint delta RMS (deg) | 0.454 ± 0.105 | 0.393 ± 0.025 |

✅ **정정 (2026-08-07) — raw 원본을 찾았다. 이 항목은 Tier B 가 아니라 Tier A 다.**
`~/Desktop/metrics_arm1.txt` 의 129 행(`full` 라벨) 뒤 **2026-06-13 세션**에 6개 trial 이
그대로 있고 `plot_metrics.py` 하드코딩값과 **정확히 일치**한다:

```
servoj  pick 26.46 / 35.89 / 47.27      chunk 2033.6 / 2087.6 / 2042.1   RTT 9.35 / 9.12 / 8.97
movj    pick 68.64 / 70.33 / 64.58      chunk 2014.9 / 2117.4 / 2014.4   RTT 11.28 / 11.44 / 10.76
```

per-joint Tracking RMSE·Joint delta RMS 6축 값까지 레코드에 들어 있어 `fig2`·`figD` 도 원본
대조가 된다. **2026-07-24 감사의 "재현 불가"는 이 파일을 보지 않은 결과다**(감사 산출물에서
`metrics_arm1` 참조 0건, `inference_metrics.txt` 만 18번).

⇒ **"researcher-reported" 표기를 뺄 수 있다.** 다만 `n=3` 은 그대로 작고, `1.86× 우위`는
n=3 에서 나온 값임을 병기할 것.

### B-2. 🔴 chunk interval 이 두 집단이다 — 섞으면 그 자체로 모순

| 집단 | chunk interval | 시점 | 기록된 `infer_hz` |
|---|---|---|---|
| n=3 (그림 fig1~fig5, figA~figD 전부) | **2054 / 2049 ms** | 2026-06-13 | **기록 없음** |
| n=25 (사이트·Tier A) | **2500.96 ms** | 2026-06-22 ~ 07-15 | 확인 안 됨(로그에 필드 없음) |

**다른 실험이다.** 그림에서 2054 를 보여주고 본문에서 2500 을 쓰면 리뷰어에게 "어느 게 맞냐"는
질문을 받고 답이 두 개가 된다.

⚠️ **차이의 원인을 단정하지 말 것.** 페이싱 유무로는 설명되지 않는다(A-3 (나) 참조 — 페이싱을
끈 실행도 2496~2502 ms). `inference_metrics.txt` 에도, n=3 쪽에도 **당시 `infer_hz` 가 기록돼
있지 않다.** 관측 사실만 적을 수 있다:
- 2500 ms 집단은 `infer_hz = 2.0` 양자화(0.5 s 격자)와 정합적이고 SD 가 13 ms 로 작다.
- 2054 ms 집단은 격자에 붙어 있지 않고 forward 시간 자체에 가깝다 ⇒ 당시 루프는 양자화되지
  않았거나 훨씬 촘촘한 격자였을 것으로 **보인다**(검증 불가).

⇒ 서술 방침: 두 집단을 **명시적으로 분리**하고, 원인은 "당시 폴링 설정이 로그에 남지 않아
귀속할 수 없다"로 적는다. 숨기지 않는다.

### B-3. 태스크 성공률

Single-object recognition 20/20 · Color recognition 20/20 · Pick-and-place 18/20.
**trial 단위 원본 증거가 이 Jetson 에 없다**(2026-07-24 감사 결론: NOTHING FOUND).
공개 사이트도 "researcher-reported controlled-evaluation values" 로 표기했으므로 논문도 같은 수위.
⚠️ 이 20/20 계열은 **§A-5 의 20회 시도와 다른 실험이다** — 섞지 말 것.

### B-5. ★ open-vocabulary grounding — 학습에 없던 명사로 접근 (사용자 육안 관찰)

**관찰 (2026-08-07 사용자 확인, trial 수·로그 없음)**

학습은 오렌지 박스 pick-and-place 뿐인데(방향 지시 포함), **학습에 없는 물체**에
**방향어 없는 맨 명사 프롬프트**만 주었을 때 팔이 그 물체로 접근했다:

```
프롬프트:  "pick up the almond"   /   "pick up the egg"     ← 방향어 없음
물체 위치: 학습 때 박스 자리와 다른 곳
```

**결과 (사용자 직접 확인, 2026-08-07)**: almond 와 charger 를 **동시에 놓고** 이름을 바꿔 부르면
각각 그쪽으로 갔다. **동시 배치 + 맨 명사** 이므로 방향어·위치기억이 함께 배제된다.

⚠️ **학습서버 초안은 같은 조합을 다르게 적었다** — *"Almond+Charger → 특정 목표로 못 가고 시작
부근에서 흔들림, Almond+Egg → Almond 2/20"*. 사용자가 직접 "동시에, 둘 다 구분"이라고 확인했으므로
**실행자 진술을 1차로 삼되**, 양쪽 다 로그가 없다는 점은 남긴다(전언이 갈린 상태).
`2/20` 같은 수치는 **출처 확인 전 인용 금지**.

⚠️ **almond·charger·marker 는 Jetson 코드에도 로그에도 없다** — `task_node.py:211-231` 에 있는 것은
**can·egg 뿐**이고 그것도 방향어 포함 버전이다. 맨 명사 프롬프트는 직접 발행해야 한다.

🔴 **사이트 `results.html` 의 80/80 이 어느 조건인지 확인이 필요하다**

게시된 값: **Object-conditioned target approach 100 % (80/80)**, 세트 = almonds · egg · red box ·
USB cable 각 20 trial. 그리고 `#two-stage-interpretation` 이 **approach 20/20 vs 완주 18/20** 로
이미 두 단계를 분리해 보고한다(*"failures belong to the physical manipulation stage rather than
initial target selection"*) — 오늘 정리한 분리와 같은 구조다.

**그런데 그 80 trial 이 단일물체였는지 다물체였는지가 게시문에 없다.** 이게 주장 강도를 가른다:
- **단일물체였다면** — 장면에 물체가 하나뿐이라 "지시된 대상으로 갔다"가 사실상 자명하다.
  conditioned approach 라는 이름에 값이 실리지 않는다
- **다물체였다면** — 80/80 이 그대로 **open-vocabulary reference resolution 의 정량 근거**가 된다

⇒ 사용자의 almond+charger 동시 배치 관찰은 후자를 시사하지만 **게시 수치와 같은 실행인지 불명**이다.
`results.html` 은 학습서버 소유가 아니라 SHARED 영역이므로 확인 후 문구를 명확히 할 것.

**대안 설명이 전부 닫힌다 — 이게 이 관찰의 힘이다**

| 대안 | 왜 배제되는가 |
|---|---|
| 방향어를 읽었다 | **프롬프트에 방향어가 없다.** 맨 명사뿐 |
| 학습 위치를 외웠다 | 학습 박스 자리가 아닌 곳에 뒀고, **두 물체 중 명사만 바꿔** 대상이 갈렸다 |
| 관절 배치를 외웠다 | **state 가 네트워크에 안 들어간다**(§ SHARED_MEMORY "state 가 모델에 아예 들어가지 않는다" — `pi0.py:151`/`:97`, `obs.state` 는 shape 만). 외울 경로 자체가 없다 |
| 문자열을 외웠다 | 박스 실험에서 **동의문 3종**(`pick up` / `grasp` / `move`)이 통했다(`task_node.py:211-231` v13 세트) |

⇒ 남는 단서는 **시각-언어 prefix 하나**다. VLM(Gemma 2B) 완전 frozen + SigLYP 9블록 rank-16 LoRA
라는 선택의 배당금이고, **vision LoRA band 결과와 연결된다** — 공간 정보의 유일한 출처가 vision
이므로 어느 band 를 적응시키는지가 표준 pi0 셋업보다 크게 작용한다(late 65 s vs early 125 s).

🔴 **주장 범위를 반드시 이 선에서 끊을 것**

- ✅ 주장 가능: **지시된 명사의 물체로 접근한다**(approach / pick target selection)
- ❌ 주장 불가: **지시대로 놓는다** — 사용자 확인 "놓는 건 랜덤이었다".
  방향 지시가 있는 pick-and-place 완주는 **학습한 박스에서만** 시연됐다
- ✅ 박스에서는 **phase 진행까지** 확인됐다(PhaseTracker 가 흡착 상태+TCP Z 로 구동되므로
  phase 가 넘어갔다는 건 시퀀스를 실제로 수행했다는 뜻)

🔴 **계란·아몬드를 못 집은 것을 인식·해상도 한계로 귀속하지 말 것**

E6 는 **흡착(suction) 엔드이펙터**다(`executor_supervisor_node.py:333`, `:687` `ToolDO`).
흡착은 밀폐가 필요해 **평면만 가능**하다 — 박스·충전기 ✅ / 곡면 계란·불규칙 아몬드 ❌.
이건 **사전에 알고 있던 하드웨어 제약**이고 정책·인식 한계가 아니다.
⚠️ 학습서버 초안의 *"실패 모드는 언어가 아니라 물체 스케일이다"*(224px, patch≈25mm) 는
**채택하지 않는다** — ① 작업영역 폭이 가정이고 E6 카메라 mm/px 실측이 없다
② 스케일과 흡착 가능성이 **같은 순서를 예측**해 구분되지 않는다
③ **almond·charger 는 실제로 구분됐다** — 인식이 실패한 게 아니다.
정확한 문장: *grasp success is bounded by the suction end-effector (flat surfaces only);
this is an end-effector constraint, not a perception one.*

⚠️ **근거 등급 = 육안 관찰.** trial 수·성공률·로그가 없다. 그리고 **구조적으로 안 남는다** —
지표가 흡착 ON 전환에만 찍히므로(`:1081`) TCP 도 그때만 기록된다(`:1144`).
**접근만 성공하고 못 집으면 레코드가 0개**다. 성공한 관찰인데 남을 자리가 없었다.

**재현 절차(다음에 로봇 앞에 앉을 때)** — 이걸로 육안이 좌표로 남는다
1. 맨 명사 프롬프트 주입: `ros2 topic pub --once /e6/task/prompt std_msgs/msg/String "data: 'pick up the almond'"`
   (또는 `/e6/voice/text_input` → `voice_command_node`). ⚠️ 맨 명사 프롬프트는 `task_node.py`
   V13_PROMPTS 에 **없다**(거긴 방향어 포함 `can_*`/`egg_*` 뿐) — 직접 발행해야 한다
2. **`record_mcap:=true`** — `/e6/robot/tcp` 가 18Hz 로 남으므로 **집지 못해도 궤적 방향이
   사후에 객관적으로 판정된다.** 흡착 기반 필드로는 안 되는 게 이래서다
3. 두 물체를 좌우로 벌려 두고 **명사만 번갈아** 발행 → 프로토콜 `named_object` /
   `reached_object`(TCP 궤적에서 유도) 로 기록. 우연이 50 % 이므로 **횟수를 셀 것**

---

### B-4. layer 26(끝단) 효과 — 사용자 실기 관찰: 차이 없음

끝점만 26↔25 로 다른 짝은 전 버전 중 **정확히 두 개**이고 학습 step 까지 맞는 것은 하나다:

| 짝 | range | layer 26 | steps | 깨끗한가 |
|---|---|---|---|---|
| **v18 vs v24** | `(0,25)` vs `(0,26)` | 없음 / 있음 | **20,000 / 20,000** | ✅ 유일한 단일변수 |
| v16 vs v26 | `(22,26)` vs `(22,25)` | 있음 / 없음 | 17,500 / **7,500** | ❌ step 교란 |

v17`(14,25)`·v23`(18,26)` 에 대응하는 `(14,26)`·`(18,25)` 는 존재하지 않는다. v16 vs v26 은
추가로 **v26 이 2026-07-24 까지 로드 불가**였다(norm_stats.json 이 실제 PNG).

**사용자가 v18·v24 를 둘 다 실기로 돌리고 **육안으로** 동작을 비교해 "사실상 차이가 없었다"고
확인했다(2026-08-07).** 수치를 재거나 trial 을 세지 않았고 `metrics_arm1.txt` 에 v18 레코드는 없다.
⇒ **근거 등급 = 육안 관찰(visual inspection).** 큰 효과를 배제하는 데는 충분하지만(있었으면 보였을
것이다) 정량 null 주장에는 부족하다. **논문에서는 "no visible difference was observed" 수준으로만
쓰고, 수치나 동등성(equivalence)을 주장하지 않는다.**
⇒ layer 26 에 관측 가능한 효과가 없다. 원래 가설(마지막 레이어가 공간 정보를 덮어쓰므로 빼면
좋아진다 — v17 이 끝단을 뺀 근거였던 논문 주장)이 **step 까지 맞춘 비교에서 성립하지 않는다.**
공개 사이트의 *"design reasoning was not confirmed by the outcome"* 의 직접 근거가 이 관찰이다.

⚠️ **근거 등급**: trial 수·수치 없는 **연구자 관찰**이고 `metrics_arm1.txt` 에 v18 레코드는 없다.
"차이가 없다"를 주장으로 쓰려면 차이를 보이는 것보다 **더 많은 trial** 이 필요하다.
관찰로 서술하거나, 주장하려면 재측정할 것.

🔴 **학습서버의 "v18 실기 하나면 block 26 mechanism 이 single-variable ablation 으로 올라간다"는
제안은 전제가 무효다** — 그 짝은 이미 돌렸고 null 이다.

---

## 3. Tier C — 쓰면 안 되는 것

| 항목 | 배제 사유 |
|---|---|
| `E6-VLA_INFERENCE/approach_logs/` 16개 JSONL | 전 파일 checkpoint 경로가 `/media/`**`billy`**`/` — 현재 계정은 `billye6`. 커밋 `87afddb` 로 merge 돼 들어온 **타 머신 산출물**, mtime 도 전부 체크아웃 시각(2026-04-20 15:16), config 도 전부 `pi0_*`(Pi-Zero 계열). **초기화 ablation·제어율 sweep 처럼 보여서 인용하고 싶어지는 게 함정이다.** |
| `docs/ROS2_LATENCY_OPTIMIZATION.md` 의 "100~190 s → 57~84 s → 40~75 s" 3단계 표 | `inference_metrics.txt` 에 **체크포인트 버전 필드가 없어** v16~v17 / v18~v19 귀속을 검증할 수 없다. 게다가 로그 범위(06-22~07-15)가 표 작성 시점(06-03)보다 늦다. |
| `~/Desktop/figures/` 4개 PNG | **빈 그림.** §4-3 참조 |
| SmolVLA vs π0.5 자원 비교 | 사용자 작업이 아님 |

---

## 4. Figure 인벤토리 (9개 전부 열어서 확인함)

### 4-1. 쓸 수 있는 것

| 파일 | 실제 내용 | n | 논문 위치 |
|---|---|---|---|
| `fig1_pick_time_rtt.png` | (a) pick time (b) Cmd RTT · 막대 + trial 점 | 3 | 실행 방식 비교 도입 |
| `fig2_tracking_rmse_per_joint.png` | J1~J6 per-joint tracking RMSE | 3 | RMSE 분석 |
| `fig3_inference_latency.png` | ServoJ 2054±24 / MovJ 2049±48 ms | 3 | §4-2 (1) 참조 |
| `fig4_summary_4panel.png` | (a) pick time (b) RMSE (c) RTT (d) chunk interval | 3 | 종합 요약 |
| `fig5_results_summary.png` | (a) 성공률 20/20·20/20·18/20 (b) task time (c) RTT | 3 + 성공률 | Results 첫 그림 |
| `figA_joint_trajectory.png` | J1~J6 시계열, 점선 = suction ON. **계단형 궤적이 그대로 보임** | 1 trial | 궤적 분석 |
| `figB_joint_velocity.png` | J1~J6 각속도. ServoJ ±10 deg/s 스파이크 다수 vs MovJ 평탄 | 1 trial | motion smoothness |
| `figD_velocity_rms.png` | 관절별 RMS 속도, **ServoJ 가 MovJ 의 약 2~3배** | 3 | smoothness 정량 |
| `figC_gripper_state.png` | 6 trial 전부 suction ON 시각 (SJ 29.4/39.6/50.6 s, MJ 71.7/74.3/67.5 s) | 6 | 부록 권장 |

### 4-2. 그림-본문 불일치 (처리 상태)

**(1) ✅ 수정 완료 — `fig3` 제목이 측정량과 달랐다.**
옛 제목 *"Inference latency per chunk (model inference + network RTT)"* ↔ 실제 플롯 변수는
`chunk_ms` = **도착 간격**이고 y축 라벨도 이미 `Chunk interval` 이라 그림 안에서 자기모순이었다.
2026-08-07 에 `plot_metrics.py:168` 을
**`"Action-chunk arrival interval\n(policy loop period, not model inference latency)"`** 로 고쳐
재생성했다. 검증: fig1/fig2/fig4 는 **변경 픽셀 0개**(스크립트가 결정론적), fig3 만 1.51 % 변경
(제목 영역). 백업은 `~/Desktop/fig_backup_20260807/`.
본문에는 "추론이 루프를 지배하므로 도착 간격이 forward 시간의 상한 근사"라고 한 줄 붙일 것.

**(2) ⏳ 미처리 — chunk interval 두 집단.** §B-2 방침대로 캡션·본문에 n=3 / n=25 명시 분리.

**(3) ✅ 수정 완료 — Tracking RMSE arm 혼합.** 공개 사이트에 n=25 혼합값 0.35° 가 올라가 있던
것을 baseline n=19 의 **0.32 ± 0.12°** 로 교체하고, arm 구성(19 + 6 = 25)과 "tick·chunk 는
arm 무관이라 통합 / RTT·RMSE 는 갈리므로 baseline 만" 을 protocol note 로 명시했다.
Cmd RTT 도 같은 이유로 baseline n=19 값 **9.10 ± 1.01 ms** 로 교체.

### 4-3. 쓸 수 없는 것 — `~/Desktop/figures/`

**빈 그림이다.** 같은 폴더 `summary_table.txt` 가 전 항목 `N/A` + `Trials: MovJ=0, ServoJ=0`,
`04_tcp_scatter.png` 는 축만 있고 점 0개, `02_joint_rmse.png` 잉크 비율 1.3 %(축·라벨뿐),
`03_exec_stability.png` 는 60×311 스텁. **파싱 실패한 실행의 잔재**이고 `fig1~fig5` 가 대체한다.
⚠️ 파일명이 `01_grasp_time.png` 처럼 정상이라 실수로 집을 위험이 있다.

`figures/04` 제목이 *End-Effector Position at Suction ON* 인 것은 EEF 정확도 그림을 만들려다
실패한 흔적이다. `paper_docx_prompt.md` 가 이미 "EEF position error 는 직접 평가하지 않았으며
FK 기반 분석은 future work" 로 한계를 명시했으므로 **그 서술을 유지하는 것이 정직하다.**

---

## 5. 노드를 5개로 나눈 근거 (코드에서 유도)

설계 의도가 아니라 **코드로 증명되는** 근거 4가지. 강한 순서.

### ① 주기가 다르면 프로세스를 나눈다 — 실측이 강제한 구조 (가장 강함)

| 노드 | 주기 | 그 주기를 정하는 것 |
|---|---|---|
| `camera_state_node` | 18 Hz | 센서 하드웨어 (HIK + ZED) |
| `inference_bridge_node` | 폴링 2 Hz → **실효 0.4 Hz** | 모델 forward 2.09 s |
| `executor_supervisor_node` | 16 Hz (62.5 ms) | 로봇 명령 주기 |
| `task_node` | 16 Hz | phase 갱신 |

**2.09 s blocking 호출과 62.5 ms 주기가 같은 프로세스에 있으면 후자는 죽는다.**
bridge 는 이걸 두 겹으로 막는다:

```python
self._executor.submit(self._run_infer, obs)   # ThreadPoolExecutor 로 분리
if self._inference_running: return            # 재진입 차단
```

⇒ 노드 분리는 취향이 아니라 **실측 2.09 s 가 강제한 것**이고, 그 결과가
`executor tick SD 0.07 ms` 다. **분리가 작동했다는 증거가 숫자로 남아 있다.**
논문에서 이 두 개(2.09 s ↔ 0.07 ms)를 나란히 놓으면 아키텍처 논증이 닫힌다.

### ② 하드웨어 소유권 분리 — 소켓이 다르다

```
camera_state_node.py:119          DobotApiFeedBack (robot_ip, 30005)   ← 읽기 전용
executor_supervisor_node.py:443   DobotApiDashboard(robot_ip, 29999)   ← 명령
```

같은 로봇 IP 인데 **소켓이 다르다.** 상태 읽기와 명령 쓰기를 물리적으로 갈라 놓았으므로
"관측 노드가 실수로 로봇을 움직이는" 경로가 **구조적으로 존재하지 않는다.**
이 원칙은 Stage 2A xArm 의 single-owner 규칙으로 그대로 계승된다 — stage 간 연속성 논거.

### ③ 순수 계산은 노드로 만들지 않았다

```
chunk_pacer.py          rclpy 참조 0건   (41줄)
trajectory_smoother.py  rclpy 참조 0건   (308줄)
```

둘 다 **노드가 아니라 모듈**이다. ROS/로봇 상태를 전혀 안 보므로 로봇 없이 단위 테스트가 된다.
`chunk_pacer.py` docstring 에 이유가 명시돼 있다:

> executor_supervisor_node 쪽 게이팅 로직과 분리해 독립적으로 단위테스트 가능하게 함

⇒ 설계 원칙이 **"기능마다 노드"가 아니라 "I/O 경계와 주기마다 노드, 순수 함수는 모듈"** 임을
코드가 보여준다. **아키텍처 섹션의 결론 문장이 된다.**

### ④ 안전 감시를 실행과 **합친** 근거 (분리하지 않은 이유)

`executor_supervisor_node` 가 1371줄로 압도적으로 크다. 설계 실패가 아니라 의도다 —
안전 판정이 명령 경로와 **같은 함수 안**에 있어야 우회 경로가 생기지 않는다.

```
FAIL_SAFETY:emergency_stop | :bad_camera | :min_tool_z(<z>mm)
_load_joint_limits_deg()      ← URDF 에서 관절 한계 자동 로드 (:86)
max_delta_deg = 3.0           ← 프레임당 속도 제한
/e6/emergency_stop (Trigger)  ← :414
```

클램프를 별도 노드로 빼면 토픽이 한 겹 더 생기고, 그 사이에 stale 프레임이 끼면 **클램프가
지난 상태로 판정**한다. 이름이 `executor_supervisor`(실행 + 감시)인 것도 이 결정을 반영한다.

🔴 리뷰어가 "1371줄 노드는 monolithic 아니냐"고 물을 때의 답이므로 **반드시 준비**할 것.

### 보조. 프롬프트 노드 분리의 대가를 상쇄한 장치

```
task_node.py:294               QoSProfile(durability=TRANSIENT_LOCAL, depth=1)
inference_bridge_node.py:89    같은 설정으로 구독
```

`task_node` 를 따로 두면 "bridge 가 먼저 떠서 프롬프트를 놓치는" 위험이 생기는데 QoS 로 막았다.
늦게 뜬 구독자도 **마지막 프롬프트를 받는다.** VLA 에서 언어 조건 소실은 치명적이라 언급 가치가 있다.

---

## 6. 표현 규칙 (`~/Desktop/paper_docx_prompt.md` + 2026-08-07 확인분)

| 개념 | 권장 | 금지 |
|---|---|---|
| 제어 주기 | "executor maintained 62.52 ± 0.07 ms (16.00 Hz)" | ~~"real-time 16 Hz inference"~~ |
| 추론 주기 | "policy produced a new chunk every ≈2.5 s, set by a 2.09 s forward pass quantized to a 0.5 s polling period" | ~~"16 Hz 추론"~~ |
| pacer 역할 | "the pacer distributes 8 steps across that interval" | ~~"the pacer sets the 2.5 s interval"~~ (실측 반증됨) |
| fig3 | "chunk arrival interval" | ~~"inference latency"~~ (측정량이 다름) |
| Tracking RMSE | **baseline arm n=19, 0.324 ± 0.115°** | ~~n=25 혼합값 0.350~~ |
| MovJ RMSE | "lower RMSE is attributed to slower execution, not higher tracking fidelity" | ~~"MovJ is more accurate"~~ |
| Joint delta RMS | "joint command smoothness" | ~~"tracking error"~~ |
| EEF | "not directly evaluated; FK-based analysis is future work" | ~~EEF 정확도 주장~~ |
| 실행 최적화 | "a follow-up execution-layer study is ongoing" | (Stage 1 본문에 QP/MPC 수치 넣지 않음) |
| `infer_hz` | 코드 기본값 **2.0** (`:56`, launch `:31`) | ~~docstring 의 1.25~~ (stale) |

---

## 7. 남은 작업

- [ ] **B-2** — 논문 캡션·본문에 chunk interval n=3 / n=25 명시 분리
- [ ] fig3 를 논문에 넣을 때 "도착 간격은 forward 시간의 상한 근사" 한 줄 추가
- [ ] Tier B 전부에 "n=3, researcher-reported" 표기
- [ ] `figC` 부록 편입 여부 결정
- [ ] **A-6** — 학습 쪽(+16 프레임 · 198/198)을 이 Jetson에서 독립 재현할지 결정.
      데이터셋이 HDD에 있어 현재 불가. 못 하면 본문에 "학습서버 측정"으로 출처 분리
- [ ] **III-3 근거 보강** — layer 26 null 은 n·수치 없는 관찰이라 §B-4 등급 유지.
      band 효과(§A-5)만으로 서술하고 끝점은 "관찰" 로 적을 것
- [ ] §0.5 를 논문 outline 으로 전개할 때 **I → II → III → IV 순서 유지**
      (III 가 눈에 띄지만 I 없이는 III 가 나올 수 없다는 것이 서사다)
