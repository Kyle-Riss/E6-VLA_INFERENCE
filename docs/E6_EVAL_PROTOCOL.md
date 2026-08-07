# E6 실기 평가 기록 프로토콜 v1

> 작성 2026-08-07. 목적은 하나다 — **실기를 돌렸다는 사실과 그 결과가 나중에 인용 가능한
> 형태로 남게 하는 것.** 이 문서가 없어서 실제로 잃은 것들이 아래 §0 에 있다.
> 근거 대장은 `STAGE1_PAPER_EVIDENCE.md`, 공용 지식은 `~/SHARED_MEMORY.md`.

---

## 0. 이 프로토콜이 막으려는 것 (전부 실제로 겪음)

| 사고 | 실제로 벌어진 일 | 이 프로토콜의 대응 |
|---|---|---|
| **부분 로깅** | vision LoRA band 실험을 조건당 **20회** 돌렸는데 로그에 **4~6개**만 남았다. 그래서 성공률을 못 쓴다 | 시도마다 레코드 1개 **필수**, 실패도 남긴다 (§4 `outcome`) |
| **실패가 안 남음** | 지표가 흡착 ON 전환에만 출력된다(`executor_supervisor_node.py:1081`) → 못 집으면 **레코드 자체가 없다.** "없음"이 실패인지 미기록인지 구분 불가 | `outcome` enum 으로 실패 사유를 명시 기록 |
| **파일을 못 찾음** | 2026-07-24 감사가 `~/Desktop/metrics_arm1.txt` 를 **한 번도 보지 않았다**(참조 0건). 그 안에 v21 band 데이터와 ServoJ/MovJ 원본이 다 있었는데 `NOT FOUND` 로 판정됐다 | 경로를 고정하고 세션마다 `session.json` 을 둬서 기계 판독 가능하게 (§2) |
| **위치 기반 그룹 라벨** | `metrics_arm1.txt` 는 `0-8` / `full` 같은 **맨 줄**로 조건을 구분했다. `full` 라벨이 뒤따르는 06-13·06-17 세션까지 삼켜 순진하게 파싱하면 5개가 44개로 잡힌다 | 조건은 **모든 레코드의 필드**로 넣는다. 섹션 헤더 금지 (§5-1) |
| **체크포인트 미확정** | `infer_resource_result.json` 의 `label: pi05_e6_v23_lora` 는 **CLI 인자**이고 `server_metadata` 가 `{}` 라 서버가 확인해 준 값이 아니다 | `checkpoint_reported` 와 `checkpoint_cli_label` 을 **분리** (§4) |
| **집단 혼합** | 서로 다른 arm(safety-clamp 19 + QP 6)을 평균해 `Tracking RMSE 0.35°` 를 게시했다. arm 별로는 0.324 vs 0.431 로 갈린다 | 집계 도구가 집단이 섞이면 **거부**한다 (§6) |
| **단위 혼동** | `dt_from_prev` 를 문서에 ms 로 적었는데 실제는 초. 변환기가 전량 SKIP 될 뻔했다 | 시간 필드는 이름에 **단위를 박는다**(`_s` / `_ms`) (§5-2) |

⚠️ 이 표는 "조심하자"는 훈계가 아니라 **전부 이 프로젝트에서 실제로 발생한 손실**이다.

---

## 1. 적용 범위

E6 로봇을 실제로 움직여 정책을 평가하는 모든 실행. 자원·지연 벤치마크(로봇 없이 forward 만
돌리는 것)는 대상이 아니다 — 그건 `scripts/infer_resource_benchmark.py` 계열이 이미 JSON 을 남긴다.

xArm 수집(`xarm_vla_collector`)은 별도 스키마(`SCHEMA_VERSION=6`, CSV 72컬럼)를 이미 쓰므로
그대로 두고, 나중에 xArm **추론** 평가를 할 때 이 프로토콜을 준용한다.

---

## 2. 파일 배치 — 경로를 고정한다

```
~/e6_eval/<YYYYMMDD_HHMMSS>_<label>/
├── session.json        세션 상수 (1개)
├── trials.jsonl        시도당 1행 (JSON Lines)
├── mcap/               (선택) 이 세션의 MCAP
└── notes.md            (선택) 자유 메모
```

**`~/Desktop` 에 두지 않는다.** 감사가 `metrics_arm1.txt` 를 놓친 게 정확히 그래서였다.
⚠️ `~/e6_eval/` 은 내부 디스크다. 외장 HDD 를 쓰지 않는 이유는 이번 프로젝트에서
**5회 조용히 단절된 이력**이 있어서다(세션 중 끊기면 그 세션이 통째로 날아간다).
세션 종료 후 백업은 별도로 한다.

---

## 3. `session.json` — 세션 내내 안 바뀌는 것

```json
{
  "schema_version": 1,
  "session_id": "20260807_143000_vision_lora_band",
  "purpose": "vision LoRA band ablation (early/mid/late/all)",
  "operator": "billye6",
  "robot": "Dobot E6",
  "robot_ip": "192.168.5.1",

  "policy_source_commit": "8b46000",
  "policy_source_repo": "/home/billye6/E6-VLA_INFERENCE",
  "openpi_variant": "local src/openpi",
  "inference_location": "local",

  "launch_args": { "action_mode": "delta", "gripper_mode": "absolute",
                   "prompt_mode": "per_frame_v16", "control_mode": "servoj",
                   "infer_hz": 2.0, "executor_hz": 16.0,
                   "steps_per_inference": 8, "max_delta_deg": 3.0,
                   "min_tool_z": 75.0 },

  "cameras": { "hik": "224x224 crop[16:240,55:279]", "zed": "224x224 crop[120:480,150:510]" },
  "lighting_note": "천장등 ON, 커튼 닫음",
  "rig_note": "오렌지 박스, 좌→우 pick-place",
  "started_at_wall": "2026-08-07 14:30:00",
  "started_at_monotonic": 123456.789
}
```

- **`inference_location`**: `"local"` 또는 `"remote:<host>"`. ⚠️ **원격 정책 서버를 쓰면
  cadence 가 달라진다**(A5000 forward 가 Orin 보다 빠르고 네트워크가 붙는다) → 로컬 실측
  (2087.9 ms / 2500.5 ms)과 **같은 표에 넣으면 안 된다.**
- **`policy_source_commit`**: `--policy.config` 는 체크포인트가 아니라 **openpi 소스의
  `TrainConfig` 를 참조**한다. 소스 트리가 다르면 같은 체크포인트도 다르게 서빙된다.

---

## 4. `trials.jsonl` — 시도마다 한 행

🔴 **성공이든 실패든 무조건 한 행.** 이게 이 프로토콜의 전부다.

```json
{
  "schema_version": 1,
  "trial_index": 7,
  "condition_id": "vision_lora_18_26",
  "condition_label": "late (18-26)",

  "checkpoint_cli_label": "pi05_e6_v23_lora",
  "checkpoint_reported": "pi05_e6_v23_lora",
  "checkpoint_dir": "/mnt/robotdata/e6_checkpoints/e6_v23_19999",
  "vision_lora_layer_range": [18, 26],

  "arm": "safety-clamp",
  "resolved_mode": "safety_clamp",
  "control_mode": "servoj",

  "started_at_wall": "2026-08-07 14:41:02",
  "started_at_monotonic": 124512.331,
  "ended_at_monotonic": 124577.512,

  "outcome": "grasped_and_placed",
  "outcome_reason": null,
  "evidence_grade": "measured",

  "time_to_suction_on_s": 65.14,
  "chunks_used": 27,
  "infer_calls": 27,
  "tcp_at_suction_mm": [202.5, -366.5, 124.9],

  "executor_tick_ms":    { "mean": 62.5, "sd": 0.3, "min": 60.5, "max": 64.6 },
  "chunk_interval_ms":   { "mean": 2500.4, "sd": 13.0 },
  "cmd_rtt_ms":          { "mean": 8.97, "sd": 2.37 },
  "tracking_rmse_deg":   { "per_joint": [0.43,0.36,0.38,0.35,0.15,0.32], "avg": 0.334 },
  "joint_delta_rms_deg": { "per_joint": [0.54,0.45,0.47,0.43,0.03,0.40], "avg": 0.388 },

  "mcap_path": "mcap/20260807_144102.mcap",
  "notes": ""
}
```

### `outcome` — enum (이 프로토콜의 핵심)

| 값 | 의미 |
|---|---|
| `grasped_and_placed` | 집고 놓기까지 완주 |
| `grasped_only` | 집었으나 놓기 실패(낙하 포함) |
| `no_grasp` | 흡착이 **한 번도 켜지지 않음** |
| `abort_safety` | `FAIL_SAFETY:*` 로 정지 → `outcome_reason` 에 원문 |
| `abort_max_steps` | `max_steps` 초과 |
| `abort_operator` | 조작자가 중단 |
| `error` | 예외·연결 끊김 등 |

⚠️ **기존 지표 `time_to_suction_on_s` 는 "집기 성공"까지다** — 흡착 ON 순간에 찍히므로
그 뒤에 떨어뜨려도 값이 남는다. 완주 여부는 `outcome` 만이 구분한다.
그래서 성공률을 쓸 때 **어느 성공인지 반드시 명시**할 것:
- 집기 성공률 = `grasped_*` / 전체
- 완주 성공률 = `grasped_and_placed` / 전체

### `evidence_grade`

| 값 | 의미 |
|---|---|
| `measured` | 계측값이 레코드에 있음 |
| `visual` | **육안 관찰만.** 수치 없음 (예: 2026-08-07 v18 vs v24) |

`visual` 레코드는 큰 효과를 배제하는 데는 쓸 수 있지만 **정량·동등성 주장에는 쓰지 않는다.**
논문 표현은 `"no visible difference was observed"` 수준까지.

---

## 5. 금지 사항

### 5-1. 조건을 섹션 헤더로 쓰지 않는다

```
❌ 0-8
   [Inference Metrics] ...
   [Inference Metrics] ...
   full                      ← 뒤따르는 다른 날 세션까지 삼킨다
```

`metrics_arm1.txt` 에서 실제로 `full` 이 06-13·06-17 세션을 삼켜 5개가 44개로 잡혔다.
**`condition_id` 를 모든 레코드에 넣는다.** 레코드 하나만 봐도 어느 조건인지 알아야 한다.

### 5-2. 시간 필드는 이름에 단위를 박는다

`_s` 또는 `_ms` 로 끝낸다. 예외 없다. `dt_from_prev` 를 ms 로 오기해 변환기가 전량 SKIP 될
뻔한 사고가 있었다. **서술용 환산값을 계약 단위표에 옮겨 적지 말 것.**

### 5-3. CLI 라벨을 체크포인트로 기록하지 않는다

`checkpoint_cli_label`(내가 준 값)과 `checkpoint_reported`(서버가 답한 값)를 **둘 다** 쓴다.
둘이 다르면 그 자체가 발견이다. 서버가 안 알려주면 `checkpoint_reported: null` 로 두고
`null` 인 이유를 `notes` 에 적는다 — **추측해서 채우지 않는다.**

### 5-4. 집단을 섞지 않는다

`arm` / `control_mode` / `inference_location` / `policy_source_commit` 이 다르면 다른 집단이다.
평균을 낼 때 섞으면 안 된다(RMSE 0.324 vs 0.431 을 섞어 0.35 를 게시한 사고). 집계 도구가
막아준다(§6).

---

## 6. 도구 — `scripts/eval_session.py`

```bash
# 스키마·필수필드·enum·단위 검사
python3 scripts/eval_session.py validate ~/e6_eval/20260807_143000_vision_lora_band

# 조건별 집계 (시도수와 기록수를 분리해서 보고)
python3 scripts/eval_session.py aggregate ~/e6_eval/*/

# 집단이 섞이면 거부한다. 의도적이면 명시적으로 허용
python3 scripts/eval_session.py aggregate ~/e6_eval/*/ --allow-mixed
```

`aggregate` 는 조건마다 다음을 낸다:

```
n_attempts        전체 레코드 수
n_grasped         grasped_* 개수
n_completed       grasped_and_placed 개수
n_timed           time_to_suction_on_s 가 있는 개수   ← 시간 통계의 실제 분모
time mean/sd/min/max, p50
outcome breakdown
```

⚠️ **`n_attempts` 와 `n_timed` 를 반드시 구분해 보고한다.** 오늘 band 데이터에서
"20 시도 / 4~6 기록"을 성공률 30 % 로 오독할 위험이 정확히 이 지점이었다.

---

## 7. 아직 적용하지 않은 코드 변경 (합의 필요)

현재 `executor_supervisor_node.py:1081` 은 **흡착 ON 전환에만** 지표를 출력한다.
이 프로토콜대로 실패도 남기려면 코드 변경이 필요하다:

1. 에피소드 종료 경로마다(`STAGE_DONE`, `FAIL_SAFETY:*`, `max_steps`, estop) **레코드 1행 emit**
2. 출력 대상을 stdout 이 아니라 `~/e6_eval/<session>/trials.jsonl` 로
3. `session.json` 은 노드 기동 시 1회 작성

⚠️ **AGENTS.md §2 에 따라 지금 손대지 않았다.** 이건 돌아가는 실행 경로를 건드리는
변경이라 감사 → 합의 → 하나씩 순서를 밟아야 한다. **당장은 사람이 손으로 채워도 이 스키마가
성립한다**(육안 관찰 레코드도 `evidence_grade: "visual"` 로 정식 표현된다).

---

## 8. 과거 데이터를 이 스키마로 옮길 때

`metrics_arm1.txt` 의 band 데이터를 옮기려면:

- `condition_id`: `vision_lora_0_8` / `_9_17` / `_18_26` / `_0_26`
- `evidence_grade`: `measured`
- `outcome`: 레코드가 있는 것은 **`grasped_only` 이상이 확실**하다(흡착이 켜졌으므로).
  완주 여부는 알 수 없으니 `grasped_only` 로 두고 `notes` 에 "완주 여부 미기록" 을 적는다.
  ⚠️ **`grasped_and_placed` 로 올려 적지 말 것** — 기록에 없는 것을 만들어내는 셈이다.
- 로그에 없는 나머지 시도는 **레코드를 만들지 않는다.** 대신 세션 `notes.md` 에
  "조건당 20회 시도, 대체로 성공, 로그는 일부만 보존(사용자 확인 2026-08-07)" 을 적는다.
  ⚠️ 없는 시도를 `no_grasp` 로 채우면 **성공률을 조작하는 것**이 된다.
