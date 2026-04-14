# Action Chunking (20Hz 기준)

## 0. π₀(pi0) 기준 청크 길이 H

- **π₀ 원래**: action chunk 길이 **H = 50 스텝**, 제어 **50Hz** → **50 step = 1초** 분량.
- **20Hz로 환산**하면 (1초당 20스텝이므로):
  - **1초 청크** → **H = 20**
  - **2초 청크** → **H = 40**
- 학습/설정 시 20Hz 제어를 쓰면 위 H로 맞추는 것이 π₀ 설계와 동일한 “시간 길이”의 청크가 됨.

## 1. 개념

- **action_horizon (H)**: 한 번 추론 시 나오는 **액션 스텝 개수** (chunk 길이).
- **20Hz**: 제어 주기가 50ms → 1초에 20번 액션 실행 → **H스텝 = H/20초**.
- Config별 기본값 (현재):
  - **pi05_droid**: `action_horizon=15` (20Hz 기준 0.75초)
  - **pi0_e6_freeze_vlm**: `action_horizon=10` (20Hz 기준 0.5초)
- π₀ 1초 청크에 맞추려면 **action_horizon=20**, 2초 청크면 **action_horizon=40** 으로 설정.

## 2. 20Hz에서의 두 가지 방식

| 방식 | steps_per_inference | 추론 주기 | 동작 |
|------|---------------------|-----------|------|
| **매 틱 재추론** | 1 | 20Hz | 매 50ms: 관측 수집 → 추론 → `actions[0]`만 실행. 반응성 높음, 연산 부하 큼. |
| **chunk 실행 후 재추론** | N (예: 10) | 20/N Hz | 한 번 추론 → `actions[0]..actions[N-1]`를 20Hz로 순서대로 실행 → N스텝 후 새 관측으로 재추론. 연산 절약, chunk 길이만큼 덜 반응적. |

## 3. run_robot_inference.py 옵션 (추론)

- **--steps_per_inference 생략**: 로봇 정책의 **action_horizon 전체**를 사용 (학습에서 1초청크 H=20이면 20스텝 실행 후 재추론).
- **--steps_per_inference=1**: 매 20Hz 틱마다 추론 (최대 반응성).
- **--steps_per_inference=N**: 한 번 추론한 chunk에서 N스텝만 20Hz로 실행한 뒤 재추론.

```bash
# 학습에서 H=20(1초) / H=40(2초) 썼으면, 생략 시 전체 청크 사용 (추론 = 학습과 동일)
python3 run_robot_inference.py --checkpoint_dir ... --hz 20

# 1초 청크 전체 사용 (H=20 학습 시, 스크립트에서 --steps_per_inference 20 주는 것과 동일)
python3 run_robot_inference.py --checkpoint_dir ... --steps_per_inference 20

# 매 틱 재추론 (반응성 최대)
python3 run_robot_inference.py --checkpoint_dir ... --steps_per_inference 1

# 0.2초마다 재추론 (더 자주 재계획)
python3 run_robot_inference.py --checkpoint_dir ... --steps_per_inference 4
```

## 4. 시간 관계 (20Hz)

- **π₀ 스타일**: H=20 → 1초 청크, H=40 → 2초 청크 (20Hz 기준).
- `action_horizon=10`, `steps_per_inference=10`  
  → 10스텝 × 50ms = **0.5초** 동안 한 chunk 재생 후 재추론.
- `action_horizon=20`, `steps_per_inference=20`  
  → 20스텝 × 50ms = **1초** (π₀ 1초 청크와 동일).
- `action_horizon=15`, `steps_per_inference=5`  
  → 5스텝 × 50ms = **0.25초**마다 재추론 (나머지 10스텝은 버림).

## 5. 확인 방법

- **action_horizon 확인**: `test_vlm_freeze_output.py` 실행 시 로그에 `action_horizon(총 chunk 길이) = N` 출력.
- **실제 chunk 크기**: `result["actions"].shape` = `(action_horizon, action_dim)` (예: (10, 8)).

요약: 20Hz 기준으로 **얼마나 chunk을 쓸지는 `--steps_per_inference`** 로 조절하면 됨. 1이면 매 틱 재추론, N이면 N스텝 실행 후 재추론.

---

## 6. 실로봇(도봇) 추천 (20Hz 기준)

- **청크 길이 K** = 한 번 추론으로 **몇 개의 50ms 액션을 미리 뽑아두는지**.
- **20Hz 실행 루프는 고정**: 매 50ms마다 액션 1개를 로봇에 전송.
- **추론은 “청크가 거의 소진될 때” 호출**: 남은 액션이 1~2개 남으면 다음 청크를 미리 계산 → 추론 지연이 있어도 제어가 끊기지 않음.

### 기본 추천: K = 4 (0.2초 묶음)

- Jetson에서 추론이 150~200ms 걸려도 “다음 0.2초 행동”을 확보해 두면 제어 루프가 안정.
- K=1: 반응성 최대이지만 추론 지연이 조금만 튀면 멈칫/튐.
- K=4: Pick&Place 등에서 **반응성과 안정성 밸런스**가 좋음.

### K 선택 가이드

| 상황 | 추천 K | 비고 |
|------|--------|------|
| **기본(실로봇)** | 4 | 0.2초 묶음, 추론 지터에 강함 |
| Jetson 느리거나 지터 큼 | 6~8 | 0.3~0.4초 미리 뽑기 |
| 빠른 반응/재시도 중요 | 2 | 0.1초, 추론이 충분히 빠를 때 |

### 정량 공식 (참고)

추론 1회 시간 **L(ms)** 를 측정했을 때:

- **K ≥ ⌈L/50⌉ + 1**
- 예: L=120ms → ceil(2.4)+1 = 4, L=180ms → 5, L=260ms → 7

### 주의

1. **학습 chunk 길이와 맞추는 게 최선.** 모델이 8개 뽑으면 앞 4개만 쓰는 건 가능; 4개 뽑는데 8개를 요구하는 건 불가.
2. **K가 길수록 안전 필터 강화 권장**: 클리핑(Δpos/Δjoint 제한), EMA, 워크스페이스 제한. 그렇지 않으면 잘못된 방향으로 0.4초가 누적될 수 있음.
