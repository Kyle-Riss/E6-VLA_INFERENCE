# 로봇 추론으로 실제 동작 보기

## 흐름 요약

1. **환경 활성화** (Jetson 터미널)
   ```bash
   cd ~/move-one/min-imum
   source move-one/bin/activate
   export JAX_PLATFORMS=cpu
   ```

2. **체크포인트 준비**
   - 폴더 하나에 `model.pth` + `assets/.../norm_stats.json` 넣기  
   - 예: `checkpoints/우리모델이름/`

3. **추론 루프 (로봇 연동)**
   - 정책 로드 → **반복**: [카메라 + 로봇 관절/그리퍼] 수집 → `policy.infer(obs)` → 액션을 로봇에 전송  
   - Dobot: 관절 6개 + 그리퍼(DO)  
   - OpenPi 출력: (action_horizon, 8) → 앞 6개 관절, 7번째(또는 8번째) 그리퍼로 매핑  
   - **π₀ 청크**: 50Hz 기준 H=50=1초 → 20Hz 환산 시 **1초 청크 H=20, 2초 청크 H=40** (자세한 환산은 `ACTION_CHUNKING_20HZ.md` 참고).

4. **실행 방법**
   - **방법 A**: `run_robot_inference.py` 사용 (아래 스크립트)
   - **방법 B**: WebSocket 정책 서버 띄우고, 클라이언트에서 관측 전송 → 액션 수신 후 로봇 제어

## Config / 체크포인트

- **Config 이름**: 학습할 때 쓴 설정과 맞춰야 함  
  - DROID 스타일(2 카메라, 7 joint + gripper): `pi05_droid`  
  - Dobot E6 (VLM freeze): `pi0_e6_freeze_vlm`
- **체크포인트 경로**: `model.pth`(또는 `model.safetensors`) + `assets` 가 있는 폴더

## Dobot 연동 요약

- **포트**: main_ui와 동일하게 **29999(Dashboard)** 에서 ClearError, EnableRobot, **MovJ**, ToolDO 전송. **30005** 에서 Feedback(QActual) 수신. (30004 Motion 포트는 사용하지 않음.)
- **관절**: `MovJ(j1..j6, coordinateMode=1, v, a)` — 도(degree) 단위. 속도는 `--movj_velocity` (기본 70, 궤적을 더 빠르게 80 등), 가속도는 `--movj_accel` (기본 60).
- **그리퍼**: `ToolDO(1, 0/1)` (action[7]>0.5 → ON).
- **현재 관절**: Feedback 소켓 30005 `QActual` (도 단위).

---

## 추론 시 로봇이 안 움직일 때 점검 (Pi0 흐름 + 원인 후보)

### 1. 흐름 연결 (Pi0 → Dobot)

1. **정책 로드**: `config_name`(예: `pi0_e6_freeze_vlm`) ↔ 체크포인트 `assets`/norm_stats 일치.
2. **관측**: `observation/exterior_image_1_left`(224×224 RGB), `observation/joint_position`(7), `observation/gripper_position`(1), `prompt`.
3. **정책 출력**: 이미 역정규화된 **상대 관절(rad)** 6개 + 그리퍼 1개. **IK solver 없음** — 관절각 직접 사용.
4. **로봇 명령**: `현재 관절(rad) + 상대 액션(rad)` → 도 변환 → `MovJ(..., 1, v=movj_velocity, a=movj_accel)` → **Dashboard(29999)** 로 전송.

### 2. 통신

- **MovJ가 30004로만 가면 일부 환경에서 동작 안 할 수 있음.** → 반드시 **29999(Dashboard)** 로 MovJ 전송 (main_ui와 동일).
- **Feedback(30005)**: 연결 실패 시 `current_joints_rad`가 None → 상대 액션이 절대값으로 해석될 수 있음. 로그에 `[2] 현재 관절각 (deg)` 가 계속 같은 값이면 로봇이 실제로 안 움직이거나, Feedback이 갱신 안 되는 것.

### 3. 카메라/색감

- 학습 시와 동일한 **색공간(RGB)**·해상도(224×224)·노출/게인 권장. `camera_capture.py`는 `camera_calibration_hikrobot.py`와 동일하게 20ms, 10dB 등 맞춤.
- 색감이 크게 다르면 정책 입력 분포가 어긋나 이상 동작할 수 있음. 필요 시 캘리브레이션/촬영 조건 재확인.

### 4. IK

- 이 파이프라인은 **관절 공간 액션**만 사용. 역기구학(IK) 계산 없음. 문제 발생 시 체크할 것은 **관절 단위 전달**(도 단위, coordinateMode=1)과 **피드백 관절각** 일치 여부.

---

## 궤적을 더 빠르게 / VLM·객체 인식

### 궤적 속도

- "살짝 내려가다 뒤로, 살짝 내려가다 뒤로"처럼 느리게만 보이면 **MovJ 속도**를 올리면 됨.
- 실행 시: `--movj_velocity 80`, `--movj_accel 65` 등 (0~100). Hz는 20 유지, 궤적만 더 빠르게 따라감.
- 스크립트: `run_inference_10000_jetson.sh`에서 이미 v=80, a=65 사용.

### VLM·카메라로 객체 인식해서 집기

- 정책(pi0_e6_freeze_vlm)은 **카메라 이미지** + **프롬프트**를 입력으로 사용. "어떤 객체를 집을지"는 VLM이 이미지를 보고 프롬프트와 맞춰 이해함.
- **프롬프트**: 장면에 맞게 구체적으로 주는 것이 좋음. 예: "pick up the red block", "pick the blue cube on the left". 학습 시 사용한 작업 설명과 비슷한 표현이면 인식이 더 안정적.
- **카메라**: 학습 시와 같은 시점·해상도(224×224 RGB)·색감이어야 "그 객체"를 올바르게 인식함. 색감/밝기가 크게 다르면 다른 걸 보거나 방향이 어긋날 수 있음.
- 객체를 제대로 안 집으면: (1) 프롬프트를 장면/객체에 맞게 수정, (2) 카메라 화각·위치가 학습 데이터와 비슷한지 확인, (3) 해당 태스크/객체로 추가 학습 또는 파인튜닝 고려.
