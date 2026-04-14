# 추론(Inference) 가이드 — move-one 로봇 제어

로봇 추론 파이프라인, 체크포인트 로드 경로, 인자, 트러블슈팅, 디버깅 순서를 정리한 문서입니다.

---

## 1. 파이프라인 개요

```
[1] 정책 로드     checkpoint_dir → model.pth 또는 model.safetensors
                 + assets/droid/norm_stats.json (같은 폴더)
[2] Dobot 연결    192.168.5.1:29999(Dashboard), 30005(Feedback)
[2.5] 카메라      HIKRobot 224×224 RGB (camera_capture.py)
[2.8] 초기 자세   (선택) 고정 TCP로 이동 — --no_init_pose 시 스킵
[3] 추론 루프     obs 수집 → policy.infer(obs) → actions → MovJ + ToolDO
```

- **관측(obs)**: exterior_image_1_left, wrist_image_left(동일 224 이미지), joint_position(7), gripper_position(1), prompt
- **정책 출력**: 이미 역정규화된 상대 액션(rad) → 0~5: 관절 6개 delta, 6~7: 그리퍼 후보
- **로봇 명령**: 현재 관절(rad) + delta → deg 변환 → MovJ(j1~j6); grip → ToolDO(1, 0/1)

---

## 2. 실행 전 필수 조건

### 2.1 가상환경 (필수)

시스템 Python이 아닌 **move-one 가상환경**에서 실행해야 합니다.  
가상환경을 쓰지 않으면 `ModuleNotFoundError: No module named 'etils'`, NumPy API 경고 등이 납니다.

```bash
cd ~/move-one
source min-imum/move-one/bin/activate
# 프롬프트에 (move-one) 이 보이면 OK
```

### 2.2 Dobot 네트워크

- 로봇 IP: **192.168.5.1**
- Jetson 이더넷(eno1)에 **같은 대역 IP**가 있어야 합니다.  
  한 번 설정해 두면 재부팅 후에도 유지하려면:

```bash
sudo nmcli con add type ethernet ifname eno1 con-name dobot-static \
  ipv4.method manual ipv4.addresses 192.168.5.100/24 autoconnect yes
sudo nmcli con up dobot-static
ping -c 2 192.168.5.1
```

### 2.3 체크포인트 폴더 구조

각 체크포인트는 **한 폴더 안에** 아래가 있어야 합니다.

| 항목 | 설명 |
|------|------|
| `model.pth` 또는 `model.safetensors` | 가중치 (safetensors 우선 사용) |
| `assets/droid/norm_stats.json` | 액션/상태 정규화 통계 (없으면 추론 시작 시 에러) |

- **norm_stats 없을 때**: 10000_jetson 등 기존 체크포인트에서 복사  
  `mkdir -p "<체크포인트>/assets/droid"`  
  `cp .../10000_jetson/assets/droid/norm_stats.json "<체크포인트>/assets/droid/"`

---

## 3. 체크포인트 경로 (새 볼륨 기준)

USB **새 볼륨** 마운트 경로: **`/media/billy/새 볼륨/`**  
(경로에 공백이 있으므로 반드시 **따옴표**로 감싼다.)

| 폴더명 | 용도 | LoRA |
|--------|------|------|
| `2000_jetson` | 2000 step 풀 | 아니오 |
| `2000_lora_jetson` | 2000 step LoRA | 예 |
| `4000_random_base_jetson` | 4000 step base | 아니오 |
| `4000_random_jetson_1500` | 4000 step LoRA 1500 | 예 |

- **Base**: `model.pth` 또는 `model.safetensors` 풀 가중치만 로드 (LoRA 병합 없음)
- **LoRA**: `model.safetensors`가 LoRA 키(lora_A, lora_B, base)를 포함하면, `model.py`에서 자동으로 병합 후 로드

---

## 4. 체크포인트 로드 경로 (디테일)

다른 체크포인트가 섞이지 않도록, **항상 `--checkpoint_dir`로 준 그 폴더만** 사용합니다.

### 4.1 run_robot_inference.py

1. `checkpoint_dir = args.checkpoint_dir` (절대 경로면 그대로, 상대면 `MINIMUM_ROOT` 기준)
2. `checkpoint_dir = Path(checkpoint_dir).resolve()`
3. `path_safetensors = checkpoint_dir / "model.safetensors"`, `path_pth = checkpoint_dir / "model.pth"`
4. **safetensors 존재 시** → 해당 폴더의 `model.safetensors` 사용  
   없으면 → 해당 폴더의 `model.pth` 사용
5. 로그 출력: `[1/3] 체크포인트 폴더 (실제 로드 경로): ...`  
   `가중치 파일: model.safetensors (우선 사용)` 또는 `model.pth`
6. `policy_config.create_trained_policy(config, checkpoint_dir)` 호출 시 **같은 checkpoint_dir** 전달

### 4.2 policy_config.create_trained_policy

1. `checkpoint_dir = download.maybe_download(str(checkpoint_dir))`  
   - 로컬 경로면 다운로드 없이 `Path(url).resolve()` 반환 → **동일 경로**
2. `path_safetensors = os.path.join(checkpoint_dir, "model.safetensors")`  
   `path_pth = os.path.join(checkpoint_dir, "model.pth")`  
   → **같은 폴더** 안에서만 경로 생성
3. `weight_path` = 이 폴더의 safetensors 또는 pth 하나
4. 로그: `가중치 로드 경로: <전체 경로>`
5. `model = train_config.model.load_pytorch(train_config, weight_path, device=...)`  
   → **그 weight_path 파일 하나**만 로드
6. `norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)`  
   → **같은 checkpoint_dir** 아래 `assets/droid/norm_stats.json` 사용 (asset_id="droid")

정리: **model 가중치**와 **norm_stats**는 모두 `--checkpoint_dir`로 지정한 **한 폴더**에서만 로드됩니다.

---

## 5. run_robot_inference.py 인자 요약

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--checkpoint_dir` | (필수) | 체크포인트 폴더 경로. 절대 경로 권장. 공백 있으면 따옴표 |
| `--config_name` | pi05_droid | 학습 시 사용한 config와 동일하게 |
| `--robot_ip` | 192.168.5.1 | Dobot IP |
| `--prompt` | pick up the red block | 작업 지시 문구 |
| `--dry_run` | False | 로봇 전송 없이 추론만 |
| `--no_camera` | False | 카메라 미사용(더미 이미지) |
| `--hz` | 20.0 | 제어/카메라 주기(Hz) |
| `--steps_per_inference` | None(=action_horizon) | 한 번 추론한 chunk에서 실행할 스텝 수 |
| `--save_frames_dir` | None | 실행 중 프레임 저장 폴더(선택) |
| `--movj_velocity` | 70 | MovJ 속도 비율 0~100 |
| `--movj_accel` | 60 | MovJ 가속도 비율 0~100 |
| `--action_scale` | 1.0 | 관절 6개 delta(rad)에 곱할 배율 |
| `--action_offset` | 0.0 | delta에 더할 오프셋(rad) |
| `--fine_stage_step` | None | 이 step 이후 매 step 재추론 + fine_stage_scale 적용 |
| `--fine_stage_scale` | 0.5 | fine stage에서 추가 배율 |
| `--grip_threshold` | 0.05 | 그리퍼 ON 임계값 (action[7] 또는 action[6]) |
| `--z_grip_trigger` | None | TCP Z(mm) 이하이고 grip>0 이면 강제 그리퍼 ON (예: 170) |
| `--no_init_pose` | False | True면 초기 자세 이동 스킵, 현재 위치에서 바로 추론 |
| `--save_step3_inference_224` | None | 3단계 비교용: 첫 추론 시 224 이미지 저장 경로 |

---

## 6. 액션 차원 및 해석

- **norm_stats.json** 기준: `actions.mean` / `std` 길이 = **8** (현재 사용 체크포인트 공통)
- **0~5**: 관절 6개 **상대 액션(rad)** → 현재 관절(rad)에 **더함** (delta)
- **6, 7**: 그리퍼 후보. 현재 코드: `action_dim > 7` → grip = action[7], `action_dim == 7` → grip = action[6]
- **역정규화**: `policy.infer()` 출력은 이미 Unnormalize 적용된 값.  
  추론 스크립트는 여기에 `action_scale` 곱·`action_offset` 더한 뒤, **현재 관절(rad) + delta**로 목표 관절을 구하고 deg로 변환해 MovJ에 전달.

---

## 7. 추론 실행 예시

### 7.1 가상환경 활성화 (필수)

```bash
cd ~/move-one
source min-imum/move-one/bin/activate
```

### 7.2 2000_lora_jetson (LoRA)

```bash
python run_robot_inference.py \
  --checkpoint_dir "/media/billy/새 볼륨/2000_lora_jetson" \
  --prompt "pick and place" \
  --hz 4 --steps_per_inference 15 \
  --action_scale 2.0 --grip_threshold 0.05 --z_grip_trigger 170 \
  --no_init_pose
```

### 7.3 4000_random_base_jetson (Base, LoRA 없음)

```bash
python run_robot_inference.py \
  --checkpoint_dir "/media/billy/새 볼륨/4000_random_base_jetson" \
  --prompt "pick and place" \
  --hz 4 --steps_per_inference 15 \
  --action_scale 2.0 --grip_threshold 0.05 --z_grip_trigger 170 \
  --no_init_pose
```

### 7.4 4000_random_jetson_1500 (LoRA)

```bash
python run_robot_inference.py \
  --checkpoint_dir "/media/billy/새 볼륨/4000_random_jetson_1500" \
  --prompt "pick and place" \
  --hz 4 --steps_per_inference 15 \
  --action_scale 2.0 --grip_threshold 0.05 --z_grip_trigger 170 \
  --no_init_pose
```

### 7.5 norm_stats 없을 때 (한 번만)

```bash
# 2000_lora_jetson 예시
mkdir -p "/media/billy/새 볼륨/2000_lora_jetson/assets/droid"
cp "$HOME/move-one/min-imum/checkpoints/10000_jetson/assets/droid/norm_stats.json" \
   "/media/billy/새 볼륨/2000_lora_jetson/assets/droid/norm_stats.json"
```

---

## 8. 트러블슈팅

| 현상 | 원인 | 조치 |
|------|------|------|
| `ModuleNotFoundError: No module named 'etils'` | 가상환경 미사용 | `source min-imum/move-one/bin/activate` 후 재실행 |
| NumPy API 경고 (0x10 vs 0xe) | 가상환경 미사용 / PyTorch와 numpy 버전 불일치 | move-one 가상환경 사용 |
| `체크포인트 폴더가 없습니다` | USB 미마운트 또는 경로 오타 | `ls "/media/billy/새 볼륨/"` 확인, 경로 따옴표 |
| `Norm stats file not found at: .../assets/droid/norm_stats.json` | 해당 체크포인트에 norm_stats 없음 | 위 7.5처럼 assets/droid 생성 후 복사 |
| `Dobot 소켓 연결 실패 ... Connection timed out` | eno1에 IP 없음 또는 로봇 전원/랜 미연결 | 2.2 네트워크 설정, `ping 192.168.5.1` |
| 로봇이 거의 안 움직임 | 액션 크기 작음 | `--action_scale 2.0` ~ 3.0 시도 |
| 그리퍼가 안 붙음 | grip 출력이 항상 낮음 | `--grip_threshold 0.05`, `--z_grip_trigger 170` 사용 |

---

## 9. 디버깅 4단계 (요약)

1. **액션 디코드**: 32D raw → 사용하는 step의 8D → 6 joint delta + denorm → 최종 MovJ 값.  
   블럭 left/center/right 바꿔서 raw·최종 명령이 바뀌는지 로그로 확인.
2. **Norm**: 실제 로드한 model 파일과 norm_stats.json이 **같은 체크포인트 폴더**인지, denorm 후 delta가 지나치게 눌리지 않는지.
3. **이미지**: 학습 시 224 vs 추론 시 224 (밝기, 블럭 크기, 그리퍼 visibility) 비교.  
   `--save_step3_inference_224`, `save_step3_training_224.py` 사용.
4. **VLM**: 1~3이 맞는 걸 전제로, 블럭 위치별 출력·마스킹 테스트로 시각 정보 반영 여부 확인.

---

## 10. 관련 스크립트

| 스크립트 | 용도 |
|----------|------|
| `run_robot_inference.py` | 메인 추론: 정책 로드 → 관측 수집 → infer → Dobot MovJ/ToolDO |
| `test_paligemma_vision_same_as_inference.py` | 추론과 동일 조건(정책+카메라+전처리)으로 PaliGemma lm_head 질의 (이미지 파일/카메라) |
| `check_red_block.py` | 카메라 1프레임에서 OpenCV로 빨간 픽셀 비율·위치 확인 |
| `save_step3_training_224.py` | 학습 프레임 하나를 224로 리사이즈해 저장 (3단계 비교용) |

---

## 11. 새 볼륨 폴더 확인 (참고)

새 체크포인트가 추가되었을 때, 아래로 실제 폴더 목록을 확인할 수 있습니다.

```bash
ls -la "/media/billy/새 볼륨/"
```

이 문서의 **§3 체크포인트 경로**는 위 출력 기준으로 유지·갱신하면 됩니다.
