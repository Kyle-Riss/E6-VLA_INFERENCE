# 체크포인트·로더 코드 정의

## 1. 체크포인트 (폴더 구조)

| 항목 | 경로/이름 | 설명 |
|------|-----------|------|
| **가중치 (PyTorch)** | `checkpoint_dir/model.safetensors` 또는 `model.pth` | 학습된 모델 파라미터. 이 중 하나만 있으면 PyTorch 경로로 로드. |
| **가중치 (JAX)** | `checkpoint_dir/params/` | Orbax 형식. PyTorch 파일이 없을 때만 사용. |
| **정규화 통계** | `checkpoint_dir/assets/<asset_id>/norm_stats.json` | 관측 정규화용. `<asset_id>` 예: `droid`. |

- **체크포인트 모델** = 위 폴더 안의 **`.pth`(또는 `.safetensors`) 파일**이 곧 체크포인트 모델(가중치)이다.
- 폴더 이름(예: `10000_jetson`)은 사용자가 정한 이름이며, **코드에서 `checkpoint_dir`로 넘기는 경로**가 이 폴더를 가리키면 된다.

---

## 2. 로더 (어디서 어떻게 로드하는지)

**파일:** `min-imum/openpi/policies/policy_config.py`

```python
def create_trained_policy(train_config, checkpoint_dir, ...):
    # 1) 가중치 경로 결정: model.safetensors 우선, 없으면 model.pth
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    if not os.path.exists(weight_path):
        weight_path = os.path.join(checkpoint_dir, "model.pth")
    is_pytorch = os.path.exists(weight_path)

    # 2) 모델 로드
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
    else:
        model = train_config.model.load(restore_params(checkpoint_dir / "params", ...))

    # 3) norm_stats: checkpoint_dir/assets + config의 asset_id
    norm_stats = load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
```

**파일:** `min-imum/openpi/models/model.py`

```python
def load_pytorch(self, train_config, weight_path: str):
    model = pi0_pytorch.PI0Pytorch(config=train_config.model)
    if weight_path.endswith(".pth"):
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
    else:
        import safetensors
        safetensors.torch.load_model(model, weight_path)
    return model
```

- **정의:** `checkpoint_dir` 안에는 **다른 체크포인트(예: pi05_droid)를 참조하지 않고**, 이 폴더 안의 `model.pth`(또는 `model.safetensors`) + `assets` 만 사용한다.

---

## 3. Config 이름 vs 체크포인트 경로

- **config_name**: 학습 시 쓴 설정과 맞춰야 함. 예: `pi05_droid`, `pi0_e6_freeze_vlm`.
- **checkpoint_dir**: 실제 폴더 경로. 예: `./checkpoints/10000_jetson`.

```python
config = get_config("pi05_droid")           # 또는 pi0_e6_freeze_vlm
policy = create_trained_policy(config, "./checkpoints/10000_jetson")
```

---

## 4. 추론 입출력

**입력 `obs` (dict):**

- `observation/exterior_image_1_left`: (224, 224, 3) uint8
- `observation/wrist_image_left`: (224, 224, 3) uint8
- `observation/joint_position`: (7,) or (8,) float32
- `observation/gripper_position`: (1,) float32
- `prompt`: str

**출력:**

- `result = policy.infer(obs)`
- `actions = result["actions"]`: shape `(action_horizon, action_dim)` (예: 16×8)

---

## 5. 상수 정의 (코드에서 쓸 때)

`move-one/checkpoint_def.py` 에서 다음 상수/함수로 위 구조를 코드로 정의해 두었다.

- `WEIGHT_PTH`, `WEIGHT_SAFETENSORS`, `ASSETS_DIR`, `PARAMS_DIR`
- `get_weight_path(checkpoint_dir)` : 로드할 가중치 파일 경로
- `is_pytorch_checkpoint(checkpoint_dir)` : PyTorch 체크포인트 여부
- `OBS_KEYS` : 추론 입력 키 목록

요약: **체크포인트 모델 = `checkpoint_dir` 안의 `model.pth`(또는 `model.safetensors`) 파일**이고, 로드는 `create_trained_policy(config, checkpoint_dir)` 한 번으로 이루어진다.
