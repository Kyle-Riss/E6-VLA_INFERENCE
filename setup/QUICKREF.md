## e6-vla — Quick Reference

### 가상환경 활성화 (move-one venv 재사용)
```bash
source ~/move-one/min-imum/move-one/bin/activate
# 프롬프트에 (move-one) 이 보이면 OK
```

### openpi import 테스트
```bash
python -c "from openpi.training import config; print('SUCCESS: OpenPi OK')"
```

### 정책 서버 실행 (run_server.sh)
```bash
bash ~/e6-vla/run_server.sh \
  ~/checkpoints/pi05_e6_v1_lora/e6_2cam_lora_v1_pytorch/31999_pytorch
```

### run_e6_client.py 실행 (클라이언트)
```bash
source ~/move-one/min-imum/move-one/bin/activate
python ~/e6-vla/examples/e6/run_e6_client.py \
  --server_host 127.0.0.1 \
  --robot_ip 192.168.5.1 \
  --prompt "approach red object"

# 로봇/카메라 없이 추론값만 확인
python ~/e6-vla/examples/e6/run_e6_client.py \
  --server_host 127.0.0.1 \
  --dry_run \
  --show_actions \
  --prompt "approach red object"
```

### 자동 approach↔return 순환
```bash
python ~/e6-vla/examples/e6/run_e6_client.py \
  --server_host 127.0.0.1 \
  --robot_ip 192.168.5.1 \
  --auto_cycle_return_approach \
  --approach_prompt "approach red object" \
  --return_prompt "return" \
  --approach_cycle_sec 8 \
  --return_cycle_sec 2
```

### 카메라만 테스트
```bash
# HIKRobot MVS SDK 경로 설정 필요
export MVCAM_COMMON_RUNENV=/opt/MVS/lib
python ~/e6-vla/hardware/utils/test_robot_connection.py
```

### 주요 경로
| 항목 | 경로 |
|------|------|
| 추론 클라이언트 | `examples/e6/run_e6_client.py` |
| 정책 서버 | `scripts/serve_policy.py` |
| E6 정책 정의 | `src/openpi/policies/e6_policy.py` |
| 학습 config | `src/openpi/training/config.py` |
| 카메라 캡처 | `hardware/camera_capture.py` |
| Dobot API | `hardware/dobot/dobot_api.py` |

### 설치된 주요 패키지 (move-one venv 기준)
- JAX/Flax: 0.6.2 / 0.10.7
- PyTorch: 2.5.0a0 (Jetson aarch64)
- NumPy: 1.26.4
- transformers: 4.53.2
