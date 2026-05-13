#!/usr/bin/env python3
"""
C 검증: 학습 데이터 이미지로 직접 추론해서 모델 자체 이상 vs 도메인 차이 구분.

실행:
  cd ~/E6-VLA_INFERENCE
  source ~/move-one/min-imum/move-one/bin/activate
  export PYTHONPATH="$PWD/src:$PYTHONPATH"
  export LD_LIBRARY_PATH="$HOME/DobotControl/min-imum/move-one/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH"
  python scripts/test_train_image_infer.py
"""
import sys
import csv
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/openpi-client/src"))
from openpi_client.websocket_client_policy import WebsocketClientPolicy  # type: ignore

DATA_ROOT = Path("/media/billye6/새 볼륨/Dobot/2CAM-Orange")
LEFT_PROMPT  = "pick up the orange box from the left side and place it on the right side"
RIGHT_PROMPT = "pick up the orange box from the right side and place it on the left side"

HOST, PORT = "127.0.0.1", 8000
# pick_from_left 에피소드 번호 (j5 < 0)
TEST_EPS = [1, 2, 3, 10, 50]


def load_ep(ep_num: int, frame_idx: int = 0):
    ep = DATA_ROOT / str(ep_num)
    with open(ep / "robot_data.csv") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    row = rows[frame_idx]
    state = np.array([float(row['j1']), float(row['j2']), float(row['j3']),
                      float(row['j4']), float(row['j5']), float(row['j6']),
                      float(row['gripper_tooldo1'])], dtype=np.float32)

    hik = cv2.imread(str(ep / f"images/hik/frame_{frame_idx:06d}.jpg"))
    hik = cv2.cvtColor(hik, cv2.COLOR_BGR2RGB)
    zed = cv2.imread(str(ep / f"images/zed/frame_{frame_idx:06d}.jpg"))
    zed = cv2.cvtColor(zed, cv2.COLOR_BGR2RGB)
    return hik, zed, state


def infer(policy, hik, zed, state, prompt):
    obs = {
        "observation/exterior_image_1_left": hik,
        "observation/exterior_image_2_left": zed,
        "observation/state": state,
        "prompt": prompt,
    }
    result = policy.infer(obs)
    actions = np.asarray(result["actions"], dtype=np.float32)  # (16, 7)
    return actions


def report(tag, state, actions):
    j2_s = state[1]
    j2_0 = actions[0, 1]
    j2_8 = actions[8, 1]
    j2_15 = actions[15, 1]
    delta0 = actions[0, :6] - state[:6]
    print(f"  [{tag}]")
    print(f"    state j2={j2_s:.1f}")
    print(f"    action[0]  j2={j2_0:.1f}  Δj2={j2_0-j2_s:+.1f}")
    print(f"    action[8]  j2={j2_8:.1f}  Δj2={j2_8-j2_s:+.1f}")
    print(f"    action[15] j2={j2_15:.1f}  Δj2={j2_15-j2_s:+.1f}")
    print(f"    delta[0] max={np.abs(delta0).max():.1f}°  {['%+.1f'%d for d in delta0]}")
    going_down = (j2_0 - j2_s) > 1.0
    print(f"    → {'✅ j2 증가 (내려가는 방향)' if going_down else '❌ j2 감소 또는 정체 (start 자세 유지)'}")


def main():
    print(f"정책 서버 연결: ws://{HOST}:{PORT}")
    policy = WebsocketClientPolicy(host=HOST, port=PORT)
    meta = policy.get_server_metadata()
    print(f"서버 메타: {meta}\n")

    # ── A: 학습 이미지로 추론 (여러 에피소드) ─────────────────────────────────
    print("=" * 60)
    print("A. 학습 이미지 + 학습 state → 추론")
    print("   기대: action[0]에서 j2 증가 (+) 이어야 함")
    print("=" * 60)
    for ep_num in TEST_EPS:
        try:
            hik, zed, state = load_ep(ep_num, frame_idx=0)
            print(f"\n  [EP {ep_num}] state={np.round(state[:6],1).tolist()}")
            actions = infer(policy, hik, zed, state, LEFT_PROMPT)
            report(f"EP{ep_num} train_img+train_state", state, actions)
        except Exception as e:
            print(f"  EP{ep_num} 실패: {e}")

    # ── B: 학습 state만 쓰고 이미지를 0 / random으로 대체 (vision 영향도 테스트) ──
    print("\n" + "=" * 60)
    print("B. EP1 state 고정, 이미지만 교체 → vision 영향도 확인")
    print("   train_img vs zeros vs random 세 가지 결과가 같으면 vision 무시 중")
    print("=" * 60)
    try:
        hik_train, zed_train, state = load_ep(1, frame_idx=0)
        hik_zero   = np.zeros_like(hik_train)
        zed_zero   = np.zeros_like(zed_train)
        hik_rand   = np.random.randint(0, 256, hik_train.shape, dtype=np.uint8)
        zed_rand   = np.random.randint(0, 256, zed_train.shape, dtype=np.uint8)

        print()
        a = infer(policy, hik_train, zed_train, state, LEFT_PROMPT)
        report("train_img", state, a)
        print()
        a = infer(policy, hik_zero, zed_zero, state, LEFT_PROMPT)
        report("zeros_img", state, a)
        print()
        a = infer(policy, hik_rand, zed_rand, state, LEFT_PROMPT)
        report("random_img", state, a)
    except Exception as e:
        print(f"  B 실패: {e}")

    # ── C: 현재 로봇 실제 state + 학습 이미지 조합 ──────────────────────────
    # 현재 로봇이 거의 init pose (j2≈37.7) 이므로 학습 EP1 frame0과 비슷한 state로 테스트
    print("\n" + "=" * 60)
    print("C. 현재 init pose state + 학습 이미지 → 추론")
    print("=" * 60)
    try:
        hik_train, zed_train, _ = load_ep(1, frame_idx=0)
        init_state = np.array([91.303, 37.731, 55.668, -6.652, -87.852, 173.284, 0.0], dtype=np.float32)
        print(f"  init_state: {np.round(init_state[:6],1).tolist()}")
        a = infer(policy, hik_train, zed_train, init_state, LEFT_PROMPT)
        report("init_state+train_img", init_state, a)
    except Exception as e:
        print(f"  C 실패: {e}")

    print("\n" + "=" * 60)
    print("해석:")
    print("  A에서 j2 증가 → 모델 OK, 현재 카메라/환경이 문제")
    print("  A에서도 j2 감소 → 모델 자체 문제 (LoRA 매핑, mode collapse)")
    print("  B에서 세 결과 같음 → 모델이 vision 무시, state만 보고 출력")
    print("=" * 60)


if __name__ == "__main__":
    main()
