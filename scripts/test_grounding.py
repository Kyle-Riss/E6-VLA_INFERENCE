#!/usr/bin/env python3
"""
"가짜 grounding" 증명 테스트:
  - 학습 trajectory의 여러 프레임(시작/중간/끝)을 state로 넣어서
    모델이 state 위치에 따라 다른 행동을 예측하는지 확인
  - 카메라 없이 (no_camera=True) zeros 이미지로 실행 가능
  - j4 포함 전체 joint delta 출력

실행:
  python scripts/test_grounding.py              # 카메라 이미지 포함
  python scripts/test_grounding.py --no_camera  # 카메라 없이 (zeros)
"""
import sys, csv, argparse, numpy as np, cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/openpi-client/src"))
from openpi_client.websocket_client_policy import WebsocketClientPolicy  # type: ignore

DATA_ROOT = Path("/media/billye6/새 볼륨/Dobot/2CAM-Orange")
LEFT_PROMPT = "pick up the orange box from the left side and place it on the right side"


def load_frame(ep_num: int, frame_idx: int, no_camera: bool):
    ep = DATA_ROOT / str(ep_num)
    with open(ep / "robot_data.csv") as f:
        rows = list(csv.DictReader(f))
    row = rows[min(frame_idx, len(rows)-1)]
    state = np.array([float(row[k]) for k in
                      ['j1','j2','j3','j4','j5','j6','gripper_tooldo1']], dtype=np.float32)
    if no_camera:
        hik = np.zeros((224, 224, 3), dtype=np.uint8)
        zed = np.zeros((224, 224, 3), dtype=np.uint8)
    else:
        hik = cv2.cvtColor(cv2.imread(str(ep/f"images/hik/frame_{frame_idx:06d}.jpg")), cv2.COLOR_BGR2RGB)
        zed = cv2.cvtColor(cv2.imread(str(ep/f"images/zed/frame_{frame_idx:06d}.jpg")), cv2.COLOR_BGR2RGB)
    return state, hik, zed


def infer_once(policy, hik, zed, state):
    a = np.asarray(policy.infer({
        "observation/exterior_image_1_left": hik,
        "observation/exterior_image_2_left": zed,
        "observation/state": state,
        "prompt": LEFT_PROMPT,
    })["actions"], dtype=np.float32)
    return a


def print_row(label, state, a):
    d = a[0, :6] - state[:6]
    j2_dir = "↑UP" if d[1] > 1.0 else ("↓FLIP" if d[1] < -20 else "→flat")
    print(f"  {label:30s}  j2={state[1]:5.1f}→{a[0,1]:5.1f}(Δ{d[1]:+5.1f}) "
          f"j3Δ{d[2]:+5.1f} j4Δ{d[3]:+5.1f}  {j2_dir}  max_delta={np.abs(d).max():.1f}°")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_camera", action="store_true", help="이미지를 zeros로 대체")
    parser.add_argument("--ep", type=int, default=1)
    args = parser.parse_args()

    policy = WebsocketClientPolicy(host="127.0.0.1", port=8000)
    policy.get_server_metadata()
    mode = "zeros(카메라 없음)" if args.no_camera else "실제 학습 이미지"
    print(f"모드: {mode}  EP={args.ep}\n")

    # ── 테스트 1: trajectory 상의 여러 프레임 위치에서 추론 ─────────────────
    print("=" * 70)
    print("TEST 1: trajectory 위치별 추론 (가짜 grounding 확인)")
    print("  진짜 grounding → frame 위치에 따라 j2 방향이 달라야 함")
    print("  가짜 grounding → frame 위치 무관하게 항상 같은 방향")
    print("=" * 70)

    ep = DATA_ROOT / str(args.ep)
    with open(ep / "robot_data.csv") as f:
        total = sum(1 for _ in f) - 1
    print(f"  EP{args.ep} 총 프레임: {total}")
    print()
    print(f"  {'label':30s}  j2 현재→예측(Δ)    j3Δ   j4Δ   방향  max_delta")
    print(f"  {'-'*68}")

    for frac, name in [(0.0, "frame_0(start)"), (0.1, "frame_10%"),
                       (0.2, "frame_20%"), (0.3, "frame_30%(pick_down)"),
                       (0.4, "frame_40%"), (0.5, "frame_50%(mid)"),
                       (0.6, "frame_60%"), (0.8, "frame_80%")]:
        fidx = int(frac * total)
        state, hik, zed = load_frame(args.ep, fidx, args.no_camera)
        a = infer_once(policy, hik, zed, state)
        print_row(f"{name}(f{fidx})", state, a)

    # ── 테스트 2: init pose에서 no_camera vs real camera 비교 ────────────────
    print()
    print("=" * 70)
    print("TEST 2: init pose 고정, no_camera vs 학습이미지 비교")
    print("  결과 차이 있음 → vision 사용 중 (grounding 시도는 함)")
    print("  결과 같음 → vision 완전 무시")
    print("=" * 70)
    init_state = np.array([91.303, 37.731, 55.668, -6.652, -87.852, 173.284, 0.0], dtype=np.float32)
    _, hik_train, zed_train = load_frame(args.ep, 0, no_camera=False)
    hik_zero = np.zeros((224, 224, 3), dtype=np.uint8)
    zed_zero = np.zeros((224, 224, 3), dtype=np.uint8)

    print(f"\n  {'label':30s}  j2 현재→예측(Δ)    j3Δ   j4Δ   방향  max_delta")
    print(f"  {'-'*68}")
    for tag, h, z in [("init+학습이미지", hik_train, zed_train),
                       ("init+zeros(no_camera)", hik_zero, zed_zero)]:
        a = infer_once(policy, h, z, init_state)
        print_row(tag, init_state, a)

    # ── 테스트 3: j4 집중 분석 (20 샘플, init pose) ─────────────────────────
    print()
    print("=" * 70)
    print("TEST 3: j4 분포 확인 (init pose, 20샘플)")
    print("  학습 j4 범위: -35 ~ +5° (approach 구간)")
    print("=" * 70)
    _, hik_tr, zed_tr = load_frame(args.ep, 0, no_camera=False)
    j4_vals = []
    j2_up = 0
    for i in range(20):
        h = hik_zero if args.no_camera else hik_tr
        z = zed_zero if args.no_camera else zed_tr
        a = infer_once(policy, h, z, init_state)
        j4_vals.append(a[0, 3])
        if a[0, 1] - init_state[1] > 1.0:
            j2_up += 1
        print(f"  [{i:02d}] j2={a[0,1]:6.1f}(Δ{a[0,1]-init_state[1]:+5.1f})  "
              f"j3={a[0,2]:6.1f}(Δ{a[0,2]-init_state[2]:+5.1f})  "
              f"j4={a[0,3]:6.1f}(Δ{a[0,3]-init_state[3]:+5.1f})")

    print()
    print(f"  j4 예측값: mean={np.mean(j4_vals):.1f}  min={np.min(j4_vals):.1f}  max={np.max(j4_vals):.1f}  std={np.std(j4_vals):.1f}")
    print(f"  학습 j4 정상범위: -35 ~ +5°  → {'정상' if -40 < np.mean(j4_vals) < 10 else '⚠️ OOD'}")
    print(f"  j2 증가 샘플: {j2_up}/20")


if __name__ == "__main__":
    main()
