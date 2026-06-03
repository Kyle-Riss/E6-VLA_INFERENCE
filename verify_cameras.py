#!/usr/bin/env python3
"""
카메라 + 추론 입력 검증 스크립트

검증 항목:
  1. HIK 카메라 — 프레임 캡처 + 이미지 저장
  2. ZED 카메라 — 프레임 캡처 + 이미지 저장
  3. 추론 서버 — obs 전송 후 action_chunk 수신 확인
  4. 저장된 이미지로 시각 확인

실행:
  source ~/move-one/min-imum/move-one/bin/activate
  export MVCAM_COMMON_RUNENV=/opt/MVS/lib
  python verify_cameras.py              # 카메라만 검증
  python verify_cameras.py --infer      # 추론 서버까지 검증
  python verify_cameras.py --no_zed     # HIK만
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent
_HARDWARE = _REPO / "hardware"
_CLIENT_SRC = _REPO / "packages" / "openpi-client" / "src"
for _p in [str(_HARDWARE), str(_CLIENT_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

SAVE_DIR = Path.home() / "verify_cameras"


def save_img(arr: np.ndarray, name: str) -> Path:
    try:
        import cv2
        path = SAVE_DIR / name
        cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        return path
    except Exception as e:
        print(f"  [WARN] 이미지 저장 실패: {e}")
        return SAVE_DIR / name


def check_frame(arr: np.ndarray, label: str) -> bool:
    mean = float(arr.mean())
    mn, mx = int(arr.min()), int(arr.max())
    ok = mean > 10.0  # 완전 검정 아니면 OK
    status = "OK" if ok else "FAIL(검정 화면)"
    print(f"  [{label}] shape={arr.shape} dtype={arr.dtype} mean={mean:.1f} min={mn} max={mx} → {status}")
    return ok


def test_hik() -> np.ndarray | None:
    print("\n[1] HIK 카메라 테스트")
    try:
        import camera_capture
        cam = camera_capture.CameraCapture()
        print(f"  드라이버: {cam._name}")
        if cam._cam is None:
            print("  FAIL — 카메라 연결 안됨")
            return None
        # 워밍업 (첫 몇 프레임은 노출 불안정)
        for _ in range(3):
            cam.get_frame()
            time.sleep(0.1)
        frame = cam.get_frame()
        cam.close()
        if frame is None:
            print("  FAIL — 프레임 읽기 실패")
            return None
        arr = np.asarray(frame, dtype=np.uint8)
        ok = check_frame(arr, "HIK")
        path = save_img(arr, "hik_frame.jpg")
        print(f"  저장: {path}")
        return arr if ok else None
    except Exception as e:
        print(f"  FAIL — 예외: {e}")
        import traceback; traceback.print_exc()
        return None


def test_zed() -> np.ndarray | None:
    print("\n[2] ZED 카메라 테스트")
    try:
        import cv2
        import pyzed.sl as sl
        zed = sl.Camera()
        init = sl.InitParameters()
        init.depth_mode = sl.DEPTH_MODE.NONE
        init.camera_resolution = sl.RESOLUTION.HD1080
        init.camera_fps = 30
        status = zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"  FAIL — ZED 오픈 실패: {status}")
            return None
        sn = zed.get_camera_information().serial_number
        print(f"  SN={sn}")
        mat = sl.Mat()
        # 워밍업
        for _ in range(5):
            zed.grab()
            time.sleep(0.05)
        if zed.grab() != sl.ERROR_CODE.SUCCESS:
            print("  FAIL — grab 실패")
            zed.close()
            return None
        zed.retrieve_image(mat, sl.VIEW.LEFT)
        raw = mat.get_data()[:, :, :3][:, :, ::-1].copy()  # BGR→RGB
        print(f"  원본 해상도: {raw.shape}")

        # 학습과 동일한 전처리
        frame = cv2.resize(raw, (640, 480))
        frame = frame[120:480, 150:510]   # 360×360
        frame = cv2.resize(frame, (224, 224))
        arr = frame.astype(np.uint8)

        # 원본도 저장 (참고용)
        raw_resized = cv2.resize(raw, (640, 480)).astype(np.uint8)
        save_img(raw_resized, "zed_raw_640x480.jpg")

        ok = check_frame(arr, "ZED(전처리후)")
        path = save_img(arr, "zed_frame_224x224.jpg")
        print(f"  저장: {path} (원본도 저장됨)")
        zed.close()
        return arr if ok else None
    except ImportError:
        print("  SKIP — pyzed 미설치")
        return None
    except Exception as e:
        print(f"  FAIL — 예외: {e}")
        import traceback; traceback.print_exc()
        return None


def test_inference(hik: np.ndarray | None, zed: np.ndarray | None,
                   host: str, port: int) -> bool:
    print(f"\n[3] 추론 서버 테스트 (ws://{host}:{port})")
    try:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        policy = WebsocketClientPolicy(host=host, port=port)
        meta = policy.get_server_metadata()
        print(f"  서버 연결 OK: {meta}")
    except Exception as e:
        print(f"  FAIL — 서버 연결 실패: {e}")
        return False

    hik_frame = hik if hik is not None else np.zeros((224, 224, 3), dtype=np.uint8)
    zed_frame = zed if zed is not None else np.zeros((224, 224, 3), dtype=np.uint8)
    state = np.array([90.128, 42.907, 59.355, -11.702, -87.582, 177.813, 0.0], dtype=np.float32)

    obs = {
        "observation/exterior_image_1_left": hik_frame,
        "observation/exterior_image_2_left": zed_frame,
        "observation/state": state,
        "prompt": "approach red object",
    }

    print(f"  obs 구성:")
    print(f"    exterior_image_1_left (HIK): {hik_frame.shape} mean={hik_frame.mean():.1f} {'[실제 카메라]' if hik is not None else '[zeros - 카메라 없음]'}")
    print(f"    exterior_image_2_left (ZED): {zed_frame.shape} mean={zed_frame.mean():.1f} {'[실제 카메라]' if zed is not None else '[zeros - 카메라 없음]'}")
    print(f"    state: {state}")
    print(f"    prompt: 'approach red object'")

    try:
        t0 = time.monotonic()
        result = policy.infer(obs)
        elapsed = (time.monotonic() - t0) * 1000
        actions = np.asarray(result["actions"], dtype=np.float32)
        print(f"\n  추론 완료 ({elapsed:.0f}ms)")
        print(f"  action shape: {actions.shape}  (기대: (16, 7))")
        print(f"  action[0] (첫 스텝): {np.round(actions[0], 2)}")
        print(f"    j1~j6 (°): {np.round(actions[0, :6], 2)}")
        print(f"    gripper  : {actions[0, 6]:.3f}")

        # 액션 범위 체크 (학습 데이터 기준 정상 범위)
        j_mean = actions[:, :6].mean(axis=0)
        print(f"  action 평균 관절각: {np.round(j_mean, 1)}")
        reasonable = np.all(np.abs(j_mean) < 200)
        print(f"  액션 범위 {'OK (±200° 이내)' if reasonable else 'WARN — 비정상적으로 큰 값'}")
        return True
    except Exception as e:
        print(f"  FAIL — 추론 실패: {e}")
        import traceback; traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_zed", action="store_true")
    parser.add_argument("--no_hik", action="store_true")
    parser.add_argument("--infer", action="store_true", help="추론 서버 검증 포함")
    parser.add_argument("--server_host", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=8000)
    args = parser.parse_args()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"이미지 저장 위치: {SAVE_DIR}")

    hik = None if args.no_hik else test_hik()
    zed = None if args.no_zed else test_zed()

    print("\n" + "=" * 50)
    print("카메라 검증 결과:")
    print(f"  HIK: {'OK' if hik is not None else 'FAIL/SKIP'}")
    print(f"  ZED: {'OK' if zed is not None else 'FAIL/SKIP'}")

    if args.infer:
        ok = test_inference(hik, zed, args.server_host, args.server_port)
        print(f"  추론 서버: {'OK' if ok else 'FAIL'}")

    print("=" * 50)
    print(f"\n저장된 이미지 확인:")
    for f in sorted(SAVE_DIR.glob("*.jpg")):
        print(f"  {f}")
    print(f"\n빠른 확인 (터미널에서 직접):")
    print(f"  eog {SAVE_DIR}/*.jpg   # 이미지 뷰어")
    print(f"  ls -lh {SAVE_DIR}/")


if __name__ == "__main__":
    main()
