#!/usr/bin/env python3
"""HIK + ZED 카메라 이미지 저장해서 확인용."""
import sys, os
os.environ.setdefault("DISPLAY", ":0")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hardware"))

import cv2
import numpy as np

out = Path.home() / "camera_check"
out.mkdir(exist_ok=True)

# ── HIK ──────────────────────────────────────────────────────────────────────
print("HIK 초기화 중...")
from camera_capture import CameraCapture
cam = CameraCapture(use_hikrobot=True)
hik_frame = cam.get_frame()
if hik_frame is not None:
    cv2.imwrite(str(out / "hik_224.jpg"), cv2.cvtColor(hik_frame, cv2.COLOR_RGB2BGR))
    print(f"HIK : shape={hik_frame.shape} dtype={hik_frame.dtype} min={hik_frame.min()} max={hik_frame.max()}")
else:
    print("HIK 프레임 없음")
cam.close()

# ── ZED ──────────────────────────────────────────────────────────────────────
print("ZED 초기화 중...")
try:
    import pyzed.sl as sl
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.depth_mode = sl.DEPTH_MODE.NONE
    status = zed.open(init)
    if status == sl.ERROR_CODE.SUCCESS:
        import time; time.sleep(1.0)
        mat = sl.Mat()
        rt = sl.RuntimeParameters()
        if zed.grab(rt) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(mat, sl.VIEW.LEFT)
            frame = mat.get_data()[:, :, :3]
            frame_rgb = frame[:, :, ::-1].copy()   # BGR → RGB
            frame_640 = cv2.resize(frame_rgb, (640, 480))
            crop = frame_640[120:480, 150:510]      # 360×360
            final = cv2.resize(crop, (224, 224))
            cv2.imwrite(str(out / "zed_full.jpg"),    cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out / "zed_224.jpg"),     cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
            print(f"ZED full : shape={frame_rgb.shape}")
            print(f"ZED 224  : shape={final.shape} min={final.min()} max={final.max()}")
        else:
            print("ZED grab 실패")
        zed.close()
    else:
        print(f"ZED 열기 실패: {status}")
except Exception as e:
    print(f"ZED 예외: {e}")

print(f"\n저장 완료: {out}/")
for f in sorted(out.glob("*.jpg")):
    print(f"  {f.name}")
