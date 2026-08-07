#!/usr/bin/env python3
"""
check_setup.py — 설치 상태 점검 스크립트
==========================================
seg_gui.py 실행에 필요한 것들이 제대로 설치됐는지 확인하고,
YOLO 모델 가중치(약 24MB)가 없으면 미리 다운로드해 둔다.

사용법:
    ./.venv/bin/python check_setup.py
"""
import sys

print("Python:", sys.version.split()[0])

try:
    import tkinter
    print("[OK] tkinter", tkinter.TkVersion)
except ImportError as e:
    print("[FAIL] tkinter:", e)
    sys.exit(1)

try:
    import cv2
    print("[OK] OpenCV", cv2.__version__, "| aruco 모듈:", "있음" if hasattr(cv2, "aruco") else "없음!")
    if not hasattr(cv2, "aruco"):
        print("       -> pip3 uninstall opencv-python 후 opencv-contrib-python 재설치 필요")
        sys.exit(1)
except ImportError as e:
    print("[FAIL] OpenCV:", e)
    sys.exit(1)

try:
    from backends.dl_yolo import YoloNailBackend
    backend = YoloNailBackend()  # 가중치 없으면 여기서 자동 다운로드
    print("[OK] YOLO 백엔드 로드 완료 (모델 가중치 준비됨)")
except Exception as e:
    print("[WARN] YOLO 백엔드 로드 실패 (GUI는 classical 폴백으로 동작):", e)

print("\n설치 점검 완료 — python3 seg_gui.py 로 실행하세요")
