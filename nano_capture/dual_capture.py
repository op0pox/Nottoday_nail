"""
dual_capture.py
================
[1. 역할]
    Jetson Nano에 고정된 CSI 카메라 2대(위=top, 측면=side)의 미리보기를
    X11 포워딩으로 노트북 화면에 나란히 띄우고, 화면의 CAPTURE 버튼을
    마우스로 클릭(또는 스페이스키)하면 두 카메라의 사진을 동시에
    ~/captures/ 폴더에 저장한다.

    저장 파일명 (번호 자동 증가):
        capture_0001_1_top.jpg   <- 위 카메라
        capture_0001_2_side.jpg  <- 측면 카메라

[2. 실행 명령어]
    (노트북에서 X11 포워딩으로 접속한 뒤)
        ssh -Y -C home@192.168.55.1
        python3 dual_capture.py

    옵션:
        python3 dual_capture.py --preview-width 480   # 미리보기 더 작게(더 부드럽게)
        python3 dual_capture.py --flip-top 2          # 위 카메라 화면 180도 회전
        python3 dual_capture.py --flip-side 2         # 측면 카메라 화면 180도 회전
        python3 dual_capture.py --outdir ~/my_photos  # 저장 폴더 변경

[3. 어디에 입력해야 하는가]
    -> 노트북에서 SSH(X11 포워딩, ssh -Y)로 접속한 Jetson Nano 터미널에서 실행.

[4. 정상적으로 실행되면]
    "Nail Capture" 창에 왼쪽=TOP, 오른쪽=SIDE 미리보기가 보이고,
    아래에 [ CAPTURE ] 버튼이 보인다.
    - 버튼 클릭 또는 스페이스키/'c' : 사진 2장 저장 (터미널에 [SAVED] 출력)
    - 'q' 또는 ESC : 종료
    카메라가 연결 안 된 쪽은 "NO SIGNAL"로 표시된다 (그 쪽은 저장 안 됨).

[5. 오류가 발생하면 확인할 것]
    - 창이 안 뜬다: 노트북에서 XQuartz 실행 중인지, ssh -Y 로 접속했는지,
      나노 터미널에서 echo $DISPLAY 가 비어있지 않은지 확인.
    - 두 카메라 다 NO SIGNAL: ls /dev/video* 로 장치가 보이는지 확인.
      장치가 없으면 카메라 케이블 연결 후 나노를 재부팅해야 한다
      (CSI 카메라는 부팅할 때만 인식된다).
    - 카메라가 있는데도 안 열린다(Argus 에러): 카메라 데몬이 꼬인 것.
      sudo systemctl restart nvargus-daemon 실행 후 다시 시도.
    - 나노가 갑자기 꺼진다: 전원 부족. 마이크로 USB 대신 5V/4A 어댑터
      (배럴잭 + J48 점퍼) 사용을 권장.
"""

import argparse
import os
import re

import cv2
import numpy as np

WINDOW_NAME = "Nail Capture"
CAPTURE_WIDTH = 1920   # 저장되는 사진 해상도 (IMX219 1080p 크롭 모드)
CAPTURE_HEIGHT = 1080
SENSOR_MODE = 2        # IMX219 1920x1080@30fps - 120fps 모드로 인한 전원 문제 방지
FRAMERATE = 21


def gst_pipeline(sensor_id, flip_method=0):
    """CSI 카메라용 GStreamer 파이프라인 (1080p, 저전력 센서 모드 고정)."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode={SENSOR_MODE} ! "
        f"video/x-raw(memory:NVMM), width={CAPTURE_WIDTH}, height={CAPTURE_HEIGHT}, "
        f"format=NV12, framerate={FRAMERATE}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        "video/x-raw, format=BGRx ! videoconvert ! "
        "video/x-raw, format=BGR ! appsink drop=true max-buffers=1"
    )


def open_camera(name, sensor_id, flip_method):
    print(f"[INFO] {name} 카메라 여는 중... (sensor-id={sensor_id})")
    cap = cv2.VideoCapture(gst_pipeline(sensor_id, flip_method), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print(f"[WARN] {name} 카메라를 열지 못했습니다. NO SIGNAL로 표시합니다.")
        return None
    print(f"[OK] {name} 카메라 열림")
    return cap


def next_capture_index(outdir):
    """폴더 안의 capture_NNNN_* 파일을 보고 다음 번호를 정한다."""
    max_idx = 0
    if os.path.isdir(outdir):
        for f in os.listdir(outdir):
            m = re.match(r"capture_(\d+)_", f)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def make_preview(frame, label, pw, ph):
    """프레임을 미리보기 크기로 줄이고 라벨을 얹는다. 없으면 NO SIGNAL 화면."""
    if frame is None:
        tile = np.zeros((ph, pw, 3), dtype=np.uint8)
        cv2.putText(tile, f"{label}: NO SIGNAL", (20, ph // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    else:
        tile = cv2.resize(frame, (pw, ph))
        cv2.putText(tile, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return tile


def parse_args():
    parser = argparse.ArgumentParser(description="카메라 2대 미리보기 + 버튼 클릭 촬영")
    parser.add_argument("--preview-width", type=int, default=640,
                        help="미리보기 한 쪽의 가로 크기 (기본 640, 끊기면 480)")
    parser.add_argument("--flip-top", type=int, default=0,
                        help="위 카메라 회전 (0=그대로, 2=180도)")
    parser.add_argument("--flip-side", type=int, default=0,
                        help="측면 카메라 회전 (0=그대로, 2=180도)")
    parser.add_argument("--outdir", default=os.path.expanduser("~/captures"),
                        help="사진 저장 폴더 (기본 ~/captures)")
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = os.path.expanduser(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    pw = args.preview_width
    ph = pw * CAPTURE_HEIGHT // CAPTURE_WIDTH
    bar_h = 70  # 아래 버튼 영역 높이

    cap_top = open_camera("TOP(위)", 0, args.flip_top)
    cap_side = open_camera("SIDE(측면)", 1, args.flip_side)
    if cap_top is None and cap_side is None:
        print("[WARN] 카메라가 하나도 안 열렸습니다. 미리보기 창은 뜨지만 촬영은 안 됩니다.")

    # 버튼 위치 (합쳐진 화면 좌표 기준)
    total_w = pw * 2
    btn_w, btn_h = 260, 46
    btn_x1 = (total_w - btn_w) // 2
    btn_y1 = ph + (bar_h - btn_h) // 2
    btn_x2, btn_y2 = btn_x1 + btn_w, btn_y1 + btn_h

    state = {"capture_requested": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if btn_x1 <= x <= btn_x2 and btn_y1 <= y <= btn_y2:
                state["capture_requested"] = True

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    idx = next_capture_index(outdir)
    saved_msg = ""     # 화면에 잠깐 띄울 저장 알림
    saved_msg_ttl = 0  # 몇 프레임 동안 보여줄지

    print(f"[INFO] 저장 폴더: {outdir} (다음 번호: {idx:04d})")
    print("[INFO] 촬영: 화면의 CAPTURE 버튼 클릭 또는 스페이스/'c' 키 | 종료: 'q' 또는 ESC")

    try:
        while True:
            ok1, frame_top = (cap_top.read() if cap_top else (False, None))
            ok2, frame_side = (cap_side.read() if cap_side else (False, None))
            if not ok1:
                frame_top = None
            if not ok2:
                frame_side = None

            preview = cv2.hconcat([
                make_preview(frame_top, "TOP", pw, ph),
                make_preview(frame_side, "SIDE", pw, ph),
            ])
            canvas = np.zeros((ph + bar_h, total_w, 3), dtype=np.uint8)
            canvas[:ph] = preview

            # CAPTURE 버튼 그리기
            cv2.rectangle(canvas, (btn_x1, btn_y1), (btn_x2, btn_y2), (60, 160, 60), -1)
            cv2.rectangle(canvas, (btn_x1, btn_y1), (btn_x2, btn_y2), (255, 255, 255), 2)
            cv2.putText(canvas, "CAPTURE (SPACE)", (btn_x1 + 22, btn_y1 + 31),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(canvas, "q: quit", (10, ph + bar_h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            if saved_msg_ttl > 0:
                cv2.putText(canvas, saved_msg, (10, ph + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                saved_msg_ttl -= 1

            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):  # q 또는 ESC
                print("[INFO] 종료합니다.")
                break
            if key in (ord(" "), ord("c")) or state["capture_requested"]:
                state["capture_requested"] = False
                saved = []
                if frame_top is not None:
                    p = os.path.join(outdir, f"capture_{idx:04d}_1_top.jpg")
                    cv2.imwrite(p, frame_top)
                    saved.append(p)
                if frame_side is not None:
                    p = os.path.join(outdir, f"capture_{idx:04d}_2_side.jpg")
                    cv2.imwrite(p, frame_side)
                    saved.append(p)
                if saved:
                    for p in saved:
                        print(f"[SAVED] {p}")
                    if len(saved) < 2:
                        print("[WARN] 한쪽 카메라가 NO SIGNAL이라 1장만 저장됐습니다.")
                    saved_msg = f"SAVED #{idx:04d} ({len(saved)}/2)"
                    saved_msg_ttl = 30
                    idx += 1
                else:
                    print("[WARN] 저장할 프레임이 없습니다 (양쪽 다 NO SIGNAL).")
                    saved_msg = "NO FRAME - NOT SAVED"
                    saved_msg_ttl = 30

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C 로 종료했습니다.")
    finally:
        for cap in (cap_top, cap_side):
            if cap is not None:
                cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
