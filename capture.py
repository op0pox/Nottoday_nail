#!/usr/bin/env python3
import os
import time
import cv2


def gstreamer_pipeline(sensor_id=0, capture_width=3264, capture_height=2464,
                       framerate=21, flip_method=0):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, "
        "framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink drop=true max-buffers=1"
        % (sensor_id, capture_width, capture_height, framerate, flip_method)
    )


def open_camera():
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(
            "카메라를 열지 못했습니다. 확인 사항:\n"
            "  1) CSI 케이블이 제대로 꽂혔는지\n"
            "  2) 다른 프로세스가 카메라를 점유 중인지 "
            "(sudo systemctl restart nvargus-daemon)\n"
            "  3) OpenCV가 GStreamer 지원으로 빌드됐는지\n"
            "  4) 단독 테스트: gst-launch-1.0 nvarguscamerasrc ! fakesink"
        )
    return cap


def grab_fresh_frame(cap, warmup=5):
    frame = None
    ret = False
    for _ in range(warmup):
        ret, frame = cap.read()
    return ret, frame


def main():
    save_dir = "captures"
    os.makedirs(save_dir, exist_ok=True)

    cap = open_camera()
    print("[INFO] 카메라 열림 (헤드리스 모드)")
    print("       Enter = 촬영,  q + Enter = 종료")

    try:
        while True:
            cmd = input("> ").strip().lower()
            if cmd == "q":
                break

            ret, frame = grab_fresh_frame(cap)
            if not ret or frame is None:
                print("[WARN] 프레임 읽기 실패")
                continue

            fname = os.path.join(
                save_dir, "nail_%s.png" % time.strftime("%Y%m%d_%H%M%S")
            )
            cv2.imwrite(fname, frame)
            print("[SAVE] %s  (%dx%d)" % (fname, frame.shape[1], frame.shape[0]))
    finally:
        cap.release()
        print("[INFO] 종료")


if __name__ == "__main__":
    main()