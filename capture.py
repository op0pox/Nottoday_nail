#!/usr/bin/env python3
"""
capture.py — Jetson Nano CSI 카메라 정지 이미지 캡처 CLI
=========================================================
[역할]
    Jetson Nano에 연결된 CSI 카메라로 사진을 찍어 captures/ 폴더에
    PNG 파일로 저장하는 간단한 명령줄 도구.
    화면(GUI) 없이 터미널만으로 동작한다 (헤드리스 모드).

[사용법]
    python3 capture.py
    > (Enter만 누르면)  사진 촬영 + 저장
    > q (입력 후 Enter) 종료

[동작 원리]
    Jetson Nano의 CSI 카메라는 일반 웹캠과 달리 cv2.VideoCapture(0)로
    바로 열 수 없다. NVIDIA 전용 GStreamer 파이프라인 문자열을 만들어
    OpenCV에 넘겨야 한다 (아래 gstreamer_pipeline 함수).
"""
import os      # 폴더 생성
import time    # 파일명에 넣을 현재 시각
import cv2     # OpenCV — 카메라 열기/프레임 읽기/저장


def gstreamer_pipeline(sensor_id=0, capture_width=3264, capture_height=2464,
                       framerate=21, flip_method=0):
    """
    CSI 카메라를 열기 위한 GStreamer 파이프라인 문자열을 만든다.

    파이프라인이란? "카메라 → 포맷 변환 → OpenCV로 전달"까지의
    처리 단계를 '!' 로 이어 쓴 명령 문자열이다. 각 단계 의미:
      nvarguscamerasrc : NVIDIA CSI 카메라 입력 (Jetson 전용)
      nvvidconv        : GPU로 이미지 포맷/회전 변환
      videoconvert     : CPU에서 최종 BGR(OpenCV 형식)로 변환
      appsink          : 결과를 응용 프로그램(여기서는 OpenCV)에 전달
      drop=true max-buffers=1 : 오래된 프레임은 버리고 최신 것만 유지

    매개변수:
      sensor_id     : 카메라 번호 (CSI 포트가 2개인 보드에서 0 또는 1)
      capture_width/height : 촬영 해상도 (3264x2464 = IMX219 센서 최대)
      framerate     : 초당 프레임 수
      flip_method   : 화면 회전 (0=그대로, 2=180도 — 거꾸로 장착 시 사용)
    """
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
    """
    카메라를 열어서 cv2.VideoCapture 객체를 반환한다.
    실패하면 확인할 사항을 담은 RuntimeError를 던진다.
    """
    # CAP_GSTREAMER = "이 문자열은 GStreamer 파이프라인이다"라고 OpenCV에 알림
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():     # 열기 실패 (케이블/점유/빌드 문제 등)
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
    """
    최신 프레임 한 장을 얻는다.

    왜 여러 번(warmup) 읽나? 카메라 버퍼에 이전 프레임이 남아있을 수
    있어서, 몇 장을 버리고 마지막 것을 써야 "지금 이 순간" 화면이 나온다.
    (자동 노출/화이트밸런스가 안정되는 효과도 있다)

    반환값: (성공 여부 bool, 프레임 numpy 배열)
    """
    frame = None
    ret = False
    for _ in range(warmup):        # warmup번 반복해서 읽고
        ret, frame = cap.read()    # 마지막으로 읽은 프레임만 남긴다
    return ret, frame


def main():
    # 저장 폴더 준비 (이미 있으면 그대로 사용)
    save_dir = "captures"
    os.makedirs(save_dir, exist_ok=True)

    cap = open_camera()
    print("[INFO] 카메라 열림 (헤드리스 모드)")
    print("       Enter = 촬영,  q + Enter = 종료")

    # try...finally: 중간에 오류가 나거나 q로 나가도
    # finally 블록(카메라 해제)은 반드시 실행된다
    try:
        while True:                              # 무한 반복 (q 입력 시 탈출)
            cmd = input("> ").strip().lower()    # 사용자 입력 대기
            if cmd == "q":
                break

            ret, frame = grab_fresh_frame(cap)
            if not ret or frame is None:         # 프레임을 못 읽은 경우
                print("[WARN] 프레임 읽기 실패")
                continue                         # 다시 입력 대기로

            # 파일명에 현재 시각을 넣어 중복을 방지 (예: nail_20260807_223512.png)
            fname = os.path.join(
                save_dir, "nail_%s.png" % time.strftime("%Y%m%d_%H%M%S")
            )
            cv2.imwrite(fname, frame)            # numpy 배열 -> PNG 파일
            print("[SAVE] %s  (%dx%d)" % (fname, frame.shape[1], frame.shape[0]))
    finally:
        cap.release()                            # 카메라 점유 해제 (중요!)
        print("[INFO] 종료")


if __name__ == "__main__":     # 직접 실행했을 때만 main() 호출
    main()
