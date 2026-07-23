"""
camera_test.py
==============
[1. 역할]
    Jetson Nano에 연결한 카메라(CSI 또는 USB)가 정상적으로 열리고
    영상을 받아올 수 있는지 확인하는 가장 간단한 테스트 스크립트이다.
    화면(GUI)이 있으면 실시간 미리보기 창을 띄우고,
    화면이 없는 순수 SSH 환경이면 프레임을 못 읽었는지/읽었는지만
    글자로 알려준다.

[2. 실행 명령어]
    python3 camera_test.py                     # CSI 카메라 기본 테스트
    python3 camera_test.py --camera usb        # USB 카메라 테스트
    python3 camera_test.py --camera usb --device 1   # USB 카메라 index 1번
    python3 camera_test.py --width 1280 --height 720

[3. 어디에 입력해야 하는가]
    -> Jetson Nano 터미널(SSH 접속 터미널)에서 실행한다.
    -> 내 노트북 터미널이 아니라 "ssh 사용자명@Jetson_IP"로 접속한 뒤의
       화면(프롬프트가 jetson@jetson-nano:~$ 처럼 바뀐 상태)에서 실행해야 한다.

[4. 정상적으로 실행되면]
    - GUI 창을 띄울 수 있는 환경(SSH -X 접속 또는 모니터 직접 연결)이면:
        "Camera Test" 라는 제목의 창이 뜨고 카메라 실시간 화면이 보인다.
        창을 선택한 상태에서 'q'를 누르면 종료된다.
    - GUI 창이 없는 순수 SSH 환경이면:
        화면 대신 "[OK] 프레임 읽기 성공 (프레임 크기: 1280x720)" 같은
        메시지가 터미널에 여러 번 출력된다. Ctrl+C로 종료한다.

[5. 오류가 발생하면 확인할 것]
    - "카메라 열기 실패" 메시지가 뜨면:
        1) CSI 케이블이 제대로 꽂혀 있는지 (파란 면 방향 확인, README 참고)
        2) `ls /dev/video*` 로 카메라 장치가 보이는지
        3) USB 카메라라면 --camera usb --device 번호가 맞는지
        4) 다른 프로그램이 카메라를 이미 쓰고 있지 않은지
    - 창은 뜨는데 새까맣게 나오면:
        1) 렌즈에 보호 필름이 붙어있지 않은지
        2) 조명이 너무 어둡지 않은지
        3) Auto Focus 카메라는 초점이 안 맞아 뿌옇게 나올 수 있음(정상, 이후 조정)
"""

import argparse
import time

import cv2

from utils import open_camera, check_gui_available


def parse_args():
    parser = argparse.ArgumentParser(description="Jetson Nano 카메라 연결 테스트")
    parser.add_argument(
        "--camera",
        choices=["csi", "usb"],
        default="csi",
        help="카메라 종류: csi(기본, CSI 리본케이블 카메라) 또는 usb",
    )
    parser.add_argument("--device", type=int, default=0, help="USB 카메라 index (기본 0)")
    parser.add_argument("--width", type=int, default=1280, help="캡처 가로 해상도")
    parser.add_argument("--height", type=int, default=720, help="캡처 세로 해상도")
    parser.add_argument(
        "--flip-method",
        type=int,
        default=0,
        help="CSI 카메라 화면이 뒤집혀 보일 때 조정 (0=그대로, 2=180도 회전)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[INFO] 카메라를 여는 중... (camera={args.camera}, "
          f"width={args.width}, height={args.height})")

    try:
        cap = open_camera(
            camera_type=args.camera,
            device=args.device,
            width=args.width,
            height=args.height,
            flip_method=args.flip_method,
        )
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return

    print("[OK] 카메라가 정상적으로 열렸습니다.")

    gui_ok = check_gui_available()
    window_name = "Camera Test"
    if gui_ok:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    fail_count = 0
    start_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                fail_count += 1
                print(f"[WARN] 프레임 읽기 실패 ({fail_count}회 연속 실패 가능)")
                if fail_count > 30:
                    print("[ERROR] 프레임을 계속 읽지 못합니다. 카메라 연결을 확인하세요.")
                    break
                time.sleep(0.1)
                continue

            fail_count = 0
            frame_count += 1

            if gui_ok:
                cv2.putText(
                    frame,
                    "Camera OK - press 'q' to quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[INFO] 사용자가 'q'를 눌러 종료했습니다.")
                    break
            else:
                # 화면이 없는 환경: 5프레임마다 한 번씩 상태만 출력
                if frame_count % 5 == 0:
                    h, w = frame.shape[:2]
                    print(f"[OK] 프레임 읽기 성공 (프레임 크기: {w}x{h}, 누적 {frame_count} 프레임)")
                time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C 로 종료했습니다.")
    finally:
        elapsed = time.time() - start_time
        cap.release()
        if gui_ok:
            cv2.destroyAllWindows()
        print(f"[INFO] 테스트 종료. 총 {frame_count} 프레임, {elapsed:.1f}초 동안 확인함.")


if __name__ == "__main__":
    main()
