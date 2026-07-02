"""
capture_image.py
=================
[1. 역할]
    카메라 실시간 화면을 띄우고, 's' 키를 누르면 현재 화면을 사진으로
    저장하고, 'q' 키를 누르면 프로그램을 종료한다.
    저장된 사진은 손톱 길이 측정(grid_calibration.py, manual_measure.py)의
    입력 이미지로 사용된다.

[2. 실행 명령어]
    python3 capture_image.py                          # CSI 카메라 사용
    python3 capture_image.py --camera usb              # USB 카메라 사용
    python3 capture_image.py --camera usb --device 1
    python3 capture_image.py --output-dir data/captured --prefix sample

[3. 어디에 입력해야 하는가]
    -> Jetson Nano 터미널(SSH 접속 상태)에서 실행한다.
    -> 화면 창이 필요하므로 SSH -X(X11 포워딩) 접속이거나,
       Jetson Nano에 모니터를 직접 연결한 상태여야 한다.
       (화면이 아예 없는 환경이면 --no-preview 옵션으로 카운트다운 촬영도 가능)

[4. 정상적으로 실행되면]
    "Capture" 라는 제목의 창에 카메라 실시간 화면이 보인다.
    화면 상단에 "s: 저장 | q: 종료" 안내 문구가 함께 보인다.
    's'를 누르면 "[SAVED] data/captured/capture_20260702_193045.jpg" 처럼
    저장 경로가 터미널에 출력되고, 화면에는 잠깐 "Saved!" 문구가 표시된다.
    'q'를 누르면 창이 닫히고 프로그램이 종료된다.

[5. 오류가 발생하면 확인할 것]
    - 창이 아예 안 뜬다면: camera_test.py를 먼저 실행해서 카메라/화면
      환경이 정상인지 확인한다.
    - "카메라 열기 실패": README의 '카메라 연결' 항목과
      camera_test.py의 오류 해결 방법을 참고한다.
    - 저장은 되는데 사진이 이상하면: 조명, 초점, 렌즈 보호필름을 확인한다.
"""

import argparse
import os

import cv2

from utils import open_camera, check_gui_available, ensure_dir, timestamp_filename


def parse_args():
    parser = argparse.ArgumentParser(description="손톱 촬영용 카메라 캡처 프로그램")
    parser.add_argument(
        "--camera",
        choices=["csi", "usb"],
        default="csi",
        help="카메라 종류: csi(기본) 또는 usb",
    )
    parser.add_argument("--device", type=int, default=0, help="USB 카메라 index (기본 0)")
    parser.add_argument("--width", type=int, default=1920, help="캡처 가로 해상도")
    parser.add_argument("--height", type=int, default=1080, help="캡처 세로 해상도")
    parser.add_argument(
        "--flip-method", type=int, default=0, help="CSI 카메라 화면 회전 (0=그대로, 2=180도)"
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("data", "captured"),
        help="사진이 저장될 폴더 (기본: data/captured)",
    )
    parser.add_argument(
        "--prefix", default="capture", help="저장 파일명 접두어 (기본: capture)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    if not check_gui_available():
        print(
            "[ERROR] 이 스크립트는 화면(GUI) 미리보기가 필요합니다.\n"
            "        SSH -X 옵션으로 재접속하거나 Jetson Nano에 모니터를 연결하세요."
        )
        return

    print(f"[INFO] 카메라를 여는 중... (camera={args.camera})")
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

    print("[OK] 카메라 연결 완료. 's' = 저장, 'q' = 종료")

    window_name = "Capture"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    saved_count = 0
    flash_counter = 0
    last_saved_path = ""

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] 프레임을 읽지 못했습니다. 카메라 연결을 확인하세요.")
                continue

            display = frame.copy()
            cv2.putText(
                display,
                "s: save | q: quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            if flash_counter > 0:
                cv2.putText(
                    display,
                    "Saved!",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                )
                flash_counter -= 1

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                filename = timestamp_filename(prefix=args.prefix, ext="jpg")
                save_path = os.path.join(args.output_dir, filename)
                cv2.imwrite(save_path, frame)
                saved_count += 1
                flash_counter = 15
                last_saved_path = save_path
                print(f"[SAVED] {save_path}")
            elif key == ord("q"):
                print("[INFO] 종료합니다.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C 로 종료했습니다.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[INFO] 총 {saved_count}장 저장했습니다.")
        if last_saved_path:
            print(f"[INFO] 마지막 저장 파일: {last_saved_path}")


if __name__ == "__main__":
    main()
