"""
manual_measure.py
==================
[1. 역할]
    charuco_calibration.py가 만든 호모그래피 JSON을 이용해서, 사용자가
    이미지에서 5개 손가락(엄지, 검지, 중지, 약지, 소지)의 "손톱 뿌리 ->
    손톱 끝" 두 점을 클릭하면, 그 두 점을 호모그래피로 mm 평면 좌표로
    변환한 뒤 실제 거리(mm)를 계산한다. 손가락이 보드 평면보다 살짝
    떠 있는 것을 보정하는 시차(parallax) 보정도 함께 적용한다.

    v1에서는 단일 스칼라 px_per_mm로 나눠서 mm를 구했지만, v2는
    호모그래피(cv2.perspectiveTransform)로 두 점을 각각 mm 평면에
    투영한 뒤 유클리드 거리를 구한다. 이 방식은 사진이 완벽히 수직으로
    찍히지 않아도 원근 왜곡을 어느 정도 보정해준다.

[2. 실행 명령어]
    python3 manual_measure.py --image data/captured/sample.jpg --calib data/results/sample_charuco.json
    python3 manual_measure.py --image data/captured/sample.jpg --calib data/results/sample_charuco.json --nail-height 3

[3. 어디에 입력해야 하는가]
    -> Jetson Nano 터미널(SSH 접속) 또는 노트북 터미널에서 실행한다.
    -> 마우스 클릭이 필요하므로 화면(GUI)이 있는 환경(SSH -X 또는 모니터 직결)에서 실행해야 한다.

[4. 정상적으로 실행되면]
    이미지 창이 뜨고, 화면 위쪽에 "[엄지] 손톱 뿌리를 클릭하세요" 안내가 나온다.
    뿌리를 클릭하면 "[엄지] 손톱 끝을 클릭하세요"로 바뀐다.
    끝점까지 클릭하면 자동으로 다음 손가락(검지)으로 넘어간다.
    5개 손가락을 모두 측정하면 아래처럼 표가 출력되고 CSV로 저장된다.

        [RESULT] 엄지(thumb): 157.92px -> 7.47mm
        [RESULT] 검지(index): 233.81px -> 11.07mm
        ...
        [SAVED] data/results/measurements.csv

[5. 오류가 발생하면 확인할 것]
    - "보정 파일을 찾을 수 없습니다": --calib 경로 확인, charuco_calibration.py를
      먼저 실행해서 JSON을 만들었는지 확인
    - "보정 파일에 homography 값이 없습니다": v1의 grid_calibration.py가 만든
      옛날 JSON(px_per_mm 방식)을 넣은 경우 발생한다. charuco_calibration.py로
      새로 보정해서 만든 JSON을 사용해야 한다.
    - 손톱 위치를 잘못 클릭했다: 각 손가락 측정 중 'r' 키로 마지막 점을 되돌릴 수 있고,
      전체를 취소하려면 'q'를 누른 뒤 프로그램을 다시 실행한다.
    - 길이 값이 이상하다: --calib JSON의 rms_reprojection_mm 값이 너무 크지
      않은지, --nail-height 값이 이 촬영 상황에 맞는지 확인한다.
"""

import argparse
import math
import os

import cv2

from utils import (
    collect_points,
    load_json,
    measure_length_mm,
    append_csv_rows,
    MEASUREMENTS_CSV_HEADER,
    FINGERS,
    FINGER_LABELS_KO,
)


def parse_args():
    parser = argparse.ArgumentParser(description="5개 손가락 손톱 길이 수동 측정 (ChArUco 호모그래피 기반)")
    parser.add_argument("--image", required=True, help="측정할 손톱 사진 경로")
    parser.add_argument(
        "--calib", required=True, help="charuco_calibration.py로 만든 보정 JSON 파일 경로 (homography 포함)"
    )
    parser.add_argument(
        "--output",
        default=os.path.join("data", "results", "measurements.csv"),
        help="측정 결과 CSV 저장 경로 (기본: data/results/measurements.csv)",
    )
    parser.add_argument(
        "--fingers",
        nargs="+",
        default=FINGERS,
        choices=FINGERS,
        help="측정할 손가락만 선택 (기본: 5개 전부)",
    )
    parser.add_argument(
        "--nail-height",
        type=float,
        default=0.0,
        help="보드 평면 ~ 손톱까지의 높이 (mm, 기본 0=보정 없음). 손가락이 보드 위로 뜬 정도.",
    )
    return parser.parse_args()


def measure_one_finger(image, finger_key):
    """한 손가락에 대해 뿌리/끝 두 점을 클릭받아 픽셀 좌표를 반환한다."""
    label_ko = FINGER_LABELS_KO[finger_key]
    labels = [
        f"[{label_ko}] 손톱 뿌리를 클릭하세요",
        f"[{label_ko}] 손톱 끝을 클릭하세요",
    ]
    points = collect_points(
        image, num_points=2, point_labels=labels, window_name=f"Measure - {label_ko}"
    )
    if points is None:
        return None
    return {"root": points[0], "tip": points[1]}


def main():
    args = parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"[ERROR] 이미지를 열 수 없습니다: {args.image}")
        return

    calib = load_json(args.calib)
    if calib is None:
        print(f"[ERROR] 보정 파일을 찾을 수 없습니다: {args.calib}")
        print("        먼저 charuco_calibration.py를 실행해서 보정 JSON을 만드세요.")
        return

    homography = calib.get("homography")
    if not homography:
        print(f"[ERROR] 보정 파일에 homography 값이 없습니다: {args.calib}")
        print("        v1(grid_calibration.py)의 옛 보정 파일이 아닌지 확인하세요.")
        print("        charuco_calibration.py로 만든 보정 JSON을 사용해야 합니다.")
        return

    camera_height_mm = calib.get("camera_height_mm", 295.0)
    rms_mm = calib.get("rms_reprojection_mm")

    print(f"[INFO] 보정 파일: {args.calib} (camera_height={camera_height_mm}mm, RMS={rms_mm}mm)")
    print(f"[INFO] 시차 보정: nail_height={args.nail_height}mm")
    print("[INFO] 각 손가락마다 '손톱 뿌리' -> '손톱 끝' 순서로 클릭하세요.")

    results = []
    for finger_key in args.fingers:
        label_ko = FINGER_LABELS_KO[finger_key]
        measured = measure_one_finger(image, finger_key)
        if measured is None:
            print(f"[INFO] '{label_ko}' 측정을 건너뛰었습니다(취소).")
            continue

        rx, ry = measured["root"]
        tx, ty = measured["tip"]
        pixel_distance = math.hypot(tx - rx, ty - ry)
        length_mm = measure_length_mm(
            homography, (rx, ry), (tx, ty), camera_height_mm=camera_height_mm, nail_height_mm=args.nail_height
        )

        print(f"[RESULT] {label_ko}({finger_key}): {pixel_distance:.2f}px -> {length_mm:.2f}mm")

        results.append(
            {
                "image_path": args.image,
                "finger": finger_key,
                "root_x": rx,
                "root_y": ry,
                "tip_x": tx,
                "tip_y": ty,
                "pixel_distance": round(pixel_distance, 4),
                "length_mm": round(length_mm, 4),
                "calibration_method": "charuco",
                "backend": "manual",
                "camera_height_mm": camera_height_mm,
                "nail_height_mm": args.nail_height,
                "homography_rms_mm": rms_mm,
            }
        )

    cv2.destroyAllWindows()

    if not results:
        print("[INFO] 저장할 측정 결과가 없습니다.")
        return

    append_csv_rows(args.output, results, MEASUREMENTS_CSV_HEADER)
    print(f"[SAVED] {args.output} ({len(results)}개 행 추가)")


if __name__ == "__main__":
    main()
