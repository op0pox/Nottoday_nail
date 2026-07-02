"""
manual_measure.py
==================
[1. 역할]
    grid_calibration.py에서 계산한 px_per_mm 값을 이용해서,
    사진 속 5개 손가락(엄지, 검지, 중지, 약지, 소지)의 손톱 길이를
    "뿌리 -> 끝" 두 점 클릭 방식으로 측정하고 mm 단위로 계산한다.
    결과는 화면에 출력되고 CSV 파일에 누적 저장된다.

[2. 실행 명령어]
    python3 manual_measure.py --image data/captured/sample.jpg --calib data/results/sample_calib.json
    python3 manual_measure.py --image data/captured/sample.jpg --calib data/results/sample_calib.json --output data/results/measurements.csv

[3. 어디에 입력해야 하는가]
    -> Jetson Nano 터미널(SSH 접속)에서 실행한다.
    -> 마우스 클릭이 필요하므로 화면(GUI)이 있는 환경(SSH -X 또는 모니터 직결)에서 실행해야 한다.

[4. 정상적으로 실행되면]
    이미지 창이 뜨고, 화면 위쪽에 "[엄지] 손톱 뿌리를 클릭하세요" 안내가 나온다.
    뿌리를 클릭하면 "[엄지] 손톱 끝을 클릭하세요"로 바뀐다.
    끝점까지 클릭하면 자동으로 다음 손가락(검지)으로 넘어간다.
    5개 손가락을 모두 측정하면 아래처럼 표가 출력되고 CSV로 저장된다.

        [RESULT] 엄지(thumb): 300.00px -> 6.25mm
        [RESULT] 검지(index): 330.00px -> 6.87mm
        ...
        [SAVED] data/results/measurements.csv

[5. 오류가 발생하면 확인할 것]
    - "보정 파일을 찾을 수 없습니다": --calib 경로 확인, grid_calibration.py를
      먼저 실행해서 JSON 파일을 만들었는지 확인
    - 손톱 위치를 잘못 클릭했다: 각 손가락 측정 중 'r' 키로 마지막 점을 되돌릴 수 있고,
      전체를 취소하려면 'q'를 누른 뒤 프로그램을 다시 실행한다.
    - 길이 값이 너무 크거나 작게 나온다: px_per_mm 값(보정 결과)이
      이 사진에 맞는 값인지, 카메라와 모눈판 사이 거리가 촬영 내내
      동일했는지 확인한다.
"""

import argparse
import csv
import math
import os

import cv2

from utils import (
    collect_points,
    ensure_dir,
    load_json,
    FINGERS,
    FINGER_LABELS_KO,
)

CSV_HEADER = [
    "image_path",
    "finger",
    "root_x",
    "root_y",
    "tip_x",
    "tip_y",
    "pixel_distance",
    "length_mm",
]


def parse_args():
    parser = argparse.ArgumentParser(description="5개 손가락 손톱 길이 수동 측정")
    parser.add_argument("--image", required=True, help="측정할 손톱 사진 경로")
    parser.add_argument(
        "--calib", required=True, help="grid_calibration.py로 만든 보정 JSON 파일 경로"
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
    return parser.parse_args()


def measure_one_finger(image, finger_key):
    """한 손가락에 대해 뿌리/끝 두 점을 클릭받아 픽셀 거리를 계산한다."""
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

    (rx, ry), (tx, ty) = points
    pixel_distance = math.hypot(tx - rx, ty - ry)
    return {
        "root": (rx, ry),
        "tip": (tx, ty),
        "pixel_distance": pixel_distance,
    }


def append_rows_to_csv(csv_path, rows):
    ensure_dir(csv_path)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"[ERROR] 이미지를 열 수 없습니다: {args.image}")
        return

    calib = load_json(args.calib)
    if calib is None:
        print(f"[ERROR] 보정 파일을 찾을 수 없습니다: {args.calib}")
        print("        먼저 grid_calibration.py를 실행해서 보정 JSON을 만드세요.")
        return

    px_per_mm = calib.get("px_per_mm")
    if not px_per_mm:
        print(f"[ERROR] 보정 파일에 px_per_mm 값이 없습니다: {args.calib}")
        return

    print(f"[INFO] px_per_mm = {px_per_mm} (보정 파일: {args.calib})")
    print("[INFO] 각 손가락마다 '손톱 뿌리' -> '손톱 끝' 순서로 클릭하세요.")

    results = []
    for finger_key in args.fingers:
        label_ko = FINGER_LABELS_KO[finger_key]
        measured = measure_one_finger(image, finger_key)
        if measured is None:
            print(f"[INFO] '{label_ko}' 측정을 건너뛰었습니다(취소).")
            continue

        pixel_distance = measured["pixel_distance"]
        length_mm = pixel_distance / px_per_mm
        rx, ry = measured["root"]
        tx, ty = measured["tip"]

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
            }
        )

    cv2.destroyAllWindows()

    if not results:
        print("[INFO] 저장할 측정 결과가 없습니다.")
        return

    append_rows_to_csv(args.output, results)
    print(f"[SAVED] {args.output} ({len(results)}개 행 추가)")


if __name__ == "__main__":
    main()
