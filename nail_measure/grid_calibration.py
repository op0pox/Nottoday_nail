"""
grid_calibration.py
====================
[1. 역할]
    모눈판이 함께 촬영된 사진을 열어서, 사용자가 모눈 눈금의
    시작점과 끝점을 클릭하면 그 사이의 실제 칸 수(cells)를 이용해
    "픽셀 1개가 실제로 몇 mm인지(px_per_mm)"를 계산한다.
    이 값은 이후 manual_measure.py에서 손톱 길이를 mm로 변환할 때 쓰인다.

    주의: 모눈 한 칸의 실제 크기는 --cell-size-mm 으로 지정한다.
    기본값은 5mm(0.5cm)로 되어 있다. 다른 모눈판(예: 진짜 0.5mm 간격)을
    쓴다면 반드시 --cell-size-mm 값을 그 모눈판에 맞게 바꿔야 한다.
    이 값이 실제와 다르면 px_per_mm이 통째로 몇 배씩 틀어지고,
    그 결과 손톱 길이도 똑같은 배수로 틀어진다.

[2. 실행 명령어]
    python3 grid_calibration.py --image data/captured/capture_20260702_193045.jpg
    python3 grid_calibration.py --image data/captured/sample.jpg --cells 10
    python3 grid_calibration.py --image data/captured/sample.jpg --cell-size-mm 0.5   # 0.5mm 모눈판을 쓸 경우
    python3 grid_calibration.py --image data/captured/sample.jpg --output data/results/sample_calib.json

[3. 어디에 입력해야 하는가]
    -> Jetson Nano 터미널(SSH 접속)에서 실행한다.
    -> 이미지 위에서 마우스로 점을 클릭해야 하므로 화면(GUI)이 필요하다.
       (SSH -X 접속 또는 모니터 직접 연결 상태여야 한다)

[4. 정상적으로 실행되면]
    이미지가 창에 뜨고, 화면 위쪽에 "모눈 시작점을 클릭하세요" 안내가 보인다.
    시작점을 클릭하면 빨간 점이 찍히고, 이어서 "모눈 끝점을 클릭하세요"로 바뀐다.
    끝점까지 클릭한 뒤 아무 키나 누르면 창이 닫히고,
    터미널에 아래처럼 결과가 출력된다.

        [OK] 픽셀 거리: 240.00 px
        [OK] 실제 거리: 50.0 mm (모눈 10칸 x 5.0mm)
        [OK] px_per_mm = 4.8000
        [SAVED] data/results/sample_calib.json

[5. 오류가 발생하면 확인할 것]
    - "이미지를 열 수 없습니다": --image 경로가 올바른지, 파일이 실제로
      존재하는지 확인 (ls data/captured/ 로 파일명 확인)
    - 점을 두 번 다 못 찍고 끝났다: 실수로 'q'를 눌러 취소된 것일 수 있음.
      다시 실행해서 순서대로 두 점만 클릭
    - px_per_mm 값이 이상하게 크거나 작다(예: 실제보다 몇 배씩 차이):
      1) --cells 값이 실제로 클릭한 모눈 칸 수와 맞는지 확인
      2) --cell-size-mm 값이 실제 모눈판의 한 칸 크기와 맞는지 확인
         (이 프로젝트 기본 모눈판은 한 칸 5mm=0.5cm 이다)
"""

import argparse
import math
import os

import cv2

from utils import collect_points, ensure_dir, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="모눈판 기준 px_per_mm 계산")
    parser.add_argument("--image", required=True, help="모눈판이 촬영된 이미지 경로")
    parser.add_argument(
        "--cells",
        type=int,
        default=10,
        help="클릭한 시작점과 끝점 사이의 모눈 칸 수 (기본 10칸)",
    )
    parser.add_argument(
        "--cell-size-mm",
        type=float,
        default=5.0,
        help="모눈 한 칸의 실제 크기 (mm, 기본 5.0mm=0.5cm). "
        "0.5mm 간격 모눈판을 쓴다면 --cell-size-mm 0.5 로 지정할 것",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="결과 JSON 저장 경로 (기본: data/results/<이미지이름>_calib.json)",
    )
    return parser.parse_args()


def default_output_path(image_path):
    base = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join("data", "results", f"{base}_calib.json")


def main():
    args = parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"[ERROR] 이미지를 열 수 없습니다: {args.image}")
        print("        경로가 올바른지, 파일이 존재하는지 확인하세요.")
        return

    real_distance_mm = args.cells * args.cell_size_mm
    print(f"[INFO] 모눈 {args.cells}칸 = {real_distance_mm}mm 기준으로 보정합니다.")

    labels = [
        f"모눈 시작점을 클릭하세요 (여기서부터 {args.cells}칸)",
        f"모눈 끝점을 클릭하세요 ({args.cells}칸 뒤 지점)",
    ]
    points = collect_points(image, num_points=2, point_labels=labels, window_name="Grid Calibration")

    if points is None:
        print("[INFO] 사용자가 취소했습니다. (px_per_mm 계산 안 함)")
        return

    (x1, y1), (x2, y2) = points
    pixel_distance = math.hypot(x2 - x1, y2 - y1)

    if pixel_distance == 0:
        print("[ERROR] 두 점이 같은 위치입니다. 서로 다른 두 점을 클릭해야 합니다.")
        return

    px_per_mm = pixel_distance / real_distance_mm

    print(f"[OK] 픽셀 거리: {pixel_distance:.2f} px")
    print(f"[OK] 실제 거리: {real_distance_mm} mm (모눈 {args.cells}칸 x {args.cell_size_mm}mm)")
    print(f"[OK] px_per_mm = {px_per_mm:.4f}")

    output_path = args.output or default_output_path(args.image)
    result = {
        "image_path": args.image,
        "grid_cells": args.cells,
        "cell_size_mm": args.cell_size_mm,
        "real_distance_mm": real_distance_mm,
        "pixel_distance": round(pixel_distance, 4),
        "px_per_mm": round(px_per_mm, 4),
        "point_start": [x1, y1],
        "point_end": [x2, y2],
    }
    ensure_dir(output_path)
    save_json(output_path, result)
    print(f"[SAVED] {output_path}")


if __name__ == "__main__":
    main()
