"""
height_experiment.py
=====================
[1. 역할]
    카메라 높이(10cm, 15cm, 20cm...)를 바꿔가며 촬영할 때마다 촬영 조건
    (카메라 모델, 높이, 해상도, 조명, 초점 방식, 촬영 날짜, 비고)을 기록하고,
    나중에 compare_actual.py가 만든 오차(comparison.csv)와 연결해서
    "어느 높이에서 가장 오차가 적었는지" 평균을 계산해준다.

    이 스크립트는 두 가지 모드로 동작한다.
      log     : 사진 한 장에 대한 촬영 조건을 기록한다.
      summary : 지금까지 기록된 모든 사진의 높이별 평균 오차를 계산해서 보여준다.

[2. 실행 명령어]
    # (1) 사진을 찍을 때마다 촬영 조건 기록
    python3 height_experiment.py log --image data/captured/sample.jpg \\
        --camera-model "Arducam 16MP AF" --height 20 \\
        --lighting "책상 스탠드" --focus "auto" --note "3차 시도"

    # (2) 여러 번 기록/측정/비교가 끝난 뒤 높이별 평균 오차 요약
    python3 height_experiment.py summary
    python3 height_experiment.py summary --camera-model "Arducam 16MP AF"

[3. 어디에 입력해야 하는가]
    -> Jetson Nano 터미널(SSH 접속)에서 실행한다. (화면/GUI 필요 없음)

[4. 정상적으로 실행되면]
    log 모드:
        [SAVED] data/results/experiment_log.csv 에 촬영 조건 1줄 추가됨

    summary 모드: (compare_actual.py로 만든 comparison.csv와 연결해서 계산)
        카메라 모델: Arducam 16MP AF

        10cm 평균 오차: 1.20mm (표본 5개)
        15cm 평균 오차: 0.82mm (표본 5개)
        20cm 평균 오차: 0.45mm (표본 5개)
        25cm 평균 오차: 0.70mm (표본 5개)
        30cm 평균 오차: 1.10mm (표본 5개)

        최적 높이: 20cm (평균 오차 0.45mm)

[5. 오류가 발생하면 확인할 것]
    - summary에서 "비교 데이터가 없습니다"가 나오면:
      compare_actual.py를 먼저 실행해서 comparison.csv를 만들었는지 확인
    - 특정 이미지의 높이가 요약에 안 잡히면:
      해당 이미지로 log 모드를 먼저 실행해서 높이를 기록했는지,
      image_path가 measurements.csv/comparison.csv와 정확히 같은
      경로 문자열인지 확인 (상대경로/절대경로 섞이면 안 됨)
"""

import argparse
import csv
import os
from collections import defaultdict
from datetime import datetime

import cv2

from utils import ensure_dir

LOG_CSV_DEFAULT = os.path.join("data", "results", "experiment_log.csv")
COMPARISON_CSV_DEFAULT = os.path.join("data", "results", "comparison.csv")

LOG_HEADER = [
    "image_path",
    "camera_model",
    "height_cm",
    "resolution",
    "lighting",
    "focus_mode",
    "capture_date",
    "note",
]


def parse_args():
    parser = argparse.ArgumentParser(description="카메라 높이별 실험 기록 및 오차 요약")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    log_parser = subparsers.add_parser("log", help="촬영 조건 1건 기록")
    log_parser.add_argument("--image", required=True, help="촬영된 이미지 경로")
    log_parser.add_argument("--camera-model", required=True, help='예: "Arducam 16MP AF"')
    log_parser.add_argument("--height", type=float, required=True, help="카메라 높이(cm)")
    log_parser.add_argument(
        "--resolution", default=None, help="예: 1920x1080 (생략하면 이미지에서 자동 계산)"
    )
    log_parser.add_argument("--lighting", default="", help="조명 조건 (예: 책상 스탠드)")
    log_parser.add_argument("--focus", default="", help="초점 방식 (예: auto, fixed)")
    log_parser.add_argument(
        "--date", default=None, help="촬영 날짜 YYYY-MM-DD (생략하면 오늘 날짜)"
    )
    log_parser.add_argument("--note", default="", help="비고")
    log_parser.add_argument("--log-csv", default=LOG_CSV_DEFAULT, help="기록 CSV 경로")

    summary_parser = subparsers.add_parser("summary", help="높이별 평균 오차 요약")
    summary_parser.add_argument("--log-csv", default=LOG_CSV_DEFAULT, help="기록 CSV 경로")
    summary_parser.add_argument(
        "--comparison-csv", default=COMPARISON_CSV_DEFAULT, help="compare_actual.py 결과 CSV"
    )
    summary_parser.add_argument(
        "--camera-model", default=None, help="특정 카메라 모델만 필터링해서 요약"
    )

    return parser.parse_args()


def guess_resolution(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "unknown"
    h, w = img.shape[:2]
    return f"{w}x{h}"


def append_row_to_csv(csv_path, row):
    ensure_dir(csv_path)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def do_log(args):
    resolution = args.resolution or guess_resolution(args.image)
    capture_date = args.date or datetime.now().strftime("%Y-%m-%d")

    row = {
        "image_path": args.image,
        "camera_model": args.camera_model,
        "height_cm": args.height,
        "resolution": resolution,
        "lighting": args.lighting,
        "focus_mode": args.focus,
        "capture_date": capture_date,
        "note": args.note,
    }
    append_row_to_csv(args.log_csv, row)
    print(f"[SAVED] {args.log_csv} 에 기록 추가됨")
    print(f"        image={args.image}, height={args.height}cm, "
          f"camera={args.camera_model}, resolution={resolution}")


def load_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def do_summary(args):
    log_rows = load_csv_rows(args.log_csv)
    comparison_rows = load_csv_rows(args.comparison_csv)

    if not log_rows:
        print(f"[ERROR] 실험 기록이 없습니다: {args.log_csv}")
        print("        height_experiment.py log 명령으로 먼저 촬영 조건을 기록하세요.")
        return
    if not comparison_rows:
        print(f"[ERROR] 비교 데이터가 없습니다: {args.comparison_csv}")
        print("        compare_actual.py를 먼저 실행해서 오차 데이터를 만드세요.")
        return

    # image_path -> (height_cm, camera_model)
    image_info = {}
    for row in log_rows:
        if args.camera_model and row["camera_model"] != args.camera_model:
            continue
        image_info[row["image_path"]] = {
            "height_cm": row["height_cm"],
            "camera_model": row["camera_model"],
        }

    if not image_info:
        print(f"[ERROR] '{args.camera_model}' 모델로 기록된 실험이 없습니다.")
        return

    # height_cm -> [error_mm, error_mm, ...]
    errors_by_height = defaultdict(list)
    matched_camera_models = set()

    for row in comparison_rows:
        info = image_info.get(row["image_path"])
        if info is None:
            continue
        height = info["height_cm"]
        errors_by_height[height].append(float(row["error_mm"]))
        matched_camera_models.add(info["camera_model"])

    if not errors_by_height:
        print("[ERROR] 기록된 실험 정보와 일치하는 비교 데이터가 없습니다.")
        print("        image_path가 log/measurements/comparison에서 서로 동일한지 확인하세요.")
        return

    camera_label = (
        args.camera_model
        if args.camera_model
        else ", ".join(sorted(matched_camera_models))
    )
    print(f"카메라 모델: {camera_label}\n")

    def height_sort_key(h):
        try:
            return float(h)
        except ValueError:
            return float("inf")

    best_height = None
    best_avg = None
    for height in sorted(errors_by_height.keys(), key=height_sort_key):
        errs = errors_by_height[height]
        avg = sum(errs) / len(errs)
        print(f"{height}cm 평균 오차: {avg:.2f}mm (표본 {len(errs)}개)")
        if best_avg is None or avg < best_avg:
            best_avg = avg
            best_height = height

    print(f"\n최적 높이: {best_height}cm (평균 오차 {best_avg:.2f}mm)")


def main():
    args = parse_args()
    if args.mode == "log":
        do_log(args)
    elif args.mode == "summary":
        do_summary(args)


if __name__ == "__main__":
    main()
