"""
compare_curve.py
=================
[1. 역할]
    measure_curve.py로 계산한 곡면 길이(curve_length_mm)와, 줄자/유연자로
    직접 잰 실제 곡면 너비(mm)를 비교해서 오차(mm)·오차율(%)을 계산한다.
    compare_actual.py(손톱 길이용)와 같은 패턴이며, 대상만 곡면 길이로
    바뀐 것이다. 같은 사진 조합(front_image+finger)을 여러 유형(P/S/B/C)으로
    재측정했다면 --type으로 특정 유형만 골라 비교할 수 있다.

[2. 실행 명령어]
    # 대화형으로 실제 값을 입력하는 방식
    python3 compare_curve.py --front data/captured/front_thumb.jpg --finger thumb

    # 실제 값을 명령줄에서 바로 넘기는 방식 (자동화/반복 실험용)
    python3 compare_curve.py --front data/captured/front_thumb.jpg --finger thumb --actual 21.5

    # 같은 사진을 여러 유형으로 재측정했을 때 특정 유형만 비교
    python3 compare_curve.py --front data/captured/front_thumb.jpg --finger thumb --type P --actual 21.5

[3. 어디에 입력해야 하는가]
    -> Jetson Nano 터미널 또는 노트북 터미널, 화면이 없어도 실행 가능하다
       (키보드 입력만 필요, 마우스 클릭 없음).

[4. 정상적으로 실행되면]
    대화형 모드에서는 아래처럼 실제 곡면 너비를 물어본 뒤, 오차를 계산해서
    출력하고 CSV로 저장한다.

        실제 곡면 너비(mm): 21.5

        [RESULT] thumb (P형): 실제 21.50mm / 측정 20.78mm / 오차 0.72mm (3.3%)
        [SAVED] data/results/curve_comparison.csv

[5. 오류가 발생하면 확인할 것]
    - "측정 데이터가 없습니다": measure_curve.py를 먼저 실행해서 해당
      front_image + finger 조합의 곡면 길이를 curve_measurements.csv에
      저장했는지, --front 경로 문자열이 그 CSV에 저장된 값과 정확히
      같은지 확인.
    - 같은 조합을 여러 유형으로 측정해서 원하는 값이 아닌 게 비교됐다:
      --type P/S/B/C로 유형을 지정해서 좁혀본다 (지정 안 하면 가장 최근
      측정값을 사용한다).
    - 숫자가 아닌 값을 입력해서 오류가 나면: 숫자(예: 21.5)만 입력.
"""

import argparse
import csv
import os
from datetime import datetime

from utils import append_csv_rows, FINGERS, FINGER_LABELS_KO
from curve_config import NAIL_TYPES

MEASURED_CSV_DEFAULT = os.path.join("data", "results", "curve_measurements.csv")
COMPARISON_CSV_DEFAULT = os.path.join("data", "results", "curve_comparison.csv")

CSV_HEADER = [
    "timestamp",
    "front_image",
    "side_image",
    "finger",
    "nail_type",
    "actual_curve_mm",
    "measured_curve_mm",
    "error_mm",
    "error_percent",
    "camera_height_cm",
]


def parse_args():
    parser = argparse.ArgumentParser(description="실제 곡면 너비(줄자 등)와 measure_curve.py 측정값 비교")
    parser.add_argument("--front", required=True, help="measure_curve.py에 넣었던 정면 사진 경로 (front_image 매칭용)")
    parser.add_argument("--finger", required=True, choices=FINGERS, help="비교할 손가락")
    parser.add_argument("--type", dest="nail_type", default=None, choices=NAIL_TYPES, help="특정 유형만 비교 (기본: 가장 최근 측정값)")
    parser.add_argument(
        "--measured-csv",
        default=MEASURED_CSV_DEFAULT,
        help=f"measure_curve.py 결과 CSV (기본: {MEASURED_CSV_DEFAULT})",
    )
    parser.add_argument(
        "--output",
        default=COMPARISON_CSV_DEFAULT,
        help=f"비교 결과 CSV 저장 경로 (기본: {COMPARISON_CSV_DEFAULT})",
    )
    parser.add_argument("--actual", type=float, default=None, help="실제 곡면 너비(mm)를 바로 지정 (생략하면 대화형 입력)")
    return parser.parse_args()


def load_latest_curve_measurement(csv_path, front_image, finger, nail_type=None):
    """
    curve_measurements.csv에서 front_image + finger(+ nail_type)가 일치하는
    행 중 가장 마지막(최근) 값을 반환한다. 없으면 None.
    """
    if not os.path.exists(csv_path):
        return None

    latest = None
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("front_image") != front_image or row.get("finger") != finger:
                continue
            if nail_type and row.get("nail_type") != nail_type:
                continue
            latest = row
    return latest


def prompt_actual_value():
    while True:
        raw = input("실제 곡면 너비(mm): ").strip()
        try:
            return float(raw)
        except ValueError:
            print("  -> 숫자로 입력해주세요. 예: 21.5")


def main():
    args = parse_args()
    label_ko = FINGER_LABELS_KO[args.finger]

    row = load_latest_curve_measurement(args.measured_csv, args.front, args.finger, args.nail_type)
    if row is None:
        print(f"[ERROR] '{args.front}' / '{label_ko}'에 대한 곡면 측정 데이터가 없습니다.")
        print(f"        먼저 measure_curve.py로 측정 결과를 {args.measured_csv} 에 저장하세요.")
        return

    actual_mm = args.actual if args.actual is not None else prompt_actual_value()
    measured_mm = float(row["curve_length_mm"])
    error_mm = abs(actual_mm - measured_mm)
    error_percent = (error_mm / actual_mm * 100) if actual_mm != 0 else 0.0

    nail_type = row.get("nail_type", "")
    print(
        f"\n[RESULT] {args.finger} ({nail_type}형): 실제 {actual_mm:.2f}mm / 측정 {measured_mm:.2f}mm / "
        f"오차 {error_mm:.2f}mm ({error_percent:.1f}%)"
    )

    out_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "front_image": args.front,
        "side_image": row.get("side_image", ""),
        "finger": args.finger,
        "nail_type": nail_type,
        "actual_curve_mm": actual_mm,
        "measured_curve_mm": measured_mm,
        "error_mm": round(error_mm, 4),
        "error_percent": round(error_percent, 2),
        "camera_height_cm": row.get("camera_height_cm", ""),
    }
    append_csv_rows(args.output, [out_row], CSV_HEADER)
    print(f"[SAVED] {args.output}")


if __name__ == "__main__":
    main()
