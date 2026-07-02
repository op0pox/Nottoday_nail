"""
compare_actual.py
==================
[1. 역할]
    manual_measure.py로 측정한 프로그램 측정값과, 사용자가 실제 자로 잰
    값을 비교해서 오차(mm)와 오차율(%)을 계산한다.
    엄지/검지/중지/약지/소지 실제 길이를 입력하면, 같은 이미지에 대해
    저장된 최근 측정값과 비교해서 결과를 CSV로 저장한다.

[2. 실행 명령어]
    # 대화형으로 실제 길이를 입력하는 방식 (초보자에게 추천)
    python3 compare_actual.py --image data/captured/sample.jpg

    # 실제 길이를 명령줄에서 바로 넘기는 방식 (자동화/반복 실험용)
    python3 compare_actual.py --image data/captured/sample.jpg \\
        --actual thumb=12.5 index=11.0 middle=13.2 ring=12.0 pinky=9.5

[3. 어디에 입력해야 하는가]
    -> Jetson Nano 터미널(SSH 접속) 또는 화면이 없어도 실행 가능하다.
       (이 스크립트는 마우스 클릭이 필요 없고 키보드 입력만 받는다)

[4. 정상적으로 실행되면]
    대화형 모드에서는 아래처럼 손가락별로 실제 길이를 입력하라는
    안내가 순서대로 나온다.

        엄지 실제 길이(mm): 12.5
        검지 실제 길이(mm): 11.0
        중지 실제 길이(mm): 13.2
        약지 실제 길이(mm): 12.0
        소지 실제 길이(mm): 9.5

    입력이 끝나면 손가락별 오차/오차율이 표로 출력되고 CSV로 저장된다.

        [RESULT] thumb: 실제 12.5mm / 측정 12.1mm / 오차 0.4mm (3.2%)
        [SAVED] data/results/comparison.csv

[5. 오류가 발생하면 확인할 것]
    - "측정 데이터가 없습니다": manual_measure.py를 먼저 실행해서
      해당 이미지에 대한 측정값을 CSV에 저장했는지 확인
    - 숫자가 아닌 값을 입력해서 오류가 나면: 숫자(예: 12.5)만 입력
    - 특정 손가락만 비교하고 싶다면: --fingers thumb index 처럼 지정
"""

import argparse
import csv
import os

from utils import ensure_dir, FINGERS, FINGER_LABELS_KO

MEASURED_CSV_DEFAULT = os.path.join("data", "results", "measurements.csv")
COMPARISON_CSV_DEFAULT = os.path.join("data", "results", "comparison.csv")

CSV_HEADER = ["image_path", "finger", "actual_mm", "measured_mm", "error_mm", "error_percent"]


def parse_args():
    parser = argparse.ArgumentParser(description="실제 자 측정값과 프로그램 측정값 비교")
    parser.add_argument("--image", required=True, help="비교할 손톱 사진 경로")
    parser.add_argument(
        "--measured-csv",
        default=MEASURED_CSV_DEFAULT,
        help=f"manual_measure.py 결과 CSV (기본: {MEASURED_CSV_DEFAULT})",
    )
    parser.add_argument(
        "--output",
        default=COMPARISON_CSV_DEFAULT,
        help=f"비교 결과 CSV 저장 경로 (기본: {COMPARISON_CSV_DEFAULT})",
    )
    parser.add_argument(
        "--fingers",
        nargs="+",
        default=FINGERS,
        choices=FINGERS,
        help="비교할 손가락만 선택 (기본: 5개 전부)",
    )
    parser.add_argument(
        "--actual",
        nargs="+",
        default=None,
        help="실제 길이를 대화형 입력 대신 바로 지정. 예: thumb=12.5 index=11.0",
    )
    return parser.parse_args()


def load_latest_measurements(csv_path, image_path):
    """
    measurements.csv에서 image_path가 일치하는 행 중,
    손가락별로 가장 마지막(최근)에 측정된 행만 골라서 dict로 반환한다.
    같은 사진을 여러 번 측정했을 경우 가장 최근 값을 사용하기 위함이다.
    """
    if not os.path.exists(csv_path):
        return {}

    latest = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("image_path") != image_path:
                continue
            finger = row.get("finger")
            latest[finger] = float(row.get("length_mm"))
    return latest


def parse_actual_args(actual_list):
    """['thumb=12.5', 'index=11.0', ...] -> {'thumb': 12.5, 'index': 11.0, ...}"""
    result = {}
    for item in actual_list:
        if "=" not in item:
            raise ValueError(f"--actual 형식이 올바르지 않습니다: {item} (예: thumb=12.5)")
        key, value = item.split("=", 1)
        key = key.strip()
        if key not in FINGERS:
            raise ValueError(f"알 수 없는 손가락 이름: {key} (가능: {', '.join(FINGERS)})")
        result[key] = float(value)
    return result


def prompt_actual_values(fingers):
    """터미널에서 손가락별 실제 길이(mm)를 하나씩 입력받는다."""
    result = {}
    print("[INFO] 실제 자로 측정한 손톱 길이를 mm 단위로 입력하세요.")
    for finger in fingers:
        label_ko = FINGER_LABELS_KO[finger]
        while True:
            raw = input(f"{label_ko} 실제 길이(mm): ").strip()
            try:
                result[finger] = float(raw)
                break
            except ValueError:
                print("  -> 숫자로 입력해주세요. 예: 12.5")
    return result


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

    measured = load_latest_measurements(args.measured_csv, args.image)
    if not measured:
        print(f"[ERROR] '{args.image}'에 대한 측정 데이터가 없습니다.")
        print(f"        먼저 manual_measure.py로 측정 결과를 {args.measured_csv} 에 저장하세요.")
        return

    missing = [f for f in args.fingers if f not in measured]
    if missing:
        missing_ko = ", ".join(FINGER_LABELS_KO[f] for f in missing)
        print(f"[WARN] 다음 손가락은 측정 데이터가 없어 건너뜁니다: {missing_ko}")

    target_fingers = [f for f in args.fingers if f in measured]
    if not target_fingers:
        print("[ERROR] 비교할 손가락 측정 데이터가 하나도 없습니다.")
        return

    if args.actual:
        try:
            actual_values = parse_actual_args(args.actual)
        except ValueError as e:
            print(f"[ERROR] {e}")
            return
    else:
        actual_values = prompt_actual_values(target_fingers)

    rows = []
    print()
    for finger in target_fingers:
        if finger not in actual_values:
            print(f"[WARN] '{FINGER_LABELS_KO[finger]}' 실제 길이가 없어 건너뜁니다.")
            continue

        actual_mm = actual_values[finger]
        measured_mm = measured[finger]
        error_mm = abs(actual_mm - measured_mm)
        error_percent = (error_mm / actual_mm * 100) if actual_mm != 0 else 0.0

        print(
            f"[RESULT] {finger}: 실제 {actual_mm}mm / 측정 {measured_mm}mm / "
            f"오차 {error_mm:.2f}mm ({error_percent:.1f}%)"
        )

        rows.append(
            {
                "image_path": args.image,
                "finger": finger,
                "actual_mm": actual_mm,
                "measured_mm": measured_mm,
                "error_mm": round(error_mm, 4),
                "error_percent": round(error_percent, 2),
            }
        )

    if not rows:
        print("[INFO] 저장할 비교 결과가 없습니다.")
        return

    append_rows_to_csv(args.output, rows)
    print(f"\n[SAVED] {args.output} ({len(rows)}개 행 추가)")


if __name__ == "__main__":
    main()
