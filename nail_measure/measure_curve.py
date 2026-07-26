"""
measure_curve.py
=================
[1. 역할]
    손톱 1개에 대해 찍은 정면 사진 1장 + 측면 사진 1장으로 손톱 곡면을
    펼쳤을 때의 너비("곡면 길이")를 계산한다. 카메라가 CSI 한 대뿐이라
    정면/측면을 동시에 못 찍기 때문에, 같은 손가락을 정면 1장 -> 측면
    1장 순서로 나눠 찍는 것을 전제로 한다. 두 사진은 각각 자기 사진의
    ChArUco 보정 JSON(charuco_calibration.py 결과)을 따로 쓴다 (사진마다
    카메라 위치가 달라질 수 있어서 사진별 호모그래피가 필수다).

    측정 방식은 manual_measure.py와 동일하게 마우스 클릭 기반이다.
      - 정면 사진: 손톱 좌/우 최대너비 2점 클릭 -> front_width_mm
      - 측면 사진: 손톱 커브가 만드는 폭 2점 클릭 -> side_width_mm
    두 값과 --type(P/S/B/C, 유형별 비율은 curve_config.py)으로
    curve_math.compute_curve_length()가 곡면 길이를 계산한다.

    --tips-csv로 "type,size_no,max_width_mm" 형태의 전개도 테이블을 주면,
    계산된 곡면 길이와 가장 가까운 팁 사이즈를 찾아 함께 기록한다
    (선택 기능. 안 주면 곡면 길이만 계산한다).

[2. 실행 명령어]
    python3 measure_curve.py \\
      --front data/captured/front_thumb.jpg  --front-calib data/results/front_thumb_charuco.json \\
      --side  data/captured/side_thumb.jpg   --side-calib  data/results/side_thumb_charuco.json \\
      --finger thumb --type P

    # 팁 사이즈 매칭까지 원할 때
    python3 measure_curve.py --front ... --front-calib ... --side ... --side-calib ... \\
      --finger thumb --type P --tips-csv data/tip_sizes.csv

[3. 어디에 입력해야 하는가]
    -> Jetson Nano 터미널(SSH -X 또는 모니터 직결) 또는 노트북 터미널에서
       실행한다. 마우스 클릭이 필요하므로 화면(GUI)이 있는 환경이어야 한다.
    -> 촬영 순서: 같은 손가락을 ChArUco 보드 위에서 정면 1장 찍고
       charuco_calibration.py로 보정 -> 손가락을 옆으로 눕혀(손날이 보드에
       닿게) 측면 1장 찍고 다시 charuco_calibration.py로 보정 -> 이 두
       사진/보정 JSON을 이 스크립트에 넣는다.

[4. 정상적으로 실행되면]
    이미지 창이 두 번(정면 -> 측면) 뜨고, 각 사진에서 2점씩 클릭하면
    아래처럼 중간값과 최종 곡면 길이가 출력되고 CSV에 저장된다.

        [INFO] 정면 사진에서 손톱 좌/우 최대너비 2점을 클릭하세요.
        [INFO] 측면 사진에서 손톱 커브 폭 2점을 클릭하세요.
        [RESULT] thumb(P형) front_width=15.02mm side_width=8.11mm
                 a=3.00mm b=8.11mm c=8.65mm h=2.16mm r=5.42mm theta=1.641rad L=8.89mm
                 curve_length = 20.78mm
        [MATCH] 가장 가까운 팁: P6 (12.00mm, 차이 -8.78mm)
        [SAVED] data/results/curve_measurements.csv
        [SAVED] data/results/debug/front_thumb_thumb_curve_front.jpg
        [SAVED] data/results/debug/side_thumb_thumb_curve_side.jpg

[5. 오류가 발생하면 확인할 것]
    - "보정 파일을 찾을 수 없습니다" / "보정 파일에 homography 값이 없습니다":
      manual_measure.py와 동일한 원인. charuco_calibration.py로 만든 JSON을
      --front-calib/--side-calib에 정확히 넣었는지 확인.
    - "수직 높이(h)가 0 이하입니다" / "반지름(r)이 0 이하입니다": 정면/측면
      클릭 위치가 손톱 너비 양 끝이 맞는지, 두 사진을 서로 바꿔 넣지
      않았는지 확인 (front/side가 뒤바뀌면 계산이 깨진다).
    - 클릭 위치를 잘못 찍었다: 'r' 키로 마지막 점 되돌리기, 전체 취소는 'q'.
    - `--tips-csv`를 줬는데 [MATCH]가 안 뜬다: CSV 안에 같은 --type 값을
      가진 행이 있는지 확인 (컬럼: type,size_no,max_width_mm).
"""

import argparse
import csv
import os
from datetime import datetime

import cv2

from utils import (
    collect_points,
    load_json,
    measure_length_mm,
    append_csv_rows,
    ensure_dir,
    FINGERS,
    FINGER_LABELS_KO,
)
from curve_math import compute_curve_length, find_closest_tip_size
from curve_config import NAIL_TYPES
from curve_classify import run_interactive_classification, append_classification_row

CURVE_CSV_HEADER = [
    "timestamp",
    "front_image",
    "side_image",
    "finger",
    "nail_type",
    "backend",
    "front_width_mm",
    "side_width_mm",
    "a_mm",
    "b_mm",
    "c_mm",
    "h_mm",
    "r_mm",
    "theta_rad",
    "L_mm",
    "curve_length_mm",
    "camera_height_cm",
    "matched_tip_size",
    "matched_tip_max_width_mm",
    "matched_tip_diff_mm",
    "note",
]


def parse_args():
    parser = argparse.ArgumentParser(description="정면+측면 사진으로 손톱 곡면 너비(곡면 길이) 측정 (수동 클릭)")
    parser.add_argument("--front", required=True, help="정면 사진 경로")
    parser.add_argument("--front-calib", required=True, help="정면 사진용 ChArUco 보정 JSON 경로")
    parser.add_argument("--side", required=True, help="측면 사진 경로")
    parser.add_argument("--side-calib", required=True, help="측면 사진용 ChArUco 보정 JSON 경로")
    parser.add_argument("--finger", required=True, choices=FINGERS, help="측정할 손가락 1개")
    parser.add_argument(
        "--type",
        dest="nail_type",
        default="P",
        choices=NAIL_TYPES + ["auto"],
        help="손톱 유형 수동 지정 (기본: P). 'auto'를 주면 Phase 2 자동 분류(정면 4점+측면 4점 추가 클릭)를 먼저 실행한다.",
    )
    parser.add_argument(
        "--nail-height",
        type=float,
        default=0.0,
        help="보드 평면 ~ 손톱까지의 높이 (mm, 기본 0=보정 없음). manual_measure.py와 동일한 시차 보정.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("data", "results", "curve_measurements.csv"),
        help="측정 결과 CSV 저장 경로 (기본: data/results/curve_measurements.csv)",
    )
    parser.add_argument(
        "--debug-dir", default=os.path.join("data", "results", "debug"), help="디버그 오버레이 이미지 저장 폴더"
    )
    parser.add_argument(
        "--tips-csv",
        default=None,
        help="선택: type,size_no,max_width_mm 컬럼을 가진 팁 사이즈 전개도 테이블 CSV 경로",
    )
    parser.add_argument(
        "--classification-output",
        default=os.path.join("data", "results", "type_classification.csv"),
        help="--type auto일 때 분류 결과를 저장할 CSV 경로 (기본: data/results/type_classification.csv)",
    )
    parser.add_argument("--note", default="", help="CSV에 함께 남길 메모 (선택)")
    return parser.parse_args()


def _load_calib(calib_path):
    calib = load_json(calib_path)
    if calib is None:
        print(f"[ERROR] 보정 파일을 찾을 수 없습니다: {calib_path}")
        print("        먼저 charuco_calibration.py를 실행해서 보정 JSON을 만드세요.")
        return None
    if not calib.get("homography"):
        print(f"[ERROR] 보정 파일에 homography 값이 없습니다: {calib_path}")
        print("        v1(grid_calibration.py)의 옛 보정 파일이 아닌지 확인하세요.")
        return None
    return calib


def _measure_width(image_path, calib, labels, window_name, nail_height_mm):
    """이미지에서 2점을 클릭받아 호모그래피 기반 mm 너비를 계산한다."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] 이미지를 열 수 없습니다: {image_path}")
        return None

    points = collect_points(image, num_points=2, point_labels=labels, window_name=window_name)
    if points is None:
        return None

    p1, p2 = points
    camera_height_mm = calib.get("camera_height_mm", 295.0)
    width_mm = measure_length_mm(
        calib["homography"], p1, p2, camera_height_mm=camera_height_mm, nail_height_mm=nail_height_mm
    )
    return {
        "image": image,
        "points": (p1, p2),
        "width_mm": width_mm,
        "camera_height_mm": camera_height_mm,
    }


def _save_debug_overlay(image, points, width_mm, label, debug_path):
    overlay = image.copy()
    p1, p2 = (int(points[0][0]), int(points[0][1])), (int(points[1][0]), int(points[1][1]))
    cv2.line(overlay, p1, p2, (0, 0, 255), 2)
    for p in (p1, p2):
        cv2.circle(overlay, p, 6, (0, 0, 255), -1)
    cv2.putText(
        overlay,
        f"{label}: {width_mm:.2f}mm",
        (p1[0], max(0, p1[1] - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    ensure_dir(debug_path)
    cv2.imwrite(debug_path, overlay)
    print(f"[SAVED] {debug_path}")


def _load_tips_table(tips_csv):
    if not tips_csv:
        return None
    if not os.path.exists(tips_csv):
        print(f"[WARN] --tips-csv 파일을 찾을 수 없어 팁 매칭을 건너뜁니다: {tips_csv}")
        return None
    with open(tips_csv, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


def main():
    args = parse_args()
    label_ko = FINGER_LABELS_KO[args.finger]

    front_calib = _load_calib(args.front_calib)
    if front_calib is None:
        return
    side_calib = _load_calib(args.side_calib)
    if side_calib is None:
        return

    print(f"[INFO] [{label_ko}] 정면 사진에서 손톱 좌/우 최대너비 2점을 클릭하세요.")
    front = _measure_width(
        args.front,
        front_calib,
        labels=[f"[{label_ko}] 정면: 손톱 왼쪽 끝을 클릭하세요", f"[{label_ko}] 정면: 손톱 오른쪽 끝을 클릭하세요"],
        window_name=f"Front - {label_ko}",
        nail_height_mm=args.nail_height,
    )
    if front is None:
        print("[INFO] 정면 측정을 취소했습니다.")
        return

    print(f"[INFO] [{label_ko}] 측면 사진에서 손톱 커브 폭 2점을 클릭하세요.")
    side = _measure_width(
        args.side,
        side_calib,
        labels=[f"[{label_ko}] 측면: 커브 시작점을 클릭하세요", f"[{label_ko}] 측면: 커브 끝점을 클릭하세요"],
        window_name=f"Side - {label_ko}",
        nail_height_mm=args.nail_height,
    )
    cv2.destroyAllWindows()
    if side is None:
        print("[INFO] 측면 측정을 취소했습니다.")
        return

    front_width_mm = front["width_mm"]
    side_width_mm = side["width_mm"]

    if args.nail_type == "auto":
        print(f"[INFO] [{label_ko}] --type auto: 유형 자동 분류를 먼저 진행합니다 (정면 4점 + 측면 4점 클릭).")
        classification = run_interactive_classification(
            front["image"], side["image"], side_calib["homography"], args.finger
        )
        cv2.destroyAllWindows()
        if classification is None:
            print("[INFO] 유형 자동 분류를 취소했습니다.")
            return
        nail_type = classification["nail_type"]
        if nail_type is None:
            print(f"[ERROR] 유형 자동 분류에 실패했습니다 (판정 불가: {classification['basis'].get('reason')}).")
            print("        --type을 P/S/B/C 중 하나로 수동 지정해서 다시 시도하세요.")
            return
        print(f"[CLASSIFY] 판정 결과: {nail_type}형")
        print(f"           근거: {classification['basis']}")
        append_classification_row(
            args.classification_output,
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "front_image": args.front,
                "side_image": args.side,
                "finger": args.finger,
                "nail_type": nail_type,
                "front_slope": round(classification["basis"].get("front_slope", 0.0), 4),
                "side_diff_34_mm": round(classification["basis"].get("diff_34_mm", 0.0), 4)
                if "diff_34_mm" in classification["basis"] else "",
                "reason": classification["basis"].get("reason", ""),
            },
        )
        print(f"[SAVED] {args.classification_output}")
    else:
        nail_type = args.nail_type

    try:
        result = compute_curve_length(front_width_mm, side_width_mm, nail_type)
    except ValueError as e:
        print(f"[ERROR] 곡면 길이 계산 실패: {e}")
        return

    print(
        f"[RESULT] {label_ko}({nail_type}형) "
        f"front_width={front_width_mm:.2f}mm side_width={side_width_mm:.2f}mm"
    )
    print(
        f"         a={result['a_mm']:.2f}mm b={result['b_mm']:.2f}mm c={result['c_mm']:.2f}mm "
        f"h={result['h_mm']:.2f}mm r={result['r_mm']:.2f}mm theta={result['theta_rad']:.3f}rad "
        f"L={result['L_mm']:.2f}mm"
    )
    print(f"         curve_length = {result['curve_length_mm']:.2f}mm")

    matched = None
    tips_rows = _load_tips_table(args.tips_csv)
    if tips_rows:
        matched = find_closest_tip_size(result["curve_length_mm"], nail_type, tips_rows)
        if matched:
            print(
                f"[MATCH] 가장 가까운 팁: {matched['size_no']} "
                f"({matched['max_width_mm']:.2f}mm, 차이 {matched['diff_mm']:+.2f}mm)"
            )
        else:
            print(f"[WARN] --tips-csv에 유형 '{nail_type}'에 해당하는 팁이 없어 매칭하지 못했습니다.")

    base_front = os.path.splitext(os.path.basename(args.front))[0]
    base_side = os.path.splitext(os.path.basename(args.side))[0]
    _save_debug_overlay(
        front["image"], front["points"], front_width_mm, "front",
        os.path.join(args.debug_dir, f"{base_front}_{args.finger}_curve_front.jpg"),
    )
    _save_debug_overlay(
        side["image"], side["points"], side_width_mm, "side",
        os.path.join(args.debug_dir, f"{base_side}_{args.finger}_curve_side.jpg"),
    )

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "front_image": args.front,
        "side_image": args.side,
        "finger": args.finger,
        "nail_type": nail_type,
        "backend": "manual",
        "front_width_mm": round(front_width_mm, 4),
        "side_width_mm": round(side_width_mm, 4),
        "a_mm": round(result["a_mm"], 4),
        "b_mm": round(result["b_mm"], 4),
        "c_mm": round(result["c_mm"], 4),
        "h_mm": round(result["h_mm"], 4),
        "r_mm": round(result["r_mm"], 4),
        "theta_rad": round(result["theta_rad"], 6),
        "L_mm": round(result["L_mm"], 4),
        "curve_length_mm": round(result["curve_length_mm"], 4),
        "camera_height_cm": round(front["camera_height_mm"] / 10, 2),
        "matched_tip_size": matched["size_no"] if matched else "",
        "matched_tip_max_width_mm": round(matched["max_width_mm"], 2) if matched else "",
        "matched_tip_diff_mm": round(matched["diff_mm"], 2) if matched else "",
        "note": args.note,
    }
    append_csv_rows(args.output, [row], CURVE_CSV_HEADER)
    print(f"[SAVED] {args.output}")


if __name__ == "__main__":
    main()
