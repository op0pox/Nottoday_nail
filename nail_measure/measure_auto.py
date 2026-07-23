"""
measure_auto.py
================
[1. 역할]
    세그멘테이션 백엔드(yolo/classical/manual)로 손톱 마스크를 자동으로
    얻고, ChArUco 호모그래피(charuco_calibration.py 결과)로 mm 길이를
    계산해서 measurements.csv에 저장한다. manual_measure.py(사람이 직접
    클릭)와 같은 CSV 포맷을 쓰기 때문에 두 결과를 곧바로 비교할 수 있다.

    이번 버전의 메인 백엔드는 yolo다 (사전학습 공개 모델 사용).
    classical/manual 백엔드는 segmentation/classical.py, segmentation/manual.py
    가 추가되는 다음 단계에서 사용할 수 있다.

[2. 실행 명령어]
    python3 measure_auto.py --image data/captured/sample.jpg --calib data/results/sample_charuco.json
    python3 measure_auto.py --image data/captured/sample.jpg --calib data/results/sample_charuco.json --conf 0.15
    python3 measure_auto.py --image data/captured/sample.jpg --calib data/results/sample_charuco.json --hand left

[3. 어디에 입력해야 하는가]
    -> --backend yolo / classical: 노트북(리눅스 랩탑) 터미널에서 실행한다.
       (ultralytics/PyTorch는 무거워서 이번 범위에서는 Jetson Nano에 설치하지 않는다)
    -> --backend manual: Jetson Nano 또는 노트북 어디서든 실행 가능하지만,
       마우스 클릭이 필요하므로 화면(GUI, SSH -X 또는 모니터 직결)이 있어야 한다.
    -> Nano로 촬영한 사진을 노트북으로 옮기는 방법은 README.md의
       "Nano 촬영 -> 노트북 측정" 절 참고 (scp 예시 포함)

[4. 정상적으로 실행되면] (yolo 백엔드 기준)
        [INFO] YOLO 검출: 5개 (필터 전 6개), conf=0.25
        [INFO] 손가락 라벨링: x좌표 순서로 자동 매칭됨 (hand=right)
        [RESULT] thumb: 11.20mm
        [RESULT] index: 13.05mm
        [RESULT] middle: 14.02mm
        [RESULT] ring: 12.44mm
        [RESULT] pinky: 9.87mm
        [SAVED] data/results/measurements.csv (5개 행 추가)
        [SAVED] data/results/debug/sample_yolo_debug.jpg

[5. 오류가 발생하면 확인할 것]
    - "ultralytics 패키지가 설치되어 있지 않습니다": 노트북에서
      pip3 install -r requirements.txt
    - "모델 자동 다운로드에 실패했습니다": 네트워크 확인, 또는 안내된 URL로 수동 다운로드
    - "보정 파일에 homography 값이 없습니다": charuco_calibration.py로 만든
      JSON을 --calib에 넣었는지 확인 (v1 grid_calibration.py 결과 JSON이 아님)
    - 검출 개수가 5개가 아니다: --conf 값을 낮춰서 다시 시도 (예: --conf 0.1)
    - 손가락 라벨이 계속 틀린다: --hand left/right 값을 반대로 지정해보거나,
      마스크가 정확히 5개가 아니면 자동으로 뜨는 수동 라벨 입력 프롬프트를 이용한다.
"""

import argparse
import os

import cv2

from utils import ensure_dir, load_json, append_csv_rows, MEASUREMENTS_CSV_HEADER
from segmentation.measure_from_mask import measure_nail_from_mask, label_fingers


def parse_args():
    parser = argparse.ArgumentParser(description="세그멘테이션 백엔드로 손톱 길이 자동 측정")
    parser.add_argument("--image", required=True, help="측정할 손톱 사진 경로")
    parser.add_argument("--calib", required=True, help="charuco_calibration.py로 만든 보정 JSON")
    parser.add_argument(
        "--backend", choices=["yolo", "classical", "manual"], default="yolo", help="세그멘테이션 백엔드"
    )
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold (yolo 백엔드만)")
    parser.add_argument("--hand", choices=["left", "right"], default="right", help="촬영된 손 (자동 라벨링용)")
    parser.add_argument(
        "--use-mediapipe", action="store_true", help="mediapipe 손 랜드마크로 손가락 라벨링 시도 (설치돼 있을 때만)"
    )
    parser.add_argument("--nail-height", type=float, default=0.0, help="보드~손톱 높이(mm, 시차 보정)")
    parser.add_argument(
        "--output", default=os.path.join("data", "results", "measurements.csv"), help="측정 결과 CSV 경로"
    )
    parser.add_argument(
        "--debug-dir", default=os.path.join("data", "results", "debug"), help="디버그 오버레이 이미지 저장 폴더"
    )
    return parser.parse_args()


def get_backend(name, conf):
    if name == "yolo":
        from segmentation.dl_yolo import YoloNailBackend

        try:
            return YoloNailBackend(conf=conf)
        except RuntimeError as e:
            print(f"[WARN] YOLO 백엔드를 불러오지 못해 classical 백엔드로 폴백합니다.\n{e}")
            from segmentation.classical import ClassicalNailBackend

            return ClassicalNailBackend()
    if name == "classical":
        from segmentation.classical import ClassicalNailBackend

        return ClassicalNailBackend()
    if name == "manual":
        from segmentation.manual import ManualNailBackend

        return ManualNailBackend()
    raise ValueError(f"알 수 없는 backend: {name}")


def save_debug_overlay(image, nail_masks, measured_list, debug_path):
    overlay = image.copy()
    for nm, measured in zip(nail_masks, measured_list):
        contours, _ = cv2.findContours(nm.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        if measured is None:
            continue
        p1 = tuple(int(v) for v in measured["point_a"])
        p2 = tuple(int(v) for v in measured["point_b"])
        cv2.line(overlay, p1, p2, (0, 0, 255), 2)
        if measured.get("width_point_a") is not None:
            w1 = tuple(int(v) for v in measured["width_point_a"])
            w2 = tuple(int(v) for v in measured["width_point_b"])
            cv2.line(overlay, w1, w2, (255, 128, 0), 2)
        width_str = (
            f" w{measured['width_mm']:.1f}mm" if measured.get("width_mm") is not None else ""
        )
        cv2.putText(
            overlay,
            f"{nm.finger}:{measured['length_mm']:.1f}mm{width_str}",
            (p1[0], max(0, p1[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
    ensure_dir(debug_path)
    cv2.imwrite(debug_path, overlay)
    print(f"[SAVED] {debug_path}")


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
        print("        charuco_calibration.py로 만든 보정 JSON을 사용하세요.")
        return
    camera_height_mm = calib.get("camera_height_mm", 295.0)
    rms_mm = calib.get("rms_reprojection_mm")

    try:
        backend = get_backend(args.backend, args.conf)
    except (RuntimeError, ImportError) as e:
        print(f"[ERROR] {e}")
        return

    nail_masks = backend.segment(image)
    if not nail_masks:
        print("[ERROR] 손톱을 하나도 찾지 못했습니다.")
        return

    if any(nm.finger is None for nm in nail_masks):
        base = os.path.splitext(os.path.basename(args.image))[0]
        debug_help_path = os.path.join(args.debug_dir, f"{base}_{backend.name}_label_help.jpg")
        nail_masks = label_fingers(
            nail_masks,
            hand=args.hand,
            use_mediapipe=args.use_mediapipe,
            debug_image=image,
            debug_path=debug_help_path,
        )

    rows = []
    measured_list = []
    for nm in nail_masks:
        measured = measure_nail_from_mask(
            nm.mask,
            homography,
            camera_height_mm=camera_height_mm,
            nail_height_mm=args.nail_height,
            finger=nm.finger,
        )
        measured_list.append(measured)
        if measured is None:
            print(f"[WARN] {nm.finger} 마스크에서 길이를 계산하지 못했습니다.")
            continue

        width_str = (
            f", 폭 {measured['width_mm']:.2f}mm" if measured.get("width_mm") is not None else ""
        )
        print(f"[RESULT] {nm.finger}: 길이 {measured['length_mm']:.2f}mm{width_str}")
        row = {
            "image_path": args.image,
            "finger": nm.finger,
            "root_x": round(measured["point_a"][0], 1),
            "root_y": round(measured["point_a"][1], 1),
            "tip_x": round(measured["point_b"][0], 1),
            "tip_y": round(measured["point_b"][1], 1),
            "pixel_distance": round(measured["pixel_distance"], 4),
            "length_mm": round(measured["length_mm"], 4),
            "calibration_method": "charuco",
            "backend": backend.name,
            "camera_height_mm": camera_height_mm,
            "nail_height_mm": args.nail_height,
            "homography_rms_mm": rms_mm,
        }
        if measured.get("width_mm") is not None:
            row.update(
                {
                    "width_left_x": round(measured["width_point_a"][0], 1),
                    "width_left_y": round(measured["width_point_a"][1], 1),
                    "width_right_x": round(measured["width_point_b"][0], 1),
                    "width_right_y": round(measured["width_point_b"][1], 1),
                    "width_pixel_distance": round(measured["width_pixel_distance"], 4),
                    "width_mm": round(measured["width_mm"], 4),
                }
            )
        rows.append(row)

    if not rows:
        print("[INFO] 저장할 측정 결과가 없습니다.")
        return

    append_csv_rows(args.output, rows, MEASUREMENTS_CSV_HEADER)
    print(f"[SAVED] {args.output} ({len(rows)}개 행 추가)")

    base = os.path.splitext(os.path.basename(args.image))[0]
    debug_path = os.path.join(args.debug_dir, f"{base}_{backend.name}_debug.jpg")
    save_debug_overlay(image, nail_masks, measured_list, debug_path)


if __name__ == "__main__":
    main()
