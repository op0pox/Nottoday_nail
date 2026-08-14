"""
distance_calibration.py
========================
[1. 역할]
    카메라를 장치에 "고정"해서 쓸 때, 렌즈~측정 대상 사이의 거리(cm)만
    입력하면 픽셀<->mm 변환 보정값을 자동으로 계산해서 JSON으로 저장한다.
    사진마다 모눈판/ChArUco 보드를 함께 찍을 필요가 없다.

    원리(핀홀 카메라 모델):
        실제 1mm가 사진에서 몇 px로 보이는지는
            px_per_mm = 초점거리(px) / 거리(mm)
        로 정해진다. 초점거리(px)는 카메라 스펙(IMX219, 라즈베리파이
        카메라 v2)에서 계산할 수 있으므로, 거리만 알면 보정이 끝난다.

    저장되는 JSON은 charuco_calibration.py 결과와 같은 형식(homography +
    camera_height_mm)이라서 measure_curve.py / manual_measure.py /
    measure_auto.py 의 --calib 류 인자에 그대로 넣을 수 있다.

[2. 실행 명령어]
    python3 distance_calibration.py --camera top    # 위 카메라 (기본 10.5cm)
    python3 distance_calibration.py --camera side   # 측면 카메라 (기본 7.0cm)

    # 거리를 다시 잰 경우 cm 단위로 직접 지정 (소수점 가능)
    python3 distance_calibration.py --camera top --distance-cm 11.2

    # 촬영 해상도가 capture_image.py 기본값(1920x1080)과 다른 경우
    python3 distance_calibration.py --camera top --width 3264 --height 2464

    # 실측으로 미세 보정하는 경우 (아래 [6] 참고)
    python3 distance_calibration.py --camera top --correction 1.03

[3. 어디에 입력해야 하는가]
    -> Jetson Nano 터미널(SSH 접속)에서 실행한다. (마우스 클릭/GUI 불필요)
    -> 카메라 고정 거리가 바뀌지 않는 한 "한 번만" 실행하면 된다.
       이후 촬영하는 모든 사진에 같은 보정 JSON을 계속 재사용한다.

[4. 정상적으로 실행되면]
        [INFO] top 카메라: 거리 10.5cm, 해상도 1920x1080
        [INFO] 센서 모드: 1920x1080 크롭 (초점거리 2714.3px)
        [OK] px_per_mm = 25.8500  (1mm = 25.85px)
        [OK] 화면에 보이는 실제 폭: 약 74.3mm
        [SAVED] data/results/top_distance_calib.json
    이 JSON을 measure_curve.py의 --front-calib / --side-calib 에 넣으면 된다.

[5. 오류가 발생하면 확인할 것]
    - "지원하지 않는 해상도": 이 스크립트는 촬영 해상도로 센서 모드를
      추정한다. 목록(3264x2464, 1920x1080, 1640x1232, 1280x720) 밖의
      해상도로 찍었다면 촬영 해상도를 목록 중 하나로 바꾸는 것을 권장.
    - 측정값이 실제와 몇 % 정도 차이난다: 거리 측정 오차이거나 렌즈
      개체차이다. 아래 [6]의 방법으로 --correction 값을 구해서 넣는다.
    - 측정값이 몇 배씩 크게 다르다: --width/--height가 실제 촬영 해상도와
      같은지, 거리가 cm 단위(mm 아님)인지 확인.

[6. 정밀도를 높이고 싶다면 (선택, 한 번만)]
    이론값은 보통 실제와 2~3% 이내로 맞지만, 더 정확히 하려면:
    1) 고정된 카메라로 모눈판(또는 자)을 한 장 찍는다.
    2) grid_calibration.py로 실측 px_per_mm을 구한다.
    3) correction = 실측 px_per_mm / 이 스크립트가 출력한 px_per_mm
    4) 그 값을 --correction 으로 넣어 다시 실행하면 끝.
    카메라를 다시 옮기지 않는 한 이 과정은 반복할 필요가 없다.

[7. 거리를 재는 기준 (중요)]
    "렌즈 표면 ~ 실제로 재려는 면(손톱 표면)"의 거리를 잰다.
    - 손톱 표면까지의 거리를 쟀다면: measure_curve.py에서 --nail-height 0
    - 손가락 받침면까지의 거리를 쟀다면: 손톱이 받침면보다 떠 있는 높이를
      --nail-height 로 넣으면 시차(parallax) 보정이 자동 적용된다.
"""

import argparse
import json
import os


def ensure_dir(path):
    """파일 경로의 상위 폴더가 없으면 만든다."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 카메라 고정 거리 프리셋 (cm)
# 장치를 조정해서 거리가 바뀌면 여기 값만 고치면 된다.
# ---------------------------------------------------------------------------
CAMERA_PRESETS = {
    "top": {"distance_cm": 10.5, "설명": "위(정면) 카메라"},
    "side": {"distance_cm": 7.0, "설명": "측면 카메라"},
}

# ---------------------------------------------------------------------------
# IMX219 (라즈베리파이 카메라 v2) 스펙 상수
#   초점거리 3.04mm, 픽셀 크기 1.12um (풀해상도 기준)
#   초점거리(px) = 3.04mm / 0.00112mm = 약 2714px
# 촬영 해상도에 따라 Jetson의 argus가 고르는 센서 모드가 다르고,
# 모드마다 비닝(2x2 픽셀 합침) 여부가 달라서 초점거리(px)가 달라진다.
# ---------------------------------------------------------------------------
FOCAL_MM = 3.04
PIXEL_UM_FULL = 1.12  # 풀해상도 픽셀 크기
FOCAL_PX_FULL = FOCAL_MM * 1000.0 / PIXEL_UM_FULL  # 약 2714px

# (가로, 세로) -> (초점거리 px, 모드 설명)
#   3264x2464 : 풀해상도(비닝 없음)      -> 2714px
#   1920x1080 : 중앙 크롭(비닝 없음)     -> 2714px
#   1640x1232 : 전체 화각 2x2 비닝       -> 1357px
#   1280x720  : 크롭 + 2x2 비닝          -> 1357px
SENSOR_MODES = {
    (3264, 2464): (FOCAL_PX_FULL, "3264x2464 풀해상도"),
    (1920, 1080): (FOCAL_PX_FULL, "1920x1080 크롭"),
    (1640, 1232): (FOCAL_PX_FULL / 2, "1640x1232 비닝(전체 화각)"),
    (1280, 720): (FOCAL_PX_FULL / 2, "1280x720 크롭+비닝"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="고정 카메라 거리(cm) 기반 자동 보정 - 모눈판 촬영 불필요"
    )
    parser.add_argument(
        "--camera",
        required=True,
        choices=list(CAMERA_PRESETS.keys()),
        help="어느 카메라인지 (top=위 카메라, side=측면 카메라)",
    )
    parser.add_argument(
        "--distance-cm",
        type=float,
        default=None,
        help="렌즈~대상 거리(cm). 생략하면 프리셋 값 사용 (top=10.5, side=7.0)",
    )
    parser.add_argument("--width", type=int, default=1920,
                        help="촬영 가로 해상도 (capture_image.py와 같게, 기본 1920)")
    parser.add_argument("--height", type=int, default=1080,
                        help="촬영 세로 해상도 (기본 1080)")
    parser.add_argument(
        "--correction",
        type=float,
        default=1.0,
        help="실측 미세 보정 배율 (기본 1.0). 구하는 법은 파일 상단 [6] 참고",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="결과 JSON 저장 경로 (기본: data/results/<camera>_distance_calib.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    preset = CAMERA_PRESETS[args.camera]
    distance_cm = args.distance_cm if args.distance_cm is not None else preset["distance_cm"]
    if distance_cm <= 0:
        print(f"[ERROR] 거리는 0보다 커야 합니다: {distance_cm}cm")
        return
    distance_mm = distance_cm * 10.0

    mode = SENSOR_MODES.get((args.width, args.height))
    if mode is None:
        supported = ", ".join(f"{w}x{h}" for (w, h) in SENSOR_MODES)
        print(f"[ERROR] 지원하지 않는 해상도입니다: {args.width}x{args.height}")
        print(f"        지원 목록: {supported}")
        print("        촬영 해상도를 위 목록 중 하나로 맞춰서 찍는 것을 권장합니다.")
        return
    focal_px, mode_name = mode

    px_per_mm = focal_px / distance_mm * args.correction
    view_width_mm = args.width / px_per_mm

    # 픽셀 -> mm 평면으로 옮기는 호모그래피 (단순 배율 행렬).
    # charuco_calibration.py 결과와 같은 키 이름을 쓰므로
    # measure_curve.py 등의 --calib 인자에 그대로 넣을 수 있다.
    s = 1.0 / px_per_mm
    homography = [[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]]

    print(f"[INFO] {args.camera} 카메라: 거리 {distance_cm}cm, 해상도 {args.width}x{args.height}")
    print(f"[INFO] 센서 모드: {mode_name} (초점거리 {focal_px:.1f}px)")
    if args.correction != 1.0:
        print(f"[INFO] 실측 보정 배율 적용: x{args.correction}")
    print(f"[OK] px_per_mm = {px_per_mm:.4f}  (1mm = {px_per_mm:.2f}px)")
    print(f"[OK] 화면에 보이는 실제 폭: 약 {view_width_mm:.1f}mm")

    result = {
        "method": "distance",
        "camera": args.camera,
        "distance_cm": distance_cm,
        "camera_height_mm": distance_mm,
        "capture_width": args.width,
        "capture_height": args.height,
        "sensor_mode": mode_name,
        "focal_px": round(focal_px, 2),
        "correction": args.correction,
        "px_per_mm": round(px_per_mm, 4),
        "homography": homography,
    }
    output_path = args.output or os.path.join(
        "data", "results", f"{args.camera}_distance_calib.json"
    )
    ensure_dir(output_path)
    save_json(output_path, result)
    print(f"[SAVED] {output_path}")
    print("        -> measure_curve.py --front-calib/--side-calib 에 이 파일을 넣으면 됩니다.")


if __name__ == "__main__":
    main()
