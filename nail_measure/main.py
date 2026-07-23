"""
main.py
=======
[1. 역할]
    지금까지 만든 모든 기능(카메라 테스트, 촬영, ChArUco 보정, 손톱 측정
    -수동/자동-, 실제값 비교, 백엔드별 오차 비교, 높이별 실험 기록/요약)을
    번호로 선택해서 실행할 수 있는 통합 메뉴 프로그램이다. 각 기능은 사실
    다른 .py 파일을 그대로 실행해주는 것이며, main.py는 그 파일들을
    순서대로 안내해주거나(메뉴 모드), --mode로 바로 하나만 실행해주는
    (스크립트 모드) 역할만 한다.

    --image, --camera, --camera-model, --backend 등 옵션으로 자주 쓰는
    값을 미리 지정해두면, 메뉴에서 매번 다시 입력하지 않고 Enter만
    눌러도 그 값이 기본값으로 사용된다.

[2. 실행 명령어]
    # 메뉴 모드 (번호 선택, 처음 쓸 때 추천)
    python3 main.py
    python3 main.py --image data/captured/sample.jpg --camera usb

    # 스크립트 모드 (--mode로 바로 실행, 반복 작업/자동화에 추천)
    python3 main.py --mode capture --camera csi
    python3 main.py --mode calibrate --image data/captured/sample.jpg
    python3 main.py --mode measure-manual --image data/captured/sample.jpg
    python3 main.py --mode measure-auto --image data/captured/sample.jpg --backend yolo --conf 0.2
    python3 main.py --mode compare   # 백엔드별 평균 오차 요약 (comparison.csv 전체 집계)

[3. 어디에 입력해야 하는가]
    -> Jetson Nano 터미널(SSH 접속) 또는 노트북 터미널에서 실행한다.
    -> 메뉴/집계 자체는 화면(GUI) 없이도 뜨지만, 촬영/보정/수동측정처럼
       마우스 클릭이 필요한 기능을 선택하면 그 순간에는 화면(GUI)이 필요하다.
    -> measure-auto(yolo/classical 백엔드)는 노트북에서 실행해야 한다
       (ultralytics/PyTorch는 Jetson Nano에 설치하지 않음). README.md의
       "Nano 촬영 -> 노트북 측정" 절 참고.

[4. 정상적으로 실행되면]
    아래와 같은 번호 메뉴가 터미널에 출력된다.

        ===== 손톱 길이 측정 프로그램 =====
        1. 카메라 연결 테스트
        2. 사진 촬영
        3. ChArUco 보정
        4. 손톱 측정 - 수동 클릭
        5. 손톱 측정 - 자동(YOLO 등)
        6. 실제 자 측정값과 비교 (사진 1장)
        7. 촬영 조건(높이 등) 기록
        8. 높이별 평균 오차 요약
        9. 백엔드별 평균 오차 비교
        0. 종료
        번호를 선택하세요:

    번호를 입력하면 해당 스크립트가 실행되고, 끝나면 다시 메뉴로 돌아온다.
    --mode를 지정하면 메뉴 없이 그 기능 하나만 실행하고 바로 종료한다.

[5. 오류가 발생하면 확인할 것]
    - 각 기능 실행 중 나는 오류는 해당 스크립트 파일
      (camera_test.py, charuco_calibration.py 등) 상단 설명의
      '오류가 발생하면 확인할 것' 항목을 참고한다.
    - "python3: command not found" 같은 오류가 main.py 실행 자체에서
      나면, Python3가 설치되어 있는지 확인한다.
"""

import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CAMERA_HEIGHT_MM = 295.0
DEFAULT_COMPARISON_CSV = os.path.join("data", "results", "comparison.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="손톱 길이 측정 프로그램 - 통합 메뉴")
    parser.add_argument(
        "--mode",
        choices=["capture", "calibrate", "measure-manual", "measure-auto", "compare"],
        default=None,
        help="지정하면 메뉴 없이 해당 기능 하나만 실행하고 종료한다 (생략하면 메뉴 모드)",
    )
    parser.add_argument("--image", default=None, help="기본으로 사용할 이미지 경로")
    parser.add_argument("--calib", default=None, help="보정 JSON 경로 (생략하면 이미지 이름으로 추정)")
    parser.add_argument("--camera", choices=["csi", "usb"], default="csi", help="기본 카메라 종류")
    parser.add_argument(
        "--camera-height",
        type=float,
        default=DEFAULT_CAMERA_HEIGHT_MM,
        help=f"카메라~보드 높이 (mm, 기본 {DEFAULT_CAMERA_HEIGHT_MM})",
    )
    parser.add_argument("--camera-model", default=None, help="기본 카메라 모델 이름")
    parser.add_argument(
        "--backend", choices=["yolo", "classical", "manual"], default="yolo", help="measure-auto에서 쓸 백엔드"
    )
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold (backend=yolo)")
    parser.add_argument("--hand", choices=["left", "right"], default="right", help="촬영된 손 (자동 라벨링용)")
    parser.add_argument("--nail-height", type=float, default=0.0, help="보드~손톱 높이(mm, 시차 보정)")
    parser.add_argument(
        "--actual", nargs="+", default=None, help="compare-actual용 실제 길이. 예: thumb=12.5 index=11.0"
    )
    parser.add_argument(
        "--comparison-csv", default=DEFAULT_COMPARISON_CSV, help="백엔드별 오차 비교에 쓸 comparison.csv 경로"
    )
    return parser.parse_args()


def ask(prompt, default=None):
    """
    사용자에게 값을 물어보되, default가 있으면 Enter만 눌러 기본값을
    그대로 사용할 수 있게 한다.
    """
    suffix = f" [기본값: {default}]" if default not in (None, "") else ""
    raw = input(f"{prompt}{suffix}: ").strip()
    if raw == "" and default not in (None, ""):
        return str(default)
    return raw


def run_script(script_name, extra_args):
    """nail_measure 폴더 안의 다른 .py 스크립트를 서브프로세스로 실행한다."""
    script_path = os.path.join(BASE_DIR, script_name)
    cmd = [sys.executable, script_path] + extra_args
    print(f"\n[RUN] {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, cwd=BASE_DIR, check=False)
    except FileNotFoundError:
        print(f"[ERROR] {script_name} 실행에 실패했습니다. 파일이 존재하는지 확인하세요.")
    print()


def _default_calib_path(image_path):
    base = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join("data", "results", f"{base}_charuco.json")


# ---------------------------------------------------------------------------
# 대화형 메뉴 핸들러
# ---------------------------------------------------------------------------
def menu_camera_test(defaults):
    camera = ask("카메라 종류 (csi/usb)", defaults["camera"])
    run_script("camera_test.py", ["--camera", camera])


def menu_capture(defaults):
    camera = ask("카메라 종류 (csi/usb)", defaults["camera"])
    run_script("capture_image.py", ["--camera", camera])


def menu_calibration(defaults):
    image = ask("보정할 이미지 경로", defaults["image"])
    if not image:
        print("[ERROR] 이미지 경로가 필요합니다.")
        return
    camera_height = ask("카메라~보드 높이(mm)", str(defaults["camera_height"]))
    run_script("charuco_calibration.py", ["--image", image, "--camera-height", camera_height])


def menu_measure_manual(defaults):
    image = ask("측정할 이미지 경로", defaults["image"])
    if not image:
        print("[ERROR] 이미지 경로가 필요합니다.")
        return
    calib = ask("보정 JSON 경로", defaults["calib"] or _default_calib_path(image))
    nail_height = ask("보드~손톱 높이(mm, 시차보정)", str(defaults["nail_height"]))
    run_script(
        "manual_measure.py",
        ["--image", image, "--calib", calib, "--nail-height", nail_height],
    )


def menu_measure_auto(defaults):
    image = ask("측정할 이미지 경로", defaults["image"])
    if not image:
        print("[ERROR] 이미지 경로가 필요합니다.")
        return
    calib = ask("보정 JSON 경로", defaults["calib"] or _default_calib_path(image))
    backend = ask("백엔드 (yolo/classical/manual)", defaults["backend"])
    hand = ask("촬영된 손 (left/right)", defaults["hand"])
    nail_height = ask("보드~손톱 높이(mm, 시차보정)", str(defaults["nail_height"]))

    args_list = [
        "--image", image,
        "--calib", calib,
        "--backend", backend,
        "--hand", hand,
        "--nail-height", nail_height,
    ]
    if backend == "yolo":
        conf = ask("YOLO confidence threshold", str(defaults["conf"]))
        args_list += ["--conf", conf]
    run_script("measure_auto.py", args_list)


def menu_compare_actual(defaults):
    image = ask("비교할 이미지 경로", defaults["image"])
    if not image:
        print("[ERROR] 이미지 경로가 필요합니다.")
        return
    run_script("compare_actual.py", ["--image", image])


def menu_experiment_log(defaults):
    image = ask("촬영 조건을 기록할 이미지 경로", defaults["image"])
    if not image:
        print("[ERROR] 이미지 경로가 필요합니다.")
        return
    camera_model = ask("카메라 모델", defaults["camera_model"])
    height_cm = ask("카메라 높이(cm)", str(defaults["camera_height"] / 10))
    lighting = ask("조명 조건", "")
    focus = ask("초점 방식 (auto/fixed 등)", "")
    note = ask("비고", "")

    args_list = [
        "log",
        "--image", image,
        "--camera-model", camera_model or "unknown",
        "--height", height_cm or "0",
    ]
    if lighting:
        args_list += ["--lighting", lighting]
    if focus:
        args_list += ["--focus", focus]
    if note:
        args_list += ["--note", note]
    run_script("height_experiment.py", args_list)


def menu_experiment_summary(defaults):
    camera_model = ask("특정 카메라 모델만 요약할까요? (비우면 전체)", "")
    args_list = ["summary"]
    if camera_model:
        args_list += ["--camera-model", camera_model]
    run_script("height_experiment.py", args_list)


def menu_backend_compare(defaults):
    print_backend_comparison(defaults["comparison_csv"])


def print_menu():
    print("\n===== 손톱 길이 측정 프로그램 =====")
    print("1. 카메라 연결 테스트")
    print("2. 사진 촬영")
    print("3. ChArUco 보정")
    print("4. 손톱 측정 - 수동 클릭")
    print("5. 손톱 측정 - 자동(YOLO 등)")
    print("6. 실제 자 측정값과 비교 (사진 1장)")
    print("7. 촬영 조건(높이 등) 기록")
    print("8. 높이별 평균 오차 요약")
    print("9. 백엔드별 평균 오차 비교")
    print("0. 종료")


# ---------------------------------------------------------------------------
# 백엔드별 평균 오차 비교 (comparison.csv 전체 집계)
# ---------------------------------------------------------------------------
def print_backend_comparison(comparison_csv):
    """
    compare_actual.py가 쌓아온 comparison.csv 전체를 백엔드별로 묶어서
    평균 오차(mm)를 계산하고 출력한다.

        [비교 결과]
        manual    평균 오차: 0.32mm (n=15)
        yolo      평균 오차: 0.41mm (n=15)
        classical 평균 오차: 0.78mm (n=15)
    """
    if not os.path.exists(comparison_csv):
        print(f"[ERROR] 비교 데이터가 없습니다: {comparison_csv}")
        print("        compare_actual.py를 먼저 실행해서 오차 데이터를 쌓아주세요.")
        return

    errors_by_backend = defaultdict(list)
    with open(comparison_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            backend = row.get("backend") or "manual"
            try:
                errors_by_backend[backend].append(float(row["error_mm"]))
            except (KeyError, ValueError, TypeError):
                continue

    if not errors_by_backend:
        print(f"[ERROR] {comparison_csv}에 유효한 오차 데이터가 없습니다.")
        return

    ranked = sorted(errors_by_backend.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
    name_width = max(len(name) for name in errors_by_backend)

    print("\n[비교 결과]")
    for backend, errs in ranked:
        avg = sum(errs) / len(errs)
        print(f"{backend.ljust(name_width)}  평균 오차: {avg:.2f}mm (n={len(errs)})")


# ---------------------------------------------------------------------------
# --mode 스크립트 모드 (메뉴 없이 바로 실행)
# ---------------------------------------------------------------------------
def dispatch_mode(args):
    if args.mode == "capture":
        run_script("capture_image.py", ["--camera", args.camera])
        return

    if args.mode == "calibrate":
        if not args.image:
            print("[ERROR] --mode calibrate 에는 --image가 필요합니다.")
            return
        run_script(
            "charuco_calibration.py",
            ["--image", args.image, "--camera-height", str(args.camera_height)],
        )
        return

    if args.mode == "measure-manual":
        if not args.image:
            print("[ERROR] --mode measure-manual 에는 --image가 필요합니다.")
            return
        calib = args.calib or _default_calib_path(args.image)
        run_script(
            "manual_measure.py",
            ["--image", args.image, "--calib", calib, "--nail-height", str(args.nail_height)],
        )
        return

    if args.mode == "measure-auto":
        if not args.image:
            print("[ERROR] --mode measure-auto 에는 --image가 필요합니다.")
            return
        calib = args.calib or _default_calib_path(args.image)
        cmd = [
            "--image", args.image,
            "--calib", calib,
            "--backend", args.backend,
            "--hand", args.hand,
            "--nail-height", str(args.nail_height),
        ]
        if args.backend == "yolo":
            cmd += ["--conf", str(args.conf)]
        run_script("measure_auto.py", cmd)
        return

    if args.mode == "compare":
        print_backend_comparison(args.comparison_csv)
        return


def main():
    args = parse_args()

    if args.mode:
        dispatch_mode(args)
        return

    defaults = {
        "image": args.image,
        "calib": args.calib,
        "camera": args.camera,
        "camera_height": args.camera_height,
        "camera_model": args.camera_model,
        "backend": args.backend,
        "conf": args.conf,
        "hand": args.hand,
        "nail_height": args.nail_height,
        "comparison_csv": args.comparison_csv,
    }

    handlers = {
        "1": menu_camera_test,
        "2": menu_capture,
        "3": menu_calibration,
        "4": menu_measure_manual,
        "5": menu_measure_auto,
        "6": menu_compare_actual,
        "7": menu_experiment_log,
        "8": menu_experiment_summary,
        "9": menu_backend_compare,
    }

    while True:
        print_menu()
        choice = input("번호를 선택하세요: ").strip()
        if choice == "0":
            print("종료합니다.")
            break
        handler = handlers.get(choice)
        if handler is None:
            print("[WARN] 올바른 번호를 입력하세요.")
            continue
        try:
            handler(defaults)
        except KeyboardInterrupt:
            print("\n[INFO] 현재 작업을 취소했습니다.")


if __name__ == "__main__":
    main()
