"""
main.py
=======
[1. 역할]
    지금까지 만든 모든 기능(카메라 테스트, 촬영, 모눈 보정, 손톱 측정,
    실제값 비교, 높이별 실험 기록/요약)을 번호로 선택해서 실행할 수 있는
    통합 메뉴 프로그램이다. 각 기능은 사실 다른 .py 파일을 그대로
    실행해주는 것이며, main.py는 그 파일들을 순서대로 안내해주는 역할만 한다.

    --image, --camera, --height, --camera-model 옵션으로 자주 쓰는 값을
    미리 지정해두면, 메뉴에서 매번 다시 입력하지 않고 Enter만 눌러도
    그 값이 기본값으로 사용된다.

[2. 실행 명령어]
    python3 main.py
    python3 main.py --image data/captured/sample.jpg --camera usb \\
        --height 20 --camera-model "Arducam 16MP AF"

[3. 어디에 입력해야 하는가]
    -> Jetson Nano 터미널(SSH 접속)에서 실행한다.
    -> 메뉴 자체는 화면(GUI) 없이도 뜨지만, 촬영/모눈 보정/손톱 측정처럼
       마우스 클릭이 필요한 기능을 선택하면 그 순간에는 화면(GUI)이 필요하다.

[4. 정상적으로 실행되면]
    아래와 같은 번호 메뉴가 터미널에 출력된다.

        ===== 손톱 길이 측정 프로그램 =====
        1. 카메라 연결 테스트
        2. 사진 촬영
        3. 모눈(0.5mm) 보정
        4. 손톱 길이 측정 (5개 손가락)
        5. 실제 자 측정값과 비교
        6. 촬영 조건(높이 등) 기록
        7. 높이별 평균 오차 요약
        0. 종료
        번호를 선택하세요:

    번호를 입력하면 해당 스크립트가 실행되고, 끝나면 다시 메뉴로 돌아온다.

[5. 오류가 발생하면 확인할 것]
    - 각 기능 실행 중 나는 오류는 해당 스크립트 파일
      (camera_test.py, capture_image.py 등) 상단 설명의
      '오류가 발생하면 확인할 것' 항목을 참고한다.
    - "python3: command not found" 같은 오류가 main.py 실행 자체에서
      나면, Jetson Nano에 Python3가 설치되어 있는지 확인한다.
      (JetPack 기본 이미지에는 보통 이미 설치되어 있음)
"""

import argparse
import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="손톱 길이 측정 프로그램 - 통합 메뉴")
    parser.add_argument("--image", default=None, help="기본으로 사용할 이미지 경로")
    parser.add_argument(
        "--camera", choices=["csi", "usb"], default="csi", help="기본 카메라 종류"
    )
    parser.add_argument("--height", type=float, default=None, help="기본 카메라 높이(cm)")
    parser.add_argument("--camera-model", default=None, help="기본 카메라 모델 이름")
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
    cells = ask("모눈 칸 수", "10")
    cell_size_mm = ask("모눈 한 칸 크기(mm)", "5.0")
    run_script(
        "grid_calibration.py",
        ["--image", image, "--cells", cells, "--cell-size-mm", cell_size_mm],
    )


def menu_measure(defaults):
    image = ask("측정할 이미지 경로", defaults["image"])
    if not image:
        print("[ERROR] 이미지 경로가 필요합니다.")
        return
    default_calib = _default_calib_path(image)
    calib = ask("보정 JSON 경로", default_calib)
    run_script("manual_measure.py", ["--image", image, "--calib", calib])


def menu_compare(defaults):
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
    height = ask("카메라 높이(cm)", defaults["height"])
    lighting = ask("조명 조건", "")
    focus = ask("초점 방식 (auto/fixed 등)", "")
    note = ask("비고", "")

    args_list = [
        "log",
        "--image", image,
        "--camera-model", camera_model or "unknown",
        "--height", height or "0",
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


def _default_calib_path(image_path):
    base = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join("data", "results", f"{base}_calib.json")


def print_menu():
    print("\n===== 손톱 길이 측정 프로그램 =====")
    print("1. 카메라 연결 테스트")
    print("2. 사진 촬영")
    print("3. 모눈(0.5mm) 보정")
    print("4. 손톱 길이 측정 (5개 손가락)")
    print("5. 실제 자 측정값과 비교")
    print("6. 촬영 조건(높이 등) 기록")
    print("7. 높이별 평균 오차 요약")
    print("0. 종료")


def main():
    args = parse_args()
    defaults = {
        "image": args.image,
        "camera": args.camera,
        "height": args.height,
        "camera_model": args.camera_model,
    }

    handlers = {
        "1": menu_camera_test,
        "2": menu_capture,
        "3": menu_calibration,
        "4": menu_measure,
        "5": menu_compare,
        "6": menu_experiment_log,
        "7": menu_experiment_summary,
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
