# -*- coding: utf-8 -*-
"""
generate_board.py
===================
[1. 역할]
    config.yaml의 charuco_board 스펙으로 ChArUco 보드 이미지를 생성해서
    PNG로 저장한다. 이 이미지를 인쇄해서 실제 캘리브레이션/스케일 보드로 쓴다.
    정면용, 측면용 보드는 스펙이 같으므로 이 스크립트로 2장 인쇄해서
    하나는 바닥에, 하나는 측면 카메라 옆에 수직으로 세워두면 된다.

[2. 실행 명령어]
    python3 generate_board.py --output charuco_board.png
    python3 generate_board.py --output charuco_board.png --dpi 300

[3. 어디에서 실행하는가]
    노트북/PC 어디서든 실행 가능하다 (인쇄할 컴퓨터에서 바로 실행하는 게 편하다).

[4. 정상적으로 실행되면]
    아래처럼 출력되고, --output 경로에 흰 배경의 체커보드+ArUco 마커 이미지가 저장된다.

        [SAVED] charuco_board.png (2000x1500 px, DPI 300 기준 실제 크기 약 125.0mm x 95.0mm)
        [INFO] 보드 스펙: 7 x 5 칸, 한 칸 15.0mm, 마커 11.0mm, 딕셔너리 DICT_4X4_50
        [중요] 프린터 설정에서 '실제 크기(100%)'로 인쇄하고, 인쇄 후 자로 한 칸 크기를
               재서 config.yaml의 square_length_mm과 일치하는지 반드시 확인하세요.

    인쇄 시 프린터 설정에서 "실제 크기(100%)" 또는 "맞춤 안 함"으로 인쇄해야
    한다. "페이지에 맞추기"로 인쇄하면 실제 크기가 달라져서 모든 mm 측정값이
    통째로 틀어진다.

[5. 오류가 발생하면 확인할 것]
    - "cv2.aruco 모듈이 없습니다": opencv-contrib-python 설치 필요 (README 참고)
    - 인쇄물 칸 크기가 config.yaml의 square_length_mm과 다르면, 실측한
      값을 config.yaml에 반영하거나 calibrate 실행 시 --square-mm로 넘긴다.
"""

import argparse
import os

import cv2

import calibration
from config_loader import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="ChArUco 보드 인쇄용 이미지 생성")
    parser.add_argument("--output", default="charuco_board.png", help="저장할 PNG 경로")
    parser.add_argument("--dpi", type=int, default=300, help="인쇄 DPI (실제 크기 환산용, 기본 300)")
    parser.add_argument("--config", default=None, help="config.yaml 경로 (생략 시 기본값)")
    parser.add_argument("--margin-mm", type=float, default=10.0, help="보드 바깥 흰 여백 (mm)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    board_cfg = config["charuco_board"]

    calibration.check_aruco_available()
    aruco_dict = calibration.get_dictionary(board_cfg["dictionary"])
    board = calibration.make_board(board_cfg, aruco_dict)

    mm_per_inch = 25.4
    px_per_mm = args.dpi / mm_per_inch

    board_w_mm = board_cfg["squares_x"] * board_cfg["square_length_mm"]
    board_h_mm = board_cfg["squares_y"] * board_cfg["square_length_mm"]
    board_w_px = int(round(board_w_mm * px_per_mm))
    board_h_px = int(round(board_h_mm * px_per_mm))

    if hasattr(board, "generateImage"):
        board_img = board.generateImage((board_w_px, board_h_px))
    else:
        board_img = board.draw((board_w_px, board_h_px))

    margin_px = int(round(args.margin_mm * px_per_mm))
    canvas = cv2.copyMakeBorder(
        board_img, margin_px, margin_px, margin_px, margin_px, cv2.BORDER_CONSTANT, value=255
    )

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(args.output, canvas)

    total_w_mm = board_w_mm + args.margin_mm * 2
    total_h_mm = board_h_mm + args.margin_mm * 2
    print(
        "[SAVED] %s (%dx%d px, DPI %d 기준 실제 크기 약 %.1fmm x %.1fmm)"
        % (args.output, canvas.shape[1], canvas.shape[0], args.dpi, total_w_mm, total_h_mm)
    )
    print(
        "[INFO] 보드 스펙: %d x %d 칸, 한 칸 %.1fmm, 마커 %.1fmm, 딕셔너리 %s"
        % (
            board_cfg["squares_x"],
            board_cfg["squares_y"],
            board_cfg["square_length_mm"],
            board_cfg["marker_length_mm"],
            board_cfg["dictionary"],
        )
    )
    print("[중요] 프린터 설정에서 '실제 크기(100%)'로 인쇄하고, 인쇄 후 자로 한 칸 크기를")
    print("       재서 config.yaml의 square_length_mm과 일치하는지 반드시 확인하세요.")


if __name__ == "__main__":
    main()
