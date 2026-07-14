# -*- coding: utf-8 -*-
"""
main.py
========
[1. 역할]
    손톱 유형 진단 프로그램의 CLI 엔트리. 4개 서브커맨드를 제공한다.
        calibrate : ChArUco 보드로 정면/측면 카메라 캘리브레이션 (1회 수행)
        measure   : 손가락 10개를 순서대로 촬영/측정하는 실전 워크플로우
        analyze   : 이미 찍어둔 정면/측면 이미지 한 쌍으로 단발성 분석 (개발/검증용)
        collect   : 향후 AI 모델 학습용 원본 이미지/마스크/키포인트 데이터 수집

    calibration.py(ChArUco/스케일), camera.py(듀얼 CSI/정지이미지),
    detection.base/contour(마스크->키포인트), analysis.*(바디기장/유형/곡면길이/
    팁매칭), visualize.py(오버레이), collect.py(데이터 저장)를 모두 여기서 엮는다.

[2. 실행 명령어 예시]
    python3 main.py calibrate --camera front
    python3 main.py calibrate --camera side --images board1.jpg board2.jpg board3.jpg
    python3 main.py measure
    python3 main.py measure --save-data
    python3 main.py measure --front-image sample_front.jpg --side-image sample_side.jpg
    python3 main.py analyze --front finger_front.jpg --side finger_side.jpg
    python3 main.py collect --session test01

[3. 어디에서 실행하는가]
    Jetson Nano B01(실제 듀얼 CSI 카메라)에서 카메라 모드로 실행하는 것이 기본이다.
    카메라가 없는 노트북/PC에서는 --front-image/--side-image(measure) 또는
    --front/--side(analyze)로 정지 이미지를 넣어 로직만 검증할 수 있다.

[4. 정상적으로 실행되면]
    calibrate: calib_front.json 또는 calib_side.json이 저장되고 재투영 오차(rms)가 출력된다.
    measure  : 손가락마다 촬영 -> 결과 오버레이 확인 -> 확정/재촬영을 반복하고,
               끝나면 data/results/measure_YYYYMMDD_HHMMSS.json 요약이 저장된다.
    analyze  : 결과 요약이 콘솔에 출력되고, 오버레이 이미지(.jpg)와 결과(.json)가
               data/results/에 저장된다.
    collect  : data/collect/session_.../ 아래에 이미지/마스크/메타데이터가 쌓인다.

[5. 오류가 발생하면 확인할 것]
    - "calib_*.json이 필요합니다": calibrate를 먼저 실행했는지 확인.
    - "카메라를 열 수 없습니다": camera.py의 안내 메시지(케이블/장치/점유 프로세스) 확인,
      카메라가 없다면 정지 이미지 모드로 대체.
    - "cv2.aruco 모듈이 없습니다": requirements.txt의 opencv-contrib-python 계열 설치 확인.
    - 측정 결과가 이상하면: 보드가 실시간으로 보이는 상태였는지(화면 좌상단
      "SCALE: live" 표시) 확인. "held"/"fallback"이면 스케일 오차가 커질 수 있다.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np

import calibration
import camera
import collect
import visualize
from config_loader import load_config, config_snapshot
from detection.base import KeypointDetector
from detection.contour import ThresholdNailSegmenter, manual_adjust_keypoints, save_debug_images
from analysis.body_length import classify_body_length
from analysis.nail_type import classify_nail_type
from analysis.curve import compute_curve_length_for_type
from analysis.tip_match import match_tip_size


# ---------------------------------------------------------------------------
# 공용 헬퍼
# ---------------------------------------------------------------------------
def _default_front_keypoints(image_shape):
    """자동 검출이 완전히 실패했을 때, 수동 보정 UI에 띄울 초기 위치(대략 중앙 부근)."""
    h, w = image_shape[:2]
    cx, cy = w / 2.0, h / 2.0
    return {
        "1": (cx, cy + h * 0.30),
        "5": (cx, cy - h * 0.30),
        "3": (cx - w * 0.25, cy),
        "7": (cx + w * 0.25, cy),
        "9": (cx, cy),
        "2": (cx - w * 0.15, cy + h * 0.15),
        "4": (cx - w * 0.15, cy - h * 0.15),
        "6": (cx + w * 0.15, cy - h * 0.15),
        "8": (cx + w * 0.15, cy + h * 0.15),
    }


def _default_side_keypoints(image_shape):
    h, w = image_shape[:2]
    cx, cy = w / 2.0, h / 2.0
    return {
        "1": (cx - w * 0.30, cy),
        "3": (cx - w * 0.05, cy - h * 0.20),
        "9": (cx - w * 0.05, cy + h * 0.20),
        "5": (cx + w * 0.30, cy - h * 0.10),
        "10": (cx + w * 0.30, cy + h * 0.10),
        "4": (cx + w * 0.10, cy - h * 0.15),
        "2": (cx + w * 0.10, cy + h * 0.15),
    }


def _to_jsonable(value):
    """dict/list/tuple/numpy 스칼라·배열을 표준 json으로 저장 가능한 형태로 재귀 변환."""
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    return value


def _result_to_json_dict(result):
    """_analyze_finger() 결과에서 이미지/마스크(numpy array, 용량 큼)를 뺀 저장용 dict."""
    return _to_jsonable(
        {
            "front_keypoints_px": {k: list(v) for k, v in result["front_keypoints_px"].items()},
            "side_keypoints_px": {k: list(v) for k, v in result["side_keypoints_px"].items()},
            "front_keypoints_mm": {k: list(v) for k, v in result["front_keypoints_mm"].items()},
            "side_keypoints_mm": {k: list(v) for k, v in result["side_keypoints_mm"].items()},
            "w_front_mm": result["w_front_mm"],
            "h_front_mm": result["h_front_mm"],
            "w_side_mm": result["w_side_mm"],
            "body_length": result["body_length"],
            "nail_type": result["nail_type"],
            "curve": result["curve"],
            "curve_length_mm": result["curve_length_mm"],
            "tip_match": result["tip_match"],
            "auto_detected": result["auto_detected"],
        }
    )


def _analyze_finger(front_bgr, side_bgr, front_homography, side_homography, calib_front, calib_side, detector, config, manual=False):
    """
    손가락 한 개(정면+측면 원본 이미지 쌍)를 왜곡보정 -> 키포인트 검출(+필요 시 수동보정)
    -> mm 변환 -> 바디기장/유형/곡면길이/팁매칭까지 한 번에 계산한다.

    front_homography/side_homography가 None이면(스케일을 전혀 모름) 계산이
    불가능하므로 RuntimeError를 낸다 (호출부가 재촬영을 유도해야 한다).
    """
    if front_homography is None or side_homography is None:
        raise RuntimeError(
            "mm 스케일을 알 수 없습니다 (ChArUco 보드가 검출되지 않았고 저장된 캘리브레이션도 없습니다).\n"
            "보드가 화면에 보이는 상태에서 다시 촬영하거나, 먼저 calibrate를 실행하세요."
        )

    front_undist = calibration.undistort(front_bgr, calib_front)
    side_undist = calibration.undistort(side_bgr, calib_side)

    front_det = detector.detect(front_undist, "front")
    side_det = detector.detect(side_undist, "side")

    front_kp_px = front_det.keypoints_px if front_det.keypoints_px else _default_front_keypoints(front_undist.shape)
    side_kp_px = side_det.keypoints_px if side_det.keypoints_px else _default_side_keypoints(side_undist.shape)

    need_manual = manual or not front_det.auto_detected or not side_det.auto_detected
    if need_manual:
        front_kp_px = manual_adjust_keypoints(front_undist, front_kp_px, "front", window_name="Manual Adjust - Front")
        side_kp_px = manual_adjust_keypoints(side_undist, side_kp_px, "side", window_name="Manual Adjust - Side")

    front_kp_mm = {k: tuple(calibration.transform_points_to_mm(front_homography, [v])[0]) for k, v in front_kp_px.items()}
    side_kp_mm = {k: tuple(calibration.transform_points_to_mm(side_homography, [v])[0]) for k, v in side_kp_px.items()}

    w_front_mm = calibration.mm_distance(front_homography, front_kp_px["3"], front_kp_px["7"])
    h_front_mm = calibration.mm_distance(front_homography, front_kp_px["1"], front_kp_px["5"])
    w_side_mm = calibration.mm_distance(side_homography, side_kp_px["3"], side_kp_px["9"])

    body = classify_body_length(h_front_mm, w_front_mm, config)
    ntype = classify_nail_type(front_kp_mm, side_kp_mm, w_side_mm, config)
    curve_result = compute_curve_length_for_type(w_front_mm, w_side_mm, ntype["final_type"], config["curve_constants"])
    tip = match_tip_size(curve_result["curve_length_mm"], ntype["final_type"], config)

    return {
        "front_image": front_undist,
        "side_image": side_undist,
        "front_mask": front_det.mask,
        "side_mask": side_det.mask,
        "front_keypoints_px": front_kp_px,
        "side_keypoints_px": side_kp_px,
        "front_keypoints_mm": front_kp_mm,
        "side_keypoints_mm": side_kp_mm,
        "w_front_mm": w_front_mm,
        "h_front_mm": h_front_mm,
        "w_side_mm": w_side_mm,
        "body_length": body,
        "nail_type": ntype,
        "curve": curve_result,
        "curve_length_mm": curve_result["curve_length_mm"],
        "tip_match": tip,
        "auto_detected": not need_manual,
    }


def _build_overlay(result, config):
    """확인 화면용: 정면/측면에 키포인트+측정선+결과 텍스트(영문)를 그려서 좌우로 이어붙인다."""
    front_overlay = visualize.draw_front_measurements(result["front_image"], result["front_keypoints_px"])
    side_overlay = visualize.draw_side_measurements(result["side_image"], result["side_keypoints_px"])
    lines = visualize.build_result_lines(result)
    front_overlay = visualize.draw_result_text(front_overlay, lines)
    front_prev = camera.downscale_for_preview(front_overlay, config)
    side_prev = camera.downscale_for_preview(side_overlay, config)
    return visualize.stack_front_side(front_prev, side_prev)


def _open_single_camera(view, config):
    cam_cfg = config["camera"]
    sensor_id = cam_cfg[view]["sensor_id"]
    pipeline = camera.gstreamer_pipeline(
        sensor_id=sensor_id,
        capture_width=cam_cfg["capture_width"],
        capture_height=cam_cfg["capture_height"],
        display_width=cam_cfg["capture_width"],
        display_height=cam_cfg["capture_height"],
        framerate=cam_cfg["framerate"],
        flip_method=cam_cfg["flip_method"],
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(
            "카메라(%s, sensor-id=%d)를 열 수 없습니다. camera.py의 DualCameraCapture 오류 메시지를 "
            "참고해 CSI 케이블/장치를 확인하거나, --images로 정지 이미지 모드를 쓰세요." % (view, sensor_id)
        )
    return cap


def _capture_calibration_images_live(view, config, num_images):
    """카메라로 보드 사진을 여러 장 촬영해서 그레이스케일 리스트로 반환한다."""
    cap = _open_single_camera(view, config)
    board_cfg = config["charuco_board"]
    grays = []
    print("[안내] 스페이스바: 현재 프레임 캡처 | Enter: 캡처 종료(최소 2장) | q: 취소")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[오류] 프레임을 읽지 못했습니다.")
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detection = calibration.detect_board(gray, board_cfg)
            status = "board: %d markers" % detection.num_markers if detection is not None else "board: not found"

            preview = camera.downscale_for_preview(frame, config)
            cv2.putText(
                preview,
                "%s | captured %d/%d (min 2)" % (status, len(grays), num_images),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("nail_analysis - calibrate (%s)" % view, preview)
            key = cv2.waitKey(20) & 0xFF
            if key == ord(" "):
                grays.append(gray)
                print("[캡처] %d/%d" % (len(grays), num_images))
                if len(grays) >= num_images:
                    break
            elif key in (13, ord("n")) and len(grays) >= 2:
                break
            elif key == ord("q"):
                raise RuntimeError("사용자가 캘리브레이션 촬영을 취소했습니다.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(grays) < 2:
        raise RuntimeError("캡처된 사진이 %d장뿐입니다 (최소 2장 필요)." % len(grays))
    return grays, grays[-1]


def _capture_loop(cap, front_scale, side_scale, config, finger_id, idx, total, labels_ko):
    """
    measure/collect가 공유하는 촬영 대기 루프. 매 프레임 실시간 스케일을
    갱신해서 화면에 SCALE 상태를 보여주고, 사용자 입력에 따라 결과를 반환한다.

    반환: (action, front_frame, side_frame, front_homography, side_homography,
            front_status, side_status)
        action: "captured" / "skip" / "quit"
    """
    label = labels_ko.get(finger_id, finger_id)
    while True:
        frame_f, frame_s = cap.read()
        if frame_f is None or frame_s is None:
            print("[오류] 카메라 프레임을 읽지 못했습니다.")
            return "quit", None, None, None, None, None, None

        gray_f = cv2.cvtColor(frame_f, cv2.COLOR_BGR2GRAY)
        gray_s = cv2.cvtColor(frame_s, cv2.COLOR_BGR2GRAY)
        h_f, status_f, _ = front_scale.update(gray_f)
        h_s, status_s, _ = side_scale.update(gray_s)

        prev_f = visualize.draw_scale_status(camera.downscale_for_preview(frame_f, config), status_f)
        prev_s = visualize.draw_scale_status(camera.downscale_for_preview(frame_s, config), status_s)
        combo = visualize.stack_front_side(prev_f, prev_s)
        cv2.putText(
            combo,
            "[%d/%d] %s (%s) - SPACE capture | s skip | q quit" % (idx + 1, total, finger_id, label),
            (10, combo.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
        cv2.imshow("nail_analysis - measure", combo)
        key = cv2.waitKey(20) & 0xFF
        if key == ord(" "):
            return "captured", frame_f.copy(), frame_s.copy(), h_f, h_s, status_f, status_s
        if key == ord("s"):
            return "skip", None, None, None, None, None, None
        if key == ord("q"):
            return "quit", None, None, None, None, None, None


def _wait_confirm(combo_image, window_name="nail_analysis - result"):
    cv2.imshow(window_name, combo_image)
    print("[안내] Enter/c: 확정 | r: 다시 촬영 | s: 이 손가락 건너뛰기 | q: 측정 중단")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, ord("c")):
            action = "confirm"
        elif key == ord("r"):
            action = "retake"
        elif key == ord("s"):
            action = "skip"
        elif key == ord("q"):
            action = "quit"
        else:
            continue
        cv2.destroyWindow(window_name)
        return action


def _save_measure_summary(results, config):
    if not results:
        print("[안내] 확정된 측정 결과가 없어 요약 파일을 저장하지 않습니다.")
        return None

    out_dir = config["paths"]["results_dir"]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": timestamp,
        "fingers": {fid: _result_to_json_dict(r) for fid, r in results.items()},
        "config_snapshot": config_snapshot(config),
    }
    path = os.path.join(out_dir, "measure_%s.json" % timestamp)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(summary), f, ensure_ascii=False, indent=2)

    print("\n[SAVED] 전체 측정 요약: %s (%d개 손가락)" % (path, len(results)))
    print("%-10s %-6s %-8s %-10s %-8s" % ("Finger", "Type", "Body", "Curve(mm)", "Tip"))
    for fid in config["fingers"]["order"]:
        if fid not in results:
            continue
        r = results[fid]
        print(
            "%-10s %-6s %-8s %-10.2f %-8s"
            % (fid, r["nail_type"]["final_type"], r["body_length"]["label"], r["curve_length_mm"], r["tip_match"]["size_label"])
        )
    return path


# ---------------------------------------------------------------------------
# 서브커맨드 구현
# ---------------------------------------------------------------------------
def cmd_calibrate(args, config):
    board_cfg = config["charuco_board"]
    view = args.camera
    default_path_key = "calib_front_path" if view == "front" else "calib_side_path"
    output_path = args.output or config["paths"][default_path_key]

    if args.images:
        gray_images = []
        for path in args.images:
            img = cv2.imread(path)
            if img is None:
                print("[경고] 이미지를 열 수 없어 건너뜁니다: %s" % path)
                continue
            gray_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if len(gray_images) < 2:
            raise RuntimeError("유효한 보드 사진이 %d장뿐입니다. --images로 2장 이상 지정하세요." % len(gray_images))
        reference_gray = gray_images[-1]
    else:
        gray_images, reference_gray = _capture_calibration_images_live(view, config, args.num_images)

    intrinsics = calibration.calibrate_intrinsics(gray_images, board_cfg)
    reference_detection = calibration.detect_board(reference_gray, board_cfg)
    if reference_detection is None:
        raise RuntimeError("기준 이미지(마지막 촬영 사진)에서 보드를 검출하지 못했습니다. 보드가 잘 보이는 사진으로 다시 시도하세요.")

    calibration.save_calibration(output_path, view, intrinsics, reference_detection, board_cfg, config)

    print("[SAVED] %s" % output_path)
    print(
        "[INFO] intrinsics rms=%.4f (사용 이미지 %d장), reference rms_mm=%.4f"
        % (intrinsics["rms"], intrinsics["num_images_used"], reference_detection.rms_mm)
    )
    warn_mm = config["calibration"]["reprojection_rms_warn_mm"]
    if reference_detection.rms_mm > warn_mm:
        print(
            "[경고] 기준 이미지 재투영 오차(%.3fmm)가 임계값(%.3fmm)보다 큽니다. "
            "보드가 평평하고 화면에 크게 잘 보이는 사진으로 다시 캘리브레이션하는 것을 권장합니다."
            % (reference_detection.rms_mm, warn_mm)
        )


def cmd_measure(args, config):
    calib_front = calibration.load_calibration(config["paths"]["calib_front_path"])
    calib_side = calibration.load_calibration(config["paths"]["calib_side_path"])
    if calib_front is None or calib_side is None:
        print(
            "[경고] 저장된 캘리브레이션이 없습니다. 'calibrate --camera front'/'--camera side'를 "
            "먼저 실행하는 것을 권장합니다. 지금은 실시간 보드 검출에만 의존해서 진행합니다."
        )

    board_cfg = config["charuco_board"]
    front_scale = calibration.ScaleProvider(calib_front, board_cfg, config, camera_view="front")
    side_scale = calibration.ScaleProvider(calib_side, board_cfg, config, camera_view="side")

    segmenter = ThresholdNailSegmenter(config, debug=args.debug)
    detector = KeypointDetector(segmenter, config)

    cap = camera.open_dual_camera(config, front_image=args.front_image, side_image=args.side_image)
    session = collect.CollectSession(config) if args.save_data else None

    fingers_order = config["fingers"]["order"]
    labels_ko = config["fingers"]["labels_ko"]
    results = {}

    try:
        idx = 0
        while idx < len(fingers_order):
            finger_id = fingers_order[idx]
            print("\n=== [%d/%d] %s (%s) ===" % (idx + 1, len(fingers_order), finger_id, labels_ko.get(finger_id, "")))

            action = None
            result = None
            while action is None:
                step, cf, cs, chf, chs, _sf, _ss = _capture_loop(
                    cap, front_scale, side_scale, config, finger_id, idx, len(fingers_order), labels_ko
                )
                if step == "skip":
                    action = "skipped"
                    break
                if step == "quit":
                    action = "quit"
                    break

                try:
                    result = _analyze_finger(cf, cs, chf, chs, calib_front, calib_side, detector, config, manual=args.manual)
                except Exception as e:
                    print("[오류] 측정 계산 실패: %s\n다시 촬영합니다." % e)
                    continue

                result["finger_id"] = finger_id
                overlay = _build_overlay(result, config)
                confirm = _wait_confirm(overlay)
                if confirm == "confirm":
                    action = "confirmed"
                elif confirm == "retake":
                    continue
                elif confirm == "skip":
                    action = "skipped"
                elif confirm == "quit":
                    action = "quit"

            if action == "confirmed":
                results[finger_id] = result
                print(
                    "[확정] %s: Type=%s Body=%s Curve=%.2fmm Tip=%s"
                    % (
                        finger_id,
                        result["nail_type"]["final_type"],
                        result["body_length"]["label"],
                        result["curve_length_mm"],
                        result["tip_match"]["size_label"],
                    )
                )
                if session is not None:
                    session.save_finger(
                        finger_id,
                        result["front_image"],
                        result["side_image"],
                        front_mask=result["front_mask"],
                        side_mask=result["side_mask"],
                        front_keypoints_px=result["front_keypoints_px"],
                        side_keypoints_px=result["side_keypoints_px"],
                        meta_extra=_result_to_json_dict(result),
                    )
            elif action == "skipped":
                print("[건너뜀] %s" % finger_id)
            elif action == "quit":
                print("[중단] 사용자가 측정을 중단했습니다.")
                break

            idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    _save_measure_summary(results, config)


def cmd_analyze(args, config):
    calib_front = calibration.load_calibration(config["paths"]["calib_front_path"])
    calib_side = calibration.load_calibration(config["paths"]["calib_side_path"])
    if (
        not calib_front
        or not calib_side
        or "reference_homography" not in calib_front
        or "reference_homography" not in calib_side
    ):
        raise RuntimeError(
            "calib_front.json / calib_side.json이 없거나 불완전합니다.\n"
            "먼저 'python main.py calibrate --camera front'와 '--camera side'를 실행하세요."
        )

    front_homography = np.array(calib_front["reference_homography"], dtype=np.float64)
    side_homography = np.array(calib_side["reference_homography"], dtype=np.float64)

    still = camera.StillImageCapture(args.front, args.side)
    frame_f, frame_s = still.read()

    segmenter = ThresholdNailSegmenter(config, debug=args.debug)
    detector = KeypointDetector(segmenter, config)

    result = _analyze_finger(
        frame_f, frame_s, front_homography, side_homography, calib_front, calib_side, detector, config, manual=args.manual
    )
    finger_id = args.finger_id or ("analyze_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    result["finger_id"] = finger_id

    print(
        "[결과] %s: Body=%s(R=%.2f) Type=%s(conf=%.2f) Curve=%.2fmm Tip=%s (diff %.2fmm)"
        % (
            finger_id,
            result["body_length"]["label"],
            result["body_length"]["ratio"],
            result["nail_type"]["final_type"],
            result["nail_type"]["confidence"],
            result["curve_length_mm"],
            result["tip_match"]["size_label"],
            result["tip_match"]["diff_mm"],
        )
    )

    out_dir = config["paths"]["results_dir"]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if args.debug:
        saved = save_debug_images(segmenter.debug_images, config["paths"]["debug_dir"], finger_id)
        if saved:
            print("[SAVED] 디버그 이미지 %d장: %s" % (len(saved), config["paths"]["debug_dir"]))

    overlay = _build_overlay(result, config)
    overlay_path = os.path.join(out_dir, "%s_overlay.jpg" % finger_id)
    cv2.imwrite(overlay_path, overlay)

    json_path = args.output_json or os.path.join(out_dir, "%s.json" % finger_id)
    payload = {"finger_id": finger_id, "result": _result_to_json_dict(result), "config_snapshot": config_snapshot(config)}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2)

    print("[SAVED] %s" % json_path)
    print("[SAVED] %s" % overlay_path)

    if not args.no_display:
        cv2.imshow("nail_analysis - analyze result", overlay)
        print("[안내] 아무 키나 누르면 종료합니다.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def cmd_collect(args, config):
    segmenter = ThresholdNailSegmenter(config, debug=args.debug)
    detector = KeypointDetector(segmenter, config)
    cap = camera.open_dual_camera(config, front_image=args.front_image, side_image=args.side_image)
    session = collect.CollectSession(config, session_name=args.session)

    board_cfg = config["charuco_board"]
    calib_front = calibration.load_calibration(config["paths"]["calib_front_path"])
    calib_side = calibration.load_calibration(config["paths"]["calib_side_path"])
    front_scale = calibration.ScaleProvider(calib_front, board_cfg, config, camera_view="front")
    side_scale = calibration.ScaleProvider(calib_side, board_cfg, config, camera_view="side")

    fingers_order = config["fingers"]["order"]
    labels_ko = config["fingers"]["labels_ko"]

    print("[안내] collect 모드: 손톱 사진(정면/측면)과 (가능하면) 자동 마스크/키포인트를 저장합니다.")
    print("       마스크/키포인트 검출이 실패해도 원본 이미지는 저장됩니다 (추후 SAM 라벨링용).")

    try:
        idx = 0
        while idx < len(fingers_order):
            finger_id = fingers_order[idx]
            print("\n=== [%d/%d] %s (%s) ===" % (idx + 1, len(fingers_order), finger_id, labels_ko.get(finger_id, "")))

            step, frame_f, frame_s, _hf, _hs, _sf, _ss = _capture_loop(
                cap, front_scale, side_scale, config, finger_id, idx, len(fingers_order), labels_ko
            )
            if step == "quit":
                print("[중단] 사용자가 데이터 수집을 중단했습니다.")
                break
            if step == "skip":
                print("[건너뜀] %s" % finger_id)
                idx += 1
                continue

            front_mask = side_mask = None
            front_kp_px = side_kp_px = None
            try:
                front_det = detector.detect(frame_f, "front")
                side_det = detector.detect(frame_s, "side")
                front_mask, side_mask = front_det.mask, side_det.mask
                front_kp_px = front_det.keypoints_px or None
                side_kp_px = side_det.keypoints_px or None
            except Exception as e:
                print("[경고] 자동 검출 실패(원본 이미지만 저장합니다): %s" % e)

            paths = session.save_finger(
                finger_id,
                frame_f,
                frame_s,
                front_mask=front_mask,
                side_mask=side_mask,
                front_keypoints_px=front_kp_px,
                side_keypoints_px=side_kp_px,
            )
            print("[저장] %s" % ", ".join(paths.values()))
            idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\n[완료] 세션 %s 에 %d개 손가락 저장됨." % (session.session_dir, len(session.list_saved_fingers())))


# ---------------------------------------------------------------------------
# CLI 파서 / 진입점
# ---------------------------------------------------------------------------
def build_arg_parser():
    parser = argparse.ArgumentParser(description="손톱 유형 진단 프로그램 (Jetson Nano B01 + 듀얼 CSI 카메라)")
    parser.add_argument("--config", default=None, help="config.yaml 경로 (생략 시 기본값)")

    sub = parser.add_subparsers(dest="command")
    sub.required = True  # Python 3.6에는 add_subparsers(required=True) 인자가 없어 수동으로 설정

    p_cal = sub.add_parser("calibrate", help="ChArUco 보드로 카메라 캘리브레이션")
    p_cal.add_argument("--camera", choices=["front", "side"], required=True)
    p_cal.add_argument("--num-images", type=int, default=15, help="라이브 촬영 시 캡처할 장수 (기본 15)")
    p_cal.add_argument("--images", nargs="+", default=None, help="정지 이미지 모드: 보드 사진 파일 경로 여러 장")
    p_cal.add_argument("--output", default=None, help="저장 경로 (생략 시 config.yaml의 paths.calib_*_path)")

    p_measure = sub.add_parser("measure", help="손가락 10개를 순서대로 촬영/측정")
    p_measure.add_argument("--save-data", action="store_true", help="확정된 측정마다 collect.py로 학습용 원본도 저장")
    p_measure.add_argument("--manual", action="store_true", help="자동 검출 성공 여부와 무관하게 항상 수동 보정 UI를 띄움")
    p_measure.add_argument("--debug", action="store_true", help="세그멘테이션 중간 단계 이미지를 저장")
    p_measure.add_argument("--front-image", default=None, help="정지 이미지 모드(개발용): 정면 이미지 경로")
    p_measure.add_argument("--side-image", default=None, help="정지 이미지 모드(개발용): 측면 이미지 경로")

    p_analyze = sub.add_parser("analyze", help="정지 이미지 한 쌍으로 단발성 분석 (카메라 없이 검증용)")
    p_analyze.add_argument("--front", required=True, help="정면 이미지 경로")
    p_analyze.add_argument("--side", required=True, help="측면 이미지 경로")
    p_analyze.add_argument("--finger-id", default=None, help="결과 파일명에 쓸 손가락 식별자 (생략 시 타임스탬프)")
    p_analyze.add_argument("--manual", action="store_true")
    p_analyze.add_argument("--debug", action="store_true")
    p_analyze.add_argument("--no-display", action="store_true", help="결과 창을 띄우지 않음 (헤드리스 환경용)")
    p_analyze.add_argument("--output-json", default=None, help="결과 JSON 저장 경로 (생략 시 data/results/<finger-id>.json)")

    p_collect = sub.add_parser("collect", help="향후 AI 모델 학습용 원본 이미지/마스크/키포인트 데이터 수집")
    p_collect.add_argument("--session", default=None, help="세션 이름 (생략 시 session_YYYYMMDD_HHMMSS)")
    p_collect.add_argument("--debug", action="store_true")
    p_collect.add_argument("--front-image", default=None, help="정지 이미지 모드(개발용): 정면 이미지 경로")
    p_collect.add_argument("--side-image", default=None, help="정지 이미지 모드(개발용): 측면 이미지 경로")

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    try:
        if args.command == "calibrate":
            cmd_calibrate(args, config)
        elif args.command == "measure":
            cmd_measure(args, config)
        elif args.command == "analyze":
            cmd_analyze(args, config)
        elif args.command == "collect":
            cmd_collect(args, config)
    except (RuntimeError, ValueError) as e:
        print("\n[오류] %s" % e)
        sys.exit(1)


if __name__ == "__main__":
    main()
