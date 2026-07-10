"""
charuco_calibration.py
=======================
[1. 역할]
    ChArUco 보드(체스보드 + ArUco 마커 조합 보정판)가 촬영된 사진에서
    보드를 자동으로 검출하고, 이미지 픽셀 좌표 <-> 보드 mm 좌표 사이의
    호모그래피(homography) 행렬을 계산한다.

    v1의 grid_calibration.py(모눈 두 점 클릭 -> 스칼라 px_per_mm)를
    대체하는 v2 보정 방식이다. 마우스 클릭이 필요 없고, 사진 속 보드를
    자동으로 인식해서 계산하기 때문에 화면(GUI) 없이도(순수 SSH만으로도)
    실행할 수 있다.

    호모그래피는 스칼라 px_per_mm 하나가 아니라 "이미지의 어느 위치든
    mm 좌표로 변환하는 함수"이기 때문에, 사진이 완벽히 수직으로
    찍히지 않아 원근感/기울어짐이 있어도 그 왜곡을 어느 정도 보정해준다.

[2. 실행 명령어]
    python3 charuco_calibration.py --image data/captured/capture_20260704_101500.jpg
    python3 charuco_calibration.py --image data/captured/sample.jpg --square-mm 9.8
    python3 charuco_calibration.py --image data/captured/sample.jpg --show   # GUI로 결과 확인

[3. 어디에 입력해야 하는가]
    -> Jetson Nano 터미널 또는 노트북 터미널 어디서든 실행 가능하다.
    -> 마우스 클릭이 필요 없으므로 순수 SSH 접속(화면 없음)만으로도 동작한다.
       (--show 옵션을 쓸 때만 GUI 창이 필요하며, 이 경우 SSH -X 접속이나
       모니터 직결 상태가 필요하다. --show 없이 실행하면 결과 확인용
       디버그 이미지가 파일로 저장되므로 굳이 GUI가 없어도 결과를 볼 수 있다)

[4. 정상적으로 실행되면]
    터미널에 아래처럼 출력된다.

        [INFO] OpenCV 4.1.1 감지 -> 구버전 aruco API 사용
        [INFO] 마커 검출: 42개 (중복 ID 3개 제거됨)
        [INFO] ChArUco 코너 검출: 68개
        [OK] 호모그래피 계산 완료. 재투영 RMS = 0.31mm
        [SAVED] data/results/capture_20260704_101500_charuco.json
        [SAVED] data/results/debug/capture_20260704_101500_charuco_debug.jpg

    RMS가 0.5mm를 넘으면 "[WARN] 재투영 오차가 큽니다..." 경고가 함께 출력된다.
    data/results/debug/ 안의 *_charuco_debug.jpg를 열어보면 검출된 마커와
    ChArUco 코너가 원본 사진 위에 표시되어 있어 보정이 잘 됐는지 눈으로
    확인할 수 있다.

[5. 오류가 발생하면 확인할 것]
    - "cv2.aruco 모듈이 없습니다": OpenCV가 opencv-contrib 없이 빌드된 것.
      `python3 -c "import cv2; print(cv2.aruco)"` 로 확인. JetPack 기본
      OpenCV에는 보통 포함되어 있고, 노트북에 pip으로 새로 깐 경우
      `pip3 install opencv-contrib-python` 이 필요할 수 있다.
    - "마커를 하나도 찾지 못했습니다": 보드 전체가 사진에 잘 나오는지,
      초점이 맞는지, 조명이 보드에 반사되어 하얗게 날아가지 않았는지 확인.
    - "중복 ID"가 계속 많이 나온다: 조명 반사나 그림자로 체크무늬 일부가
      가짜 마커로 오검출되는 것. 조명을 보드에 고르게 비추고 반사를 줄여서
      다시 촬영해보라.
    - RMS(재투영 오차)가 0.5mm보다 훨씬 크다: 보드가 평평하지 않거나
      (구겨짐/휘어짐), 카메라 각도가 너무 기울어져 있을 수 있다.
    - "--square-mm 값을 확인하라"는 오차가 계속 크게 난다면, 인쇄된 보드를
      실제 자로 재서 --square-mm 값을 실측치로 바꿔서 다시 시도한다.
"""

import argparse
import json
import os
from datetime import datetime

import cv2
import numpy as np

from utils import ensure_dir, save_json, check_gui_available

# ---------------------------------------------------------------------------
# 보드 스펙 (인쇄물과 반드시 일치해야 한다. 인쇄 후 실측했다면 --square-mm 등으로 덮어쓸 것)
# ---------------------------------------------------------------------------
SQUARES_X = 18
SQUARES_Y = 26
SQUARE_MM = 10.0   # 인쇄 후 실측값으로 교체 가능 (예: 9.8)
MARKER_MM = 7.0    # SQUARE_MM을 바꾸면 비율(0.7)을 유지해서 같이 바꿀 것
DICT_NAME = "DICT_4X4_250"

DEFAULT_CAMERA_HEIGHT_MM = 295.0
RMS_WARN_THRESHOLD_MM = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description="ChArUco 보드 자동 검출 기반 mm 보정")
    parser.add_argument("--image", required=True, help="ChArUco 보드가 촬영된 이미지 경로")
    parser.add_argument("--squares-x", type=int, default=SQUARES_X, help="보드 가로 칸 수")
    parser.add_argument("--squares-y", type=int, default=SQUARES_Y, help="보드 세로 칸 수")
    parser.add_argument(
        "--square-mm", type=float, default=SQUARE_MM, help="체스보드 한 칸 크기 (mm, 실측값 권장)"
    )
    parser.add_argument(
        "--marker-mm", type=float, default=MARKER_MM, help="ArUco 마커 한 변 크기 (mm)"
    )
    parser.add_argument("--dict", default=DICT_NAME, help="ArUco 딕셔너리 이름 (예: DICT_4X4_250)")
    parser.add_argument(
        "--camera-height",
        type=float,
        default=DEFAULT_CAMERA_HEIGHT_MM,
        help=f"카메라 렌즈~보드 평면 높이 (mm, 기본 {DEFAULT_CAMERA_HEIGHT_MM})",
    )
    parser.add_argument(
        "--output", default=None, help="결과 JSON 저장 경로 (기본: data/results/<이미지이름>_charuco.json)"
    )
    parser.add_argument(
        "--debug-dir", default=os.path.join("data", "results", "debug"), help="디버그 이미지 저장 폴더"
    )
    parser.add_argument("--show", action="store_true", help="결과를 GUI 창으로도 띄워서 확인")
    return parser.parse_args()


def default_output_path(image_path):
    base = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join("data", "results", f"{base}_charuco.json")


def default_debug_path(debug_dir, image_path):
    base = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(debug_dir, f"{base}_charuco_debug.jpg")


# ---------------------------------------------------------------------------
# cv2.aruco 구/신 API 호환 레이어
# ---------------------------------------------------------------------------
def _check_aruco_available():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco 모듈이 없습니다. OpenCV가 opencv-contrib 없이 빌드된 것으로 보입니다.\n"
            "JetPack 기본 OpenCV에는 보통 포함되어 있습니다. 노트북이라면\n"
            "  pip3 install opencv-contrib-python\n"
            "을 시도해보세요 (단, 이미 GStreamer 지원 OpenCV가 있다면 덮어쓰지 않도록 주의)."
        )


def _is_new_api():
    """OpenCV 4.7+ 에서 도입된 ArucoDetector/CharucoDetector 클래스 기반 API인지 확인"""
    return hasattr(cv2.aruco, "ArucoDetector")


def _get_dictionary(dict_name):
    dict_const = getattr(cv2.aruco, dict_name, None)
    if dict_const is None:
        raise ValueError(f"알 수 없는 ArUco 딕셔너리 이름: {dict_name}")
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dict_const)
    return cv2.aruco.Dictionary_get(dict_const)


def _make_board(squares_x, squares_y, square_mm, marker_mm, aruco_dict):
    if _is_new_api():
        return cv2.aruco.CharucoBoard((squares_x, squares_y), square_mm, marker_mm, aruco_dict)
    return cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_mm, marker_mm, aruco_dict)


def _make_detector_params():
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        return cv2.aruco.DetectorParameters_create()
    return cv2.aruco.DetectorParameters()


def _detect_markers(gray, aruco_dict, params):
    if _is_new_api():
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    return corners, ids, rejected


def _interpolate_charuco(marker_corners, marker_ids, gray, board):
    """
    필터링(중복 제거)된 마커로부터 ChArUco 코너를 보간한다.

    구버전 API는 interpolateCornersCharuco()로 직접 제어 가능하지만,
    신버전(4.7+) API는 이 자유 함수가 제거되고 CharucoDetector 클래스로
    통합되었다. 신버전에서는 우리가 직접 필터링한 마커 목록을 그대로
    주입할 방법이 공식적으로 없어서, CharucoDetector가 이미지에서 다시
    마커를 검출하도록 맡긴다 (이 경우 우리 쪽 중복 필터는 참고용 경고로만
    쓰이고, 실제 필터링은 CharucoDetector 내부 로직에 맡겨진다).
    """
    if hasattr(cv2.aruco, "interpolateCornersCharuco"):
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board
        )
        if not retval:
            return None, None
        return charuco_corners, charuco_ids

    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    return charuco_corners, charuco_ids


# ---------------------------------------------------------------------------
# 중복 마커 ID 필터
# ---------------------------------------------------------------------------
def _dedupe_markers(marker_corners, marker_ids):
    """
    같은 ID로 검출된 마커가 여러 개면(체크무늬 일부가 가짜 마커로 오검출된
    경우 흔히 발생) 면적이 가장 큰 것 하나만 남긴다. 이걸 안 하면
    ChArUco 보간이 통째로 실패하거나 왜곡된 좌표를 낼 수 있다.
    """
    if marker_ids is None or len(marker_ids) == 0:
        return marker_corners, marker_ids, 0

    best_by_id = {}
    for corners, mid in zip(marker_corners, marker_ids.flatten()):
        area = cv2.contourArea(corners.reshape(-1, 1, 2).astype(np.float32))
        mid = int(mid)
        if mid not in best_by_id or area > best_by_id[mid][1]:
            best_by_id[mid] = (corners, area)

    dup_count = len(marker_ids) - len(best_by_id)
    ids_sorted = sorted(best_by_id.keys())
    filtered_corners = [best_by_id[i][0] for i in ids_sorted]
    filtered_ids = np.array(ids_sorted, dtype=marker_ids.dtype).reshape(-1, 1)
    return filtered_corners, filtered_ids, dup_count


# ---------------------------------------------------------------------------
# ChArUco 코너 id -> 보드 mm 좌표
# ---------------------------------------------------------------------------
def charuco_id_to_mm(charuco_id, squares_x, square_mm):
    """
    OpenCV의 ChArUco 내부 코너 id는 (squares_x - 1)개 열을 기준으로
    행 우선(row-major) 순서로 매겨진다. id로부터 보드 좌상단 기준
    mm 좌표를 계산한다.
    """
    cols = squares_x - 1
    col = charuco_id % cols
    row = charuco_id // cols
    x_mm = (col + 1) * square_mm
    y_mm = (row + 1) * square_mm
    return x_mm, y_mm


# ---------------------------------------------------------------------------
# 메인 로직
# ---------------------------------------------------------------------------
def calibrate(image, args):
    _check_aruco_available()

    api_kind = "신버전(4.7+, ArucoDetector/CharucoDetector)" if _is_new_api() else "구버전(detectMarkers/interpolateCornersCharuco)"
    print(f"[INFO] OpenCV {cv2.__version__} 감지 -> {api_kind} aruco API 사용")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = _get_dictionary(args.dict)
    board = _make_board(args.squares_x, args.squares_y, args.square_mm, args.marker_mm, aruco_dict)
    params = _make_detector_params()

    marker_corners, marker_ids, _rejected = _detect_markers(gray, aruco_dict, params)

    if marker_ids is None or len(marker_ids) == 0:
        raise RuntimeError(
            "마커를 하나도 찾지 못했습니다. 보드 전체가 잘 나온 사진인지, "
            "초점/조명이 괜찮은지 확인하세요."
        )

    marker_corners, marker_ids, dup_count = _dedupe_markers(marker_corners, marker_ids)
    print(f"[INFO] 마커 검출: {len(marker_ids)}개" + (f" (중복 ID {dup_count}개 제거됨)" if dup_count else ""))

    charuco_corners, charuco_ids = _interpolate_charuco(marker_corners, marker_ids, gray, board)
    if charuco_corners is None or charuco_ids is None or len(charuco_corners) < 4:
        raise RuntimeError(
            "ChArUco 코너를 충분히 찾지 못했습니다 (최소 4개 필요). "
            "보드가 더 크게, 더 선명하게 나오도록 다시 촬영해보세요."
        )
    print(f"[INFO] ChArUco 코너 검출: {len(charuco_corners)}개")

    image_points = charuco_corners.reshape(-1, 2).astype(np.float32)
    mm_points = np.array(
        [charuco_id_to_mm(int(cid), args.squares_x, args.square_mm) for cid in charuco_ids.flatten()],
        dtype=np.float32,
    )

    H, mask = cv2.findHomography(image_points, mm_points, cv2.RANSAC, 2.0)
    if H is None:
        raise RuntimeError("호모그래피 계산에 실패했습니다. 코너가 한 직선 위에 몰려있지 않은지 확인하세요.")

    inlier_mask = mask.ravel().astype(bool)
    projected = cv2.perspectiveTransform(image_points.reshape(-1, 1, 2), H).reshape(-1, 2)
    diffs = projected[inlier_mask] - mm_points[inlier_mask]
    rms_mm = float(np.sqrt(np.mean(np.sum(diffs ** 2, axis=1))))

    print(f"[OK] 호모그래피 계산 완료. 재투영 RMS = {rms_mm:.3f}mm (인라이어 {int(inlier_mask.sum())}/{len(mm_points)})")
    if rms_mm > RMS_WARN_THRESHOLD_MM:
        print(
            f"[WARN] 재투영 오차가 큽니다 ({rms_mm:.3f}mm > {RMS_WARN_THRESHOLD_MM}mm). "
            "보드 평탄도, 카메라 각도, --square-mm 실측값을 확인해보세요."
        )

    return {
        "homography": H.tolist(),
        "num_markers_detected": int(len(marker_ids)),
        "num_duplicate_markers_removed": int(dup_count),
        "num_charuco_corners_detected": int(len(charuco_corners)),
        "num_inliers": int(inlier_mask.sum()),
        "rms_reprojection_mm": rms_mm,
        "marker_corners": marker_corners,
        "marker_ids": marker_ids,
        "charuco_corners": charuco_corners,
        "charuco_ids": charuco_ids,
    }


def save_debug_image(image, calib_result, debug_path):
    debug_img = image.copy()

    marker_ids = calib_result["marker_ids"]
    for corners, mid in zip(calib_result["marker_corners"], marker_ids.flatten()):
        pts = corners.reshape(-1, 2).astype(int)
        cv2.polylines(debug_img, [pts], True, (0, 255, 0), 2)
        center = pts.mean(axis=0).astype(int)
        cv2.putText(
            debug_img, str(int(mid)), tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    for corner in calib_result["charuco_corners"].reshape(-1, 2):
        cv2.circle(debug_img, tuple(corner.astype(int)), 4, (0, 0, 255), -1)

    cv2.putText(
        debug_img,
        f"RMS={calib_result['rms_reprojection_mm']:.3f}mm  corners={calib_result['num_charuco_corners_detected']}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    ensure_dir(debug_path)
    cv2.imwrite(debug_path, debug_img)
    print(f"[SAVED] {debug_path}")
    return debug_img


def main():
    args = parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"[ERROR] 이미지를 열 수 없습니다: {args.image}")
        print("        경로가 올바른지, 파일이 실제로 존재하는지 확인하세요.")
        return

    try:
        calib_result = calibrate(image, args)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return

    debug_path = default_debug_path(args.debug_dir, args.image)
    debug_img = save_debug_image(image, calib_result, debug_path)

    if args.show:
        if check_gui_available():
            cv2.namedWindow("ChArUco Calibration", cv2.WINDOW_NORMAL)
            cv2.imshow("ChArUco Calibration", debug_img)
            print("[INFO] 아무 키나 누르면 창을 닫습니다.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("[INFO] GUI를 사용할 수 없는 환경이라 --show 요청을 건너뛰고 파일로만 저장했습니다.")

    output_path = args.output or default_output_path(args.image)
    result_json = {
        "image_path": args.image,
        "board_spec": {
            "squares_x": args.squares_x,
            "squares_y": args.squares_y,
            "square_mm": args.square_mm,
            "marker_mm": args.marker_mm,
            "dict": args.dict,
        },
        "homography": calib_result["homography"],
        "num_markers_detected": calib_result["num_markers_detected"],
        "num_duplicate_markers_removed": calib_result["num_duplicate_markers_removed"],
        "num_charuco_corners_detected": calib_result["num_charuco_corners_detected"],
        "num_inliers": calib_result["num_inliers"],
        "rms_reprojection_mm": round(calib_result["rms_reprojection_mm"], 4),
        "camera_height_mm": args.camera_height,
        "opencv_version": cv2.__version__,
        "aruco_api": "new" if _is_new_api() else "legacy",
        "debug_image_path": debug_path,
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    ensure_dir(output_path)
    save_json(output_path, result_json)
    print(f"[SAVED] {output_path}")


if __name__ == "__main__":
    main()
