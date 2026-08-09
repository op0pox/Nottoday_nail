"""
charuco_calibration.py
=======================
[1. 역할]
    ChArUco 보드(체스보드 + ArUco 마커 조합 보정판)가 촬영된 사진에서
    보드를 자동으로 검출하고, "이미지 픽셀 좌표 <-> 보드 mm 좌표" 사이의
    호모그래피(homography) 행렬을 계산한다.

    ChArUco 보드란? 체스판 무늬의 흰 칸 안에 ArUco 마커(QR코드 비슷한
    흑백 정사각형 패턴)가 인쇄된 보정판이다. 마커마다 고유 번호(ID)가
    있어서, 보드의 일부만 사진에 나와도 "이 코너가 보드의 몇 번째
    칸인지"를 알 수 있다. -> 손이 보드를 가려도 보정이 가능한 이유!

    호모그래피는 스칼라 px_per_mm 하나가 아니라 "이미지의 어느 위치든
    mm 좌표로 변환하는 함수"이기 때문에, 사진이 완벽히 수직으로
    찍히지 않아 원근/기울어짐이 있어도 그 왜곡을 어느 정도 보정해준다.

[2. 실행 방법]
    보통은 seg_gui.py가 내부적으로 calibrate()를 호출해서 쓰지만,
    단독 CLI로도 실행할 수 있다:
        python3 charuco_calibration.py --image 사진.jpg
        python3 charuco_calibration.py --image 사진.jpg --square-mm 9.8   # 실측값으로 보정
        python3 charuco_calibration.py --image 사진.jpg --show            # 결과 GUI 확인

[3. 정상 실행 시 출력 예]
        [INFO] 마커 검출: 42개 (중복 ID 3개 제거됨)
        [INFO] ChArUco 코너 검출: 68개
        [OK] 호모그래피 계산 완료. 재투영 RMS = 0.31mm
        [SAVED] results/사진_charuco.json
        [SAVED] results/debug/사진_charuco_debug.jpg
    RMS가 0.5mm를 넘으면 경고가 출력된다. debug 이미지를 열어보면
    검출된 마커/코너가 표시되어 있어 눈으로 확인할 수 있다.

[4. 오류가 발생하면 확인할 것]
    - "cv2.aruco 모듈이 없습니다": pip3 install opencv-contrib-python
    - "마커를 하나도 찾지 못했습니다": 보드가 사진에 잘 나오는지, 초점/반사 확인
    - RMS(재투영 오차)가 크다: 보드 평탄도(구겨짐), 카메라 각도,
      --square-mm 실측값 확인
"""

import argparse       # 명령줄 옵션(--image 등) 파싱
import json           # (참고용 import — 저장은 utils.save_json 사용)
import os
from datetime import datetime

import cv2
import numpy as np

from utils import ensure_dir, save_json, check_gui_available

# ---------------------------------------------------------------------------
# 보드 스펙 (인쇄물과 반드시 일치해야 한다! 인쇄 후 실측했다면 --square-mm 등으로 덮어쓸 것)
# 이 값이 실제와 다르면 모든 mm 측정값이 같은 배수로 통째로 틀어진다.
# ---------------------------------------------------------------------------
SQUARES_X = 18     # 보드 가로 칸 수
SQUARES_Y = 26     # 보드 세로 칸 수
SQUARE_MM = 10.0   # 체스판 한 칸의 실제 크기 (인쇄 후 실측값으로 교체 가능, 예: 9.8)
MARKER_MM = 7.0    # ArUco 마커 한 변 크기 (SQUARE_MM을 바꾸면 비율 0.7 유지해서 같이 바꿀 것)
DICT_NAME = "DICT_4X4_250"   # 마커 사전: 4x4 격자 패턴, 250종류의 ID

DEFAULT_CAMERA_HEIGHT_MM = 295.0    # 카메라~보드 높이 기본값 (시차 보정용)
RMS_WARN_THRESHOLD_MM = 0.5         # 재투영 오차가 이보다 크면 경고


def parse_args():
    """명령줄 인자 정의. (seg_gui.py는 이 대신 SimpleNamespace로 값을 만들어 넘긴다)"""
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
        "--output", default=None, help="결과 JSON 저장 경로 (기본: results/<이미지이름>_charuco.json)"
    )
    parser.add_argument(
        "--debug-dir", default=os.path.join("results", "debug"), help="디버그 이미지 저장 폴더"
    )
    parser.add_argument("--show", action="store_true", help="결과를 GUI 창으로도 띄워서 확인")
    return parser.parse_args()


def default_output_path(image_path):
    """결과 JSON 기본 저장 경로: results/<이미지이름>_charuco.json"""
    base = os.path.splitext(os.path.basename(image_path))[0]   # 확장자 뗀 파일명
    return os.path.join("results", f"{base}_charuco.json")


def default_debug_path(debug_dir, image_path):
    """디버그 이미지 기본 저장 경로"""
    base = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(debug_dir, f"{base}_charuco_debug.jpg")


# ---------------------------------------------------------------------------
# cv2.aruco 구/신 API 호환 레이어
#
# OpenCV는 4.7 버전에서 aruco 관련 함수 이름/구조를 크게 바꿨다.
# (구버전: detectMarkers() 같은 자유 함수 / 신버전: ArucoDetector 클래스)
# Jetson Nano(구버전 OpenCV)와 노트북(신버전) 양쪽에서 돌아가야 하므로,
# 아래 _로 시작하는 함수들이 "어느 버전이든 같은 방식으로 호출"할 수 있게
# 감싸주는 역할을 한다. (어댑터 패턴)
# ---------------------------------------------------------------------------
def _check_aruco_available():
    """cv2.aruco 모듈이 있는지 확인. contrib 없이 빌드된 OpenCV에는 없다."""
    if not hasattr(cv2, "aruco"):     # hasattr = 속성이 존재하는지 검사
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
    """이름 문자열("DICT_4X4_250")로 ArUco 마커 사전 객체를 얻는다."""
    # getattr(cv2.aruco, 이름) = 문자열로 모듈의 상수를 동적으로 찾기
    dict_const = getattr(cv2.aruco, dict_name, None)
    if dict_const is None:
        raise ValueError(f"알 수 없는 ArUco 딕셔너리 이름: {dict_name}")
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dict_const)
    return cv2.aruco.Dictionary_get(dict_const)      # 아주 구버전용


def _make_board(squares_x, squares_y, square_mm, marker_mm, aruco_dict):
    """보드 스펙 객체 생성 (검출기가 '어떤 보드를 찾을지' 알기 위해 필요)"""
    if _is_new_api():
        return cv2.aruco.CharucoBoard((squares_x, squares_y), square_mm, marker_mm, aruco_dict)
    return cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_mm, marker_mm, aruco_dict)


def _make_detector_params():
    """마커 검출 파라미터 객체 생성 (기본값 사용)"""
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        return cv2.aruco.DetectorParameters_create()   # 구버전
    return cv2.aruco.DetectorParameters()              # 신버전


def _detect_markers(gray, aruco_dict, params):
    """
    흑백 이미지에서 ArUco 마커들을 검출한다.
    반환: (마커 코너 좌표들, 마커 ID들, 기각된 후보들)
    """
    if _is_new_api():
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    return corners, ids, rejected


def _interpolate_charuco(marker_corners, marker_ids, gray, board):
    """
    검출된 마커들로부터 ChArUco 코너(체스판 칸이 만나는 교차점)를 보간한다.
    마커보다 코너가 위치 정밀도가 훨씬 높아서, 호모그래피 계산에는
    마커 자체가 아니라 이 코너들을 쓴다.

    구버전 API는 interpolateCornersCharuco()로 직접 제어 가능하지만,
    신버전(4.7+)에서는 CharucoDetector 클래스로 통합되어, 우리가 필터링한
    마커 목록을 주입할 방법이 없어 이미지에서 다시 검출하도록 맡긴다.
    """
    if hasattr(cv2.aruco, "interpolateCornersCharuco"):    # 구버전 경로
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board
        )
        if not retval:
            return None, None
        return charuco_corners, charuco_ids

    # 신버전 경로
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

    # ID별로 "면적이 가장 큰 마커"만 남기는 딕셔너리 구축
    best_by_id = {}
    # marker_ids는 (N, 1) 모양이라 flatten()으로 1차원으로 편다
    for corners, mid in zip(marker_corners, marker_ids.flatten()):
        area = cv2.contourArea(corners.reshape(-1, 1, 2).astype(np.float32))
        mid = int(mid)
        if mid not in best_by_id or area > best_by_id[mid][1]:
            best_by_id[mid] = (corners, area)

    dup_count = len(marker_ids) - len(best_by_id)     # 제거된 중복 개수
    ids_sorted = sorted(best_by_id.keys())
    filtered_corners = [best_by_id[i][0] for i in ids_sorted]
    # OpenCV가 기대하는 (N, 1) 모양으로 복원
    filtered_ids = np.array(ids_sorted, dtype=marker_ids.dtype).reshape(-1, 1)
    return filtered_corners, filtered_ids, dup_count


# ---------------------------------------------------------------------------
# ChArUco 코너 id -> 보드 mm 좌표
# ---------------------------------------------------------------------------
def charuco_id_to_mm(charuco_id, squares_x, square_mm):
    """
    OpenCV의 ChArUco 내부 코너 id를 보드 좌상단 기준 mm 좌표로 변환한다.

    코너 id는 (squares_x - 1)개 열을 기준으로 행 우선(row-major) 순서로
    매겨진다. 즉 id를 열 개수로 나눈 나머지=열 번호, 몫=행 번호.
    (내부 코너는 칸과 칸 사이에 있으므로 (col+1), (row+1) 번째 칸 경계 위치)
    """
    cols = squares_x - 1
    col = charuco_id % cols      # 나머지 = 열 번호
    row = charuco_id // cols     # 몫 = 행 번호
    x_mm = (col + 1) * square_mm
    y_mm = (row + 1) * square_mm
    return x_mm, y_mm


# ---------------------------------------------------------------------------
# 메인 로직
# ---------------------------------------------------------------------------
def calibrate(image, args):
    """
    이 파일의 핵심 함수. 이미지에서 보드를 찾아 호모그래피를 계산한다.

    단계:
      1. 흑백 변환 후 ArUco 마커 검출
      2. 중복 ID 제거
      3. 마커들로부터 ChArUco 코너 보간 (정밀한 교차점 좌표)
      4. 코너 id -> 실제 mm 좌표 매핑 생성
      5. (픽셀 좌표, mm 좌표) 대응쌍들로 호모그래피 계산 (RANSAC)
      6. 재투영 오차(RMS) 계산으로 정확도 검증

    반환: 호모그래피, 검출 통계, 디버그용 좌표들이 담긴 딕셔너리
    """
    _check_aruco_available()

    api_kind = "신버전(4.7+, ArucoDetector/CharucoDetector)" if _is_new_api() else "구버전(detectMarkers/interpolateCornersCharuco)"
    print(f"[INFO] OpenCV {cv2.__version__} 감지 -> {api_kind} aruco API 사용")

    # ── 1) 준비: 흑백 변환 + 보드/검출기 객체 생성 ──
    # (마커 검출은 색이 필요 없어서 흑백으로 하는 게 빠르고 안정적)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = _get_dictionary(args.dict)
    board = _make_board(args.squares_x, args.squares_y, args.square_mm, args.marker_mm, aruco_dict)
    params = _make_detector_params()

    # ── 2) 마커 검출 ──
    marker_corners, marker_ids, _rejected = _detect_markers(gray, aruco_dict, params)

    if marker_ids is None or len(marker_ids) == 0:
        raise RuntimeError(
            "마커를 하나도 찾지 못했습니다. 보드 전체가 잘 나온 사진인지, "
            "초점/조명이 괜찮은지 확인하세요."
        )

    # ── 3) 중복 제거 ──
    marker_corners, marker_ids, dup_count = _dedupe_markers(marker_corners, marker_ids)
    print(f"[INFO] 마커 검출: {len(marker_ids)}개" + (f" (중복 ID {dup_count}개 제거됨)" if dup_count else ""))

    # ── 4) ChArUco 코너 보간 ──
    charuco_corners, charuco_ids = _interpolate_charuco(marker_corners, marker_ids, gray, board)
    if charuco_corners is None or charuco_ids is None or len(charuco_corners) < 4:
        # 호모그래피 계산에는 최소 4개의 대응점이 필요하다
        raise RuntimeError(
            "ChArUco 코너를 충분히 찾지 못했습니다 (최소 4개 필요). "
            "보드가 더 크게, 더 선명하게 나오도록 다시 촬영해보세요."
        )
    print(f"[INFO] ChArUco 코너 검출: {len(charuco_corners)}개")

    # ── 5) 대응쌍 만들기: 코너의 (픽셀 좌표) <-> (실제 mm 좌표) ──
    image_points = charuco_corners.reshape(-1, 2).astype(np.float32)   # 픽셀 좌표들
    mm_points = np.array(
        [charuco_id_to_mm(int(cid), args.squares_x, args.square_mm) for cid in charuco_ids.flatten()],
        dtype=np.float32,
    )

    # ── 6) 호모그래피 계산 ──
    # RANSAC = 일부 대응점이 틀렸어도(아웃라이어) 무시하고 다수가 일치하는
    # 변환을 찾는 강건한 알고리즘. 2.0 = 아웃라이어 판정 허용 오차.
    # 반환된 mask는 각 점이 인라이어(1)인지 아웃라이어(0)인지 표시.
    H, mask = cv2.findHomography(image_points, mm_points, cv2.RANSAC, 2.0)
    if H is None:
        raise RuntimeError("호모그래피 계산에 실패했습니다. 코너가 한 직선 위에 몰려있지 않은지 확인하세요.")

    # ── 7) 재투영 오차(RMS) 계산: 변환이 얼마나 정확한지 자기 검증 ──
    # 픽셀 좌표들을 방금 구한 H로 변환해보고, 정답 mm 좌표와의 차이를 잰다.
    inlier_mask = mask.ravel().astype(bool)
    projected = cv2.perspectiveTransform(image_points.reshape(-1, 1, 2), H).reshape(-1, 2)
    diffs = projected[inlier_mask] - mm_points[inlier_mask]
    # RMS = 오차 제곱 평균의 제곱근 (평균적인 오차 크기, mm 단위)
    rms_mm = float(np.sqrt(np.mean(np.sum(diffs ** 2, axis=1))))

    print(f"[OK] 호모그래피 계산 완료. 재투영 RMS = {rms_mm:.3f}mm (인라이어 {int(inlier_mask.sum())}/{len(mm_points)})")
    if rms_mm > RMS_WARN_THRESHOLD_MM:
        print(
            f"[WARN] 재투영 오차가 큽니다 ({rms_mm:.3f}mm > {RMS_WARN_THRESHOLD_MM}mm). "
            "보드 평탄도, 카메라 각도, --square-mm 실측값을 확인해보세요."
        )

    return {
        "homography": H.tolist(),         # 3x3 행렬 (JSON 저장 가능하게 리스트로)
        "num_markers_detected": int(len(marker_ids)),
        "num_duplicate_markers_removed": int(dup_count),
        "num_charuco_corners_detected": int(len(charuco_corners)),
        "num_inliers": int(inlier_mask.sum()),
        "rms_reprojection_mm": rms_mm,
        # 아래는 디버그 이미지 그리기용 원본 좌표들
        "marker_corners": marker_corners,
        "marker_ids": marker_ids,
        "charuco_corners": charuco_corners,
        "charuco_ids": charuco_ids,
    }


def save_debug_image(image, calib_result, debug_path):
    """검출 결과를 원본 위에 그려서 저장 (보정이 잘 됐는지 눈으로 확인용)"""
    debug_img = image.copy()

    # 초록 사각형 + ID 번호 = 검출된 마커들
    marker_ids = calib_result["marker_ids"]
    for corners, mid in zip(calib_result["marker_corners"], marker_ids.flatten()):
        pts = corners.reshape(-1, 2).astype(int)
        cv2.polylines(debug_img, [pts], True, (0, 255, 0), 2)   # True = 닫힌 다각형
        center = pts.mean(axis=0).astype(int)
        cv2.putText(
            debug_img, str(int(mid)), tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    # 빨간 점 = ChArUco 코너들 (호모그래피 계산에 실제로 쓰인 점)
    for corner in calib_result["charuco_corners"].reshape(-1, 2):
        cv2.circle(debug_img, tuple(corner.astype(int)), 4, (0, 0, 255), -1)

    # 좌상단에 요약 정보 (노란 텍스트)
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
    """CLI로 단독 실행했을 때의 흐름: 이미지 로드 -> 보정 -> 저장"""
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

    # 디버그 이미지 저장
    debug_path = default_debug_path(args.debug_dir, args.image)
    debug_img = save_debug_image(image, calib_result, debug_path)

    # --show 옵션이면 창으로도 표시
    if args.show:
        if check_gui_available():
            cv2.namedWindow("ChArUco Calibration", cv2.WINDOW_NORMAL)
            cv2.imshow("ChArUco Calibration", debug_img)
            print("[INFO] 아무 키나 누르면 창을 닫습니다.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("[INFO] GUI를 사용할 수 없는 환경이라 --show 요청을 건너뛰고 파일로만 저장했습니다.")

    # 결과를 JSON으로 저장 (호모그래피 + 검출 통계 + 실행 정보)
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
