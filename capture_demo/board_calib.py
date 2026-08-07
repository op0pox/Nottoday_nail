# -*- coding: utf-8 -*-
"""
board_calib.py
================
사진 한 장마다 ChArUco 보드를 새로 검출해서 이미지<->mm 호모그래피를 계산한다.
(nail_analysis/calibration.py의 detect_board()를 이 프로젝트 전용으로 가볍게
포팅한 것 — nail_analysis는 별도 requirements/상대 임포트 구조라 그대로
가져다 쓰기보다 필요한 부분만 옮겨서 독립적으로 유지한다.)

폰으로 매번 자유롭게 찍고 그때마다 보드를 같이 깔아 찍는 것을 전제로 하므로,
"한 번 보정하고 저장해서 재사용"하는 고정 카메라 모델이 아니라 사진마다
새로 계산한다.

기본 보드 스펙은 nail_measure/charuco_calibration.py의 기본값과 동일하다
(18x26칸, 10mm/칸 체스보드 + 7mm ArUco 마커, DICT_4X4_250 — 이미 인쇄해서
쓰고 있는 보드). 다른 보드를 쓸 경우 BOARD_CONFIG를 바꾸거나 detect_board()에
board_cfg를 직접 넘기면 된다.

인쇄물이 정확히 10mm/칸이 아닐 수 있으니(프린터 배율 오차), 실제 자로 한 칸을
재서 다르면 BOARD_CONFIG["square_length_mm"]을 실측값으로 바꿀 것.
"""

import math

import cv2
import numpy as np

BOARD_CONFIG = {
    "squares_x": 18,
    "squares_y": 26,
    "square_length_mm": 10.0,
    "marker_length_mm": 7.0,
    "dictionary": "DICT_4X4_250",
}

DEFAULT_RANSAC_THRESHOLD_MM = 2.0


class BoardDetectionError(RuntimeError):
    pass


def _check_aruco_available():
    if not hasattr(cv2, "aruco"):
        raise BoardDetectionError(
            "cv2.aruco 모듈이 없습니다. opencv-contrib-python이 설치된 파이썬 환경에서 실행하세요."
        )


def _is_new_aruco_api():
    return hasattr(cv2.aruco, "ArucoDetector")


def _get_dictionary(dict_name):
    dict_const = getattr(cv2.aruco, dict_name, None)
    if dict_const is None:
        raise BoardDetectionError("알 수 없는 ArUco 딕셔너리 이름: %s" % dict_name)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dict_const)
    return cv2.aruco.Dictionary_get(dict_const)


def _make_board(board_cfg, aruco_dict):
    squares_x = board_cfg["squares_x"]
    squares_y = board_cfg["squares_y"]
    square_mm = board_cfg["square_length_mm"]
    marker_mm = board_cfg["marker_length_mm"]
    if _is_new_aruco_api():
        return cv2.aruco.CharucoBoard((squares_x, squares_y), square_mm, marker_mm, aruco_dict)
    return cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_mm, marker_mm, aruco_dict)


def _make_detector_params():
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        return cv2.aruco.DetectorParameters_create()
    return cv2.aruco.DetectorParameters()


def _detect_markers(gray, aruco_dict, params):
    if _is_new_aruco_api():
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    return corners, ids, rejected


def _dedupe_markers(marker_corners, marker_ids):
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


def _marker_id_to_objpoints(board):
    """board.getIds()/getObjPoints()를 {marker_id: 4x2 mm 좌표} 딕셔너리로 변환."""
    ids = [int(i) for i in board.getIds()]
    obj_pts = board.getObjPoints()  # ids와 같은 순서, 마커당 4x3(z=0)
    return {mid: obj_pts[i][:, :2] for i, mid in enumerate(ids)}


MIN_MARKERS_FOR_HOMOGRAPHY = 3


class BoardDetection(object):
    def __init__(self, homography, num_markers, num_points, rms_mm):
        self.homography = homography
        self.num_markers = num_markers
        self.num_points = num_points  # 호모그래피 계산에 쓰인 마커 코너 점 개수 (마커당 4개)
        self.rms_mm = rms_mm


def detect_board(image_bgr, board_cfg=None):
    """
    사진 한 장(BGR)에서 ChArUco 보드를 검출해서 이미지<->mm 호모그래피를 계산한다.
    실패하면 BoardDetectionError를 낸다 (호출부가 재촬영을 유도해야 함).

    interpolateCornersCharuco/CharucoDetector로 "체스보드 내부 코너"를 보간하는
    방식은 실측 테스트에서(폰 근접 촬영 등으로) 마커는 정상 검출되는데도 내부
    코너 보간이 종종 0개로 실패했다 -> 대신 검출된 각 ArUco 마커의 4개 모서리
    좌표(보드 기준 mm 위치는 이미 알려져 있음, board.getObjPoints())를 직접
    호모그래피 대응점으로 쓴다. 마커 1개당 4점이라 대응점 수도 더 많고,
    보간 단계가 아예 없어서 더 안정적이다.
    """
    board_cfg = board_cfg or BOARD_CONFIG
    _check_aruco_available()

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    aruco_dict = _get_dictionary(board_cfg["dictionary"])
    board = _make_board(board_cfg, aruco_dict)
    params = _make_detector_params()

    marker_corners, marker_ids, _rejected = _detect_markers(gray, aruco_dict, params)
    if marker_ids is None or len(marker_ids) == 0:
        raise BoardDetectionError(
            "ChArUco 마커를 하나도 찾지 못했습니다. 보드 전체가 사진에 잘 나오는지, "
            "초점/조명이 괜찮은지 확인하고 다시 촬영해주세요."
        )

    marker_corners, marker_ids, _dup = _dedupe_markers(marker_corners, marker_ids)
    if len(marker_ids) < MIN_MARKERS_FOR_HOMOGRAPHY:
        raise BoardDetectionError(
            "검출된 마커가 %d개뿐입니다 (최소 %d개 필요). 보드가 더 크고 선명하게 "
            "나오도록 다시 촬영해주세요." % (len(marker_ids), MIN_MARKERS_FOR_HOMOGRAPHY)
        )

    id_to_objpoints = _marker_id_to_objpoints(board)
    image_points = []
    mm_points = []
    for corners, mid in zip(marker_corners, marker_ids.flatten()):
        mid = int(mid)
        if mid not in id_to_objpoints:
            continue
        img_c = corners.reshape(4, 2)
        obj_c = id_to_objpoints[mid]
        image_points.extend(img_c)
        mm_points.extend(obj_c)

    image_points = np.array(image_points, dtype=np.float32)
    mm_points = np.array(mm_points, dtype=np.float32)
    if len(image_points) < MIN_MARKERS_FOR_HOMOGRAPHY * 4:
        raise BoardDetectionError(
            "검출된 마커가 이 보드 스펙(BOARD_CONFIG)과 맞지 않습니다. "
            "보드 종류/규격을 확인해주세요."
        )

    H, mask = cv2.findHomography(image_points, mm_points, cv2.RANSAC, DEFAULT_RANSAC_THRESHOLD_MM)
    if H is None:
        raise BoardDetectionError("호모그래피 계산에 실패했습니다. 코너가 한 직선 위에 몰려있지 않은지 확인하세요.")

    inlier_mask = mask.ravel().astype(bool)
    projected = cv2.perspectiveTransform(image_points.reshape(-1, 1, 2), H).reshape(-1, 2)
    diffs = projected[inlier_mask] - mm_points[inlier_mask]
    rms_mm = float(np.sqrt(np.mean(np.sum(diffs ** 2, axis=1)))) if inlier_mask.any() else float("nan")

    return BoardDetection(
        homography=H,
        num_markers=int(len(marker_ids)),
        num_points=int(len(image_points)),
        rms_mm=rms_mm,
    )


def transform_points_to_mm(homography, points_px):
    """[(x,y), ...] 픽셀 좌표들을 mm 좌표로 변환한다."""
    pts = np.array(points_px, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, homography).reshape(-1, 2)
    return [tuple(p) for p in out]


def mm_distance(homography, point_a_px, point_b_px):
    """호모그래피 기준으로 픽셀 좌표 두 점 사이의 실제 mm 거리."""
    a_mm, b_mm = transform_points_to_mm(homography, [point_a_px, point_b_px])
    return math.hypot(a_mm[0] - b_mm[0], a_mm[1] - b_mm[1])
