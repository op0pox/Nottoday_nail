# -*- coding: utf-8 -*-
"""
calibration.py
================
[역할]
    ChArUco 보드(체커보드+ArUco)를 이용한 카메라 캘리브레이션과, 측정
    프레임마다 mm/px 스케일(호모그래피)을 계산하는 기능을 담당한다.

    - 카메라 매트릭스/왜곡계수 계산 (여러 장의 보드 사진으로 1회 수행,
      calib_front.json / calib_side.json에 저장)
    - 한 장의 사진에서 보드를 검출해 이미지<->mm 호모그래피 계산
      (findHomography 기반, 원근/기울어짐을 어느 정도 보정)
    - 측면 카메라 전용: 보드 pose(거리)로부터 "손가락 평면"의 스케일을
      보정하는 로직 (side_plane_offset_mm)
    - 실시간 측정 중 보드 검출 실패에 대비한 폴백(hold -> 저장된 calib)

    cv2.aruco의 구버전(4.6 이하)/신버전(4.7+) API를 모두 지원하는 호환
    레이어를 포함한다 (JetPack 4.6.x는 구버전 API인 경우가 많다).

[실행 위치] 이 파일은 직접 실행하는 CLI가 아니라 main.py가 불러와 쓰는
라이브러리 모듈이다 (main.py의 `calibrate --camera front|side` 참고).
"""

import json
import math
import os
from datetime import datetime

import cv2
import numpy as np

DEFAULT_RANSAC_THRESHOLD_MM = 2.0


# ---------------------------------------------------------------------------
# cv2.aruco 구/신 API 호환 레이어
# ---------------------------------------------------------------------------
def check_aruco_available():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco 모듈이 없습니다. OpenCV가 opencv-contrib 없이 빌드된 것으로 보입니다.\n"
            "JetPack 기본 OpenCV에는 보통 포함되어 있습니다. 노트북/PC라면\n"
            "  pip3 install opencv-contrib-python\n"
            "을 시도해보세요."
        )


def is_new_aruco_api():
    """OpenCV 4.7+ 에서 도입된 ArucoDetector/CharucoDetector 클래스 기반 API인지."""
    return hasattr(cv2.aruco, "ArucoDetector")


def get_dictionary(dict_name):
    dict_const = getattr(cv2.aruco, dict_name, None)
    if dict_const is None:
        raise ValueError("알 수 없는 ArUco 딕셔너리 이름: %s" % dict_name)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dict_const)
    return cv2.aruco.Dictionary_get(dict_const)


def make_board(board_cfg, aruco_dict):
    squares_x = board_cfg["squares_x"]
    squares_y = board_cfg["squares_y"]
    square_mm = board_cfg["square_length_mm"]
    marker_mm = board_cfg["marker_length_mm"]
    if is_new_aruco_api():
        return cv2.aruco.CharucoBoard((squares_x, squares_y), square_mm, marker_mm, aruco_dict)
    return cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_mm, marker_mm, aruco_dict)


def make_detector_params():
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        return cv2.aruco.DetectorParameters_create()
    return cv2.aruco.DetectorParameters()


def detect_markers(gray, aruco_dict, params):
    if is_new_aruco_api():
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    return corners, ids, rejected


def interpolate_charuco(marker_corners, marker_ids, gray, board):
    """
    구버전은 interpolateCornersCharuco()로 직접 제어 가능. 신버전(4.7+)은
    이 자유 함수가 없어져서 CharucoDetector에 다시 검출을 맡긴다(이 경우
    우리가 미리 걸러낸 중복 마커 필터는 참고 경고 용도로만 쓰인다).
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


def dedupe_markers(marker_corners, marker_ids):
    """같은 ID로 검출된 마커가 여러 개면 면적이 가장 큰 것만 남긴다."""
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


def charuco_id_to_mm(charuco_id, squares_x, square_mm):
    """ChArUco 내부 코너 id -> 보드 좌상단 기준 mm 좌표."""
    cols = squares_x - 1
    col = charuco_id % cols
    row = charuco_id // cols
    return (col + 1) * square_mm, (row + 1) * square_mm


# ---------------------------------------------------------------------------
# 보드 검출 결과 (한 프레임)
# ---------------------------------------------------------------------------
class BoardDetection(object):
    def __init__(
        self,
        homography,
        num_markers,
        num_duplicate_markers,
        num_corners,
        rms_mm,
        charuco_corners,
        charuco_ids,
    ):
        self.homography = homography
        self.num_markers = num_markers
        self.num_duplicate_markers = num_duplicate_markers
        self.num_corners = num_corners
        self.rms_mm = rms_mm
        self.charuco_corners = charuco_corners
        self.charuco_ids = charuco_ids
        self.board_distance_mm = None  # estimate_board_pose()가 채움 (옵션)


def detect_board(gray, board_cfg):
    """
    한 프레임(그레이스케일)에서 ChArUco 보드를 검출해서 이미지<->mm
    호모그래피를 계산한다. 실패하면 None을 반환한다.
    """
    check_aruco_available()
    aruco_dict = get_dictionary(board_cfg["dictionary"])
    board = make_board(board_cfg, aruco_dict)
    params = make_detector_params()

    marker_corners, marker_ids, _rejected = detect_markers(gray, aruco_dict, params)
    if marker_ids is None or len(marker_ids) == 0:
        return None

    marker_corners, marker_ids, dup_count = dedupe_markers(marker_corners, marker_ids)

    charuco_corners, charuco_ids = interpolate_charuco(marker_corners, marker_ids, gray, board)
    if charuco_corners is None or charuco_ids is None or len(charuco_corners) < 4:
        return None

    image_points = charuco_corners.reshape(-1, 2).astype(np.float32)
    mm_points = np.array(
        [charuco_id_to_mm(int(cid), board_cfg["squares_x"], board_cfg["square_length_mm"]) for cid in charuco_ids.flatten()],
        dtype=np.float32,
    )

    H, mask = cv2.findHomography(image_points, mm_points, cv2.RANSAC, DEFAULT_RANSAC_THRESHOLD_MM)
    if H is None:
        return None

    inlier_mask = mask.ravel().astype(bool)
    projected = cv2.perspectiveTransform(image_points.reshape(-1, 1, 2), H).reshape(-1, 2)
    diffs = projected[inlier_mask] - mm_points[inlier_mask]
    rms_mm = float(np.sqrt(np.mean(np.sum(diffs ** 2, axis=1)))) if inlier_mask.any() else float("nan")

    return BoardDetection(
        homography=H,
        num_markers=int(len(marker_ids)),
        num_duplicate_markers=int(dup_count),
        num_corners=int(len(charuco_corners)),
        rms_mm=rms_mm,
        charuco_corners=charuco_corners,
        charuco_ids=charuco_ids,
    )


def estimate_board_pose(detection, board_cfg, camera_matrix, dist_coeffs):
    """
    카메라 내부 파라미터가 있을 때, solvePnP로 보드까지의 거리(mm, z축 방향)를
    추정해서 detection.board_distance_mm에 채운다. 실패하면 조용히 None을 유지한다.
    """
    if camera_matrix is None or dist_coeffs is None:
        return
    try:
        obj_points = np.array(
            [
                [
                    charuco_id_to_mm(int(cid), board_cfg["squares_x"], board_cfg["square_length_mm"])[0],
                    charuco_id_to_mm(int(cid), board_cfg["squares_x"], board_cfg["square_length_mm"])[1],
                    0.0,
                ]
                for cid in detection.charuco_ids.flatten()
            ],
            dtype=np.float64,
        )
        img_points = detection.charuco_corners.reshape(-1, 2).astype(np.float64)
        ok, _rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
        if ok:
            detection.board_distance_mm = float(tvec[2][0])
    except cv2.error:
        detection.board_distance_mm = None


# ---------------------------------------------------------------------------
# 카메라 내부 파라미터(매트릭스/왜곡계수) 캘리브레이션
# ---------------------------------------------------------------------------
def calibrate_intrinsics(gray_images, board_cfg):
    """
    여러 장의 보드 사진(그레이스케일 리스트)으로 camera_matrix, dist_coeffs를 구한다.
    반환: dict(camera_matrix, dist_coeffs, rms, image_size, num_images_used) 또는
    사용 가능한 이미지가 2장 미만이면 RuntimeError.
    """
    check_aruco_available()
    aruco_dict = get_dictionary(board_cfg["dictionary"])
    board = make_board(board_cfg, aruco_dict)
    params = make_detector_params()

    all_corners = []
    all_ids = []
    image_size = None

    for gray in gray_images:
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        marker_corners, marker_ids, _ = detect_markers(gray, aruco_dict, params)
        if marker_ids is None or len(marker_ids) == 0:
            continue
        marker_corners, marker_ids, _dup = dedupe_markers(marker_corners, marker_ids)
        charuco_corners, charuco_ids = interpolate_charuco(marker_corners, marker_ids, gray, board)
        if charuco_corners is None or len(charuco_corners) < 4:
            continue
        all_corners.append(charuco_corners)
        all_ids.append(charuco_ids)

    if len(all_corners) < 2:
        raise RuntimeError(
            "카메라 캘리브레이션에 쓸 수 있는 유효한 보드 검출이 %d장뿐입니다 (최소 2장 필요).\n"
            "보드가 화면에 잘 나오도록 여러 각도에서 다시 촬영하세요." % len(all_corners)
        )

    if hasattr(cv2.aruco, "calibrateCameraCharuco"):
        rms, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.aruco.calibrateCameraCharuco(
            all_corners, all_ids, board, image_size, None, None
        )
    else:
        # 신버전 API: board.matchImagePoints()로 각 프레임의 3D-2D 대응점을 얻어
        # 일반 cv2.calibrateCamera()로 계산한다.
        object_points_list = []
        image_points_list = []
        for corners, ids in zip(all_corners, all_ids):
            obj_pts, img_pts = board.matchImagePoints(corners, ids)
            if obj_pts is None or len(obj_pts) < 4:
                continue
            object_points_list.append(obj_pts)
            image_points_list.append(img_pts)
        if len(object_points_list) < 2:
            raise RuntimeError("신버전 aruco API에서 유효한 대응점을 만들지 못했습니다.")
        rms, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
            object_points_list, image_points_list, image_size, None, None
        )

    return {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.flatten().tolist(),
        "rms": float(rms),
        "image_size": list(image_size),
        "num_images_used": len(all_corners),
    }


# ---------------------------------------------------------------------------
# calib_front.json / calib_side.json 저장/로드
# ---------------------------------------------------------------------------
def save_calibration(path, camera_view, intrinsics, reference_detection, board_cfg, config):
    """
    camera_view: "front" 또는 "side"
    intrinsics: calibrate_intrinsics() 반환값
    reference_detection: 대표 1장에서 얻은 BoardDetection (기준 호모그래피 저장용)
    """
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    data = {
        "camera_view": camera_view,
        "board_spec": board_cfg,
        "camera_matrix": intrinsics["camera_matrix"],
        "dist_coeffs": intrinsics["dist_coeffs"],
        "intrinsics_rms": intrinsics["rms"],
        "image_size": intrinsics["image_size"],
        "num_images_used": intrinsics["num_images_used"],
        "reference_homography": reference_detection.homography.tolist(),
        "reference_rms_mm": reference_detection.rms_mm,
        "reference_num_markers": reference_detection.num_markers,
        "reference_num_corners": reference_detection.num_corners,
        "side_plane_offset_mm": config["calibration"]["side_plane_offset_mm"] if camera_view == "side" else None,
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data


def load_calibration(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calibration_camera_matrix(calib_data):
    if not calib_data or "camera_matrix" not in calib_data:
        return None, None
    camera_matrix = np.array(calib_data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(calib_data["dist_coeffs"], dtype=np.float64)
    return camera_matrix, dist_coeffs


def undistort(image_bgr, calib_data):
    """calib_data에 camera_matrix/dist_coeffs가 있으면 왜곡보정, 없으면 원본 그대로."""
    camera_matrix, dist_coeffs = calibration_camera_matrix(calib_data)
    if camera_matrix is None:
        return image_bgr
    return cv2.undistort(image_bgr, camera_matrix, dist_coeffs)


# ---------------------------------------------------------------------------
# 실시간 스케일 제공자 (측정 중 매 프레임 호출)
# ---------------------------------------------------------------------------
class ScaleProvider(object):
    """
    매 프레임 ChArUco 보드를 검출해서 실시간 호모그래피(mm 스케일)를 제공한다.

    우선순위:
        1. 이번 프레임에서 보드가 검출됨 (마커 min_markers_for_live_scale개 이상)
           -> 실시간 스케일 사용 ("live"), 측면이면 평면 오프셋 보정 적용
        2. 검출 실패, 최근 성공한 스케일이 scale_hold_frames 이내
           -> 마지막 성공값 유지 ("held"), 화면에 경고 표시할 것
        3. 그것도 없으면 저장된 calib_*.json의 reference_homography로 폴백 ("fallback")
        4. 그마저 없으면 None, "none" (스케일을 전혀 알 수 없음 - 측정 불가)
    """

    def __init__(self, calib_data, board_cfg, config, camera_view="front"):
        self.calib_data = calib_data
        self.board_cfg = board_cfg
        self.config = config
        self.camera_view = camera_view
        self.camera_matrix, self.dist_coeffs = calibration_camera_matrix(calib_data)
        self.plane_offset_mm = (
            config["calibration"]["side_plane_offset_mm"] if camera_view == "side" else 0.0
        )
        self._last_good_homography = None
        self._frames_since_good = 0

    def update(self, gray):
        """반환: (homography_or_None, status, detection_or_None)"""
        cal_cfg = self.config["calibration"]
        detection = detect_board(gray, self.board_cfg)

        if detection is not None and detection.num_markers >= cal_cfg["min_markers_for_live_scale"]:
            homography = detection.homography
            if self.plane_offset_mm and self.camera_matrix is not None:
                estimate_board_pose(detection, self.board_cfg, self.camera_matrix, self.dist_coeffs)
                homography = self._apply_plane_offset(homography, detection)
            self._last_good_homography = homography
            self._frames_since_good = 0
            return homography, "live", detection

        hold_frames = cal_cfg["scale_hold_frames"]
        if self._last_good_homography is not None and self._frames_since_good < hold_frames:
            self._frames_since_good += 1
            return self._last_good_homography, "held", detection

        if self.calib_data and "reference_homography" in self.calib_data:
            return np.array(self.calib_data["reference_homography"], dtype=np.float64), "fallback", detection

        return None, "none", detection

    def _apply_plane_offset(self, homography, detection):
        """
        보드 평면과 손가락 평면 사이 거리 차이(side_plane_offset_mm)를 반영해서
        스케일을 보정한다. 스케일 ∝ 1/거리이므로:
            보정계수 = board_distance_mm / (board_distance_mm + side_plane_offset_mm)
        board_distance_mm(solvePnP)을 못 구했으면 보정 없이 그대로 반환한다.
        """
        board_distance = detection.board_distance_mm
        if not board_distance:
            return homography
        correction = board_distance / (board_distance + self.plane_offset_mm)
        return homography * correction


def transform_points_to_mm(homography, points):
    """이미지 픽셀 좌표 리스트를 호모그래피로 mm 좌표로 변환한다."""
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, homography)
    return transformed.reshape(-1, 2)


def mm_distance(homography, point_a_px, point_b_px):
    mm_pts = transform_points_to_mm(homography, [point_a_px, point_b_px])
    return float(math.hypot(mm_pts[0][0] - mm_pts[1][0], mm_pts[0][1] - mm_pts[1][1]))
